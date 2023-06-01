from copy import deepcopy
import torch
import pytorch_lightning as pl
from tools.src.model.builder import (
    build_arch,
    build_metric,
)

import math
import torch.optim.lr_scheduler as lr_scheduler

class ModelInterface(pl.LightningModule):
    def __init__(self, model, optimization):
        super().__init__()
        self.save_hyperparameters()
        self.model = deepcopy(model)
        self.optimization = deepcopy(optimization)
        self.configure_metrics()
        self.model.pop("evaluation")
        self.model = build_arch(self.model.arch)

    def forward(self, x):
        # for testing
        return self.model.forward_test(x)["output"]

    def training_step(self, batch, batch_idx):
        input, label = batch
        output = self.model.forward_train(input, label)
        loss = output["loss"]

        # logging all output
        for name, value in output.items():
            if name != "loss":
                self.log(
                    f"train_{name}", value, on_step=True, on_epoch=True, prog_bar=False
                )
        # logging lr
        for opt in self.trainer.optimizers:
            # dirty hack to get the name of the optimizer
            self.log(
                f"lr-{type(opt).__name__}",
                opt.param_groups[0]["lr"],
                prog_bar=True,
                on_step=True,
                on_epoch=False,
            )
        # logging loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input, label = batch
        output = self.model.forward_test(input, label)

        # logging all output
        for name, value in output.items():
            if name != "output" and name != "loss":
                self.log(
                    f"val_{name}", value, on_step=True, on_epoch=True, prog_bar=False
                )
        # logging loss
        self.log("val_loss", output["loss"], on_step=True, on_epoch=True, prog_bar=True)
        return {"pred": output["output"], "label": label}

    def validation_epoch_end(self, validation_step_outputs):
        # [val_step_output1, val_step_output2, ...]
        pred = torch.cat([out["pred"] for out in validation_step_outputs], dim=0)
        label = torch.cat([out["label"] for out in validation_step_outputs], dim=0)

        # Report Metrics
        for metric in self.metrics:
            res = metric(pred, label)
            self.log(
                f"val_{metric.metric_name}",
                res,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

    def test_step(self, batch, batch_idx):
        input, label = batch
        output = self.model.forward_test(input, label)
        # logging all output
        for name, value in output.items():
            if name != "output" and name != "loss":
                self.log(
                    f"test_{name}", value, on_step=True, on_epoch=True, prog_bar=False
                )
        # logging loss
        self.log(
            "test_loss", output["loss"], on_step=True, on_epoch=True, prog_bar=True
        )
        return {"pred": output["output"], "label": label}

    def test_epoch_end(self, test_step_outputs):
        # [val_step_output1, val_step_output2, ...]
        pred = torch.cat([out["pred"] for out in test_step_outputs], dim=0)
        label = torch.cat([out["label"] for out in test_step_outputs], dim=0)

        # Report Metrics
        for metric in self.metrics:
            res = metric(pred, label)
            self.log(
                f"test_{metric.metric_name}",
                res,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

    def configure_optimizers(self):
        # optimizer
        optimizer = getattr(torch.optim, self.optimization.optimizer.pop('type'))
        optimizer = optimizer(self.model.parameters(), lr=self.optimization.optimizer.lr,
                              weight_decay=self.optimization.optimizer.weight_decay)

        # scheduler
        if self.optimization.scheduler.get('warmup_steps', None) is None:
            warmup_steps = 5*len(self.trainer.datamodule.train_dataloader())
        else:
            warmup_steps = self.optimization.scheduler.warmup_steps * len(self.trainer.datamodule.train_dataloader())

        warmup_scheduler = LinearWarmupScheduler(optimizer,
                                                 warmup_steps=warmup_steps,
                                                 initial_lr=self.optimization.optimizer.get('initial_lr', 0),
                                                 max_lr=self.optimization.optimizer.lr,
                                                 T_max=self.num_max_steps,
                                                 eta_min=self.optimization.scheduler.min_lr)
        scheduler = {
            "interval": "step",
            "frequency": 1,
            "scheduler": warmup_scheduler,
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def configure_metrics(self):
        metrics = deepcopy(
            self.model.evaluation.get(
                "metrics", dict(type="TorchMetircs", metric_name="Accuracy")
            )
        )
        self.metrics = []
        if isinstance(metrics, list):
            for metric in metrics:
                self.metrics.append(build_metric(metric))
        elif isinstance(metrics, dict):
            self.metrics.append(build_metric(metrics))
        else:
            raise TypeError(
                f"Metrics must be a list or a dict, received {type(metrics)} type!"
            )

    @property
    def num_max_steps(self):
        # get max training steps inferred from datamodule and devices
        if self.trainer.max_steps > 0:
            return self.trainer.max_steps

        limit_batches = self.trainer.limit_train_batches
        batches = len(self.trainer.datamodule.train_dataloader())
        batches = (
            min(batches, limit_batches)
            if isinstance(limit_batches, int)
            else int(limit_batches * batches)
        )

        num_devices = max(1, self.trainer.num_devices)

        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        return (batches // effective_accum) * self.trainer.max_epochs


class LinearWarmupScheduler(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, initial_lr, max_lr, T_max, eta_min=0):
        self.warmup_steps = warmup_steps
        self.initial_lr = initial_lr
        self.max_lr = max_lr
        self.T_max = T_max
        self.eta_min = eta_min
        super(LinearWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch <= self.warmup_steps:
            return [self.initial_lr + (self.max_lr - self.initial_lr) * self.last_epoch / self.warmup_steps for _ in self.base_lrs]
        else:
            t = self.last_epoch - self.warmup_steps
            return [self.eta_min + (self.max_lr - self.eta_min) * (1 + math.cos(math.pi * t / self.T_max)) / 2 for _ in self.base_lrs]

