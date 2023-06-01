import os
import ast
import sys
import torch
import warnings
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning.callbacks as plc

# ugly hack to enable configs inside the package to be run
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from tools.src.model import ModelInterface
from tools.src.data import DataInterface
from argparse import ArgumentParser
from tools.src.utils import load_config
from tools.src.model.utils import OptimizerResumeHook

torch.multiprocessing.set_sharing_strategy("file_system")


def train():
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument("--cfg", type=str, help="config file path")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--gpu_ids", default="0", type=str)
    args = parser.parse_args()

    # device setting
    args.accelerator = "auto"
    # training setting for distributed training
    # args.gpu = '[0, 1, 2, 3]'
    args.gpus = ast.literal_eval(args.gpu_ids)
    if isinstance(args.gpus, int):
        args.gpus = [args.gpus]
    args.devices = len(args.gpus)
    if args.devices > 1:
        args.sync_batchnorm = True
        args.strategy = "ddp"

    # load model and data config settings from config file
    cfg = load_config(args.cfg)

    # fix random seed (seed in config has higher priority )
    seed = cfg.seed if cfg.get("seed", None) is not None else args.seed
    pl.seed_everything(seed)

    # ignore warning
    if not cfg.get("warning", True):
        warnings.filterwarnings("ignore")

    # save config file to log directory
    cfg.base_name = args.cfg.split("/")[-1]
    if os.path.exists(os.path.join(cfg.log.work_dir, cfg.log.exp_name)):
        cfg.dump(os.path.join(cfg.log.work_dir, cfg.log.exp_name, cfg.base_name))
    else:
        os.makedirs(os.path.join(cfg.log.work_dir, cfg.log.exp_name))
        cfg.dump(os.path.join(cfg.log.work_dir, cfg.log.exp_name, cfg.base_name))

    # 5 part: model(arch, loss, ) -> ModelInterface /data(file i/o, preprocess pipeline) -> DataInterface
    # /optimization(optimizer, scheduler, epoch/iter...) -> ModelInterface/
    # log(logger, checkpoint, work_dir) -> Trainer /other

    # other setting
    if cfg.get("cudnn_benchmark", True):
        args.benchmark = True

    if cfg.get("deterministic", False):
        if cfg.get("cudnn_benchmark", True):
            print("cudnn_benchmark will be disabled")
        args.deterministic = True
        args.benchmark = False

    # data
    data_module = DataInterface(cfg.data)
    # set ddp sampler
    if args.devices > 1:
        # if cfg.data.train.get('sampler', None) is None:
        args.replace_sampler_ddp = True

    # optimization
    if cfg.optimization.type == "epoch":
        args.max_epochs = cfg.optimization.max_iters
    elif cfg.optimization.type == "iter":
        args.max_steps = cfg.optimization.max_iters
    else:
        raise NotImplementedError(
            "You must choose optimziation update step from (epoch/iter)"
        )

    # for models need setting readout layer with dataloader informatios
    if cfg.model.get("archs", None) is not None:
        for arch in cfg.model.archs:
            if arch.pop("need_dataloader", False):
                data_module.setup(stage="fit")
                arch.dataloader = data_module.train_dataloader()
    else:
        if cfg.model.pop("need_dataloader", False):
            data_module.setup(stage="fit")
            cfg.model.dataloader = data_module.train_dataloader()

    if cfg.get("resume_from", None) is None:
        model = ModelInterface(cfg.model, cfg.optimization)
    else:
        model = ModelInterface.load_from_checkpoint(checkpoint_path=cfg.resume_from)

    # log
    # callbacks
    callbacks = []
    if cfg.get("resume_from", None) is not None:
        callbacks.append(OptimizerResumeHook())

    # accumulation of gradients
    if cfg.optimization.get("accumulation_steps", 1) != 1:
        if isinstance(cfg.optimization.accumulation_steps, int):
            callbacks.append(
                plc.GradientAccumulationScheduler(
                    scheduling={0: cfg.optimization.accumulation_steps}
                )
            )
        else:
            # dict of scheduling {epoch: accumulation_steps, ...}
            callbacks.append(
                plc.GradientAccumulationScheduler(
                    scheduling=cfg.optimization.accumulation_steps
                )
            )

    # used to control early stopping
    if cfg.log.get("earlystopping", None) is not None:
        callbacks.append(
            plc.EarlyStopping(
                monitor=cfg.log.get("monitor", "val_loss"),
                mode=cfg.log.earlystopping.get("mode", "max"),
                strict=cfg.log.earlystopping.get("strict", False),
                patience=cfg.log.earlystopping.get("patience", 5),
                min_delta=cfg.log.earlystopping.get("min_delta", 1e-5),
                check_finite=cfg.log.earlystopping.get("check_finite", True),
                verbose=cfg.log.earlystopping.get("verbose", True),
            )
        )
    # used to save the best model
    if cfg.log.checkpoint is not None:
        if cfg.log.checkpoint.type == "ModelCheckpoint":
            dirpath = cfg.log.checkpoint.get(
                "dirpath", os.path.join(cfg.log.work_dir, cfg.log.exp_name, "ckpts")
            )
            filename = cfg.log.checkpoint.get(
                "filename",
                f"exp_name={cfg.log.exp_name}-" + f'cfg={cfg.base_name.strip(".py")}-' +
                f"bs={cfg.data.train_batch_size}-" + f"seed={seed}-" + f"{{{cfg.log.monitor}:.4f}}",
            )
            callbacks.append(
                plc.ModelCheckpoint(
                    monitor=cfg.log.monitor,
                    dirpath=dirpath,
                    filename=filename,
                    save_top_k=cfg.log.checkpoint.top_k,
                    mode=cfg.log.checkpoint.mode,
                    verbose=cfg.log.checkpoint.verbose,
                    save_last=cfg.log.checkpoint.save_last,
                )
            )
        else:
            raise NotImplementedError("Other kind of checkpoints haven't implemented!")

    # Disable ProgressBar
    # callbacks.append(plc.progress.TQDMProgressBar(
    #     refresh_rate=0,
    # ))

    args.callbacks = callbacks

    # logger
    if cfg.log.get("logger", False):
        save_dir = os.path.join(cfg.log.work_dir, cfg.log.exp_name, "log")
        args.log_every_n_steps = cfg.log.logger_interval
        cfg.log.exp_name = cfg.log.exp_name
        args.logger = [WandbLogger(name=cfg.log.exp_name, project=cfg.log.project_name, save_dir=save_dir)]

    # load trainer
    trainer = Trainer.from_argparse_args(args)

    trainer.fit(model, data_module, ckpt_path=cfg.get("resume_from", None))

    trainer.test(model, data_module)


if __name__ == "__main__":
    train()
