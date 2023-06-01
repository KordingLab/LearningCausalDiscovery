import torchmetrics
import pytorch_lightning as pl
from ..builder import METRICS
from ..utils import pascal_case_to_snake_case


@METRICS.register_module()
class TorchMetrics(pl.LightningModule):
    def __init__(
        self,
        metric_name=None,
        step_reduction=None,
        multi_label=False,
        prob=True,
        **kwargs
    ):
        super(TorchMetrics, self).__init__()
        if metric_name is None:
            raise ValueError("loss_name is required")
        assert step_reduction in ["mean", "sum", None]
        self.step_reduction = step_reduction
        self.prob = prob
        self.multi_label = multi_label
        self.metrics = getattr(torchmetrics, metric_name)(**kwargs)
        self.metric_name = pascal_case_to_snake_case(metric_name)

    def forward(self, pred, target):
        if self.prob:
            pred = pred.softmax(dim=-1)
        if self.step_reduction == "mean":
            pred = pred.mean(dim=1)
        elif self.step_reduction == "sum":
            pred = pred.sum(dim=1)
        # for binary classification (AUROC)
        if getattr(self.metrics, "pos_label", None) is not None:
            pred = pred[..., self.metrics.pos_label]
        # for multilabel classification
        if self.multi_label:
            pred = pred.view(-1)
            target = target.view(-1)
        return self.metrics(pred.detach().cpu(), target.detach().cpu())
