import pytorch_lightning as pl
from ..utils import add_prefix
from ..builder import ARCHS
from .commons import PatchEmbeddingLayer, MeanPooler


@ARCHS.register_module()
class Base(pl.LightningModule):
    """
    Base architecture with patch embedding layer
    """
    def __init__(self, in_channels=None, embedding_size=None, hidden_size=None, patch_size=None, stride=None,
                 num_classes=2, input_length=None, dropout=0.1, input_norm=False,
                 losses=dict(type="TorchLoss", loss_name="CrossEntropyLoss", loss_weight=1.0),
                 **kwargs):
        super().__init__()
        self.name = "Transformer"
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        if embedding_size is None:
            self.embedding_size = hidden_size
        else:
            self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.input_length = input_length
        self.dropout = dropout
        self.patch_size = patch_size
        self.stride = stride
        self.input_norm = input_norm
        self.losses = losses

        # build embedding layer
        self.embedding = PatchEmbeddingLayer(
            in_channels=self.in_channels,
            input_length=self.input_length,
            embedding_size=self.embedding_size,
            input_norm=self.input_norm,
            patch_size=self.patch_size,
            stride=self.stride,
        )
        # build backbone

        # build decode head
        self.head = MeanPooler(in_channels=self.embedding_size, num_classes=self.num_classes, losses=losses)

    def exact_feat(self, x):
        x = self.embedding(x)
        x = self.backbone(x)
        return x

    def forward_train(self, x, label):
        loss = dict()
        feat = self.exact_feat(x)

        decode_loss = self.head.forward_train(feat, label)
        loss.update(add_prefix(f"mainhead", decode_loss))

        # sum up all losses
        loss.update(
            {"loss": sum([loss[k] for k in loss.keys() if "loss" in k.lower()])}
        )

        # pack the output and losses
        return loss

    def forward_test(self, x, label=None):
        feat = self.exact_feat(x)
        res = self.head.forward_test(feat, label)

        # sum up all losses
        if label is not None:
            res.update(
                {"loss": sum([res[k] for k in res.keys() if "loss" in k.lower()])}
            )
        else:
            res.update({"loss": "Not available"})
        return res
