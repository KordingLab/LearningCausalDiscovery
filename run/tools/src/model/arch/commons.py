import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
from ..builder import build_loss


class PatchEmbeddingLayer(nn.Module):
    """
    PatchEmbedding is a module that takes a sequence and returns an embedding of its patches.
    """
    def __init__(
            self,
            in_channels=2,
            input_length=512,
            embedding_size=128,
            patch_size=8,
            stride=None,
            input_norm=False,
    ):
        super().__init__()
        self.input_length = input_length
        self.embedding_size = embedding_size
        if stride is None:
            stride = patch_size
        else:
            assert stride > 0, "stride must be positive"
        # 1d patch embedding
        self.projection = nn.Sequential(
            Rearrange(f"b l c -> b c l"),
            nn.Conv1d(
                in_channels,
                embedding_size,
                kernel_size=patch_size,
                stride=stride
            ),
            Rearrange(f"b c l -> b l c"),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_size))
        if stride == patch_size:
            self.positions = nn.Parameter(
                torch.randn((input_length // patch_size) + 1, embedding_size)
            )
        else:
            self.positions = nn.Parameter(
                torch.randn(
                    ((input_length - patch_size) // stride + 1) + 1, embedding_size
                )
            )

        # input batch norm
        if input_norm:
            self.input_norm = nn.Sequential(
                Rearrange(f"b l c -> b c l"),
                nn.BatchNorm1d(num_features=2),
                Rearrange(f"b c l -> b l c")
                )
        else:
            self.input_norm = nn.Identity()

    def forward(self, x):
        x = self.input_norm(x)
        b = x.shape[0]
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, "() n e -> b n e", b=b)
        # add the cls token to the beginning of the embedding
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.positions
        return x


class LinearHead(nn.Module):
    def __init__(self, losses=None, in_channels=512, num_classes=2):
        super().__init__()
        assert losses is not None, "losses must be specified"
        self.losses = []
        self.in_channels = in_channels
        self.num_classes = num_classes
        # build loss
        if isinstance(losses, dict):
            self.losses.append(build_loss(losses))
        elif isinstance(losses, list):
            for loss in losses:
                self.losses.append(build_loss(loss))
        else:
            raise TypeError(
                f"losses must be a dict or sequence of dict,\
                               but got {type(losses)}"
            )

        self.model = nn.Sequential(nn.LayerNorm(self.in_channels),
                                   nn.Linear(self.in_channels, self.num_classes))

    def forward(self, x):
        return self.model(x)

    def forward_train(self, input, label):
        """forward for training"""
        output = self.forward(input)
        losses = self.parse_losses(output, label)

        return losses

    def forward_test(self, input, label=None):
        """forward for testing"""
        output = self.forward(input)
        if label is not None:
            losses = self.parse_losses(output, label)
            return {**{"output": output}, **losses}
        else:
            return {"output": output}

    def parse_losses(self, pred, label):
        loss = dict()
        for _loss in self.losses:
            if _loss.loss_name not in loss:
                loss[_loss.loss_name] = _loss(pred, label) * _loss.loss_weight
            else:
                loss[_loss.loss_name] += _loss(pred, label) * _loss.loss_weight

        return loss


class MeanPooler(LinearHead):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def forward(self, x):
        return self.model(x.mean(dim=1))


class AttentionPooler(LinearHead):
    def __init__(self, dropout=0.3, **kwargs):
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.attention = nn.Sequential(
                        nn.LayerNorm(self.in_channels),
                        nn.Linear(self.in_channels, self.in_channels, bias=False),
                        nn.GELU(),
                        nn.Linear(self.in_channels, 1),
        )

    def forward(self, x):
        x = self.dropout(x)
        w = self.attention(x).float()
        w = torch.softmax(w, 1)
        attention_embeddings = torch.sum(w * x, dim=1)
        return self.model(attention_embeddings)


class DecoderHead(LinearHead):
    def __init__(self, rec_error_threshold=0, **kwargs):
        super().__init__(**kwargs)
        self.rec_error_threshold = rec_error_threshold

    def forward(self, x):
        return self.model(x)

    def forward_train(self, input, label):
        """forward for training"""
        output = self.forward(input)
        losses = self.parse_losses(output, label)

        return losses

    def forward_test(self, input, label):
        """forward for testing"""
        output = self.forward(input)

        # reconstruction error
        losses = self.parse_losses(output.reshape(-1), label.reshape(-1))
        output = ((label - output) ** 2).mean(axis=-1).mean(axis=-1)
        output = (output - self.rec_error_threshold).float()
        output = torch.where(output > 0, output, torch.zeros_like(output))
        return {**{"output": output}, **losses}