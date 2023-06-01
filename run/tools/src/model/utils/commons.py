import re
import torch
import torch.nn as nn


def add_prefix(prefix, name_dict, seperator="_"):
    return {prefix + seperator + key: value for key, value in name_dict.items()}


def pascal_case_to_snake_case(camel_case):
    # if input like 'AUROC':
    if camel_case.isupper():
        return camel_case.lower()
    else:
        snake_case = re.sub(r"(?P<key>[A-Z])", r"_\g<key>", camel_case)
        return snake_case.lower().strip("_")


def snake_case_to_pascal_case(snake_case):
    if snake_case.islower():
        return snake_case.title()
    else:
        words = snake_case.split("_")
        return "".join(word.title() for word in words)


def build_conv_layer(
    conv_cfg,
    in_channels,
    out_channels,
    kernel_size,
    norm_cfg="BatchNorm2d",
    act_cfg="ReLU",
    stride=1,
    padding=0,
    bias=True,
):
    conv = getattr(nn, conv_cfg)
    act = getattr(nn, act_cfg)
    norm = getattr(nn, norm_cfg)
    conv_layer = nn.Sequential(
        conv(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
        norm(num_features=out_channels),
        act(inplace=True),
    )
    return conv_layer


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class ResidualConcat(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        return torch.cat([x, res], dim=-1)
