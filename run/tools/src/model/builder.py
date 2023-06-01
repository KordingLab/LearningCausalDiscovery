from ..utils.config import Registry, build_from_cfg

ARCHS = Registry("arch")
LOSSES = Registry("loss")
METRICS = Registry("metric")


def build_arch(cfg):
    """build arch with given config"""
    return build_from_cfg(cfg, ARCHS)


def build_loss(cfg):
    """build loss with given config"""
    return build_from_cfg(cfg, LOSSES)


def build_metric(cfg):
    """build metric with given config"""
    return build_from_cfg(cfg, METRICS)
