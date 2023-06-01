from ..utils.config import Registry, build_from_cfg

DATASETS = Registry("dataset")


def build_dataset(cfg):
    """build dataset with given config"""
    return build_from_cfg(cfg, DATASETS)

