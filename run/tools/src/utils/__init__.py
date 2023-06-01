from .config import build_from_cfg, Registry, Config


# load config from config file
def load_config(cfg_path=None):
    import os
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"{cfg_path} not existed!")
    cfg = Config.fromfile(
        cfg_path,
    )
    return cfg


__all__ = ["build_from_cfg", "Registry"]
