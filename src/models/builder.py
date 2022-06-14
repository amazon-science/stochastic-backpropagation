from dlhammer.registry import Registry

MODELS = Registry()


def build_model(cfg):
    return MODELS[cfg.MODEL.NAME](cfg)
