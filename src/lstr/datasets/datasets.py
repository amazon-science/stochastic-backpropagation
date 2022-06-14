from dlhammer.registry import Registry

DATA_LAYERS = Registry()


def build_dataset(cfg, phase, tag=''):
    data_layer = DATA_LAYERS[cfg.MODEL.MODEL_NAME + tag + cfg.DATA.DATA_NAME]
    return data_layer(cfg, phase)
