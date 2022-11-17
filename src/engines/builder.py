from dlhammer.registry import Registry

TRAINERS = Registry()


def build_trainer(cfg):
    return TRAINERS[f'{cfg.TASK}']
