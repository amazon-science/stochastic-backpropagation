__all__ = ["build_model"]

from dlhammer.registry import Registry
from dlhammer.weight_init import weights_init

META_ARCHITECTURES = Registry()


def build_model(cfg, device=None):
    model = META_ARCHITECTURES[cfg.MODEL.MODEL_NAME](cfg)
    model.apply(weights_init)
    return model.to(device)
