__all__ = [
    "build_data_loaders",
    "build_models",
    "build_optimizers",
    "build_schedulers",
    "get_trainer_class",
]

import torch.optim as optim
from easydict import EasyDict
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils import data

from dlhammer.lr_schedulers import build_scheduler

from .datasets import build_dataset
from .models import build_model
from .models.video_feature_extractor import FeatureExtractor
from .trainer.e2e_perframe_det_trainer import do_e2e_perframe_det_train
from .utils.model_util import freeze_bn_parameters


def build_data_loader(cfg, dataset, phase):

    shuffle = True if phase == "train" else False

    if cfg.DDP.ENABLE:
        sampler = data.distributed.DistributedSampler(
            dataset, shuffle=shuffle, drop_last=True
        )
        shuffle = False
        batch_size = cfg.DATA_LOADER[f"{phase.upper()}_BATCH_SIZE"] // (
            cfg.DDP.NUM_GPUS * cfg.DDP.NUM_SHARDS
        )
    else:
        sampler = None
        batch_size = cfg.DATA_LOADER[f"{phase.upper()}_BATCH_SIZE"]

    data_loader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        sampler=sampler,
    )
    return data_loader


def build_data_loaders(cfg):
    tag = ""
    datasets = {phase: build_dataset(cfg, phase, tag) for phase in cfg.SOLVER.PHASES}
    data_loaders = {
        phase: build_data_loader(cfg, dataset, phase)
        for phase, dataset in datasets.items()
    }
    return data_loaders


def build_models(cfg, device, local_rank):
    models = EasyDict()

    models.spatial_model = FeatureExtractor(cfg.MODEL.SPATIAL, pooling=True)
    models.temporal_model = build_model(cfg)
    if cfg.MODEL.SPATIAL.FREEZE_BN:
        freeze_bn_parameters(models.spatial_model)

    ## to cuda
    models = EasyDict({name: model.to(device) for name, model in models.items()})
    if cfg.DDP.ENABLE and local_rank >= 0:
        models = EasyDict(
            {
                name: DDP(model, device_ids=[local_rank])
                for name, model in models.items()
            }
        )
    return models


def build_optimizer(optim_cfg, model):
    if optim_cfg.OPTIMIZER == "sgd":
        optimizer = optim.SGD(
            [{"params": model.parameters(), "initial_lr": optim_cfg.BASE_LR}],
            lr=optim_cfg.BASE_LR,
            weight_decay=optim_cfg.WEIGHT_DECAY,
            momentum=optim_cfg.MOMENTUM,
        )
    elif optim_cfg.OPTIMIZER == "adam":
        optimizer = optim.Adam(
            [{"params": model.parameters(), "initial_lr": optim_cfg.BASE_LR}],
            lr=optim_cfg.BASE_LR,
            weight_decay=optim_cfg.WEIGHT_DECAY,
        )
    elif optim_cfg.OPTIMIZER == "adamw":
        optimizer = optim.AdamW(
            [{"params": model.parameters(), "initial_lr": optim_cfg.BASE_LR}],
            lr=optim_cfg.BASE_LR,
            weight_decay=optim_cfg.WEIGHT_DECAY,
        )
    else:
        raise RuntimeError("Unknown optimizer: {}".format(optim_cfg.OPTIMIZER))
    return optimizer


def build_optimizers(cfg, models):
    optimizers = EasyDict()
    optimizers.spatial_model = build_optimizer(cfg.SOLVER.SPATIAL, models.spatial_model)
    optimizers.temporal_model = build_optimizer(
        cfg.SOLVER.TEMPORAL, models.temporal_model
    )
    return optimizers


def build_schedulers(cfg, data_loaders, optimizers):
    schedulers = EasyDict()
    if "train" in cfg.SOLVER.PHASES:
        schedulers.spatial_model = build_scheduler(
            optimizers.spatial_model,
            cfg.SOLVER.SPATIAL.SCHEDULER,
            cfg.SOLVER.START_EPOCH,
            cfg.SOLVER.NUM_EPOCHS,
            len(data_loaders["train"]),
        )
        schedulers.temporal_model = build_scheduler(
            optimizers.temporal_model,
            cfg.SOLVER.TEMPORAL.SCHEDULER,
            cfg.SOLVER.START_EPOCH,
            cfg.SOLVER.NUM_EPOCHS,
            len(data_loaders["train"]),
        )
    return schedulers


def get_trainer_class(cfg):
    model_name = cfg.MODEL.MODEL_NAME
    if model_name == "E2E_LSTR":
        return do_e2e_perframe_det_train
    else:
        raise ValueError("Not Implemented")
