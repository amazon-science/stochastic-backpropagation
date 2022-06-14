import os

import torch
import torch.multiprocessing as mp

from dlhammer import distributed as du
from dlhammer.bootstrap import bootstrap, logger
from dlhammer.checkpointer import setup_checkpointer
from dlhammer.criterions import build_criterion
from dlhammer.env import setup_random_seed


def main(local_rank):
    cfg = bootstrap(print_cfg=False)

    if cfg.TASK == "oad":
        from lstr.builder import (build_data_loaders, build_models,
                                  build_optimizers, build_schedulers,
                                  get_trainer_class)
        from lstr.utils.parser import postprocess_cfg

        cfg = postprocess_cfg(cfg)
    else:
        raise ValueError("Task: {cfg.TASK} is not supported yet")

    # init ddp
    if cfg.DDP.ENABLE:
        cfg.DDP.LOCAL_RANK = local_rank
        du.init_process_group(
            local_rank,
            cfg.DDP.NUM_GPUS,
            cfg.DDP.SHARD_ID,
            cfg.DDP.NUM_SHARDS,
            init_method="env://",
            dist_backend="nccl",
        )
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_random_seed(cfg.SEED)

    # %%
    # build dataloader
    data_loaders = build_data_loaders(cfg)

    # build model
    models = build_models(cfg, device, local_rank)

    # build optimizers
    optimizers = build_optimizers(cfg, models)

    # build checkpointer
    checkpointer = setup_checkpointer(cfg, phase="train")
    checkpointer.load(models, optimizers)

    # build criterion
    criterions = build_criterion(cfg, device)

    # build schedulers
    schedulers = build_schedulers(cfg, data_loaders, optimizers)

    # train
    trainer = get_trainer_class(cfg)
    trainer(
        cfg,
        data_loaders,
        models,
        criterions,
        optimizers,
        schedulers,
        device,
        checkpointer,
        logger,
    )


if __name__ == "__main__":
    cfg = bootstrap(print_cfg=True)

    if cfg.DDP.ENABLE:
        local_world_size = cfg.DDP.NUM_GPUS
        os.environ["MASTER_ADDR"] = str(cfg.DDP.MASTER_ADDR)
        os.environ["MASTER_PORT"] = str(cfg.DDP.MASTER_PORT)
        mp.spawn(main, nprocs=local_world_size, join=True)
    else:
        main(0)
