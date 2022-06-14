import os
import sys
sys.path.insert(0, 'src')

import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as ddp

from dlhammer.bootstrap import bootstrap, logger
from dlhammer import distributed as du
from dlhammer.env import setup_random_seed
from dlhammer.checkpointer import setup_checkpointer

from factory import (build_data_loader, build_model, build_criterion,
                     build_trainer, build_optimizer, build_scheduler)


def main(local_rank):
    cfg = bootstrap(print_cfg=False)

    # Setup ddp
    if cfg.DDP.ENABLE:
        cfg.DDP.LOCAL_RANK = local_rank
        du.init_process_group(local_rank,
                              cfg.DDP.NUM_GPUS,
                              cfg.DDP.SHARD_ID,
                              cfg.DDP.NUM_SHARDS,
                              init_method='env://',
                              dist_backend='nccl')
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Setup seed
    setup_random_seed(cfg.SEED)

    # Setup checkpointer
    checkpointer = setup_checkpointer(cfg, phase='train')

    # Build data loaders
    data_loaders = {
        phase: build_data_loader(cfg, phase)
        for phase in cfg.SOLVER.PHASES
    }

    # Build model
    model = build_model(cfg, device)

    # Build criterion
    criterion = build_criterion(cfg, device)

    # Build optimizer
    optimizer = build_optimizer(cfg, model)

    # Load pretrained model and optimizer
    checkpointer.load(model, optimizer)

    # Build scheduler
    scheduler = build_scheduler(
        cfg, optimizer, len(data_loaders['train']))

    if cfg.DDP.ENABLE:
        model = ddp(model, device_ids=[local_rank])

    ####################################################################
    # START TRAINING
    ####################################################################
    trainer = build_trainer(cfg)(
        cfg,
        data_loaders,
        model,
        criterion,
        optimizer,
        scheduler,
        checkpointer,
        device,
        local_rank,
        logger)
    trainer.run()


if __name__ == '__main__':
    cfg = bootstrap(print_cfg=True)

    if cfg.DDP.ENABLE:
        local_world_size = cfg.DDP.NUM_GPUS
        os.environ['MASTER_ADDR'] = str(cfg.DDP.MASTER_ADDR)
        os.environ['MASTER_PORT'] = str(cfg.DDP.MASTER_PORT)
        mp.spawn(main, nprocs=local_world_size, join=True)
    else:
        main(0)
