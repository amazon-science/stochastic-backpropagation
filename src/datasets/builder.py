import torch.utils.data as data

from dlhammer.registry import Registry

DATASETS = Registry()


def build_data_loader(cfg, phase):
    dataset = DATASETS[cfg.DATA.NAME](cfg, phase)

    shuffle = True if phase == 'train' else False

    if cfg.DDP.ENABLE:
        sampler = data.distributed.DistributedSampler(dataset, shuffle=shuffle, drop_last=True)
        shuffle = False
        batch_size = cfg.DATA_LOADER[f'{phase.upper()}_BATCH_SIZE'] // (cfg.DDP.NUM_GPUS * cfg.DDP.NUM_SHARDS)
    else:
        sampler = None
        batch_size = cfg.DATA_LOADER[f'{phase.upper()}_BATCH_SIZE']

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=cfg.DATA_LOADER.NUM_WORKERS,
                                  pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
                                  sampler=sampler)
    return data_loader
