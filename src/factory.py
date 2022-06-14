def build_data_loader(cfg, phase):
    from datasets import build_data_loader
    data_loader = build_data_loader(cfg, phase)
    return data_loader


def build_model(cfg, device):
    from models import build_model
    model = build_model(cfg).to(device)
    return model


def build_criterion(cfg, device):
    from dlhammer.criterions import build_criterion
    criterion = build_criterion(cfg, device)
    return criterion


def build_optimizer(cfg, model):
    from dlhammer.optimizers import build_optimizer
    optimizer = build_optimizer(model, cfg.SOLVER.OPTIMIZER)
    return optimizer


def build_scheduler(cfg, optimizers, num_iters_per_epoch):
    from dlhammer.lr_schedulers import build_scheduler
    scheduler = build_scheduler(optimizers,
                                cfg.SOLVER.SCHEDULER,
                                cfg.SOLVER.START_EPOCH,
                                cfg.SOLVER.NUM_EPOCHS,
                                num_iters_per_epoch)
    return scheduler


def build_trainer(cfg):
    from engines import build_trainer
    trainer = build_trainer(cfg)
    return trainer
