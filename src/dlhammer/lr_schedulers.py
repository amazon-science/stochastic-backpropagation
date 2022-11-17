import math
from bisect import bisect_right

from torch.optim.lr_scheduler import _LRScheduler


def _get_warmup_factor_at_iter(warmup_method,
                               this_iter,
                               warmup_iters,
                               warmup_factor):
    if this_iter >= warmup_iters:
        return 1.0

    if warmup_method == 'constant':
        return warmup_factor
    elif warmup_method == 'linear':
        alpha = this_iter / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    else:
        raise ValueError('Unknown warmup method: {}'.format(warmup_method))


class MultiStepLR(_LRScheduler):

    def __init__(self,
                 optimizer,
                 milestones,
                 gamma=0.1,
                 last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                'Milestones should be a list of increasing integers. Got {}'.format(milestones)
            )
        self.milestones = milestones
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            base_lr
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]

    def _compute_values(self):
        # The new interface
        return self.get_lr()


class WarmupMultiStepLR(_LRScheduler):

    def __init__(self,
                 optimizer,
                 milestones,
                 gamma=0.1,
                 warmup_factor=0.3,
                 warmup_iters=500,
                 warmup_method='linear',
                 last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                'Milestones should be a list of increasing integers. Got {}'.format(milestones)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = _get_warmup_factor_at_iter(
            self.warmup_method,
            self.last_epoch,
            self.warmup_iters,
            self.warmup_factor,
        )
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]

    def _compute_values(self):
        # The new interface
        return self.get_lr()


class CosineLR(_LRScheduler):

    def __init__(self,
                 optimizer,
                 max_iters,
                 last_epoch=-1):
        self.max_iters = max_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            base_lr
            * 0.5
            * (1.0 + math.cos(math.pi * self.last_epoch / self.max_iters))
            for base_lr in self.base_lrs
        ]

    def _compute_values(self):
        # The new interface
        return self.get_lr()


class WarmupCosineLR(_LRScheduler):

    def __init__(self,
                 optimizer,
                 max_iters,
                 warmup_factor=0.3,
                 warmup_iters=500,
                 warmup_method='linear',
                 last_epoch=-1):
        self.max_iters = max_iters
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = _get_warmup_factor_at_iter(
            self.warmup_method,
            self.last_epoch,
            self.warmup_iters,
            self.warmup_factor,
        )
        return [
            base_lr
            * warmup_factor
            * 0.5
            * (1.0 + math.cos(math.pi * self.last_epoch / self.max_iters))
            for base_lr in self.base_lrs
        ]

    def _compute_values(self):
        # The new interface
        return self.get_lr()


def build_scheduler(optimizer, scheduler_cfg, start_epoch, num_epochs, num_iters_per_epoch):
    """Unlike the PyTorch version, our schedulers adjust the learning rate
    according to iteration rather than epoch.
    """
    # Set last epoch (here 'epoch' is actually 'iters')
    last_epoch = (start_epoch - 1) * num_iters_per_epoch - 1

    if scheduler_cfg.NAME == 'multistep':
        # Convert milestones epochs to iters
        milestones = [(i - 1) * num_iters_per_epoch for i in scheduler_cfg.MILESTONES]

        scheduler = MultiStepLR(optimizer,
                                milestones=milestones,
                                gamma=scheduler_cfg.GAMMA,
                                last_epoch=last_epoch)
    elif scheduler_cfg.NAME == 'warmup_multistep':
        # Convert milestones epochs to iters
        milestones = [(i - 1) * num_iters_per_epoch for i in scheduler_cfg.MILESTONES]

        # Convert warmup epochs to iters
        scheduler_cfg.WARMUP_ITERS = scheduler_cfg.WARMUP_EPOCHS * num_iters_per_epoch

        scheduler = WarmupMultiStepLR(optimizer,
                                      milestones=milestones,
                                      gamma=scheduler_cfg.GAMMA,
                                      warmup_factor=scheduler_cfg.WARMUP_FACTOR,
                                      warmup_iters=scheduler_cfg.WARMUP_ITERS,
                                      warmup_method=scheduler_cfg.WARMUP_METHOD,
                                      last_epoch=last_epoch)
    elif scheduler_cfg.NAME == 'cosine':
        # Get max number of iters
        max_iters = num_epochs * num_iters_per_epoch

        scheduler = CosineLR(optimizer,
                             max_iters=max_iters,
                             last_epoch=last_epoch)
    elif scheduler_cfg.NAME == 'warmup_cosine':
        # Get max number of iters
        max_iters = num_epochs * num_iters_per_epoch

        # Convert warmup epochs to iters
        scheduler_cfg.WARMUP_ITERS = scheduler_cfg.WARMUP_EPOCHS * num_iters_per_epoch

        scheduler = WarmupCosineLR(optimizer,
                                   max_iters=max_iters,
                                   warmup_factor=scheduler_cfg.WARMUP_FACTOR,
                                   warmup_iters=scheduler_cfg.WARMUP_ITERS,
                                   warmup_method=scheduler_cfg.WARMUP_METHOD,
                                   last_epoch=last_epoch)
    else:
        raise RuntimeError('Unknown lr scheduler: {}'.format(scheduler_cfg.NAME))
    return scheduler
