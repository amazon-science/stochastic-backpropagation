import time
from tqdm import tqdm

import torch

from .nested import nested_call, nested_to_device


class BaseTrainer(object):
    """Base trainer"""

    def __init__(self,
                 cfg,
                 data_loaders,
                 model,
                 criterion,
                 optimizer,
                 scheduler,
                 checkpointer,
                 device,
                 local_rank=0,
                 logger=None):
        super(BaseTrainer, self).__init__()

        self.cfg = cfg
        self.data_loaders = data_loaders
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpointer = checkpointer
        self.device = device
        self.local_rank = local_rank
        self.logger = logger

        self.phase = None
        self.eval_meters = None
        self.loss_meters = None
        self.batch_logs = None
        self.epoch_logs = None

    def training(self):
        return True if self.phase == 'train' else False

    def reset_meters(self):
        if self.eval_meters is not None:
            nested_call(self.eval_meters, 'reset')
        if self.loss_meters is not None:
            nested_call(self.loss_meters, 'reset')

        self.batch_logs = {}
        self.epoch_logs = {
            phase: {} for phase in self.cfg.SOLVER.PHASES
        }

    def update_meters(self):
        total_meters = self.eval_meters[self.phase].compute()
        self.epoch_logs[self.phase].update({
            key: value for key, value in total_meters.items()
            if not key.startswith('no-print')
        })

    def reset_data_loader(self):
        for data_loader in self.data_loaders.values():
            if self.cfg.DDP.ENABLE and data_loader.sampler is not None:
                data_loader.sampler.set_epoch(self.epoch)

    def train_one_epoch(self):
        self.phase = 'train'

        nested_call(self.model, 'train')

        self.reset_data_loader()

        self.run_one_epoch()

    def test_one_epoch(self):
        self.phase = 'test'

        nested_call(self.model, 'eval')

        self.run_one_epoch()

    def run_one_epoch(self):
        step_func = self.train_step if self.training() else self.test_step

        start_time = time.time()
        with torch.set_grad_enabled(self.training()):
            if self.local_rank == 0:
                pbar = tqdm(self.data_loaders[self.phase],
                            desc=f'{self.phase.capitalize()}ing epoch {self.epoch}')
            else:
                pbar = self.data_loaders[self.phase]

            for batch_idx, batch in enumerate(pbar, start=1):
                batch = nested_to_device(batch, self.device, non_blocking=True)

                loss, batch_logs = step_func(batch)

                self.loss_meters[self.phase].update(loss.item())

                if self.local_rank == 0:
                    self.log_batch(pbar, loss, batch_logs)
        end_time = time.time()

        self.update_meters()

        self.log_epoch(end_time - start_time)

    def log_batch(self, pbar, loss, batch_logs):
        batch_logs['lr'] = f'{self.scheduler.get_last_lr()[0]:.7f}'
        batch_logs['loss'] = f'{loss.item():.5f}'
        pbar.set_postfix(batch_logs)

    def log_epoch(self, time_cost=0.0):
        epoch_loss = self.loss_meters[self.phase].compute()

        if self.local_rank == 0:
            log = f'Epoch {self.epoch:2d}'
            log += f' | {self.phase.capitalize()}'
            for key, value in self.epoch_logs[self.phase].items():
                log += f' {key}: {value}'
            log += f' | Loss: {epoch_loss:.5f}'
            log += f' | Time cost: {time_cost:.3f} sec'
            self.logger.info(log)

    def run(self, test_interval=1):
        for self.epoch in range(self.cfg.SOLVER.START_EPOCH, self.cfg.SOLVER.NUM_EPOCHS + 1):
            self.reset_meters()

            self.train_one_epoch()

            if 'test' in self.cfg.SOLVER.PHASES and self.epoch % test_interval == 0:
                self.test_one_epoch()

            self.checkpointer.save(self.epoch, self.model, self.optimizer)

    def train_step(self, batch):
        raise NotImplementedError

    def test_step(self, batch):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError
