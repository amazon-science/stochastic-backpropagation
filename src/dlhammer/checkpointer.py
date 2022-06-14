import os

import torch

from .bootstrap import logger


class Checkpointer(object):

    def __init__(self, cfg, phase):
        # Get checkpoint path
        self.checkpoint_path = cfg.MODEL.get('CHECKPOINT', '')
        if self.checkpoint_path != '' and not os.path.isfile(self.checkpoint_path):
            raise ValueError(f'Cannot find checkpoint: {self.checkpoint_path}')

        # Load checkpoint
        self.checkpoint = self._load_checkpoint()

        if self.checkpoint is not None:
            if 'train' in cfg.SOLVER.PHASES and phase == 'train':
                cfg.SOLVER.START_EPOCH = self.checkpoint.get('epoch', 0) + 1
            else:
                cfg.SOLVER.START_EPOCH = self.checkpoint.get('epoch', 0)
        elif self.checkpoint is None and phase != 'train':
            raise ValueError(f'Cannot find checkpoint {self.checkpoint_path}')
        else:
            cfg.SOLVER.START_EPOCH = 1

        self.output_dir = os.path.join(cfg.OUTPUT_DIR, 'ckpts')
        if phase == 'train':
            os.makedirs(self.output_dir, exist_ok=True)

    def load_models(self, models):
        if self.checkpoint is not None:
            if isinstance(models, dict):
                for name, model in models.items():
                    model.load_state_dict(
                        self.checkpoint[f'model_{name}_state_dict'])
            else:
                models.load_state_dict(self.checkpoint['model_state_dict'])
            logger.info(f'=> loaded models successfully from {self.checkpoint_path}')

    def load_optimizers(self, optimizers):
        if self.checkpoint is not None and optimizers is not None:
            if isinstance(optimizers, dict):
                for name, optimizer in optimizers.items():
                    optimizer.load_state_dict(
                        self.checkpoint[f'optimizer_{name}_state_dict'])
            else:
                optimizers.load_state_dict(self.checkpoint['optimizer_state_dict'])
            logger.info(f'=> loaded optimizers successfully from {self.checkpoint_path}')

    def load(self, models, optimizers=None):
        """
        Args:
            models (nn.Module, dict(nn.Module)): model or nested model.
            optimizers (Optimizer, dict(Optimizer)): optimizer or nested optimizer.
        """
        if self.checkpoint is not None:
            self.load_models(models)
            self.load_optimizers(optimizers)

    def save(self, epoch, models, optimizers):
        """
        Args:
            models (nn.Module, Dict(nn.Module)): model or nested model.
            optimizers (Optimizer, Dict(Optimizer)): optimizer or nested optimizer.
        """
        save_dict = {'epoch': epoch}

        # Save models
        get_model_state_dict = lambda model: model.module.state_dict(
        ) if torch.cuda.device_count() > 1 else model.state_dict()
        if isinstance(models, dict):
            save_dict.update({
                f'model_{name}_state_dict': get_model_state_dict(model)
                for name, model in models.items()
            })
        else:
            save_dict.update({'model_state_dict': get_model_state_dict(models)})

        # Save optimizers
        if isinstance(optimizers, dict):
            save_dict.update({
                f'optimizer_{name}_state_dict': optimizer.state_dict()
                for name, optimizer in optimizers.items()
            })
        else:
            save_dict.update({f'optimizer_state_dict': optimizers.state_dict()})

        # Do save
        torch.save(save_dict,
                   os.path.join(self.output_dir, 'epoch-{}.pth'.format(epoch)))

    def _load_checkpoint(self):
        if os.path.isfile(self.checkpoint_path):
            return torch.load(self.checkpoint_path, map_location=torch.device('cpu'))
        return None


def setup_checkpointer(cfg, phase):
    return Checkpointer(cfg, phase)
