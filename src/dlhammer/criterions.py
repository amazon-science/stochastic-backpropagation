import torch
import torch.nn as nn

from .registry import Registry

CRITERIONS = Registry()


@CRITERIONS.register('BCE')
class BinaryCrossEntropyLoss(nn.Module):

    def __init__(self, reduction='mean', ignore_index=-100):
        super(BinaryCrossEntropyLoss, self).__init__()

        self.criterion = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, input, target):
        return self.criterion(input, target)


@CRITERIONS.register('SCE')
class SingleCrossEntropyLoss(nn.Module):

    def __init__(self, reduction='mean', ignore_index=-100):
        super(SingleCrossEntropyLoss, self).__init__()

        self.criterion = nn.CrossEntropyLoss(reduction=reduction, ignore_index=ignore_index)

    def forward(self, input, target):
        return self.criterion(input, target)


@CRITERIONS.register('MCE')
class MultipCrossEntropyLoss(nn.Module):
    """Implemented by (Xu et al., 2019) for multi-label action detection.
    """

    def __init__(self, reduction='mean', ignore_index=-100):
        super(MultipCrossEntropyLoss, self).__init__()

        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, input, target):
        logsoftmax = nn.LogSoftmax(dim=1).to(input.device)

        if self.ignore_index >= 0:
            notice_index = [i for i in range(target.shape[-1]) if i != self.ignore_index]
            output = torch.sum(-target[:, notice_index] * logsoftmax(input[:, notice_index]), dim=1)

            if self.reduction == 'mean':
                return torch.mean(output[target[:, self.ignore_index] != 1])
            elif self.reduction == 'sum':
                return torch.sum(output[target[:, self.ignore_index] != 1])
            else:
                return output[target[:, self.ignore_index] != 1]
        else:
            output = torch.sum(-target * logsoftmax(input), dim=1)

            if self.reduction == 'mean':
                return torch.mean(output)
            elif self.reduction == 'sum':
                return torch.sum(output)
            else:
                return output


def build_criterion(cfg, device=None):
    criterion = {}
    for name, params in cfg.MODEL.CRITERIONS:
        if name in CRITERIONS:
            if 'ignore_index' not in params and 'IGNORE_INDEX' in cfg.DATA:
                params['ignore_index'] = cfg.DATA.IGNORE_INDEX
            criterion[name] = CRITERIONS[name](**params).to(device)
        else:
            raise RuntimeError('Unknown criterion: {}'.format(name))
    return criterion
