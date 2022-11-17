from easydict import EasyDict

import torch.optim as optim


def to_lowercase_key(easydict):
    """Convert the key in the easydict to lowercase.
    Args:
        easydict (EasyDict): the easydict to be changed.
    Returns: Another EasyDict with lowercase key.
    """
    if not isinstance(easydict, EasyDict):
        raise TypeError(f'Only support EasyDict as input. Got {type(easydict)}')
    outputs = EasyDict()
    for k, element in easydict.items():
        if isinstance(k, str):
            new_k = k.lower()
        else:
            new_k = k
        outputs[new_k] = element
    return outputs


def build_optimizer(model, solver_cfg):
    solver_cfg = to_lowercase_key(solver_cfg)

    if solver_cfg.type.startswith('mmcv_'):
        from mmcv.runner import build_optimizer
        solver_cfg['type'] = solver_cfg['type'].replace('mmcv_', '')
        return build_optimizer(model, dict(solver_cfg))
    else:
        pytorch_optimizer_params = {
            'params': [
                {'params': model.parameters(), 'initial_lr': solver_cfg.lr}
            ]
        }
        pytorch_optimizer_params.update(solver_cfg)
        del pytorch_optimizer_params['type']
        return getattr(optim, solver_cfg.type)(**pytorch_optimizer_params)
