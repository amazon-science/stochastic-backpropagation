from easydict import EasyDict

import torch


def map_structure(func, structure, args=(), kwargs={}):
    """Apply func to each element in structure.
    Args:
        func (callable): The function.
        structure (dict or list or tuple): the structure to be mapped.
    Kwargs:
        args (list or tuple): The args to the function.
        kwargs (dict): The kwargs to the function.
    Returns: The same structure as `structure`.
    """
    if structure is None:
        return None
    elif isinstance(structure, (list, tuple)):
        return [
            map_structure(element, *args, **kwargs)
            for element in structure
        ]
    elif isinstance(structure, dict):
        outputs = EasyDict()
        for k, element in structure.items():
            outputs[k] = map_structure(element, *args, **kwargs)
        return outputs
    else:
        return func(structure, *args, **kwargs)


def nested_call(structure, func_name, args=(), kwargs={}):
    """Call function for each element in nested structure.
    Args:
        structure (dict,tuple,list,object): If the structure is dict or list, then call func_name 
        for each values in the structure. If structure is None, then the function will do nothing.
        func_name (string): function to call.
    Kwargs:
        args (list or tuple): The args to the called function.
        kwargs (dict): The kwargs to the function.
    Returns: The same structure as `structure` but each element calls `func_name`.
    """
    if structure is None:
        return
    elif isinstance(structure, (list, tuple)):
        return [
            nested_call(element, func_name, *args, **kwargs)
            for element in structure
        ]
    elif isinstance(structure, dict):
        outputs = EasyDict()
        for k, element in structure.items():
            outputs[k] = nested_call(element, func_name, *args, **kwargs)
        return outputs
    else:
        return getattr(structure, func_name)(*args, **kwargs)


def nested_to_device(structure, device, non_blocking=True):
    """Transfer data to device. The data is a nested structure
    Args:
        data (dict or list or tuple or torch.Tensor): nested data.
        device (torch.device): The target device.
    Kwargs: 
        non_blocking (bool): Wether to block.
    Returns: The same structure as `structure` but each element is transfered to `device`.
    """
    if isinstance(structure, torch.Tensor):
        return structure.to(device, non_blocking=non_blocking)
    elif isinstance(structure, (list, tuple)):
        return [
            nested_to_device(element, device, non_blocking)
            for element in structure
        ]
    elif isinstance(structure, dict):
        outputs = EasyDict()
        for k, element in structure.items():
            outputs[k] = nested_to_device(element, device, non_blocking)
        return outputs
    else:
        raise ValueError(f'Type {type(structure)} is not supported.')
