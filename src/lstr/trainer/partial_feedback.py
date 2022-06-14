import warnings

import numpy as np
import torch


class FrameSampler(object):
    """sample frames"""

    @staticmethod
    def uniform_sample(T, ratio):
        """TODO: Docstring for uniform_sample.

        Args:
            T (int): The number of frames
            ratio (float): sample ratio. between (0,1)

        Returns: (l1, l2). l1 and l2 are lists. l1 is the sampled indexs.
                    l2 is the remaining indexs.

        """
        l = list(range(T))
        stride = int(1 / ratio)
        if stride < T:
            l1 = list(range(0, T, stride))
        else:  # if only sample 1 frame, take the center one.
            l1 = [T // 2]
        l2 = [x for x in l if x not in l1]
        return (l1, l2)

    @staticmethod
    def uniform_jitter_sample(T, ratio):
        """sample each chunk(chunk length: 1/ratio). randomly sample in each chunk.

        Args:
            T (int): The number of frames
            ratio (float): sample ratio. between (0,1)

        Returns: (l1, l2). l1 and l2 are lists. l1 is the sampled indexs.
                    l2 is the remaining indexs.

        """
        l = list(range(T))
        stride = int(1 / ratio)
        l1 = np.arange(0, T, stride)
        jitters = torch.randint(0, stride, (len(l1),)).numpy()
        l1 = l1 + jitters
        l1 = np.clip(l1, 0, T - 1)

        l1 = l1.tolist()
        l2 = [x for x in l if x not in l1]
        return (l1, l2)


def split_inference(model, inp, split_size, rng_state=None):
    """inference the input with split.

    Args:
        model (nn.Module): The model.
        inp (torch.Tensor[bs,T,...]): The inputs to be inferenced.
        split_size (int): split size.
        rng_state (random state): If not None, the pytorch rng_state will be set
                                to the given one before the model is inferenced.

    Returns: The features. torch.Tensor[bs,T,...]

    """
    bs, T = inp.shape[:2]
    if split_size == -1:
        split_size = T
    if bs * T % split_size != 0:
        warnings.warn(
            f"If split inference, bs*T % split_size must be 0. Got: bs:{bs}, T:{T}, split_size:{split_size}"
        )
        split_size = T
        warnings.warn(f"Set split_size to T:{T} to continue")

    mbs = bs * T // split_size
    new_inp = inp.reshape(mbs, split_size, *inp.shape[2:])
    features = []
    for i in range(mbs):
        if rng_state is not None:
            torch.set_rng_state(rng_state)
        feature = model(new_inp[i : i + 1])
        features.append(feature)

    features = torch.cat(features, dim=0)  # (mbs, split_size, C)
    return features.reshape(bs, T, *features.shape[2:])  # (B, T, C)


def vanilla_forward(models, inp, freeze_backbone=True, split_size=-1):
    """vanilla forward

    Args:
        models (dict): The model.
        inp (tuple): (visual_inputs,motion_inputs,...), each is of shape (B,T,C,H,W).
                        tar is the target, shape: (B,T,NUM_CLASSES).
        split_size (int): If positive, then will split the inputs to smaller batches to do the forward.
                        Else feed the whole input to the model. Only use this when no_grad is required.

    Returns: (torch.Tensor). The predicted logits. shape (B,T,NUM_CLASSES)

    """
    split_size = -1
    if freeze_backbone:
        with torch.no_grad():
            if split_size > 0:
                features = split_inference(models.spatial_model, inp[0], split_size)
            else:
                features = models.spatial_model(inp[0])  # (B, T, C)
    else:
        if split_size > 0:
            features = split_inference(models.spatial_model, inp[0], split_size)
        else:
            features = models.spatial_model(inp[0])  # (B, T, C)
    temporal_inp = [features] + inp[1:]
    det_logits = models.temporal_model(*temporal_inp)  # (B, T, NUM_CLASSES)
    return det_logits


def partial_feedback_forward(models, inp, sample_rate, sample_strategy="uniform"):
    """perform partial feedback on spatial_model
    Partial feedback is the old name. The new name is Stochastic Backpropagation.

    Args:
        models (dict): The model.
        inp (tuple): may be (rgb,) or (rgb,flow), each is of shape (B,T,C,H,W).
                        tar is the target, shape: (B,T,NUM_CLASSES).
        sample_rate (float): The ratio of sampled frames in one video segment.
        sample_strategy (string): The strategy to sample from frames.
                        Options: 'uniform', 'uniform_jitter'.

    Returns: (torch.Tensor): the predicted logits. shape: (B,T,NUM_CLASSES)
    """
    rgb_inputs = inp[0]
    bs, T = rgb_inputs.shape[:2]

    assert sample_strategy in [
        "uniform",
        "uniform_jitter",
    ], f'sample_strategy should be in ["uniform", "uniform_jitter"]. Got: {sample_strategy}'

    ## forward backbone with partial gradient tracking.
    if sample_strategy == "uniform":
        grad_idx, no_grad_idx = FrameSampler.uniform_sample(T, sample_rate)
    elif sample_strategy == "uniform_jitter":
        grad_idx, no_grad_idx = FrameSampler.uniform_jitter_sample(T, sample_rate)
    else:
        raise NotImplementedError(f"Not Implemented: {sample_strategy}")

    grad_frames = rgb_inputs[:, grad_idx, :, :, :].clone()  # (B, T1, C,H,W)
    no_grad_frames = rgb_inputs[:, no_grad_idx, :, :, :].clone()  # (B, T2, C,H,W)
    split = True
    split_size = -1
    with torch.no_grad():
        if split:  # split to save memory
            no_grad_features = split_inference(
                models.spatial_model,
                no_grad_frames,
                split_size=split_size,
                rng_state=None,
            )
        else:
            no_grad_features = models.spatial_model(no_grad_frames)  # (B,T2,C)
    with torch.enable_grad():
        grad_features = models.spatial_model(grad_frames)  # (B, T1, C)
    feature_dim = grad_features.shape[-1]
    features = torch.zeros(
        [bs, T, feature_dim], device=grad_features.device, dtype=grad_features.dtype
    )
    features[:, grad_idx, :] = grad_features
    features[:, no_grad_idx, :] = no_grad_features

    # forward temporal model
    temporal_inp = [features] + inp[1:]
    det_logits = models.temporal_model(*temporal_inp)  # (B,T,C)
    return det_logits
