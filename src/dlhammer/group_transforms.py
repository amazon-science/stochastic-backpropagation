import torch
from torchvision.transforms import functional as F
import numpy as np
from PIL import Image


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data


class IdentityTransform(object):

    def __call__(self, data):
        return data


class TemporalSampling(object):
    """Perform temporal sampling for a group of images."""

    def __init__(self, sampling_method, sampling_rate=0.5):
        super(TemporalSampling, self).__init__()
        self.sampling_method = sampling_method
        self.sampling_rate = sampling_rate

    def __call__(self, img_group):
        """
        Args:
            img_group (list(PIL.Image) or Tensor([C,T,H,W])): a list of image to sampling.

        Returns (The same type as input).
        """
        if self.sampling_rate == 1:
            return img_group

        n = len(img_group)
        if self.sampling_method == 'random':
            size = int(n * self.sampling_rate)
            idx = np.random.choice(n, size, replace=False)
            idx = np.sort(idx)
        elif self.sampling_method == 'uniform':
            step = int(1 / self.sampling_rate)
            idx = np.arange(0, n, step)
        else:
            raise ValueError(f'sampling_method {self.sampling_method} is not supported')

        idx = idx.tolist()
        if isinstance(img_group, (list, tuple)):
            new_img_group = [img_group[i] for i in idx]
        elif isinstance(img_group, (torch.Tensor, np.ndarray)):
            new_img_group = img_group[idx]
        else:
            raise ValueError(f'img_group type {type(img_group)} is not supported')
        return new_img_group


class GroupStack(object):
    """Stack a list of images to ndarray."""

    def __init__(self, roll=False):
        self.roll = roll # BGR -> RGB

    def __call__(self, img_group):
        """
        Args:
            img_group (list(PIL.Image) or list(np.ndarray)):
            a list of image of shape (H, W, C) for rgb or (H, W) for grayscale.

        Returns (np.ndarray): stack the images to ndarray of shape (T, H, W, C).
        """
        if isinstance(img_group[0], np.ndarray):
            imgs = np.stack(img_group, axis=0)
            if imgs.ndim == 3: # THW
                imgs = imgs[:, :, :, np.newaxis] # THWC
            return imgs
        else:
            if img_group[0].mode == 'L':
                return np.stack([np.expand_dims(x, 2) for x in img_group], axis=0) # THWC
            elif img_group[0].mode == 'RGB':
                if self.roll:
                    return np.stack([np.array(x)[:, :, ::-1] for x in img_group], axis=0) # THWC
                else:
                    return np.stack(img_group, axis=2) # THWC


class GroupToTensor(object):

    def __init__(self, channel_first=True, div=True):
        self.channel_first = channel_first
        self.div = div

    def __call__(self, img_group):
        """
        Args:
            img_group (np.ndarray): the stacked images of shape (T, H, W, C)

        Returns (torch.Tensor): convert dtype to torch.float32.
                If `channel_first` is True, the images are permuted to shape (C, T, H, W).
                If `div` is True, the images are divided by 255 and normalized to [0.0, 1.0].
        """
        if isinstance(img_group, np.ndarray):
            img_group = torch.from_numpy(img_group).float()
        else:
            raise ValueError(f'Only support np.ndarray. Got {type(img_group)}')
        if self.channel_first:
            img_group = img_group.permute(3, 0, 1, 2).contiguous() # THWC -> CTHW
        else:
            img_group = img_group.permute(0,3,1,2).contiguous() # THWC -> TCHW
        if self.div:
            img_group = img_group.div(255.)
        return img_group


class GroupNormalize(object):

    def __init__(self, mean, std, channel_dim=0):
        
        new_shape = [1,1,1,1] 
        new_shape[channel_dim] = len(mean)
        mean = np.array(mean, dtype=np.float32)
        # self.mean = mean[:, np.newaxis, np.newaxis, np.newaxis] # C -> CTHW
        self.mean = np.reshape(mean, new_shape)
        std = np.array(std, dtype=np.float32)
        # self.std = std[:, np.newaxis, np.newaxis, np.newaxis] # C -> CTHW
        self.std = np.reshape(std, new_shape)

    def __call__(self, tensor):
        """
        Args:
            tensor (torch.Tensor): the stacked images of shape (C, T, H, W).

        Returns (torch.Tensor): sub mean and div std along the channel dim.
        """
        if isinstance(tensor, torch.Tensor):
            mean = torch.from_numpy(self.mean).to(tensor.device)
            std = torch.from_numpy(self.std).to(tensor.device)
        else:
            mean = self.mean
            std = self.std
        return (tensor - mean) / std


class GroupRandomHorizontalFlip(object):
    """Randomly horizontal flip for a group of images."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img_group):
        flip = torch.rand([]) < self.p
        if flip:
            if isinstance(img_group, torch.Tensor):
                assert img_group.dim() == 4, \
                    f'img_group should be dim-4 (CTHW) or (TCHW). Got shape: {img_group.shape}'
                return F.hflip(img_group)
            elif isinstance(img_group[0], Image):
                return [F.hflip(img) for img in img_group]
            else:
                raise NotImplementedError(f'img_group of type {type(img_group)} is not supported')
        else:
            return img_group


class GroupResize(object):
    """Randomly resize for a group of images."""

    def __init__(self, scale_range, keep_aspect_ratio=True):
        """
        Args:
            scale_range (list or typle): The shorter side of the image will be resized to
                                            1) a range if `scale_range` is a list or tuple.
                                            2) a specific size if `scale_range` contains only 1 element.
        """
        super(GroupResize, self).__init__()
        self.scale_range = scale_range
        self.keep_aspect_ratio = keep_aspect_ratio

    def __call__(self, img_group):
        """
        Args:
            img_group (list(PIL.Image) or Tensor([C,T,H,W])): a list of image to resize.

        Returns: The same type as input. resized img_group.
            
        """
        # resize
        if len(self.scale_range) == 1:
            size = self.scale_range[0]
        else:
            size = torch.randint(self.scale_range[0], self.scale_range[1], [])
        size= int(size)
        size = size if self.keep_aspect_ratio else (size, size)

        if isinstance(img_group, torch.Tensor):
            assert img_group.dim() == 4, \
                f'img_group should be dim-4 (CTHW) or (TCHW). Got shape: {img_group.shape}'
            return F.resize(img_group, size)
        elif isinstance(img_group[0], Image):
            return [F.resize(img, size) for img in img_group]
        else:
            raise NotImplementedError(f'img_group of type {type(img_group)} is not supported')


class GroupCrop(object):
    """Crop for a group of images."""

    def __init__(self, crop_size, crop_pos='random'):
        super(GroupCrop, self).__init__()
        self.crop_size = crop_size
        self.crop_pos = crop_pos
        assert crop_pos in ['random', 'top_left', 'center', 'bottom_right'], \
            f'crop_pos {self.crop_pos} is not supported'

    def __call__(self, img_group):
        """
        Args:
            img_group (list(PIL.Image) or torch.Tensor([C,T,H,W])): a list of images to crop.

        Returns (The same type as input).
        """
        crop_w, crop_h = self.crop_size
        if isinstance(img_group, torch.Tensor):
            assert img_group.dim() == 4, \
                f'img_group should be dim-4 (TCHW) or (CTHW). Got shape: {img_group.shape}'
            img_h, img_w = img_group.shape[-2:]
            offset_h, offset_w = self.get_crop_offset(img_h, img_w, crop_h, crop_w)
            return F.crop(img_group, offset_h, offset_w, crop_h, crop_w)
        elif isinstance(img_group[0], Image):
            img_w, img_h = img_group[0].size
            offset_h, offset_w = self.get_crop_offset(img_h, img_w, crop_h, crop_w)
            return [F.crop(img, offset_h, offset_w, crop_h, crop_w) for img in img_group]
        else:
            raise NotImplementedError(f'img_group of type {type(img_group)} is not supported')

    def get_crop_offset(self, img_h, img_w, crop_h, crop_w):
        """Get the offset to crop.

        Returns (offset_h, offset_w).
        """
        if self.crop_pos == 'random':
            offset_h = int(torch.randint(img_h - crop_h, []))
            offset_w = int(torch.randint(img_w - crop_w, []))
        elif self.crop_pos == 'top_left':
            offset_h = 0
            offset_w = 0
        elif self.crop_pos == 'center':
            offset_h = int((img_h - crop_h) // 2)
            offset_w = int((img_w - crop_w) // 2)
        elif self.crop_pos == 'bottom_right':
            offset_h = img_h - crop_h
            offset_w = img_w - crop_w
        else:
            raise NotImplementedError(f'crop_pos {self.crop_pos} is not supported')
        return (offset_h, offset_w)


class GroupMultiScaleCrop(object):

    @staticmethod
    def fill_fix_offset(image_w, image_h, crop_w, crop_h, more_fix_crop=False):
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        ret = list()
        ret.append((0, 0))  # upper left
        ret.append((4 * w_step, 0))  # upper right
        ret.append((0, 4 * h_step))  # lower left
        ret.append((4 * w_step, 4 * h_step))  # lower right
        ret.append((2 * w_step, 2 * h_step))  # center

        if more_fix_crop:
            ret.append((0, 2 * h_step))  # center left
            ret.append((4 * w_step, 2 * h_step))  # center right
            ret.append((2 * w_step, 4 * h_step))  # lower center
            ret.append((2 * w_step, 0 * h_step))  # upper center

            ret.append((1 * w_step, 1 * h_step))  # upper left quarter
            ret.append((3 * w_step, 1 * h_step))  # upper right quarter
            ret.append((1 * w_step, 3 * h_step))  # lower left quarter
            ret.append((3 * w_step, 3 * h_step))  # lower righ quarter

        return ret
