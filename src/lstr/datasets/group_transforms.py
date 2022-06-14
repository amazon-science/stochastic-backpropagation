import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data


"""
Group: A list of image. i.e. a video segment.
"""


class GroupOperation(object):
    """perform operation for all the images in img_group."""

    def __init__(self, functional):
        """TODO: to be defined.

        Args:
            functional (function or callable class): function, i.e. torchvision.transforms.functional.resize.


        """
        self.functional = functional

    def __call__(self, img_group):
        return [self.functional(img) for img in img_group]


class GroupResize(object):
    """perfrom group random resize"""

    def __init__(self, scale_range, keep_aspect_ratio=True):
        """
        Args:
            scale_range (tuple/list): The shorter side of the image will be resized to
                                        1) the range if `scale_range` is a tuple
                                        2) the specific size if `scale_range` contains only 1 element.
        """
        super(GroupResize, self).__init__()
        self.scale_range = scale_range
        self.keep_aspect_ratio = keep_aspect_ratio

    def __call__(self, img_group):
        """
        Args:
            img_group (List(PIL.Image) or Tensor([T,C,H,W])): a list of image to resize.

        Returns: The same type as input. resized img_group.
            
        """
        # resize
        size = self.scale_range[0] if len(
            self.scale_range) == 1 else torch.randint(self.scale_range[0],
                                                        self.scale_range[1], [])
        size = int(size)
        size = size if self.keep_aspect_ratio else (size, size)

        if isinstance(img_group, torch.Tensor):
            assert img_group.dim(
            ) == 4, f'input img_group should be dim-4(TCHW). Got shape: {img_group.shape}'
            return F.resize(img_group, size)
        elif isinstance(img_group[0], Image):
            return [F.resize(img, size) for img in img_group]
        else:
            raise NotImplementedError(
                f'img_group of type: {type(img_group)} is not supported')


class GroupCrop(object):
    """perfrom group crop'"""

    def __init__(self, crop_size, crop_pos='random'):
        super(GroupCrop, self).__init__()
        self.crop_size = crop_size
        self.crop_pos = crop_pos
        assert crop_pos in ['random', 'top_left', 'center', 'bottom_right'
                           ], f'crop_pos: {self.crop_pos} not implemented'

    def __call__(self, img_group):
        """
        Args:
            img_group (List(PIL.Image) or List(torch.Tensor)): a list of image to crop.

        Returns: The same type as input. cropped img_group.
            
        """
        crop_w, crop_h = self.crop_size
        if isinstance(img_group, torch.Tensor):
            assert img_group.dim(
            ) == 4, f'input img_group should be dim-4(TCHW). Got shape: {img_group.shape}'
            img_h, img_w = img_group.shape[-2:]
            offset_h, offset_w = self.get_crop_offset(img_w, img_h, crop_w,
                                                        crop_h)
            return F.crop(img_group, offset_h, offset_w, crop_h, crop_w)
        elif isinstance(img_group[0], Image):
            img_w, img_h = img_group[0].size
            offset_h, offset_w = self.get_crop_offset(img_w, img_h, crop_w,
                                                        crop_h)
            return [
                F.crop(img, offset_h, offset_w, crop_h, crop_w)
                for img in img_group
            ]
        else:
            raise NotImplementedError(
                f'img_group of type: {type(img_group)} is not supported')

    def get_crop_offset(self, img_w, img_h, crop_w, crop_h):
        """get the offset to crop.

        Returns: (offset_h, offset_w).

        """
        if self.crop_pos == 'random':
            offset_h = int(torch.randint(img_h - crop_h, []))
            offset_w = int(torch.randint(img_w - crop_w, []))
        elif self.crop_pos == 'top_left':
            offset_h = offset_w = 0
        elif self.crop_pos == 'center':
            offset_h = int((img_h - crop_h) // 2)
            offset_w = int((img_w - crop_w) // 2)
        elif self.crop_pos == 'bottom_right':
            offset_h = img_h - crop_h
            offset_w = img_w - crop_w
        else:
            raise NotImplementedError(
                f'crop_pos: {self.crop_pos} not implemented')
        return (offset_h, offset_w)


class GroupRandomScaleCrop(object):
    """perform group scale and crop"""

    def __init__(self,
                    scale_range,
                    crop_size,
                    crop_pos='random',
                    keep_aspect_ratio=True):
        """

        Args:
            scale_range (tuple/list): The shorter side of the image will be resized to
                                        1) the range if `scale_range` is a tuple
                                        2) the specific size if `scale_range` contains only 1 element.
            crop_size (int or tuple): If tuple, crop_size = (crop_w, crop_h);
                                        else crop_size = crop_w = crop_h.

        Kwargs:
            crop_pos (string): 'random', 'center'.


        """
        self.scale_range = scale_range
        self.crop_size = crop_size if not isinstance(crop_size, int) else (
            crop_size, crop_size)

    def __call__(self, img_group):

        # resize
        size = self.scale_range[0] if len(
            self.scale_range) == 1 else torch.randint(self.scale_range[0],
                                                        self.scale_range[1], [])
        size = int(size)
        img_group = [F.resize(img, (size, size)) for img in img_group]

        # crop
        crop_w, crop_h = self.crop_size
        img_w, img_h = img_group[0].size
        offset_h = int(torch.randint(img_h - crop_h, []))
        offset_w = int(torch.randint(img_w - crop_w, []))
        img_group = [
            F.crop(img, offset_h, offset_w, crop_h, crop_w) for img in img_group
        ]
        return img_group


class GroupRandomHorizontalFlip(object):
    """randomly h_flip for a group of images."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img_group):
        flip = torch.rand([]) < self.p
        if flip:
            if isinstance(img_group, torch.Tensor):
                assert img_group.dim(
                ) == 4, f'input img_group should be dim-4(TCHW). Got shape: {img_group.shape}'
                return F.hflip(img_group)
            elif isinstance(img_group[0], Image):
                return [F.hflip(img) for img in img_group]
            else:
                raise NotImplementedError(
                    f'img_group of type: {type(img_group)} is not supported')
        else:
            return img_group


class GroupStack(object):
    """Stack to numpy array and convert  THWC to TCHW"""

    def __call__(self, img_group):
        """
        Args:
            img_group (List(PIL.Image)): a list of image. image shape: (H,W,C) for rgb, (H,W) for grayscale

        Returns: np.ndarray. stack the images along the first axis.
                shape: (T, C, H, W).
        """
        if isinstance(img_group, (list, tuple)):
            imgs = np.stack(img_group, axis=0)
        else:
            imgs = img_group
        if imgs.ndim == 3:    # THW
            imgs = imgs[:, np.newaxis, :, :]    # TCHW
        else:    # THWC
            imgs = np.transpose(imgs, [0, 3, 1, 2])    # THWC -> TCHW
        return imgs


class GroupToTensor(object):

    def __init__(self, div=True):
        self.div = div

    def __call__(self, imgs):
        """
        Args:
            imgs (np.ndarray) : The stacked images. shape: (T, C, H, W). dtype: np.uint8

        Returns: (torch.Tensor). convert dtype to torch.float32.
                If `div` is True, the images are divided by 255 to be normalized to [0,1]
            
        """
        imgs = torch.from_numpy(imgs).float()
        if self.div:
            imgs = imgs.div(255.)
        return imgs


class GroupNormalize(object):

    def __init__(self, mean, std):

        if len(mean) == 1:    # flow
            assert len(
                std
            ) == 1, f'flow input must have std of length 1. got {len(std)}'
            self.mean = mean[0]
            self.std = std[0]
        else:
            mean = np.array(mean, dtype=np.float32)
            self.mean = mean[np.newaxis, :, np.newaxis,
                                np.newaxis]    # shape: (C,) ->(T,C,H,W)
            std = np.array(std, dtype=np.float32)
            self.std = std[np.newaxis, :, np.newaxis,
                            np.newaxis]    # shape: (C,) ->(T,C,H,W)

    def __call__(self, tensor):
        """
        Args:
            tensor (torch.Tensor) : The stacked images. shape: (T, C, H, W).

        Returns: (torch.Tensor).  sub mean and div std along the channel dim.
            
        """
        return (tensor - self.mean) / self.std
