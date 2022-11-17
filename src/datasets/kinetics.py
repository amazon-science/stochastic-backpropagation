import os

import torch
import torch.utils.data as data
import numpy as np
import pandas as pd
from decord import VideoReader
from decord import cpu

from dlhammer.bootstrap import logger
from dlhammer import group_transforms as GT
from .builder import DATASETS as registry


@registry.register('kinetics')
class KineticsDataset(data.Dataset):

    def __init__(self, cfg, phase):
        """
        Args:
            cfg (dict): the configs.
            phase (string): either 'train' or 'test'.
        """
        self.cfg = cfg
        self.data_cfg = cfg.DATA[phase.upper()]
        self.training = phase == 'train'

        self.num_retries = self.cfg.DATA.NUM_RETRIES
        self.num_clips = self.data_cfg.NUM_CLIPS
        self.clip_sample_strategy = self.data_cfg.CLIP_SAMPLE_STRATEGY
        self.clip_sample_rate = self.data_cfg.SAMPLING_RATE

        self.init_transforms()
        self.init_dataset()

    def init_transforms(self):
        if self.training:
            transforms = [
                GT.TemporalSampling(**self.cfg.DATA.AUG.TEMPORAL_SAMPLING),
                GT.GroupStack(),
                GT.GroupToTensor(div=self.cfg.DATA.DIV_255),
                GT.GroupNormalize(self.cfg.DATA.MEAN, self.cfg.DATA.STD),
                GT.GroupRandomHorizontalFlip(self.cfg.DATA.AUG.FLIP_P),
                GT.GroupResize(self.cfg.DATA.TRAIN.RESIZE_SCALE),
                GT.GroupCrop(self.cfg.DATA.TRAIN.CROP_SIZE,
                             crop_pos=self.cfg.DATA.TRAIN.get('CROP_POS', 'random'))
            ]
        else:
            transforms = [
                GT.GroupStack(),
                GT.GroupToTensor(div=self.cfg.DATA.DIV_255),
                GT.GroupNormalize(self.cfg.DATA.MEAN, self.cfg.DATA.STD),
                GT.GroupResize(self.cfg.DATA.TEST.RESIZE_SCALE),
                GT.GroupCrop(self.cfg.DATA.TEST.CROP_SIZE,
                             crop_pos=self.cfg.DATA.TEST.get('CROP_POS', 'center'))
            ]
        self.transforms = GT.Compose(transforms)

    def init_dataset(self):
        """create samples.

        Returns: List. Each element is a tuple (video_path, clip_idx, label).

        """
        data = pd.read_csv(self.data_cfg.LABEL_FILE, header=None, delimiter=' ')
        video_names = list(data.values[:, 0])
        video_paths = [os.path.join(self.data_cfg.DATA_DIR, video_name) for video_name in video_names]
        labels = list(data.values[:, 2])

        samples = []
        for video_path, label in zip(video_paths, labels):
            for idx in range(self.num_clips):
                samples.append((video_path, idx, label))
        self.samples = samples

    def __getitem__(self, index):
        video_path, clip_idx, label = self.samples[index]

        for _ in range(self.num_retries):
            video_clip, frame_ids = self.loadvideo_decord(video_path, clip_idx)
            if video_clip is not None:
                video_clip = self.transforms(video_clip)
                return (video_clip, frame_ids), label
            index = torch.randint(0,len(self.samples),[])
            video_path, clip_idx, label = self.samples[index]

        raise ValueError(
            f'Retry to load video {self.num_retries} times and still get a broken video. Go to check your dataset.'
        )

    def get_clip_frame_ids(self, video_length, clip_id, num_clips, num_frames, clip_sample_rate,
                            clip_sample_strategy):
        """

        Args:
            video_length (int): The number of frames in a video.
            clip_id (int): The id of the clip.
            num_clips (int): Divide the video with the number of clips.
            num_frames (int): The number of frames to sample from a clip.
            clip_sample_rate (int): The sample rate inside the clip.
            clip_sample_strategy (string): The strategy to sample clip from the video. Options: 'random', 'uniform', 'center'.

        Returns: List[int]. The frame indexs to sample from the video.

        """
        clip_length = num_frames * clip_sample_rate
        if clip_length >= video_length:
            clip_start = 0
        else:
            if clip_sample_strategy == 'random':
                clip_start = int(torch.randint(0, video_length - clip_length, []))
            elif clip_sample_strategy == 'center':
                clip_start = (video_length - clip_length) // 2
            elif clip_sample_strategy == 'uniform':
                clip_start = (video_length - clip_length) // num_clips * clip_id

        clip_ids = np.arange(clip_start, clip_start + clip_length, clip_sample_rate)
        clip_ids = np.clip(clip_ids, 0, video_length - 1)
        return clip_ids

    def loadvideo_decord(self, video_path, clip_id):

        if not (os.path.exists(video_path)):
            logger.info(f'Video cannot be loaded by decord: {video_path}')
            return None, None

        # avoid hanging issue
        if os.path.getsize(video_path) < 1 * 1024:
            logger.info(f'SKIP: {video_path} - {os.path.getsize(video_path)}')
            return None, None

        # read video
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
        except:
            logger.info(f'Video cannot be loaded by decord: {video_path}')
            return None, None

        vr.seek(0)
        frame_ids = self.get_clip_frame_ids(len(vr), clip_id, self.data_cfg.NUM_CLIPS,
                                            self.data_cfg.NUM_FRAMES, self.data_cfg.SAMPLING_RATE,
                                            self.data_cfg.CLIP_SAMPLE_STRATEGY)
        buffer = vr.get_batch(frame_ids).asnumpy()
        return buffer, frame_ids

    def __len__(self):
        return len(self.samples)
