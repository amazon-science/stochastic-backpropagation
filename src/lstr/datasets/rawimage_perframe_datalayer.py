import os.path as osp
from bisect import bisect_right

import numpy as np
import torch
import torch.utils.data as data

from dlhammer import group_transforms as GT

from .datasets import DATA_LAYERS as registry


@registry.register("E2E_TRNTHUMOS")
@registry.register("E2E_TRNTVSeries")
@registry.register("E2E_TRNANet")
@registry.register("E2E_TRNHACS")
class TRNDataLayer(data.Dataset):
    def __init__(self, cfg, phase="train"):
        self.data_root = cfg.DATA.DATA_ROOT
        self.visual_feature = cfg.INPUT.VISUAL_FEATURE
        self.motion_feature = cfg.INPUT.MOTION_FEATURE
        self.target_perframe = cfg.INPUT.TARGET_PERFRAME
        self.sessions = getattr(cfg.DATA, phase.upper() + "_SESSION_SET")
        self.enc_steps = cfg.MODEL.RNN.ENC_STEPS
        self.dec_steps = cfg.MODEL.RNN.DEC_STEPS
        self.training = phase == "train"

        self.transforms = self.get_transforms(cfg, phase)
        self._init_dataset(0)

    def shuffle(self, seed):
        self._init_dataset(seed)

    @staticmethod
    def get_transforms(cfg, phase):
        if phase == "train":
            transforms = [
                GT.GroupRandomHorizontalFlip(cfg.INPUT.TRAIN.FLIP_P),
                GT.GroupResize(
                    cfg.INPUT.TRAIN.RESIZE_SCALE,
                    keep_aspect_ratio=cfg.INPUT.KEEP_ASPECT,
                ),
                GT.GroupCrop(cfg.INPUT.TRAIN.CROP_SIZE, crop_pos="random"),
            ]
        else:
            transforms = [
                GT.GroupResize(
                    cfg.INPUT.TEST.RESIZE_SCALE, keep_aspect_ratio=cfg.INPUT.KEEP_ASPECT
                ),
                GT.GroupCrop(
                    cfg.INPUT.TEST.CROP_SIZE,
                    crop_pos=cfg.INPUT.TEST.get("CROP_POS", "center"),
                ),
            ]

        if not cfg.INPUT.DO_RESIZE_CROP:
            transforms = []

        transforms = GT.Compose(
            [
                GT.GroupStack(),  # stack to numpy array.
                GT.GroupToTensor(div=cfg.INPUT.DIV_255),  # to torch.Tensor
                GT.GroupNormalize(cfg.INPUT.MEAN, cfg.INPUT.STD),
            ]
            + transforms
        )
        return transforms

    def _init_dataset(self, seed):
        self.inputs = []
        rng = np.random.default_rng(seed)
        for session in self.sessions:
            target = np.load(
                osp.join(self.data_root, self.target_perframe, session + ".npy"),
                mmap_mode="r",
            )

            seed = rng.integers(low=0, high=self.enc_steps) if self.training else 0
            # ensure dataset has the same length at each epoch.
            if self.training:
                stop = (
                    (target.shape[0] - self.enc_steps - self.dec_steps)
                    // self.enc_steps
                    * self.enc_steps
                )
            else:
                stop = target.shape[0] - self.enc_steps - self.dec_steps
            for start in range(seed, stop, self.enc_steps):
                end = start + self.enc_steps
                enc_target = target[start:end]
                dec_target = self._get_dec_target(target[start : end + self.dec_steps])
                self.inputs.append([session, start, end, enc_target, dec_target])

    def _get_dec_target(self, target_vector):
        target_matrix = np.zeros(
            (self.enc_steps, self.dec_steps, target_vector.shape[-1])
        )
        for i in range(self.enc_steps):
            for j in range(self.dec_steps):
                # 0 -> [1, 2, 3]
                # target_matrix[i, j] = target_vector[i + j + 1, :]
                # 0 -> [0, 1, 2]
                target_matrix[i, j] = target_vector[i + j, :]
        return target_matrix

    def load_rgb(self, session, ids):
        """load the RGB for a video segment from a video.

        Args:
            session (string): The video filename.
            ids (list): The index to sample.

        Returns: np.array[T,C,H,W]. The image list for the segment.

        """
        # load from rgb_npy
        video_path = osp.join(self.data_root, self.visual_feature, session + ".npy")
        segment = np.load(video_path, mmap_mode="r")[ids]
        return segment

    def __getitem__(self, index):
        session, start, end, enc_target, dec_target = self.inputs[index]

        visual_inputs = self.load_rgb(session, list(range(start, end)))
        visual_inputs = self.transforms(visual_inputs)

        enc_target = torch.as_tensor(enc_target.astype(np.float32))
        dec_target = torch.as_tensor(dec_target.astype(np.float32))

        return visual_inputs, -1, enc_target, dec_target.view(-1, enc_target.shape[-1])

    def __len__(self):
        return len(self.inputs)


@registry.register("E2E_TENTHUMOS")
@registry.register("E2E_TENTVSeries")
@registry.register("E2E_TENANet")
@registry.register("E2E_TENHACS")
@registry.register("E2E_TDNTHUMOS")
@registry.register("E2E_TDNTVSeries")
@registry.register("E2E_TDNANet")
@registry.register("E2E_TDNHACS")
@registry.register("E2E_LSTRTHUMOS")
@registry.register("E2E_LSTRTVSeries")
@registry.register("E2E_LSTRANet")
@registry.register("E2E_LSTRHACS")
class LSTRRawImageDataLayer(data.Dataset):
    def __init__(self, cfg, phase="train"):
        self.cfg = cfg
        self.phase = phase
        self.data_root = cfg.DATA.DATA_ROOT
        self.visual_feature = cfg.INPUT.VISUAL_FEATURE
        self.motion_feature = cfg.INPUT.MOTION_FEATURE
        self.target_perframe = cfg.INPUT.TARGET_PERFRAME
        self.sessions = getattr(cfg.DATA, phase.upper() + "_SESSION_SET")
        self.num_memories = cfg.MODEL.LSTR.NUM_MEMORIES
        self.memory_sample_rate = cfg.MODEL.LSTR.MEMORY_SAMPLE_RATE
        self.memory_max_length = cfg.MODEL.LSTR.MEMORY_MAX_LENGTH
        self.num_queries = cfg.MODEL.LSTR.WORK_MODULE[0]
        self.training = phase == "train"

        stride = cfg.INPUT[phase.upper()].STRIDE
        self.stride = self.num_queries if stride < 0 else stride

        self.transforms = self.get_transforms(cfg, phase)
        self._init_dataset(0)

    def shuffle(self, seed):
        self._init_dataset(seed)

    @staticmethod
    def get_transforms(cfg, phase):
        if phase == "train":
            transforms = [
                GT.GroupRandomHorizontalFlip(cfg.INPUT.TRAIN.FLIP_P),
                GT.GroupResize(
                    cfg.INPUT.TRAIN.RESIZE_SCALE,
                    keep_aspect_ratio=cfg.INPUT.KEEP_ASPECT,
                ),
                GT.GroupCrop(cfg.INPUT.TRAIN.CROP_SIZE, crop_pos="random"),
            ]
        else:
            transforms = [
                GT.GroupResize(
                    cfg.INPUT.TEST.RESIZE_SCALE, keep_aspect_ratio=cfg.INPUT.KEEP_ASPECT
                ),
                GT.GroupCrop(
                    cfg.INPUT.TEST.CROP_SIZE,
                    crop_pos=cfg.INPUT.TEST.get("CROP_POS", "center"),
                ),
            ]

        if not cfg.INPUT.DO_RESIZE_CROP:
            transforms = []

        transforms = GT.Compose(
            [
                GT.GroupStack(),  # stack to numpy array.
                GT.GroupToTensor(
                    channel_first=False, div=cfg.INPUT.DIV_255
                ),  # to torch.Tensor
                GT.GroupNormalize(cfg.INPUT.MEAN, cfg.INPUT.STD,channel_dim=1),
            ]
            + transforms
        )
        return transforms

    def _init_dataset(self, seed):
        self.inputs = []
        rng = np.random.default_rng(seed)
        for session in self.sessions:
            target = np.load(
                osp.join(self.data_root, self.target_perframe, session + ".npy"),
                mmap_mode="r",
            )

            seed = rng.integers(low=0, high=self.num_queries) if self.training else 0

            if self.training:
                end = (target.shape[0] - self.num_queries) // self.stride * self.stride
            else:
                end = target.shape[0] - self.num_queries

            for start in range(seed, end, self.stride):
                end = start + self.num_queries
                self.inputs.append([session, start, end, target[start:end]])

    def segment_sampler(self, start, end):
        indices = np.linspace(start, end, num=self.num_memories)
        return np.sort(indices).astype(np.int32)

    def uniform_sampler(self, start, end):
        indices = np.arange(start, end + 1)[:: self.memory_sample_rate]
        padding = self.num_memories - indices.shape[0]
        if padding > 0:
            indices = np.concatenate((np.zeros(padding), indices))
        return np.sort(indices).astype(np.int32)

    def load_rgb(self, session, ids):
        """load the RGB for a video segment from a video.

        Args:
            session (string): The video filename.
            ids (list): The index to sample.

        Returns: List(np.array). The image list for the segment.

        """
        # load from rgb_npy
        video_path = osp.join(self.data_root, self.visual_feature, session + ".npy")
        segment = np.load(video_path, mmap_mode="r")[ids]  # shape: [N,H,W,C]

        # convert to PIL.Image.
        # img_group = []
        # for i in range(len(segment)):
        #     img_group.append(Image.fromarray(segment[i]))
        # return img_group
        return segment

    def __getitem__(self, index):
        session, start, end, target = self.inputs[index]

        # Get query
        query_indices = np.arange(start, end).clip(0)
        query_visual_inputs = self.load_rgb(session, query_indices)
        query_visual_inputs = self.transforms(query_visual_inputs)  # torch.Tensor[TCHW]

        # Get memory
        if self.num_memories > 0:
            memory_start, memory_end = max(0, start - self.memory_max_length), start - 1
            if self.training:
                memory_indices = self.segment_sampler(memory_start, memory_end).clip(0)
            else:
                memory_indices = self.uniform_sampler(memory_start, memory_end).clip(0)
            memory_visual_inputs = self.load_rgb(session, memory_indices)
            memory_visual_inputs = self.transforms(
                memory_visual_inputs
            )  # torch.Tensor(TCHW)

            # Get memory key padding mask
            memory_key_padding_mask = np.zeros(memory_indices.shape[0])
            last_zero = bisect_right(memory_indices, 0) - 1
            if last_zero > 0:
                memory_key_padding_mask[:last_zero] = float("-inf")

            visual_inputs = torch.cat(
                (memory_visual_inputs, query_visual_inputs), dim=0
            )  # torch.Tensor.  shape: (T1+T2,C,H,W)
            memory_key_padding_mask = torch.as_tensor(
                memory_key_padding_mask.astype(np.float32)
            )
        else:
            visual_inputs = query_visual_inputs
            memory_key_padding_mask = None

        target = torch.as_tensor(target.astype(np.float32))

        if memory_key_padding_mask is not None:
            return (visual_inputs, -1, memory_key_padding_mask), target
        else:
            return (visual_inputs, -1), target

    def __len__(self):
        return len(self.inputs)


@registry.register("E2E_TENBatchInferenceTHUMOS")
@registry.register("E2E_TENBatchInferenceTVSeries")
@registry.register("E2E_TENBatchInferenceANet")
@registry.register("E2E_TENBatchInferenceHACS")
@registry.register("E2E_TDNBatchInferenceTHUMOS")
@registry.register("E2E_TDNBatchInferenceTVSeries")
@registry.register("E2E_TDNBatchInferenceANet")
@registry.register("E2E_TDNBatchInferenceHACS")
@registry.register("E2E_LSTRBatchInferenceTHUMOS")
@registry.register("E2E_LSTRBatchInferenceTVSeries")
@registry.register("E2E_LSTRBatchInferenceANet")
@registry.register("E2E_LSTRBatchInferenceHACS")
class LSTRRawImageBatchInferenceDataLayer(data.Dataset):
    def __init__(self, cfg, phase="test"):
        self.data_root = cfg.DATA.DATA_ROOT
        self.visual_feature = cfg.INPUT.VISUAL_FEATURE
        self.motion_feature = cfg.INPUT.MOTION_FEATURE
        self.target_perframe = cfg.INPUT.TARGET_PERFRAME
        self.sessions = cfg.DATA.TEST_SESSION_SET
        self.num_memories = cfg.MODEL.LSTR.NUM_MEMORIES
        self.memory_sample_rate = cfg.MODEL.LSTR.MEMORY_SAMPLE_RATE
        self.memory_max_length = cfg.MODEL.LSTR.MEMORY_MAX_LENGTH
        self.num_queries = cfg.MODEL.LSTR.WORK_MODULE[0]

        assert phase == "test", "This data layer only supports batch inference"

        self.transforms = LSTRRawImageDataLayer.get_transforms(cfg, phase)

        self.inputs = []
        for session in self.sessions:
            target = np.load(
                osp.join(self.data_root, self.target_perframe, session + ".npy")
            )
            for start, end in zip(
                range(0, target.shape[0] + 1),
                range(self.num_queries, target.shape[0] + 1),
            ):
                self.inputs.append(
                    [session, start, end, target[start:end], target.shape[0]]
                )

    def uniform_sampler(self, start, end):
        indices = np.arange(start, end + 1)[:: self.memory_sample_rate]
        padding = self.num_memories - indices.shape[0]
        if padding > 0:
            indices = np.concatenate((np.zeros(padding), indices))
        return np.sort(indices).astype(np.int32)

    def __getitem__(self, index):
        session, start, end, target, num_frames = self.inputs[index]

        visual_inputs = np.load(
            osp.join(self.data_root, self.visual_feature, session + ".npy"),
            mmap_mode="r",
        )
        # motion_inputs = np.load(osp.join(self.data_root, self.motion_feature,
        #                                     session + '.npy'),
        #                         mmap_mode='r')

        # Get query
        query_indices = np.arange(start, end).clip(0)
        query_visual_inputs = visual_inputs[query_indices]
        query_visual_inputs = self.transforms(query_visual_inputs)
        # query_motion_inputs = motion_inputs[query_indices]

        # Get memory
        if self.num_memories > 0:
            memory_start, memory_end = max(0, start - self.memory_max_length), start - 1
            memory_indices = self.uniform_sampler(memory_start, memory_end).clip(0)
            memory_visual_inputs = visual_inputs[memory_indices]
            # memory_motion_inputs = motion_inputs[memory_indices]
            memory_visual_inputs = self.transforms(
                memory_visual_inputs
            )  # torch.Tensor(TCHW)

            # Get memory key padding mask
            memory_key_padding_mask = np.zeros(memory_indices.shape[0])
            last_zero = bisect_right(memory_indices, 0) - 1
            if last_zero > 0:
                memory_key_padding_mask[:last_zero] = float("-inf")

            visual_inputs = np.concatenate((memory_visual_inputs, query_visual_inputs))
            # motion_inputs = np.concatenate(
            #     (memory_motion_inputs, query_motion_inputs))
            # memory_key_padding_mask = torch.as_tensor(
            #     memory_key_padding_mask.astype(np.float32))
        else:
            visual_inputs = query_visual_inputs
            # motion_inputs = query_motion_inputs

        target = torch.as_tensor(target.astype(np.float32))

        if self.num_memories > 0:
            return (
                visual_inputs,
                -1,
                memory_key_padding_mask,
                target,
                session,
                query_indices,
                num_frames,
            )
        else:
            return (visual_inputs, -1, target, session, query_indices, num_frames)

    def __len__(self):
        return len(self.inputs)
