__all__ = ["build_feature_head"]

import torch
import torch.nn as nn

from dlhammer.registry import Registry

FEATURE_HEADS = Registry()
FEATURE_SIZES = {
    "rgb_anet_resnet50": 2048,
    "flow_anet_resnet50": 2048,
    "rgb_kinetics_bninception": 1024,
    "flow_kinetics_bninception": 1024,
    "rgb_kinetics_resnet50": 2048,
    "rgb_igib_omni_kinetics_resnet50": 2048,
    "rgb_imagenet_omni_kinetics_resnet50": 2048,
}


@FEATURE_HEADS.register("THUMOS")
@FEATURE_HEADS.register("TVSeries")
@FEATURE_HEADS.register("ANet")
@FEATURE_HEADS.register("HACS")
class BaseFeatureHead(nn.Module):
    def __init__(self, cfg):
        super(BaseFeatureHead, self).__init__()

        if cfg.INPUT.MODALITY in ["visual", "motion", "twostream"]:
            self.with_visual = "motion" not in cfg.INPUT.MODALITY
            self.with_motion = "visual" not in cfg.INPUT.MODALITY
        else:
            raise RuntimeError("Unknown modality of {}".format(cfg.INPUT.MODALITY))

        if self.with_visual and self.with_motion:
            visual_size = FEATURE_SIZES[cfg.MODEL.SPATIAL.NAME]
            motion_size = FEATURE_SIZES[cfg.INPUT.MOTION_FEATURE]
            fusion_size = visual_size + motion_size
        elif self.with_visual:
            fusion_size = FEATURE_SIZES[cfg.MODEL.SPATIAL.NAME]
        elif self.with_motion:
            fusion_size = FEATURE_SIZES[cfg.INPUT.MOTION_FEATURE]
        else:
            raise ValueError("unexcepted error.")

        self.d_model = fusion_size

        if cfg.MODEL.FEATURE_HEAD.LINEAR_ENABLED:
            if cfg.MODEL.FEATURE_HEAD.LINEAR_OUT_FEATURES != -1:
                self.d_model = cfg.MODEL.FEATURE_HEAD.LINEAR_OUT_FEATURES
            self.input_linear = nn.Sequential(
                nn.Linear(fusion_size, self.d_model),
                nn.ReLU(inplace=False),
            )
        else:
            self.input_linear = nn.Identity()

    def forward(self, visual_input, motion_input):
        if self.with_visual and self.with_motion:
            fusion_input = torch.cat((visual_input, motion_input), dim=-1)
        elif self.with_visual:
            fusion_input = visual_input
        elif self.with_motion:
            fusion_input = motion_input
        else:
            raise ValueError(
                "visual_input and motion_input should not be None at the same time."
            )
        fusion_input = self.input_linear(fusion_input)
        return fusion_input


def build_feature_head(cfg):
    feature_head = FEATURE_HEADS[cfg.DATA.DATA_NAME]
    return feature_head(cfg)
