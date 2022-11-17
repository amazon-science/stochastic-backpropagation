__all__ = ["build_backbone"]

import torch
from torch import nn

from dlhammer.registry import Registry

from .bninception import BNInception
from .mmaction2_resnet import ResNet as MMAction2ResNet

BACKBONES = Registry()


@BACKBONES.register("rgb_anet_resnet50")
def rgb_anet_resnet50():
    feature_extractor = MMAction2ResNet(depth=50, in_channels=3)
    MODEL_PATH = "pretrained_models/mmaction2_resnet/tsn_r50_320p_1x1x8_50e_activitynet_clip_rgb_20200804-f7bcbf34.pth"
    state_dict = torch.load(MODEL_PATH)["model_state_dict"]
    feature_extractor.load_state_dict(state_dict)
    params = {
        "input_mean": [0.485, 0.456, 0.406],
        "input_std": [0.229, 0.224, 0.225],
        "scale_size": 256,
        "input_size": 224,
        "in_channels": 3,
        "out_channels": 2048,
        "num_crops": 1,
        "div": True,
        "roll": False,
    }
    return feature_extractor, params


@BACKBONES.register("flow_anet_resnet50")
def flow_anet_resnet50():
    feature_extractor = MMAction2ResNet(depth=50, in_channels=10)
    MODEL_PATH = "pretrained_models/mmaction2_resnet/tsn_r50_320p_1x1x8_150e_activitynet_clip_flow_20200804-8622cf38.pth"
    state_dict = torch.load(MODEL_PATH)["model_state_dict"]
    feature_extractor.load_state_dict(state_dict)
    params = {
        "input_mean": [0.5],
        "input_std": [0.5],
        "scale_size": 256,
        "input_size": 224,
        "in_channels": 10,
        "out_channels": 2048,
        "num_crops": 1,
        "div": True,
        "roll": False,
    }
    return feature_extractor, params


@BACKBONES.register("rgb_kinetics_bninception")
def rgb_kinetics_bninception():
    feature_extractor = BNInception(in_channels=3)
    MODEL_PATH = "pretrained_models/bninception/kinetics_tsn_rgb.pth"
    state_dict = torch.load(MODEL_PATH)
    feature_extractor.load_state_dict(state_dict)
    params = {
        "input_mean": [104, 117, 128],
        "input_std": [1, 1, 1],
        "scale_size": 256,
        "input_size": 224,
        "in_channels": 3,
        "out_channels": 1024,
        "num_crops": 10,
        "div": False,
        "roll": True,
    }
    return feature_extractor, params


@BACKBONES.register("flow_kinetics_bninception")
def flow_kinetics_bninception():
    feature_extractor = BNInception(in_channels=10)
    MODEL_PATH = "pretrained_models/bninception/kinetics_tsn_flow.pth"
    state_dict = torch.load(MODEL_PATH)
    feature_extractor.load_state_dict(state_dict)
    params = {
        "input_mean": [128],
        "input_std": [1],
        "scale_size": 256,
        "input_size": 224,
        "in_channels": 10,
        "out_channels": 1024,
        "num_crops": 10,
        "div": False,
        "roll": True,
    }
    return feature_extractor, params


@BACKBONES.register("rgb_kinetics_resnet50")
def rgb_kinetics_resnet50():
    feature_extractor = MMAction2ResNet(depth=50, in_channels=3)
    # MODEL_PATH = 'pretrained_models/mmaction2_resnet/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth'
    MODEL_PATH = "pretrained_models/mmaction2_resnet/tsn_r50_320p_1x1x8_100e_kinetics400_rgb_20200702-ef80e3d7.pth"
    print(f"load checkpoint:{MODEL_PATH}")
    s = torch.load(MODEL_PATH)
    if "model_state_dict" in s.keys():
        state_dict = s["model_state_dict"]
    elif "state_dict" in s.keys():
        old_state_dict = s["state_dict"]
        state_dict = {}
        for k, v in old_state_dict.items():
            if "cls_head" in k:
                continue
            state_dict[k.replace("backbone.", "")] = v
    else:
        raise ValueError(
            f'pretrained_models don\'t have key "model_state_dict" or "state_dict". Got: {s.keys()} '
        )

    feature_extractor.load_state_dict(state_dict)
    params = {
        "input_mean": [0.485, 0.456, 0.406],
        "input_std": [0.229, 0.224, 0.225],
        "scale_size": 256,
        "input_size": 224,
        "in_channels": 3,
        "out_channels": 2048,
        "num_crops": 1,
        "div": True,
        "roll": False,
    }
    return feature_extractor, params


@BACKBONES.register("rgb_imagenet_omni_kinetics_resnet50")
def rgb_imagenet_omni_kinetics_resnet50():
    feature_extractor = MMAction2ResNet(depth=50, in_channels=3)
    MODEL_PATH = "pretrained_models/mmaction2_resnet/tsn_1G1B_pretrained_r50_omni_1x1x3_kinetics400_rgb_20200926-2863fed0.pth"
    state_dict = torch.load(MODEL_PATH)["model_state_dict"]
    feature_extractor.load_state_dict(state_dict)
    params = {
        "input_mean": [0.485, 0.456, 0.406],
        "input_std": [0.229, 0.224, 0.225],
        "scale_size": 256,
        "input_size": 224,
        "in_channels": 3,
        "out_channels": 2048,
        "num_crops": 1,
        "div": True,
        "roll": False,
    }
    return feature_extractor, params


@BACKBONES.register("rgb_igib_omni_kinetics_resnet50")
def rgb_igib_omni_kinetics_resnet50():
    feature_extractor = MMAction2ResNet(depth=50, in_channels=3)
    MODEL_PATH = "pretrained_models/mmaction2_resnet/tsn_1G1B_pretrained_r50_omni_1x1x3_kinetics400_rgb_20200926-2863fed0.pth"
    state_dict = torch.load(MODEL_PATH)["model_state_dict"]
    feature_extractor.load_state_dict(state_dict)
    params = {
        "input_mean": [0.485, 0.456, 0.406],
        "input_std": [0.229, 0.224, 0.225],
        "scale_size": 256,
        "input_size": 224,
        "in_channels": 3,
        "out_channels": 2048,
        "num_crops": 1,
        "div": True,
        "roll": False,
    }
    return feature_extractor, params


class ModelMMCLS(nn.Module):

    """models from mmcls"""

    def __init__(self, cfg):
        """TODO: to be defined."""
        super().__init__()
        from mmcls.models import build_backbone as mmcls_build_backbone
        self.backbone = mmcls_build_backbone(cfg)

    def forward(self, x):
        outs = self.backbone(x)
        return outs[0]

    @classmethod
    def build(cls, cfg, model_path):
        model = cls(cfg)
        state_dict = torch.load(model_path)
        info = model.load_state_dict(state_dict["state_dict"], strict=False)
        print(info)
        return model


@BACKBONES.register("rgb_kinetics_resnext101_32x4d")
def rgb_kinetics_resnext101_32x4d():
    cfg = {
        "type": "ResNeXt",
        "depth": 101,
        "num_stages": 4,
        "out_indices": (3,),
        "groups": 32,
        "width_per_group": 4,
        "style": "pytorch",
    }
    MODEL_PATH = "pretrained_models/mmaction2_resnet/tsn_rn101_32x4d_320p_1x1x3_100e_kinetics400_rgb-16a8b561.pth"
    feature_extractor = ModelMMCLS.build(cfg, MODEL_PATH)
    params = {
        "input_mean": [0.485, 0.456, 0.406],
        "input_std": [0.229, 0.224, 0.225],
        "scale_size": 256,
        "input_size": 224,
        "in_channels": 3,
        "out_channels": 2048,
        "num_crops": 1,
        "div": True,
        "roll": False,
    }
    return feature_extractor, params


_BACKBONES = {
    "rgb": {
        "anet": {
            "resnet50": rgb_anet_resnet50,
        },
        "kinetics": {
            "resnet50": rgb_kinetics_resnet50,
            "bninception": rgb_kinetics_bninception,
        },
        "imagenet_omni_kinetics": {
            "resnet50": rgb_imagenet_omni_kinetics_resnet50,
        },
        "igib_omni_kinetics": {
            "resnet50": rgb_igib_omni_kinetics_resnet50,
        },
    },
    "flow": {
        "anet": {
            "resnet50": flow_anet_resnet50,
        },
        "kinetics": {
            "bninception": flow_kinetics_bninception,
        },
    },
}


def build_backbone(backbone_cfg):
    backbone_name = backbone_cfg.NAME
    if backbone_name in BACKBONES:
        return BACKBONES[backbone_name]()
    else:
        import timm

        if backbone_name == "vit_base_patch16_224":
            kwargs = {
                "pretrained": backbone_cfg.PRETRAINED,
                "num_classes": 0,
                "drop_path_rate": backbone_cfg.DROP_PATH_RATE,
                "drop_rate": backbone_cfg.DROP_RATE,
            }
        return timm.create_model(backbone_name, **kwargs), {}
