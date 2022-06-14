import torch
import torch.nn as nn

from .backbones import build_backbone


class FeatureExtractor(nn.Module):
    def __init__(self, model_config, pooling=False, pool_size=(1, 1)):
        """
        Args:
            backbone_name (string): The name of the backbone.
            pooling (bool): Whether to perform spatial pooling to the backbone features.
        """
        super(FeatureExtractor, self).__init__()

        self.backbone, params = build_backbone(model_config)
        self.params = params
        if pooling:
            self.avgpool = nn.AdaptiveAvgPool2d(pool_size)

        self.pooling = pooling

        # self.input_mean = params['input_mean']
        # self.input_std = params['input_std']
        # self.scale_size = params['scale_size']
        # self.input_size = params['input_size']
        # self.in_channels = params['in_channels']
        # self.out_channels = params['out_channels']
        # self.num_crops = params['num_crops']
        # self.div = params['div']
        # self.roll = params['roll']

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): The input video segments. shape: (B,T,C,H,W)
        """
        batch_size, T = x.shape[:2]
        x = x.view(-1, *x.shape[2:])  # (BxT, C,H,W)

        x = self.backbone(x)  # (BxT, C', H', W'), or (BXT, C')
        if self.pooling:
            x = self.avgpool(x)  # (BxT, C', 1, 1)
            x = x.flatten(1)
        x = x.reshape(batch_size, T, -1)  # (B, T, C')
        return x


class TwoStreamFeatureExtractor(nn.Module):
    """two stream feature extractor"""

    def __init__(self, cfg, pooling=False, pool_size=(1, 1)):
        super(TwoStreamFeatureExtractor, self).__init__()

        self.rgb_feature_extractor = FeatureExtractor(
            cfg.MODEL.SPATIAL, pooling=pooling, pool_size=pool_size
        )
        self.flow_feature_extractor = FeatureExtractor(
            cfg.MODEL.SPATIAL.FLOW, pooling=pooling, pool_size=pool_size
        )

    def forward(self, inp):
        """

        Args:
            inp (tuple): (rgb, flow) each of shape [B,T,C,H,W]

        Returns: torch.Tensor[B,T,C]. concated feature.

        """
        rgb, flow = inp
        rgb_feature = self.rgb_feature_extractor(rgb)
        flow_feature = self.flow_feature_extractor(flow)
        return torch.cat([rgb_feature, flow_feature], dim=-1)  # (B,T,C')
