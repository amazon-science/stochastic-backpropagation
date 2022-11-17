from torch import nn


class I3DHead(nn.Module):
    """Classification head with average pooling from I3D.
    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels of input feature.
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
    """

    def __init__(self, head_cfg, init_mean=0, init_std=0.01, init_bias=0):
        super(I3DHead, self).__init__()

        self.num_classes = head_cfg.NUM_CLASSES
        self.in_channels = head_cfg.IN_CHANNELS
        self.spatial_type = head_cfg.SPATIAL_TYPE
        if self.spatial_type == 'avg':
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool = nn.Identity()
        self.dropout_ratio = head_cfg.DROPOUT_RATIO
        if self.dropout_ratio > 0:
            self.dropout = nn.Dropout(self.dropout_ratio)
        else:
            self.dropout = nn.Identity()
        self.init_mean = init_mean
        self.init_std = init_std
        self.init_bias = init_bias

        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)

        self.init_weights()

    def init_weights(self):
        """Initiate weights and bias of `fc_cls` from scratch."""
        if self.fc_cls.weight is not None:
            nn.init.normal_(self.fc_cls.weight, self.init_mean, self.init_std)
        if self.fc_cls.bias is not None:
            nn.init.constant_(self.fc_cls.bias, self.init_bias)

    def forward(self, x):
        # [N, in_channels, 4, 7, 7]
        x = self.avg_pool(x)
        # [N, in_channels, 1, 1, 1]
        x = self.dropout(x)
        # [N, in_channels, 1, 1, 1]
        x = x.view(x.shape[0], -1)
        # [N, in_channels]
        cls_score = self.fc_cls(x)
        # [N, num_classes]
        return cls_score
