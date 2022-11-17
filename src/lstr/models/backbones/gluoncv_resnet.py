import torch.nn as nn

from gluoncv.torch.engine.config import get_cfg_defaults
from gluoncv.torch.model_zoo import get_model


class ResNet(nn.Module):

    def __init__(self, config_file):
        super(ResNet, self).__init__()

        cfg = get_cfg_defaults()
        cfg.merge_from_file(config_file)
        self.model = get_model(cfg)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        return x
