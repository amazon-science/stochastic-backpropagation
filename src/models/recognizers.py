from torch import nn

from dlhammer.bootstrap import logger

from .builder import MODELS as registry


@registry.register('swin')
class VideoSwinTransformer(nn.Module):
    """Model of video swin transformer"""

    def __init__(self, cfg):
        super(VideoSwinTransformer, self).__init__()

        if hasattr(cfg, 'MEMSAVE') and cfg.MEMSAVE.ENABLE:
            cfg.MODEL.SWIN.BACKBONE['graddrop_config'] = cfg.MEMSAVE.GRADDROP_CFG

            if cfg.MEMSAVE.VERSION == 'v1':
                from .archs.vswin.swin_sbp import SwinTransformer3D
            elif cfg.MEMSAVE.VERSION == 'v2':
                from .archs.vswin.swin_sbp_checkboard import SwinTransformer3D
            else:
                raise NameError(f'Unknow MEMSAVE version: {cfg.MEMSAVE.VERSION}')
        else:
            from .archs.vswin.swin import SwinTransformer3D

        from .archs.vswin.heads import I3DHead

        self.backbone = SwinTransformer3D(**cfg.MODEL.SWIN.BACKBONE)

        try:
            self.backbone.inflate_weights(logger)
        except Exception:
            logger.info(f'Not found ImageNet pretrained weights, continue...')

        self.head = I3DHead(cfg.MODEL.SWIN.HEAD)

    def forward(self, inp):
        x, _ = inp
        x = self.backbone(x)
        logits = self.head(x)
        return logits
