

import torch
import torch.nn as nn
from functools import partial

import timm
assert timm.__version__ == "0.3.2"

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.resnet import resnet26d, resnet50d
from timm.models.registry import register_model

import numpy as np

try:
    from mask_ops import *
except:
    from .mask_ops import *


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


class MlpSBP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 drop_path=nn.Identity(), drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm = norm_layer(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.drop_path = drop_path

    def forward_mlp(self, x):  # x: B, L=H*W, C
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def forward_mlp_sbp(self, x, grad_keep_idx, grad_drop_idx):
        # keep gradients
        x_keep = self.forward_mlp(x[:, grad_keep_idx, :])
        x_out = torch.zeros(x.shape, dtype=x_keep.dtype, device=x.device)
        x_out[:, grad_keep_idx, :] = x_keep

        # drop gradients
        with torch.no_grad():
            x_out[:, grad_drop_idx, :] = self.forward_mlp(x[:, grad_drop_idx, :])
        return x_out

    def forward(self, x, with_gd=False, grad_keep_idx=[], grad_drop_idx=[]):
        if not with_gd:
            x_mlp = self.forward_mlp(x)
        else:
            x_mlp = self.forward_mlp_sbp(x, grad_keep_idx, grad_drop_idx)
        return x + self.drop_path(x_mlp)
        

class AttentionSBP(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward_attn_map(self, q, k, v):
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        return attn @ v  # has the same shape with q

    def forward_attn_map_sbp(self, q, k, v, grad_keep_idx, grad_drop_idx, drop_qkv=False):
        # keep gradients
        if drop_qkv:
            attn_keep = self.forward_attn_map(
                q[:, :, grad_keep_idx, :],
                k[:, :, grad_keep_idx, :],
                v[:, :, grad_keep_idx, :]
            )
        else:
            attn_keep = self.forward_attn_map(q[:, :, grad_keep_idx, :], k, v)  # q: B, num_heads, num_tokens, head_dim
        attn = torch.zeros(q.shape, dtype=attn_keep.dtype, device=q.device)
        attn[:, :, grad_keep_idx, :] = attn_keep
        
        # drop gradients
        with torch.no_grad():
            attn[:, :, grad_drop_idx, :] = self.forward_attn_map(q[:, :, grad_drop_idx, :], k, v)
        return attn

    def forward_attn_map_sbp_drop_head(self, q, k, v, grad_keep_idx, grad_drop_idx):
        keep_ratio = len(grad_keep_idx) / (len(grad_keep_idx) + len(grad_drop_idx))
        num_heads = q.shape[1]
        random_nums = np.random.rand(num_heads)
        head_grad_keep_idx = random_nums <= keep_ratio
        head_grad_drop_idx = random_nums > keep_ratio

        # keep gradients
        attn_keep = self.forward_attn_map(
            q[:, head_grad_keep_idx, :, :],
            k[:, head_grad_keep_idx, :, :],
            v[:, head_grad_keep_idx, :, :]
        )  # q: B, num_heads, num_tokens, head_dim
        attn = torch.zeros(q.shape, dtype=attn_keep.dtype, device=q.device)
        attn[:, head_grad_keep_idx, :, :] = attn_keep

        # drop gradients
        with torch.no_grad():
            attn[:, head_grad_drop_idx, :, :] = self.forward_attn_map(
                q[:, head_grad_drop_idx, :, :],
                k[:, head_grad_drop_idx, :, :],
                v[:, head_grad_drop_idx, :, :]
            )
        return attn

    def forward(self, x, with_gd=False, grad_keep_idx=[], grad_drop_idx=[], drop_attn_option='qkv'):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        use_index = True
        if not with_gd:
            x = self.forward_attn_map(q, k, v)
        else:
            if drop_attn_option == 'qkv':
                x = self.forward_attn_map_sbp(q, k, v, grad_keep_idx, grad_drop_idx, drop_qkv=True)
            elif drop_attn_option == 'query':
                x = self.forward_attn_map_sbp(q, k, v, grad_keep_idx, grad_drop_idx, drop_qkv=False)
            elif drop_attn_option == 'head':
                x = self.forward_attn_map_sbp_drop_head(q, k, v, grad_keep_idx, grad_drop_idx)
            else:
                raise NotImplementedError(
                    "drop_attn_option {} not implemented, only support for qkv, query, and head!".format(drop_attn_option)
                    )

        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class BlockSBP(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, 
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, 
                 no_gd_on_attn=False, no_gd_on_mlp=False, drop_attn_option='qkv'):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = AttentionSBP(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MlpSBP(
            in_features=dim, hidden_features=mlp_hidden_dim,
            act_layer=act_layer, norm_layer=norm_layer,
            drop_path=self.drop_path, drop=drop
        )

        self.no_gd_on_attn = no_gd_on_attn
        self.no_gd_on_mlp = no_gd_on_mlp
        self.drop_attn_option = drop_attn_option

    def forward(self, x, with_gd=False, grad_keep_idx=[], grad_drop_idx=[]):
        # attn
        if self.no_gd_on_attn or not with_gd:
            x = x + self.drop_path(self.attn(self.norm1(x)))
        else:
            attn = self.attn(self.norm1(x), with_gd, grad_keep_idx, grad_drop_idx, drop_attn_option=self.drop_attn_option)
            x = x + self.drop_path(attn)

        # mlp
        if self.no_gd_on_mlp or not with_gd:
            x = self.mlp(x)
        else:
            x = self.mlp(x, with_gd, grad_keep_idx, grad_drop_idx)
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """

    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class VisionTransformerSBP(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm,
                 grad_keep_ratio=1.0, grad_drop_list=[], grad_mask_sampling_method='grid', 
                 no_gd_on_attn=False, no_gd_on_mlp=False, drop_attn_option='qkv',
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # sbp set up
        self.sbp_input_size = img_size // patch_size
        self.grad_keep_ratio = grad_keep_ratio
        self.grad_mask_sampling_method = grad_mask_sampling_method
        if len(grad_drop_list) == 0:
            self.grad_drop_list = [0] * depth
        else:
            self.grad_drop_list = grad_drop_list
            assert len(self.grad_drop_list) == depth

        print('SBP set up')
        print('self.grad_keep_ratio: ', self.grad_keep_ratio)
        print('self.grad_drop_list: ', self.grad_drop_list)
        print('self.grad_mask_sampling_method: ', self.grad_mask_sampling_method)
        print('no_gd_on_attn, no_gd_on_mlp, drop_attn_option: ', no_gd_on_attn, no_gd_on_mlp, drop_attn_option)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            BlockSBP(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                no_gd_on_attn=no_gd_on_attn, no_gd_on_mlp=no_gd_on_mlp, drop_attn_option=drop_attn_option,
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)  
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # gradient drop
        if self.grad_keep_ratio < 1:
            H = W = self.sbp_input_size
            # generate base mask
            if self.grad_mask_sampling_method == 'grid': # grid-wise mask sampling
                # grid mask with random start point
                if self.grad_keep_ratio > 0.5:
                    grid_stride = int(np.round(1.0 / (1 - self.grad_keep_ratio)))
                else:
                    grid_stride = int(np.round(1.0 / self.grad_keep_ratio))
                assert grid_stride > 1, 'gradient keep ratio should in (0, 1) for generating grid mask'
                gird_start_point = np.random.randint(grid_stride)
                _, grad_keep_idx, grad_drop_idx = generate_grid_mask2d(H, W, self.grad_keep_ratio, gird_start_point)
            elif self.grad_mask_sampling_method == 'random':
                _, grad_keep_idx, grad_drop_idx = generate_random_mask2d(H, W, self.grad_keep_ratio)
            else:
                raise NotImplementedError("grad_drop_option {} not implemented!".format(self.grad_drop_option))

            # correct the indices by shifting one for the cls token
            grad_keep_idx = [0] + [idx + 1 for idx in grad_keep_idx]
            grad_drop_idx = [idx + 1 for idx in grad_drop_idx]

        # forward blocks
        for i, blk in enumerate(self.blocks):
            if self.grad_keep_ratio < 1:
                with_gd = self.grad_drop_list[i]
                x = blk(x, with_gd, grad_keep_idx=grad_keep_idx, grad_drop_idx=grad_drop_idx)
            else:
                x = blk(x)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict


@register_model
def vit_tiny_sbp05(pretrained=False, **kwargs):
    model = VisionTransformerSBP(
        patch_size=16, embed_dim=192, depth=12, num_heads=3,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), qkv_bias=True,
        grad_keep_ratio=0.5, grad_drop_list=[0] * 4 + [1] * 8,
        grad_mask_sampling_method='grid', drop_attn_option='qkv',
        **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def vit_base_sbp05(pretrained=False, **kwargs):
    model = VisionTransformerSBP(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), qkv_bias=True,
        grad_keep_ratio=0.5, grad_drop_list=[0] * 4 + [1] * 8,
        grad_mask_sampling_method='grid', drop_attn_option='qkv',
        **kwargs)
    model.default_cfg = _cfg()
    return model

if __name__ == '__main__':
    
    img = torch.randn(5, 3, 224, 224)
    # model = vit_tiny_sbp05()
    model = vit_base_sbp05(drop_path_rate=0.3)
    
    out = model(img)
    print(out.shape)
    print(model)
    print()

