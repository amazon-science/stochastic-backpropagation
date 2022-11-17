

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model

try:
    from .mask_ops import *
except:
    from mask_ops import *


class BlockSBP(nn.Module):
    """ ConvNeXt Block with SBP implementation
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward_mlp(self, x):
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        return x

    def forward_mlp_sbp(self, x, keep_idx, drop_idx):
        # keep gradients
        x_keep = self.forward_mlp(x[:, keep_idx, :])
        x_out = torch.zeros(x.shape, dtype=x_keep.dtype, device=x.device)
        x_out[:, keep_idx, :] = x_keep

        # drop gradients
        with torch.no_grad():
            x_out[:, drop_idx, :] = self.forward_mlp(x[:, drop_idx, :])
        return x_out

    def forward(self, x, with_gd=False, keep_idx=[], drop_idx=[]):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (B, C, H, W) -> (B, H, W, C)

        if not with_gd:
            x = self.forward_mlp(x)
        else: # with gradient drop: SBP
            B, H, W, C = x.shape
            x = x.view(B, H * W, C)
            x = self.forward_mlp_sbp(x, keep_idx, drop_idx)
            x = x.view(B, H, W, C)

        x = x.permute(0, 3, 1, 2) # (B, H, W, C) -> (B, C, H, W)

        x = input + self.drop_path(x)
        return x


class DownSampleSBP(nn.Module):
    """ DownSample Block with SBP implementation
    """
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.norm = LayerNorm(dim_in, eps=1e-6)
        self.reduction = nn.Linear(4 * dim_in, dim_out)
        self.dim_out = dim_out

    def forward(self, x, with_gd=False, keep_idx=[], drop_idx=[]):
        x = x.permute(0, 2, 3, 1) # (B, C, H, W) -> (B, H, W, C)
        x = self.norm(x)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        
        if not with_gd:
            x_out = self.reduction(x)
        else:
            B, H, W, C = x.shape
            x = x.view(B, -1, C)  
            x_keep = self.reduction(x[:, keep_idx, :])
            x_out = torch.zeros((B, H * W, self.dim_out), dtype=x_keep.dtype, device=x.device)
            x_out[:, keep_idx, :] = x_keep

            # drop gradients
            with torch.no_grad():
                x_out[:, drop_idx, :] = self.reduction(x[:, drop_idx, :])
            x_out = x_out.view(B, H, W, self.dim_out)
        
        x_out = x_out.permute(0, 3, 1, 2).contiguous() # (B, H, W, C) -> (B, C, H, W)
        return x_out


class ConvNeXtSBP(nn.Module):
    """ ConvNeXt model with SBP implementation
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 grad_keep_ratio=1.0, grad_drop_list=[], 
                 grad_drop_downsample_layers=True, linear_downsample=False, 
                 grad_mask_sampling_method='grid', 
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            if grad_drop_downsample_layers or linear_downsample:
                downsample_layer = DownSampleSBP(dims[i], dims[i+1])
            else:
                downsample_layer = nn.Sequential(
                        LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                        nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
                )
            self.downsample_layers.append(downsample_layer)
        
        # SBP set up
        self.grad_keep_ratio = grad_keep_ratio
        if len(grad_drop_list) == 0:
            self.grad_drop_list = [[0] * depths[i] for i in range(4)]
        else:
            self.grad_drop_list = grad_drop_list
            assert all(len(self.grad_drop_list[i]) == depths[i] for i in range(4))
        self.grad_mask_sampling_method = grad_mask_sampling_method
        self.grad_drop_downsample_layers = grad_drop_downsample_layers
        assert grad_mask_sampling_method in ['grid', 'random'], 'grad_mask_sampling_method should be either grid or random'

        print('SBP set up')
        print('self.grad_keep_ratio: ', self.grad_keep_ratio)
        print('self.grad_drop_list: ', self.grad_drop_list)
        print('self.grad_mask_sampling_method: ', self.grad_mask_sampling_method)
        print('self.grad_drop_downsample_layers: ', self.grad_drop_downsample_layers)
        
        # generate base mask on the latest stage that applies SBP
        self.sbp_base_down_ratio = 2 ** (len(depths) + 1)
        for i in range(len(depths)):
            if sum(self.grad_drop_list[len(depths) - 1 - i]) == 0:
                self.sbp_base_down_ratio //= 2
            else: # found the latest stage that applies SBP
                break
        print('self.sbp_base_down_ratio: ', self.sbp_base_down_ratio)
        print('base mask size should be {} for input size 224'.format(224 // self.sbp_base_down_ratio))

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[BlockSBP(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        # SBP, gradient keep/drop base mask 
        if self.grad_keep_ratio < 1:
            B, C, H, W = x.shape
            base_H, base_W = H // self.sbp_base_down_ratio, W // self.sbp_base_down_ratio
            # generate base mask
            if self.grad_mask_sampling_method == 'grid': # grid-wise mask sampling
                # grid mask with random start point
                if self.grad_keep_ratio > 0.5:
                    grid_stride = int(np.round(1.0 / (1 - self.grad_keep_ratio)))
                else:
                    grid_stride = int(np.round(1.0 / self.grad_keep_ratio))
                assert grid_stride > 1, 'gradient keep ratio should in (0, 1) for generating grid mask'
                gird_start_point = np.random.randint(grid_stride)
                base_keep_mask, _, _ = generate_grid_mask2d(base_H, base_W, self.grad_keep_ratio, gird_start_point)
            elif self.grad_mask_sampling_method == 'random':
                base_keep_mask, _, _ = generate_random_mask2d(base_H, base_W, self.grad_keep_ratio)
            else:
                raise NotImplementedError("grad_mask_sampling_method {} not implemented!".format(self.grad_mask_sampling_method))

        for i in range(4):
            # apply SBP
            if self.grad_keep_ratio < 1.0 and sum(self.grad_drop_list[i]) > 0:
                _, _, H, W = x.shape 
                downsample_stride = 4 if i == 0 else 2
                H, W = H // downsample_stride, W // downsample_stride
                block_size = max(1, H // base_H)
                # make sure the spatial drop locations are the same with base mask across stages/layers
                _, keep_idx, drop_idx = generate_mask2d_with_base(H, W, block_size, base_keep_mask)
                # downsample
                if i > 0 and self.grad_drop_downsample_layers and self.grad_drop_list[i][0]:
                    x = self.downsample_layers[i](x, with_gd=True, keep_idx=keep_idx, drop_idx=drop_idx)
                else:
                    x = self.downsample_layers[i](x)
                # blocks
                for j, layer_j in enumerate(self.stages[i]):
                    with_gd = self.grad_drop_list[i][j]
                    x = layer_j(x, with_gd=with_gd, keep_idx=keep_idx, drop_idx=drop_idx)
            else:
                x = self.downsample_layers[i](x)
                x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


@register_model
def convnext_tiny_sbp05(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXtSBP(
        depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
        grad_keep_ratio=0.5, grad_drop_list=[[1] * 3, [1] * 3, [1] * 6 + [0] * 3, [0] * 3], 
        grad_drop_downsample_layers=True, grad_mask_sampling_method='grid', 
        **kwargs)
    return model

@register_model
def convnext_base_sbp05(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXtSBP(
        depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024],
        grad_keep_ratio=0.5, grad_drop_list=[[1] * 3, [1] * 3, [1] * (27 - 6) + [0] * 6, [0] * 3], 
        grad_drop_downsample_layers=True, grad_mask_sampling_method='grid', 
        **kwargs)
    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


if __name__ == '__main__':
    
    img = torch.randn(5, 3, 224, 224)
    model = convnext_tiny_sbp05(drop_path_rate=0.1)
    
    out = model(img)
    print(out.shape)
    print(model)
    print(count_parameters(model))