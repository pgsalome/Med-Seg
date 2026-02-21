from typing import Type

import torch
from dynamic_network_architectures.building_blocks.helper import (
    convert_dim_to_conv_op,
    get_matching_instancenorm,
)
from dynamic_network_architectures.building_blocks.plain_conv_encoder import (
    PlainConvEncoder,
)
from monai.networks.nets.swin_unetr import SwinTransformer as SwinViT
from monai.utils import ensure_tuple_rep
from torch import nn
from torch.nn.modules.conv import _ConvNd


class TriadSwinEncoder(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        feature_size: int = 48,
        drop_path_rate: float = 0.0,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        patch_size = ensure_tuple_rep(2, spatial_dims)
        window_size = ensure_tuple_rep(7, spatial_dims)

        self.swin = SwinViT(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=[2, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=drop_path_rate,
            norm_layer=torch.nn.LayerNorm,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
            use_v2=True,
        )

    def forward(self, x):
        return self.swin(x)


class TriadPlainConvEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage,
        conv_op: Type[_ConvNd],
        kernel_sizes,
        strides,
        n_conv_per_stage,
    ):
        super().__init__()
        self.encoder = PlainConvEncoder(
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_conv_per_stage=n_conv_per_stage,
            conv_bias=True,
            norm_op=get_matching_instancenorm(conv_op),
            norm_op_kwargs={"eps": 1e-5, "affine": True},
            dropout_op=None,
            dropout_op_kwargs=None,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={"inplace": True},
            return_skips=True,
            nonlin_first=False,
        )

    def forward(self, x):
        return self.encoder(x)


def build_triad_plain_encoder(in_channels: int = 1):
    unet_base = 32
    unet_max = 320
    conv_kernel_sizes = [[3, 3, 3]] * 6
    pool_strides = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
    n_conv = [2, 2, 2, 2, 2, 2]

    dim = len(conv_kernel_sizes[0])
    conv_op = convert_dim_to_conv_op(dim)
    num_stages = len(conv_kernel_sizes)
    features = [min(unet_base * (2**i), unet_max) for i in range(num_stages)]

    return TriadPlainConvEncoder(
        input_channels=in_channels,
        n_stages=num_stages,
        features_per_stage=features,
        conv_op=conv_op,
        kernel_sizes=conv_kernel_sizes,
        strides=pool_strides,
        n_conv_per_stage=n_conv,
    )
