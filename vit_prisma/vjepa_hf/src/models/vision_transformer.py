# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math
from functools import partial

import torch
import torch.nn as nn

from vit_prisma.vjepa_hf.src.models.utils.patch_embed import PatchEmbed, PatchEmbed3D
from vit_prisma.vjepa_hf.src.models.utils.modules import Block
from vit_prisma.vjepa_hf.src.models.utils.pos_embs import get_2d_sincos_pos_embed, get_3d_sincos_pos_embed
from vit_prisma.vjepa_hf.src.utils.tensors import trunc_normal_
from vit_prisma.vjepa_hf.src.masks.utils import apply_masks


class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(
        self,
        input_size=(224, 224),
        patch_size=16,
        num_frames=1,
        tubelet_size=2,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        out_layers=None,
        uniform_power=False,
        use_SiLU=False,
        wide_SiLU=True,
        is_causal=False,
        use_sdpa=True,
        use_activation_checkpointing=False,
        **kwargs
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.out_layers = out_layers

        if type(input_size) is int:
            input_size = (input_size, input_size)
        self.img_height, self.img_width = input_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.is_video = num_frames > 1

        self.use_activation_checkpointing = use_activation_checkpointing

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        # Tokenize pixels with convolution
        if self.is_video:
            self.patch_embed = PatchEmbed3D(
                patch_size=patch_size,
                tubelet_size=tubelet_size,
                in_chans=in_chans,
                embed_dim=embed_dim)
            self.num_patches = (
                (num_frames // tubelet_size)
                * (input_size[0] // patch_size)
                * (input_size[1] // patch_size)
            )
        else:
            self.patch_embed = PatchEmbed(
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim)
            self.num_patches = (
                (input_size[0] // patch_size)
                * (input_size[1] // patch_size)
            )

        # Position embedding
        self.uniform_power = uniform_power
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim),
            requires_grad=False)

        # Attention Blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                is_causal=is_causal,
                use_sdpa=use_sdpa,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                act_layer=nn.SiLU if use_SiLU else nn.GELU,
                wide_SiLU=wide_SiLU,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # ------ initialize weights
        if self.pos_embed is not None:
            self._init_pos_embed(self.pos_embed.data)  # sincos pos-embed
        self.init_std = init_std
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _init_pos_embed(self, pos_embed):
        embed_dim = pos_embed.size(-1)
        grid_size = self.img_height // self.patch_size  # TODO: update; initialization currently assumes square input
        if self.is_video:
            grid_depth = self.num_frames // self.tubelet_size
            sincos = get_3d_sincos_pos_embed(
                embed_dim,
                grid_size,
                grid_depth,
                cls_token=False,
                uniform_power=self.uniform_power
            )
        else:
            sincos = get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False)
        pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv3d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def get_num_layers(self):
        return len(self.blocks)

    def no_weight_decay(self):
        return {}

    def forward(self, x, masks=None):
        """
        :param x: input image/video
        :param masks: indices of patch tokens to mask (remove)
        """
        if masks is not None and not isinstance(masks, list):
            masks = [masks]

        # Tokenize input
        pos_embed = self.pos_embed
        if pos_embed is not None:
            pos_embed = self.interpolate_pos_encoding(x, pos_embed)
        x = self.patch_embed(x)
        if pos_embed is not None:
            x += pos_embed
        B, N, D = x.shape

        # Mask away unwanted tokens (if masks provided)
        if masks is not None:
            x = apply_masks(x, masks)
            masks = torch.cat(masks, dim=0)

        # Fwd prop
        outs = []
        for i, blk in enumerate(self.blocks):
            if self.use_activation_checkpointing:
                x = torch.utils.checkpoint.checkpoint(blk, x, False, masks, use_reentrant=False)
            else:
                x = blk(x, mask=masks)
            if self.out_layers is not None and i in self.out_layers:
                outs.append(self.norm(x))

        if self.out_layers is not None:
            return outs

        if self.norm is not None:
            x = self.norm(x)

        return x

    def interpolate_pos_encoding(self, x, pos_embed):

        _, N, dim = pos_embed.shape

        if self.is_video:

            # If pos_embed already corret size, just return
            _, _, T, H, W = x.shape
            if H == self.img_height and W == self.img_width and T == self.num_frames:
                return pos_embed

            # Convert depth, height, width of input to be measured in patches
            # instead of pixels/frames
            T = T // self.tubelet_size
            H = H // self.patch_size
            W = W // self.patch_size

            # Compute the initialized shape of the positional embedding measured
            # in patches
            N_t = self.num_frames // self.tubelet_size
            N_h = self.img_height // self.patch_size
            N_w = self.img_width // self.patch_size
            assert N_h * N_w * N_t == N, 'Positional embedding initialized incorrectly'

            # Compute scale factor for spatio-temporal interpolation
            scale_factor = (T/N_t, H/N_h, W/N_w)

            pos_embed = nn.functional.interpolate(
                pos_embed.reshape(1, N_t, N_h, N_w, dim).permute(0, 4, 1, 2, 3),
                scale_factor=scale_factor,
                mode='trilinear')
            pos_embed = pos_embed.permute(0, 2, 3, 4, 1).view(1, -1, dim)
            return pos_embed

        else:

            # If pos_embed already corret size, just return
            _, _, H, W = x.shape
            if H == self.img_height and W == self.img_width:
                return pos_embed

            # Compute scale factor for spatial interpolation
            npatch = (H // self.patch_size) * (W // self.patch_size)
            scale_factor = math.sqrt(npatch / N)

            pos_embed = nn.functional.interpolate(
                pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
                scale_factor=scale_factor,
                mode='bicubic')
            pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
            return pos_embed


def vit_synthetic(patch_size=16, **kwargs):
    # For performance testing only
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=1, depth=1, num_heads=1, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_tiny(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_small(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_giant(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=1408, depth=40, num_heads=16, mlp_ratio=48/11,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_gigantic(patch_size=14, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=1664, depth=48, num_heads=16, mpl_ratio=64/13,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model


VIT_EMBED_DIMS = {
    'vit_synthetic': 1,
    'vit_tiny': 192,
    'vit_small': 384,
    'vit_base': 768,
    'vit_large': 1024,
    'vit_huge': 1280,
    'vit_giant': 1408,
    'vit_gigantic': 1664,
}
