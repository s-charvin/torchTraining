# -*- coding: utf-8 -*-


import types
import re
from functools import partial
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn.init import trunc_normal_

from .common import TwoStreamFusion
from .attention import MultiScaleBlock
from .reversible_mvit import ReversibleMViT
from .utils import (
    calc_mvit_feature_geometry,
    get_3d_sincos_pos_embed,
    round_width,
)
from . import head_helper
from . import stem_helper

try:
    from fairscale.nn.checkpoint import checkpoint_wrapper
except ImportError:
    checkpoint_wrapper = None


class MViT(nn.Module):
    """
    Model builder for MViTv1 and MViTv2.

    "MViTv2: Improved Multiscale Vision Transformers for Classification and Detection"
    Yanghao Li, Chao-Yuan Wu, Haoqi Fan, Karttikeya Mangalam, Bo Xiong, Jitendra Malik, Christoph Feichtenhofer
    https://arxiv.org/abs/2112.01526
    "Multiscale Vision Transformers"
    Haoqi Fan, Bo Xiong, Karttikeya Mangalam, Yanghao Li, Zhicheng Yan, Jitendra Malik, Christoph Feichtenhofer
    https://arxiv.org/abs/2104.11227
    """

    def __init__(self,

                 crop_size,

                 bn_num_sync_devices=1,
                 bn_global_sync=False,

                 num_mlp_layers=1,
                 mlp_dim=2048,
                 bn_mlp=False,

                 bn_sync_mlp=False,

                 input_channel_num=[3, 3],
                 num_classes=400,
                 model_dropout_rate=0.5,


                 head_act="softmax",
                 act_checkpoint=False,
                 detach_final_fc=False,

                 mode="conv",
                 pool_first=False,
                 has_cls_embed=True,
                 patch_kernel=[3, 7, 7],
                 patch_stride=[2, 4, 4],
                 patch_padding=[2, 4, 4],
                 use_2d_patch=False,
                 embed_dim=96,
                 num_heads=1,
                 mlp_ratio=4.0,
                 qkv_bias=True,
                 droppath_rate=0.1,
                 layer_scale_init_value=0.0,
                 depth=16,
                 norm="layernorm",
                 dim_mul=[],
                 head_mul=[],
                 pool_kv_stride=[],
                 pool_kv_stride_adaptive=None,
                 pool_q_stride=[],
                 pool_kvq_kernel=None,
                 zero_decay_pos_cls=True,
                 norm_stem=False,
                 sep_pos_embed=False,
                 dropout_rate=0.0,
                 use_abs_pos=True,
                 rel_pos_spatial=False,
                 rel_pos_temporal=False,
                 rel_pos_zero_init=False,
                 residual_pooling=False,
                 dim_mul_in_att=False,
                 separate_qkv=False,
                 head_init_scale=1.0,
                 use_mean_pooling=False,
                 use_fixed_sincos_pos=False,
                 num_frames=8,
                 enable_detection=False,
                 detection_aligned=True,
                 detection_spatial_scale_factor=16,
                 detection_roi_xform_resolution=7,

                 enable_rev=False,
                 rev_respath_fuse="concat",
                 rev_buffer_layers=[],
                 rev_res_path="conv",
                 rev_pre_q_fusion="avg",

                 #  frozen_bn=False,
                 #  fp16_allreduce=False,
                 #  use_precise_stats=False,
                 #  bn_num_batches_precise=200,
                 #  bn_weight_decay=0.0,
                 #  bn_norm_type="batchnorm",
                 #  bn_num_splits=1,
                 #  dropconnect_rate=0.0,
                 #  fc_init_std=0.01,
                 ):
        super().__init__()
        # Get parameters.
        pool_first = pool_first

        # Prepare input.
        assert min(crop_size) == max(crop_size)
        crop_size = crop_size[0]
        spatial_size = crop_size
        temporal_size = num_frames

        in_chans = input_channel_num[0]
        self.use_2d_patch = use_2d_patch
        self.enable_detection = enable_detection
        self.enable_rev = enable_rev
        self.patch_stride = patch_stride
        if self.use_2d_patch:
            self.patch_stride = [1] + self.patch_stride

        self.T = temporal_size / self.patch_stride[0]
        self.H = math.ceil(spatial_size / self.patch_stride[1])
        self.W = math.ceil(spatial_size / self.patch_stride[2])

        # Prepare output.
        num_classes = num_classes
        embed_dim = embed_dim

        # Prepare backbone
        num_heads = num_heads
        mlp_ratio = mlp_ratio
        qkv_bias = qkv_bias
        self.dropout_rate = dropout_rate
        depth = depth
        layer_scale_init_value = layer_scale_init_value
        head_init_scale = head_init_scale
        mode = mode
        self.has_cls_embed = has_cls_embed
        self.use_mean_pooling = use_mean_pooling

        # Params for positional embedding
        self.use_abs_pos = use_abs_pos
        self.use_fixed_sincos_pos = use_fixed_sincos_pos
        self.sep_pos_embed = sep_pos_embed
        self.rel_pos_spatial = rel_pos_spatial
        self.rel_pos_temporal = rel_pos_temporal
        if norm == "layernorm":
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        else:
            raise NotImplementedError("Only supports layernorm.")
        self.num_classes = num_classes
        self.patch_embed = stem_helper.PatchEmbed(
            dim_in=in_chans,
            dim_out=embed_dim,
            kernel=patch_kernel,
            stride=patch_stride,
            padding=patch_padding,
            conv_2d=self.use_2d_patch,
        )
        self.zero_decay_pos_cls = zero_decay_pos_cls
        if act_checkpoint:
            self.patch_embed = checkpoint_wrapper(self.patch_embed)
        self.input_dims = [temporal_size, spatial_size, spatial_size]
        assert self.input_dims[1] == self.input_dims[2]
        self.patch_dims = [
            math.ceil(self.input_dims[i] / self.patch_stride[i])
            for i in range(len(self.input_dims))
        ]
        num_patches = math.prod(self.patch_dims)

        dpr = [
            x.item() for x in torch.linspace(0, droppath_rate, depth)
        ]  # stochastic depth decay rule

        if self.has_cls_embed:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            pos_embed_dim = num_patches + 1
        else:
            pos_embed_dim = num_patches

        if self.use_abs_pos:
            if self.sep_pos_embed:
                self.pos_embed_spatial = nn.Parameter(
                    torch.zeros(
                        1, self.patch_dims[1] * self.patch_dims[2], embed_dim
                    )
                )
                self.pos_embed_temporal = nn.Parameter(
                    torch.zeros(1, self.patch_dims[0], embed_dim)
                )
                if self.has_cls_embed:
                    self.pos_embed_class = nn.Parameter(
                        torch.zeros(1, 1, embed_dim)
                    )
            else:
                self.pos_embed = nn.Parameter(
                    torch.zeros(
                        1,
                        pos_embed_dim,
                        embed_dim,
                    ),
                    requires_grad=not self.use_fixed_sincos_pos,
                )

        if self.dropout_rate > 0.0:
            self.pos_drop = nn.Dropout(p=self.dropout_rate)

        dim_mul_, head_mul_ = torch.ones(depth + 1), torch.ones(depth + 1)
        for i in range(len(dim_mul)):
            dim_mul_[dim_mul[i][0]] = dim_mul[i][1]
        for i in range(len(head_mul)):
            head_mul_[head_mul[i][0]] = head_mul[i][1]

        pool_q = [[] for i in range(depth)]
        pool_kv = [[] for i in range(depth)]
        stride_q = [[] for i in range(depth)]
        stride_kv = [[] for i in range(depth)]

        for i in range(len(pool_q_stride)):
            stride_q[pool_q_stride[i][0]] = pool_q_stride[i][
                1:
            ]
            if pool_kvq_kernel is not None:
                pool_q[pool_q_stride[i][0]] = pool_kvq_kernel
            else:
                pool_q[pool_q_stride[i][0]] = [
                    s + 1 if s > 1 else s for s in pool_q_stride[i][1:]
                ]

        # If POOL_KV_STRIDE_ADAPTIVE is not None, initialize POOL_KV_STRIDE.
        if pool_kv_stride_adaptive is not None:
            _stride_kv = pool_kv_stride_adaptive
            pool_kv_stride = []
            for i in range(depth):
                if len(stride_q[i]) > 0:
                    _stride_kv = [
                        max(_stride_kv[d] // stride_q[i][d], 1)
                        for d in range(len(_stride_kv))
                    ]
                pool_kv_stride.append([i] + _stride_kv)

        for i in range(len(pool_kv_stride)):
            stride_kv[pool_kv_stride[i][0]] = pool_kv_stride[
                i
            ][1:]
            if pool_kvq_kernel is not None:
                pool_kv[
                    pool_kv_stride[i][0]
                ] = pool_kvq_kernel
            else:
                pool_kv[pool_kv_stride[i][0]] = [
                    s + 1 if s > 1 else s
                    for s in pool_kv_stride[i][1:]
                ]

        self.pool_q = pool_q
        self.pool_kv = pool_kv
        self.stride_q = stride_q
        self.stride_kv = stride_kv

        self.norm_stem = norm_layer(embed_dim) if norm_stem else None

        input_size = self.patch_dims

        if self.enable_rev:

            # rev does not allow cls token
            assert not self.has_cls_embed

            self.rev_backbone = ReversibleMViT(
                model=self,
                embed_dim=embed_dim,
                depth=depth,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                res_path=rev_res_path,
                dropout_rate=dropout_rate,
                droppath_rate=droppath_rate,
                has_cls_embed=has_cls_embed,
                mode=mode,
                pool_first=pool_first,
                rel_pos_spatial=rel_pos_spatial,
                rel_pos_temporal=rel_pos_temporal,
                rel_pos_zero_init=rel_pos_zero_init,
                residual_pooling=residual_pooling,
                separate_qkv=separate_qkv,
                pre_q_fusion=rev_pre_q_fusion,
                norm=norm,
                dim_mul=dim_mul,
                head_mul=head_mul,
                buffer_layers=rev_buffer_layers,)

            embed_dim = round_width(
                embed_dim, dim_mul_.prod(), divisor=num_heads
            )

            self.fuse = TwoStreamFusion(
                rev_respath_fuse, dim=2 * embed_dim
            )

            if "concat" in rev_respath_fuse:
                self.norm = norm_layer(2 * embed_dim)
            else:
                self.norm = norm_layer(embed_dim)

        else:

            self.blocks = nn.ModuleList()

            for i in range(depth):
                num_heads = round_width(num_heads, head_mul_[i])
                if dim_mul_in_att:
                    dim_out = round_width(
                        embed_dim,
                        dim_mul_[i],
                        divisor=round_width(num_heads, head_mul_[i]),
                    )
                else:
                    dim_out = round_width(
                        embed_dim,
                        dim_mul_[i + 1],
                        divisor=round_width(num_heads, head_mul_[i + 1]),
                    )
                attention_block = MultiScaleBlock(
                    dim=embed_dim,
                    dim_out=dim_out,
                    num_heads=num_heads,
                    input_size=input_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop_rate=self.dropout_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    kernel_q=pool_q[i] if len(pool_q) > i else [],
                    kernel_kv=pool_kv[i] if len(pool_kv) > i else [],
                    stride_q=stride_q[i] if len(stride_q) > i else [],
                    stride_kv=stride_kv[i] if len(stride_kv) > i else [],
                    mode=mode,
                    has_cls_embed=self.has_cls_embed,
                    pool_first=pool_first,
                    rel_pos_spatial=self.rel_pos_spatial,
                    rel_pos_temporal=self.rel_pos_temporal,
                    rel_pos_zero_init=rel_pos_zero_init,
                    residual_pooling=residual_pooling,
                    dim_mul_in_att=dim_mul_in_att,
                    separate_qkv=separate_qkv,
                )

                if act_checkpoint:
                    attention_block = checkpoint_wrapper(attention_block)
                self.blocks.append(attention_block)
                if len(stride_q[i]) > 0:
                    input_size = [
                        size // stride
                        for size, stride in zip(input_size, stride_q[i])
                    ]

                embed_dim = dim_out

            self.norm = norm_layer(embed_dim)

        if self.enable_detection:
            self.head = head_helper.ResNetRoIHead(
                dim_in=[embed_dim],
                num_classes=num_classes,
                pool_size=[[temporal_size // self.patch_stride[0], 1, 1]],
                resolution=[[detection_roi_xform_resolution] * 2],
                scale_factor=[detection_spatial_scale_factor],
                dropout_rate=model_dropout_rate,
                act_func=head_act,
                aligned=detection_aligned,
            )
        else:
            self.head = head_helper.TransformerBasicHead(
                2 *
                embed_dim if (
                    "concat" in rev_respath_fuse and self.enable_rev) else embed_dim,
                num_classes,
                dropout_rate=model_dropout_rate,
                act_func=head_act,
                num_mlp_layers=num_mlp_layers,
                mlp_dim=mlp_dim,
                bn_mlp=bn_mlp,
                bn_sync_mlp=bn_sync_mlp,
                num_sync_devices=bn_num_sync_devices,
                global_sync=bn_global_sync,
                detach_final_fc=detach_final_fc,
            )
        if self.use_abs_pos:
            if self.sep_pos_embed:
                trunc_normal_(self.pos_embed_spatial, std=0.02)
                trunc_normal_(self.pos_embed_temporal, std=0.02)
                if self.has_cls_embed:
                    trunc_normal_(self.pos_embed_class, std=0.02)
            else:
                trunc_normal_(self.pos_embed, std=0.02)
                if self.use_fixed_sincos_pos:
                    pos_embed = get_3d_sincos_pos_embed(
                        self.pos_embed.shape[-1],
                        self.H,
                        self.T,
                        cls_token=self.has_cls_embed,
                    )
                    self.pos_embed.data.copy_(
                        torch.from_numpy(pos_embed).float().unsqueeze(0)
                    )

        if self.has_cls_embed:
            trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

        self.head.projection.weight.data.mul_(head_init_scale)
        self.head.projection.bias.data.mul_(head_init_scale)

        self.feat_size, self.feat_stride = calc_mvit_feature_geometry(
            num_frames=num_frames,
            crop_size=crop_size,
            depth=depth,
            patch_stride=patch_stride,
            pool_q_stride=pool_q_stride,
        )

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.02)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        names = []
        if self.zero_decay_pos_cls:
            if self.use_abs_pos:
                if self.sep_pos_embed:
                    names.extend(
                        [
                            "pos_embed_spatial",
                            "pos_embed_temporal",
                            "pos_embed_class",
                        ]
                    )
                else:
                    names.append("pos_embed")
            if self.rel_pos_spatial:
                names.extend(["rel_pos_h", "rel_pos_w", "rel_pos_hw"])
            if self.rel_pos_temporal:
                names.extend(["rel_pos_t"])
            if self.has_cls_embed:
                names.append("cls_token")

        return names

    def _get_pos_embed(self, pos_embed, bcthw):

        if len(bcthw) == 4:
            t, h, w = 1, bcthw[-2], bcthw[-1]
        else:
            t, h, w = bcthw[-3], bcthw[-2], bcthw[-1]
        if self.has_cls_embed:
            cls_pos_embed = pos_embed[:, 0:1, :]
            pos_embed = pos_embed[:, 1:]
        txy_num = pos_embed.shape[1]
        p_t, p_h, p_w = self.patch_dims
        assert p_t * p_h * p_w == txy_num

        if (p_t, p_h, p_w) != (t, h, w):
            new_pos_embed = F.interpolate(
                pos_embed[:, :, :]
                .reshape(1, p_t, p_h, p_w, -1)
                .permute(0, 4, 1, 2, 3),
                size=(t, h, w),
                mode="trilinear",
            )
            pos_embed = new_pos_embed.reshape(
                1, -1, t * h * w).permute(0, 2, 1)

        if self.has_cls_embed:
            pos_embed = torch.cat((cls_pos_embed, pos_embed), dim=1)

        return pos_embed

    def _forward_reversible(self, x):
        """
        Reversible specific code for forward computation.
        """
        # rev does not support cls token or detection
        assert not self.has_cls_embed
        assert not self.enable_detection

        x = self.rev_backbone(x)

        if self.use_mean_pooling:
            x = self.fuse(x)
            x = x.mean(1)
            x = self.norm(x)
        else:
            x = self.norm(x)
            x = self.fuse(x)
            x = x.mean(1)

        x = self.head(x)

        return x

    def forward(self, x, bboxes=None, return_attn=False):
        # [B * vf_seg_len,C,  vf_seq_len, W, H]
        x, bcthw = self.patch_embed(x)
        bcthw = list(bcthw)
        if len(bcthw) == 4:  # Fix bcthw in case of 4D tensor
            bcthw.insert(2, torch.tensor(self.T))
        T, H, W = bcthw[-3], bcthw[-2], bcthw[-1]
        assert len(bcthw) == 5 and (T, H, W) == (self.T, self.H, self.W), bcthw
        B, N, C = x.shape

        s = 1 if self.has_cls_embed else 0
        if self.use_fixed_sincos_pos:
            x += self.pos_embed[:, s:, :]  # s: on/off cls token

        if self.has_cls_embed:
            cls_tokens = self.cls_token.expand(
                B, -1, -1
            )  # stole cls_tokens impl from Phil Wang, thanks
            if self.use_fixed_sincos_pos:
                cls_tokens = cls_tokens + self.pos_embed[:, :s, :]
            x = torch.cat((cls_tokens, x), dim=1)

        if self.use_abs_pos:
            if self.sep_pos_embed:
                pos_embed = self.pos_embed_spatial.repeat(
                    1, self.patch_dims[0], 1
                ) + torch.repeat_interleave(
                    self.pos_embed_temporal,
                    self.patch_dims[1] * self.patch_dims[2],
                    dim=1,
                )
                if self.has_cls_embed:
                    pos_embed = torch.cat([self.pos_embed_class, pos_embed], 1)
                x += self._get_pos_embed(pos_embed, bcthw)
            else:
                x += self._get_pos_embed(self.pos_embed, bcthw)

        if self.dropout_rate:
            x = self.pos_drop(x)

        if self.norm_stem:
            x = self.norm_stem(x)

        thw = [T, H, W]

        if self.enable_rev:
            x = self._forward_reversible(x)

        else:
            for blk in self.blocks:
                x, thw = blk(x, thw)

            if self.enable_detection:
                assert not self.enable_rev

                x = self.norm(x)
                if self.has_cls_embed:
                    x = x[:, 1:]

                B, _, C = x.shape
                x = x.transpose(1, 2).reshape(B, C, thw[0], thw[1], thw[2])

                x = self.head([x], bboxes)

            else:
                if self.use_mean_pooling:
                    if self.has_cls_embed:
                        x = x[:, 1:]
                    x = x.mean(1)
                    x = self.norm(x)
                elif self.has_cls_embed:
                    x = self.norm(x)
                    x = x[:, 0]
                else:  # this is default, [norm->mean]
                    x = self.norm(x)
                    x = x.mean(1)
                x = self.head(x)

        return x


REV_MVIT_B_16x4_CONV = partial(
    MViT,
    # crop_size=240,
    # input_channel_num=[3, 3],
    # num_frames=30,
    # num_classes=4,

    model_dropout_rate=0.5,
    zero_decay_pos_cls=False,
    has_cls_embed=False,
    sep_pos_embed=True,
    use_abs_pos=True,
    depth=16,
    num_heads=1,
    embed_dim=96,
    patch_kernel=[3, 7, 7],
    patch_stride=[2, 4, 4],
    patch_padding=[1, 3, 3],
    mlp_ratio=4.0,
    qkv_bias=False,
    droppath_rate=0.05,
    norm="layernorm",
    mode="conv",
    dim_mul=[[1, 2.0], [3, 2.0], [14, 2.0]],
    head_mul=[[1, 2.0], [3, 2.0], [14, 2.0]],
    pool_kvq_kernel=[3, 3, 3],
    pool_kv_stride_adaptive=[1, 8, 8],
    pool_q_stride=[[1, 1, 2, 2], [3, 1, 2, 2], [14, 1, 2, 2]],
    enable_rev=True,
    rev_respath_fuse="concat",
    rev_buffer_layers=[1, 3, 14],
    rev_res_path="conv",
)


REV_MVIT_B = partial(
    MViT,
    # crop_size=224,
    # input_channel_num=[3, 3],
    # num_classes=4,

    num_frames=1,
    use_2d_patch=True,
    has_cls_embed=False,
    embed_dim=96,
    patch_kernel=[7, 7],
    patch_stride=[4, 4],
    patch_padding=[3, 3],
    num_heads=1,
    mlp_ratio=4.0,
    qkv_bias=True,
    droppath_rate=0.1,
    depth=16,
    norm="layernorm",
    mode="conv",
    dim_mul=[[1, 2.0], [3, 2.0], [14, 2.0]],
    head_mul=[[1, 2.0], [3, 2.0], [14, 2.0]],
    pool_kvq_kernel=[1, 3, 3],
    pool_kv_stride_adaptive=[1, 4, 4],
    pool_q_stride=[[1, 1, 2, 2], [3, 1, 2, 2], [14, 1, 2, 2]],
    separate_qkv=True,
    enable_rev=True,
    rev_respath_fuse="concat",
    rev_buffer_layers=[1, 3, 14],
    rev_res_path="conv",
    rev_pre_q_fusion="concat_linear_2",
    model_dropout_rate=0.0,
    zero_decay_pos_cls=False,
)

model_urls = {
    'REV_MVIT_B': 'https://dl.fbaipublicfiles.com/pyslowfast/rev/REV_MVIT_B.pyth',
    "REV_MVIT_B_16x4_CONV": "https://dl.fbaipublicfiles.com/pyslowfast/rev/REV_MVIT_B_16x4.pyth"
}


def update_state_dict(state_dict):
    # '.'s are no longer allowed in module names, but pervious _DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls. This pattern is used
    # to find such keys.
    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    return state_dict


class EmptyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def modify_mvit(model):
    # Modify attributs

    if model.head.projection._get_name() == "Linear":
        model.last_linear = model.head.projection
        model.head.projection = EmptyModel()
    elif model.head.projection._get_name() == "MLPHead":
        model.last_linear = model.head.projection.projection[-1]
        model.head.projection.projection[-1] = EmptyModel()

    def _forward_reversible(self, x):
        """
        Reversible specific code for forward computation.
        """
        # rev does not support cls token or detection
        assert not self.has_cls_embed
        assert not self.enable_detection

        x = self.rev_backbone(x)

        if self.use_mean_pooling:
            x = self.fuse(x)
            x = x.mean(1)
            x = self.norm(x)
        else:
            x = self.norm(x)
            x = self.fuse(x)
            x = x.mean(1)

        x = self.head(x)
        x = self.last_linear(x)

        return x

    def forward(self, x, bboxes=None, return_attn=False):
        # [B * vf_seg_len,C,  vf_seq_len, W, H]
        x, bcthw = self.patch_embed(x)
        bcthw = list(bcthw)
        if len(bcthw) == 4:  # Fix bcthw in case of 4D tensor
            bcthw.insert(2, self.T)
        T, H, W = bcthw[-3], bcthw[-2], bcthw[-1]
        assert len(bcthw) == 5 and (T, H, W) == (self.T, self.H, self.W), bcthw
        B, N, C = x.shape

        s = 1 if self.has_cls_embed else 0
        if self.use_fixed_sincos_pos:
            x += self.pos_embed[:, s:, :]  # s: on/off cls token

        if self.has_cls_embed:
            cls_tokens = self.cls_token.expand(
                B, -1, -1
            )  # stole cls_tokens impl from Phil Wang, thanks
            if self.use_fixed_sincos_pos:
                cls_tokens = cls_tokens + self.pos_embed[:, :s, :]
            x = torch.cat((cls_tokens, x), dim=1)

        if self.use_abs_pos:
            if self.sep_pos_embed:
                pos_embed = self.pos_embed_spatial.repeat(
                    1, self.patch_dims[0], 1
                ) + torch.repeat_interleave(
                    self.pos_embed_temporal,
                    self.patch_dims[1] * self.patch_dims[2],
                    dim=1,
                )
                if self.has_cls_embed:
                    pos_embed = torch.cat([self.pos_embed_class, pos_embed], 1)
                x += self._get_pos_embed(pos_embed, bcthw)
            else:
                x += self._get_pos_embed(self.pos_embed, bcthw)

        if self.dropout_rate:
            x = self.pos_drop(x)

        if self.norm_stem:
            x = self.norm_stem(x)

        thw = [T, H, W]

        if self.enable_rev:
            x = self._forward_reversible(x)

        else:
            for blk in self.blocks:
                x, thw = blk(x, thw)

            if self.enable_detection:
                assert not self.enable_rev

                x = self.norm(x)
                if self.has_cls_embed:
                    x = x[:, 1:]

                B, _, C = x.shape
                x = x.transpose(1, 2).reshape(B, C, thw[0], thw[1], thw[2])

                x = self.head([x], bboxes)
                x = self.last_linear(x)
            else:
                if self.use_mean_pooling:
                    if self.has_cls_embed:
                        x = x[:, 1:]
                    x = x.mean(1)
                    x = self.norm(x)
                elif self.has_cls_embed:
                    x = self.norm(x)
                    x = x[:, 0]
                else:  # this is default, [norm->mean]
                    x = self.norm(x)
                    x = x.mean(1)
                x = self.head(x)
                x = self.last_linear(x)
        return x

    model.forward = types.MethodType(forward, model)
    model._forward_reversible = types.MethodType(_forward_reversible, model)
    return model


def rev_mvit_b_16x4_conv3d(num_classes=400, pretrained=False, crop_size=[224, 224], input_channel_num=[3, 3], num_frames=30,):
    model = REV_MVIT_B_16x4_CONV(
        crop_size=crop_size,
        input_channel_num=input_channel_num,
        num_classes=num_classes,
        num_frames=num_frames
    )
    if pretrained:
        model = REV_MVIT_B_16x4_CONV(
            crop_size=crop_size,
            input_channel_num=input_channel_num,
            num_classes=400,
            num_frames=num_frames
        )
        state_dict = model_zoo.load_url(model_urls["REV_MVIT_B_16x4_CONV"])
        state_dict = update_state_dict(state_dict)
        del state_dict["model_state"]["cls_token"]
        del state_dict["model_state"]["pos_embed_class"]

        pos_num = min(state_dict["model_state"]["pos_embed_spatial"].shape[1], model.state_dict()[
                      "pos_embed_spatial"].shape[1])
        if pos_num < state_dict["model_state"]["pos_embed_spatial"].shape[1]:
            state_dict["model_state"]["pos_embed_spatial"] = state_dict["model_state"]["pos_embed_spatial"][:, :pos_num, :]
        else:
            state_dict["model_state"]["pos_embed_spatial"] = model.state_dict()[
                "pos_embed_spatial"]

        pos_num = min(state_dict["model_state"]["pos_embed_temporal"].shape[1], model.state_dict()[
                      "pos_embed_temporal"].shape[1])
        if pos_num < state_dict["model_state"]["pos_embed_temporal"].shape[1]:
            state_dict["model_state"]["pos_embed_temporal"] = state_dict["model_state"]["pos_embed_temporal"][:, :pos_num, :]
        else:
            state_dict["model_state"]["pos_embed_temporal"] = model.state_dict()[
                "pos_embed_temporal"]

        model.load_state_dict(state_dict["model_state"])
    else:
        model = REV_MVIT_B_16x4_CONV(
            crop_size=crop_size,
            input_channel_num=input_channel_num,
            num_classes=num_classes,
            num_frames=num_frames
        )

    model = modify_mvit(model)
    return model


def rev_mvit_b_conv2d(num_classes=1000, pretrained=False, crop_size=[224, 224], input_channel_num=[3, 3]):

    if pretrained:
        model = REV_MVIT_B(
            crop_size=crop_size,
            input_channel_num=input_channel_num,
            num_classes=1000,
        )
        state_dict = model_zoo.load_url(model_urls["REV_MVIT_B"])
        state_dict = update_state_dict(state_dict)
        pos_num = min(state_dict["model_state"]["pos_embed"].shape[1], model.state_dict()[
                      "pos_embed"].shape[1])
        if pos_num < state_dict["model_state"]["pos_embed"].shape[1]:
            state_dict["model_state"]["pos_embed"] = state_dict["model_state"]["pos_embed"][:, :pos_num, :]

        model.load_state_dict(state_dict["model_state"])
    else:
        model = REV_MVIT_B(
            crop_size=crop_size,
            input_channel_num=input_channel_num,
            num_classes=num_classes,
        )
    model = modify_mvit(model)
    return model
