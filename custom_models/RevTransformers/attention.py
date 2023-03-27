#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved. All Rights Reserved.


import numpy
import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """
    Stochastic Depth per sample.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()  # binarize
    output = x.div(keep_prob) * mask
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop_rate=0.0,
    ):
        super().__init__()
        self.drop_rate = drop_rate
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        if self.drop_rate > 0.0:
            self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        if self.drop_rate > 0.0:
            x = self.drop(x)
        x = self.fc2(x)
        if self.drop_rate > 0.0:
            x = self.drop(x)
        return x


def attention_pool(tensor, pool, hw_shape, norm=None):

    if pool is None:
        return tensor, hw_shape

    tensor_dim = tensor.ndim
    if tensor_dim == 4:
        pass
    elif tensor_dim == 3:
        tensor = tensor.unsqueeze(1)
    else:
        raise NotImplementedError(
            f"Unsupported input dimension {tensor.shape}")

    B, N, L, C = tensor.shape
    H, W = hw_shape

    # B Head S*F C/Head -> B*Head C/Head S F
    tensor = tensor.reshape(B * N, H, W, C).permute(0, 3, 1, 2).contiguous()
    tensor = pool(tensor)  # 池化

    hw_shape = [tensor.shape[2], tensor.shape[3]]
    L_pooled = tensor.shape[2] * tensor.shape[3]
    # B*Head C/Head S F -> B Head S*F C/Head
    tensor = tensor.reshape(B, N, C, L_pooled).transpose(2, 3)

    if norm is not None:
        tensor = norm(tensor)

    if tensor_dim == 3:
        tensor = tensor.squeeze(1)
    return tensor, hw_shape


def cal_rel_pos_spatial(
    attn,
    q,
    q_shape,
    k_shape,
    rel_pos_h,
    rel_pos_w,
):
    """
    Spatial Relative Positional Embeddings.
    """
    sp_idx = 0
    q_h, q_w = q_shape
    k_h, k_w = k_shape

    # Scale up rel pos if shapes for q and k are different.
    q_h_ratio = max(k_h / q_h, 1.0)
    k_h_ratio = max(q_h / k_h, 1.0)
    dist_h = (
        torch.arange(q_h)[:, None] * q_h_ratio -
        torch.arange(k_h)[None, :] * k_h_ratio
    )
    dist_h += (k_h - 1) * k_h_ratio
    q_w_ratio = max(k_w / q_w, 1.0)
    k_w_ratio = max(q_w / k_w, 1.0)
    dist_w = (
        torch.arange(q_w)[:, None] * q_w_ratio -
        torch.arange(k_w)[None, :] * k_w_ratio
    )
    dist_w += (k_w - 1) * k_w_ratio

    Rh = rel_pos_h[dist_h.long()]
    Rw = rel_pos_w[dist_w.long()]

    B, n_head, q_N, dim = q.shape

    r_q = q[:, :, sp_idx:].reshape(B, n_head, q_h, q_w, dim)
    rel_h = torch.einsum("byhwc,hkc->byhwk", r_q, Rh)
    rel_w = torch.einsum("byhwc,wkc->byhwk", r_q, Rw)

    attn[:, :, sp_idx:, sp_idx:] = (
        attn[:, :, sp_idx:, sp_idx:].view(B, -1, q_h, q_w, k_h, k_w)
        + rel_h[:, :, :, :, :, None]
        + rel_w[:, :, :, :, None, :]
    ).view(B, -1, q_h * q_w, k_h * k_w)

    return attn


class MultiScaleAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        num_heads=8,
        qkv_bias=False,
        kernel_q=(1, 1),
        kernel_kv=(1, 1),
        stride_q=(1, 1),
        stride_kv=(1, 1),
        norm_layer=nn.LayerNorm,
        mode="conv",

    ):
        super().__init__()

        self.num_heads = num_heads
        self.dim_out = dim_out
        head_dim = dim_out // num_heads
        self.scale = head_dim**-0.5

        padding_q = [int(q // 2) for q in kernel_q]
        padding_kv = [int(kv // 2) for kv in kernel_kv]

        self.qkv = nn.Linear(dim, dim_out * 3, bias=qkv_bias)

        self.proj = nn.Linear(dim_out, dim_out)

        # Skip pooling with kernel and stride size of (1, 1, 1).
        if numpy.prod(kernel_q) == 1 and numpy.prod(stride_q) == 1:
            kernel_q = ()
        if numpy.prod(kernel_kv) == 1 and numpy.prod(stride_kv) == 1:
            kernel_kv = ()
        self.mode = mode

        dim_conv = dim_out // num_heads if mode == "conv" else dim_out
        self.pool_q = (
            nn.Conv2d(
                dim_conv,
                dim_conv,
                kernel_q,
                stride=stride_q,
                padding=padding_q,
                groups=dim_conv,
                bias=False,
            )
            if len(kernel_q) > 0
            else None
        )
        self.norm_q = norm_layer(dim_conv) if len(kernel_q) > 0 else None
        self.pool_k = (
            nn.Conv2d(
                dim_conv,
                dim_conv,
                kernel_kv,
                stride=stride_kv,
                padding=padding_kv,
                groups=dim_conv,
                bias=False,
            )
            if len(kernel_kv) > 0
            else None
        )
        self.norm_k = norm_layer(dim_conv) if len(kernel_kv) > 0 else None
        self.pool_v = (
            nn.Conv2d(
                dim_conv,
                dim_conv,
                kernel_kv,
                stride=stride_kv,
                padding=padding_kv,
                groups=dim_conv,
                bias=False,
            )
            if len(kernel_kv) > 0
            else None
        )
        self.norm_v = norm_layer(dim_conv) if len(kernel_kv) > 0 else None

        # relative pos embedding

    def forward(self, x, hw_shape):
        B, N, _ = x.shape
        # B H*W C , (S, F)

        # B H*W C -> 3 B Head H*W C/Head
        qkv = (
            self.qkv(x).reshape(
                B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        # B Head H*W C/Head -> B Head H`*W` C/Head
        q, q_shape = attention_pool(
            q,
            self.pool_q,
            hw_shape,
            norm=self.norm_q if hasattr(self, "norm_q") else None,
        )

        # B Head H*W C/Head -> B Head H`*W` C/Head
        k, k_shape = attention_pool(
            k,
            self.pool_k,
            hw_shape,
            norm=self.norm_k if hasattr(self, "norm_k") else None,
        )

        # B Head H*W C/Head -> B Head H`*W` C/Head
        v, v_shape = attention_pool(
            v,
            self.pool_v,
            hw_shape,
            norm=self.norm_v if hasattr(self, "norm_v") else None,
        )

        N = q.shape[2]

        # 注意力计算
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        x = attn @ v
        # 注意力计算结束

        # 残差
        x = x + q

        # B Head H`*W` C/Head -> B H`*W` C
        x = x.transpose(1, 2).reshape(B, -1, self.dim_out)
        # B H`*W` C > B H`*W` C
        x = self.proj(x)

        # B H`*W` C, (H`, W`)
        return x, q_shape


class MultiScaleBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        kernel_q=(1, 1),
        kernel_kv=(1, 1),
        stride_q=(1, 1),
        stride_kv=(1, 1),
        mode="conv",

    ):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim)

        att_dim = dim
        self.attn = MultiScaleAttention(
            dim,
            att_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            kernel_q=kernel_q,
            kernel_kv=kernel_kv,
            stride_q=stride_q,
            stride_kv=stride_kv,
            norm_layer=norm_layer,
            mode=mode,
        )
        self.drop_path = DropPath(
            drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(att_dim)
        mlp_hidden_dim = int(att_dim * mlp_ratio)

        mlp_dim_out = dim_out
        self.mlp = Mlp(
            in_features=att_dim,
            hidden_features=mlp_hidden_dim,
            out_features=mlp_dim_out,
        )
        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

        if len(stride_q) > 0 and numpy.prod(stride_q) > 1:
            kernel_skip = [s + 1 if s > 1 else s for s in stride_q]
            stride_skip = stride_q
            padding_skip = [int(skip // 2) for skip in kernel_skip]
            self.pool_skip = nn.MaxPool2d(
                kernel_skip, stride_skip, padding_skip, ceil_mode=False
            )
        else:
            self.pool_skip = None

    def forward(self, x, hw_shape):
        # print(
        #     f"{self.__class__.__name__},  Input: {x.shape}, {hw_shape[0], hw_shape[1]}")

        # B H*W C , (H, W)
        x_norm = self.norm1(x)
        # B H*W C , (H, W) -> B H`*W` C, (H`, W`)
        x_block, hw_shape_new = self.attn(x_norm, hw_shape)

        # 对输入做一次池化
        x_res, _ = attention_pool(
            x, self.pool_skip, hw_shape
        )

        x = x_res + self.drop_path(x_block)
        x_norm = self.norm2(x)
        x_mlp = self.mlp(x_norm)

        if self.dim != self.dim_out:
            x = self.proj(x_norm)
        x = x + self.drop_path(x_mlp)
        return x, hw_shape_new
