

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

from .common import DropPath, Mlp


def attention_pool(tensor, pool, thw_shape, has_cls_embed=True, norm=None):
    """
    将输入张量的空间维度进行池化, 并返回池化后的张量以及更新后的空间维度.

    Args:
        tensor (torch.Tensor): 输入张量, 维度为 (B, N, L, C) 或 (B, L, C), 其中 B 表示 batch size, N 表示样本数, L 表示序列长度, C 表示特征通道数.
        pool (torch.nn.Module): 用于池化的模块, 可以是 nn.AvgPool2d 或 nn.AdaptiveAvgPool2d 等.
        thw_shape (list): 一个包含 3 个整数的列表, 表示原始张量的空间维度, 格式为 [T, H, W], 其中 T 表示时间维度, H 和 W 表示图像的高和宽.
        has_cls_embed (bool): 是否有类别嵌入, 默认为 True.
        norm (torch.nn.Module): 对池化后的张量进行标准化的模块, 可以是 nn.LayerNorm 等.

    Returns:
        tensor (torch.Tensor): 经过池化后的张量, 维度为 (B, N, C', L_pooled), 其中 C' 表示池化后的特征通道数, L_pooled 表示池化后的序列长度.
        thw_shape (list): 更新后的空间维度, 格式为 [T_pooled, H_pooled, W_pooled].
    """

    if pool is None:
        return tensor, thw_shape
    tensor_dim = tensor.ndim
    if tensor_dim == 4:
        pass
    elif tensor_dim == 3:
        tensor = tensor.unsqueeze(1)
    else:
        raise NotImplementedError(
            f"Unsupported input dimension {tensor.shape}")

    if has_cls_embed:
        cls_tok, tensor = tensor[:, :, :1, :], tensor[:, :, 1:, :]

    B, N, L, C = tensor.shape
    T, H, W = thw_shape
    tensor = (
        tensor.reshape(B * N, T, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
    )

    tensor = pool(tensor)

    thw_shape = [tensor.shape[2], tensor.shape[3], tensor.shape[4]]
    L_pooled = tensor.shape[2] * tensor.shape[3] * tensor.shape[4]
    tensor = tensor.reshape(B, N, C, L_pooled).transpose(2, 3)
    if has_cls_embed:
        tensor = torch.cat((cls_tok, tensor), dim=2)
    if norm is not None:
        tensor = norm(tensor)
    # Assert tensor_dim in [3, 4]
    if tensor_dim == 4:
        pass
    else:  # tensor_dim == 3:
        tensor = tensor.squeeze(1)
    return tensor, thw_shape


def get_rel_pos(rel_pos, d):
    """
    对相对位置编码进行插值或截断, 以适应不同维度的位置嵌入.

    Args:
        rel_pos (torch.Tensor): 相对位置编码, 维度为 (ori_d, ori_d, C).
        d (int): 新的位置编码维度.

    Returns:
        torch.Tensor: 插值或截断后的相对位置编码, 维度为 (d, d, C).

    Raises:
        AssertionError: 如果输入 rel_pos 不是 3 维张量, 或者 rel_pos.shape[0] 不等于 rel_pos.shape[1], 则抛出 AssertionError.

    """
    if isinstance(d, int):
        ori_d = rel_pos.shape[0]
        if ori_d == d:
            return rel_pos
        else:
            # Interpolate rel pos.
            new_pos_embed = F.interpolate(
                rel_pos.reshape(1, ori_d, -1).permute(0, 2, 1),
                size=d,
                mode="linear",
            )

            return new_pos_embed.reshape(-1, d).permute(1, 0)


def cal_rel_pos_spatial(
    attn, q, k, has_cls_embed, q_shape, k_shape, rel_pos_h, rel_pos_w
):
    """
    Decomposed Spatial Relative Positional Embeddings.
    对于给定的注意力张量, 使用空间相对位置嵌入来更新它.

    Args:
        attn (tensor): 输入的注意力张量, 形状为 [B, H, q_N, k_N].
        q (tensor): 查询张量, 形状为 [B, H, q_N, dim].
        k (tensor): 键张量, 形状为 [B, H, k_N, dim].
        has_cls_embed (bool): 是否有类别嵌入.
        q_shape (tuple): 查询张量形状 [q_t, q_h, q_w].
        k_shape (tuple): 键张量形状 [k_t, k_h, k_w].
        rel_pos_h (tensor): 水平相对位置嵌入, 形状为 [2 * max(q_h, k_h) - 1, dim].
        rel_pos_w (tensor): 垂直相对位置嵌入, 形状为 [2 * max(q_w, k_w) - 1, dim].

    Returns:
        tensor: 更新后的注意力张量, 形状与输入相同.
    """
    sp_idx = 1 if has_cls_embed else 0
    q_t, q_h, q_w = q_shape
    k_t, k_h, k_w = k_shape
    dh = int(2 * max(q_h, k_h) - 1)
    dw = int(2 * max(q_w, k_w) - 1)

    # Scale up rel pos if shapes for q and k are different.
    q_h_ratio = max(k_h / q_h, 1.0)
    k_h_ratio = max(q_h / k_h, 1.0)
    dist_h = (
        torch.arange(q_h)[:, None] * q_h_ratio
        - torch.arange(k_h)[None, :] * k_h_ratio
    )
    dist_h += (k_h - 1) * k_h_ratio
    q_w_ratio = max(k_w / q_w, 1.0)
    k_w_ratio = max(q_w / k_w, 1.0)
    dist_w = (
        torch.arange(q_w)[:, None] * q_w_ratio
        - torch.arange(k_w)[None, :] * k_w_ratio
    )
    dist_w += (k_w - 1) * k_w_ratio

    # Intepolate rel pos if needed.
    rel_pos_h = get_rel_pos(rel_pos_h, dh)
    rel_pos_w = get_rel_pos(rel_pos_w, dw)
    Rh = rel_pos_h[dist_h.long()]
    Rw = rel_pos_w[dist_w.long()]

    B, n_head, q_N, dim = q.shape

    r_q = q[:, :, sp_idx:].reshape(B, n_head, q_t, q_h, q_w, dim)
    rel_h_q = torch.einsum(
        "bythwc,hkc->bythwk", r_q, Rh
    )  # [B, H, q_t, qh, qw, k_h]
    rel_w_q = torch.einsum(
        "bythwc,wkc->bythwk", r_q, Rw
    )  # [B, H, q_t, qh, qw, k_w]

    attn[:, :, sp_idx:, sp_idx:] = (
        attn[:, :, sp_idx:, sp_idx:].view(B, -1, q_t, q_h, q_w, k_t, k_h, k_w)
        + rel_h_q[:, :, :, :, :, None, :, None]
        + rel_w_q[:, :, :, :, :, None, None, :]
    ).view(B, -1, q_t * q_h * q_w, k_t * k_h * k_w)

    return attn


def cal_rel_pos_temporal(attn, q, has_cls_embed, q_shape, k_shape, rel_pos_t):
    """
    Temporal Relative Positional Embeddings.
    计算时序相对位置嵌入.

    Args:
    attn: torch.Tensor, 输入的注意力张量, 形状为[B, n_head, N, N], 其中B为批次大小, n_head为头数, 
          N为序列长度, 可包含CLS嵌入.
    q: torch.Tensor, 查询张量, 形状为[B, n_head, q_t, q_h, q_w, dim], 其中B为批次大小, n_head为头数, 
       q_t、q_h、q_w为查询张量的时序、高度和宽度维度, dim为嵌入维度.
    has_cls_embed: bool, 是否包含CLS嵌入.
    q_shape: tuple, 查询张量的形状, 形状为(q_t, q_h, q_w).
    k_shape: tuple, 键值张量的形状, 形状为(k_t, k_h, k_w).
    rel_pos_t: torch.Tensor, 时序相对位置嵌入, 形状为[2*max(q_t, k_t)-1, dim], 其中dim为嵌入维度.

    Returns:
    attn: torch.Tensor, 输出的注意力张量, 形状为[B, n_head, N, N].
    """
    sp_idx = 1 if has_cls_embed else 0
    q_t, q_h, q_w = q_shape
    k_t, k_h, k_w = k_shape
    dt = int(2 * max(q_t, k_t) - 1)
    # Intepolate rel pos if needed.
    rel_pos_t = get_rel_pos(rel_pos_t, dt)

    # Scale up rel pos if shapes for q and k are different.
    q_t_ratio = max(k_t / q_t, 1.0)
    k_t_ratio = max(q_t / k_t, 1.0)
    dist_t = (
        torch.arange(q_t)[:, None] * q_t_ratio
        - torch.arange(k_t)[None, :] * k_t_ratio
    )
    dist_t += (k_t - 1) * k_t_ratio
    Rt = rel_pos_t[dist_t.long()]

    B, n_head, q_N, dim = q.shape

    r_q = q[:, :, sp_idx:].reshape(B, n_head, q_t, q_h, q_w, dim)
    # [B, H, q_t, q_h, q_w, dim] -> [q_t, B, H, q_h, q_w, dim] -> [q_t, B*H*q_h*q_w, dim]
    r_q = r_q.permute(2, 0, 1, 3, 4, 5).reshape(
        q_t, B * n_head * q_h * q_w, dim
    )

    # [q_t, B*H*q_h*q_w, dim] * [q_t, dim, k_t] = [q_t, B*H*q_h*q_w, k_t] -> [B*H*q_h*q_w, q_t, k_t]
    rel = torch.matmul(r_q, Rt.transpose(1, 2)).transpose(0, 1)
    # [B*H*q_h*q_w, q_t, k_t] -> [B, H, q_t, q_h, q_w, k_t]
    rel = rel.view(B, n_head, q_h, q_w, q_t, k_t).permute(0, 1, 4, 2, 3, 5)

    attn[:, :, sp_idx:, sp_idx:] = (
        attn[:, :, sp_idx:, sp_idx:].view(B, -1, q_t, q_h, q_w, k_t, k_h, k_w)
        + rel[:, :, :, :, :, :, None, None]
    ).view(B, -1, q_t * q_h * q_w, k_t * k_h * k_w)

    return attn


class MultiScaleAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        input_size,
        num_heads=8,
        qkv_bias=False,
        dropout_rate=0.0,
        kernel_q=(1, 1, 1),
        kernel_kv=(1, 1, 1),
        stride_q=(1, 1, 1),
        stride_kv=(1, 1, 1),
        norm_layer=nn.LayerNorm,
        has_cls_embed=True,
        # Options include `conv`, `avg`, and `max`.
        mode="conv",
        # If True, perform pool before projection.
        pool_first=False,
        rel_pos_spatial=False,
        rel_pos_temporal=False,
        rel_pos_zero_init=False,
        residual_pooling=False,
        separate_qkv=False,
    ):
        """_summary_

        Args:
            dim (_type_): 输入张量的特征维度大小.
            dim_out (_type_): 输出张量的特征维度大小.

            input_size (_type_): 输入张量的空间尺寸和时间长度，类型为 (C, H, W, T) 的四元组.
            num_heads (int, optional): 多头注意力中注意力头的数量.. Defaults to 8.
            qkv_bias (bool, optional): 是否在线性变换中加入偏置.. Defaults to False.
            drop_rate (float, optional): dropout 概率.. Defaults to 0.0.
            kernel_q (tuple, optional): 用于进行查询线性变换的卷积核大小，类型为 (D_q, H_q, W_q) 的三元组.. Defaults to (1, 1, 1).
            kernel_kv (tuple, optional): 用于进行键值线性变换的卷积核大小，类型为 (D_kv, H_kv, W_kv) 的三元组.. Defaults to (1, 1, 1).
            stride_q (tuple, optional): 用于进行查询线性变换的卷积核的步长大小，类型为 (D_q, H_q, W_q) 的三元组.. Defaults to (1, 1, 1).
            stride_kv (tuple, optional): 用于进行键值线性变换的卷积核的步长大小，类型为 (D_kv, H_kv, W_kv) 的三元组.. Defaults to (1, 1, 1).
            norm_layer (_type_, optional): 用于进行归一化的函数.. Defaults to nn.LayerNorm.
            has_cls_embed (bool, optional): 是否对输入张量添加“类别”嵌入向量.. Defaults to True.
            mode (str, optional): 用于对查询、键、值张量进行下采样的方式，包括 conv (卷积) 、avg (平均池化) 、max (最大池化) 三种选项.. Defaults to "conv".
            pool_first (bool, optional): 是否在线性变换之前对查询、键、值张量进行下采样.. Defaults to False.
            rel_pos_spatial (bool, optional): 是否在多尺度注意力中考虑空间位置信息.. Defaults to False.
            rel_pos_temporal (bool, optional): 是否在多尺度注意力中考虑时间位置信息.. Defaults to False.
            rel_pos_zero_init (bool, optional): 是否将位置编码张量初始化为零.. Defaults to False.
            residual_pooling (bool, optional): 是否使用残差连接对下采样的张量进行处理.. Defaults to False.
            separate_qkv (bool, optional): 是否对查询、键、值进行分别处理.. Defaults to False.

        Raises:
            NotImplementedError: _description_
        """
        super().__init__()
        self.pool_first = pool_first
        self.separate_qkv = separate_qkv
        self.dropout_rate = dropout_rate
        self.num_heads = num_heads
        self.dim_out = dim_out
        head_dim = dim_out // num_heads
        self.scale = head_dim**-0.5
        self.has_cls_embed = has_cls_embed
        self.mode = mode
        padding_q = [int(q // 2) for q in kernel_q]
        padding_kv = [int(kv // 2) for kv in kernel_kv]

        if pool_first or separate_qkv:
            self.q = nn.Linear(dim, dim_out, bias=qkv_bias)
            self.k = nn.Linear(dim, dim_out, bias=qkv_bias)
            self.v = nn.Linear(dim, dim_out, bias=qkv_bias)
        else:
            self.qkv = nn.Linear(dim, dim_out * 3, bias=qkv_bias)

        self.proj = nn.Linear(dim_out, dim_out)
        if dropout_rate > 0.0:
            self.proj_drop = nn.Dropout(dropout_rate)

        # Skip pooling with kernel and stride size of (1, 1, 1).
        if numpy.prod(kernel_q) == 1 and numpy.prod(stride_q) == 1:
            kernel_q = ()
        if numpy.prod(kernel_kv) == 1 and numpy.prod(stride_kv) == 1:
            kernel_kv = ()

        if mode in ("avg", "max"):
            pool_op = nn.MaxPool3d if mode == "max" else nn.AvgPool3d
            self.pool_q = (
                pool_op(kernel_q, stride_q, padding_q, ceil_mode=False)
                if len(kernel_q) > 0
                else None
            )
            self.pool_k = (
                pool_op(kernel_kv, stride_kv, padding_kv, ceil_mode=False)
                if len(kernel_kv) > 0
                else None
            )
            self.pool_v = (
                pool_op(kernel_kv, stride_kv, padding_kv, ceil_mode=False)
                if len(kernel_kv) > 0
                else None
            )
        elif mode == "conv" or mode == "conv_unshared":
            if pool_first:
                dim_conv = dim // num_heads if mode == "conv" else dim
            else:
                dim_conv = dim_out // num_heads if mode == "conv" else dim_out
            self.pool_q = (
                nn.Conv3d(
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
                nn.Conv3d(
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
                nn.Conv3d(
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
        else:
            raise NotImplementedError(f"Unsupported model {mode}")

        self.rel_pos_spatial = rel_pos_spatial
        self.rel_pos_temporal = rel_pos_temporal
        if self.rel_pos_spatial:
            assert input_size[1] == input_size[2]
            size = input_size[1]
            q_size = size // stride_q[1] if len(stride_q) > 0 else size
            kv_size = size // stride_kv[1] if len(stride_kv) > 0 else size
            rel_sp_dim = 2 * max(q_size, kv_size) - 1

            self.rel_pos_h = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))
            if not rel_pos_zero_init:
                trunc_normal_(self.rel_pos_h, std=0.02)
                trunc_normal_(self.rel_pos_w, std=0.02)
        if self.rel_pos_temporal:
            self.rel_pos_t = nn.Parameter(
                torch.zeros(2 * input_size[0] - 1, head_dim)
            )
            if not rel_pos_zero_init:
                trunc_normal_(self.rel_pos_t, std=0.02)

        self.residual_pooling = residual_pooling

    def forward(self, x, thw_shape):
        B, N, _ = x.shape

        if self.pool_first:
            if self.mode == "conv_unshared":
                fold_dim = 1
            else:
                fold_dim = self.num_heads
            x = x.reshape(B, N, fold_dim, -1).permute(0, 2, 1, 3)
            q = k = v = x
        else:
            assert self.mode != "conv_unshared"
            if not self.separate_qkv:
                qkv = (
                    self.qkv(x)
                    .reshape(B, N, 3, self.num_heads, -1)
                    .permute(2, 0, 3, 1, 4)
                )
                q, k, v = qkv[0], qkv[1], qkv[2]
            else:
                q = k = v = x
                q = (
                    self.q(q)
                    .reshape(B, N, self.num_heads, -1)
                    .permute(0, 2, 1, 3)
                )
                k = (
                    self.k(k)
                    .reshape(B, N, self.num_heads, -1)
                    .permute(0, 2, 1, 3)
                )
                v = (
                    self.v(v)
                    .reshape(B, N, self.num_heads, -1)
                    .permute(0, 2, 1, 3)
                )

        q, q_shape = attention_pool(
            q,
            self.pool_q,
            thw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_q if hasattr(self, "norm_q") else None,
        )
        k, k_shape = attention_pool(
            k,
            self.pool_k,
            thw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_k if hasattr(self, "norm_k") else None,
        )
        v, v_shape = attention_pool(
            v,
            self.pool_v,
            thw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_v if hasattr(self, "norm_v") else None,
        )

        if self.pool_first:
            q_N = (
                numpy.prod(q_shape) + 1
                if self.has_cls_embed
                else numpy.prod(q_shape)
            )
            k_N = (
                numpy.prod(k_shape) + 1
                if self.has_cls_embed
                else numpy.prod(k_shape)
            )
            v_N = (
                numpy.prod(v_shape) + 1
                if self.has_cls_embed
                else numpy.prod(v_shape)
            )

            q = q.permute(0, 2, 1, 3).reshape(B, q_N, -1)
            q = (
                self.q(q)
                .reshape(B, q_N, self.num_heads, -1)
                .permute(0, 2, 1, 3)
            )

            v = v.permute(0, 2, 1, 3).reshape(B, v_N, -1)
            v = (
                self.v(v)
                .reshape(B, v_N, self.num_heads, -1)
                .permute(0, 2, 1, 3)
            )

            k = k.permute(0, 2, 1, 3).reshape(B, k_N, -1)
            k = (
                self.k(k)
                .reshape(B, k_N, self.num_heads, -1)
                .permute(0, 2, 1, 3)
            )

        N = q.shape[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)
        if self.rel_pos_spatial:
            attn = cal_rel_pos_spatial(
                attn,
                q,
                k,
                self.has_cls_embed,
                q_shape,
                k_shape,
                self.rel_pos_h,
                self.rel_pos_w,
            )

        if self.rel_pos_temporal:
            attn = cal_rel_pos_temporal(
                attn,
                q,
                self.has_cls_embed,
                q_shape,
                k_shape,
                self.rel_pos_t,
            )
        attn = attn.softmax(dim=-1)

        x = attn @ v

        if self.residual_pooling:
            if self.has_cls_embed:
                x[:, :, 1:, :] += q[:, :, 1:, :]
            else:
                x = x + q

        x = x.transpose(1, 2).reshape(B, -1, self.dim_out)
        x = self.proj(x)

        if self.dropout_rate > 0.0:
            x = self.proj_drop(x)
        return x, q_shape


class MultiScaleBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        num_heads,
        input_size,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        drop_path=0.0,
        layer_scale_init_value=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        up_rate=None,
        kernel_q=(1, 1, 1),
        kernel_kv=(1, 1, 1),
        stride_q=(1, 1, 1),
        stride_kv=(1, 1, 1),
        mode="conv",
        has_cls_embed=True,
        pool_first=False,
        rel_pos_spatial=False,
        rel_pos_temporal=False,
        rel_pos_zero_init=False,
        residual_pooling=False,
        dim_mul_in_att=False,
        separate_qkv=False,
    ):
        """_summary_

        Args:
            dim (_type_): 输入特征的通道数
            dim_out (_type_): 输出特征的通道数
            num_heads (_type_): 注意力头数
            input_size (_type_): 输入特征的尺寸
            mlp_ratio (float, optional): MLP 中间层维度和输入层的比例. Defaults to 4.0.
            qkv_bias (bool, optional): 是否对注意力权值和偏置进行初始化. Defaults to False.
            qk_scale (_type_, optional): 注意力权值的缩放因子. Defaults to None.
            drop_rate (float, optional): Dropout 概率. Defaults to 0.0.
            drop_path (float, optional): DropPath 概率. Defaults to 0.0.
            layer_scale_init_value (float, optional): Gamma 初始化值. Defaults to 0.0.
            act_layer (_type_, optional): 激活函数类型. Defaults to nn.GELU.
            norm_layer (_type_, optional): 归一化函数类型. Defaults to nn.LayerNorm.
            up_rate (_type_, optional): 上采样倍率. Defaults to None.
            kernel_q (tuple, optional): 注意力查询卷积核大小. Defaults to (1, 1, 1).
            kernel_kv (tuple, optional): 注意力键值卷积核大小. Defaults to (1, 1, 1).
            stride_q (tuple, optional): 注意力查询卷积步长. Defaults to (1, 1, 1).
            stride_kv (tuple, optional): 注意力键值卷积步长. Defaults to (1, 1, 1).
            mode (str, optional): 注意力卷积的类型. Defaults to "conv".
            has_cls_embed (bool, optional): 是否含有类别嵌入. Defaults to True.
            pool_first (bool, optional): 是否先进行池化操作. Defaults to False.
            rel_pos_spatial (bool, optional): 是否使用空间相对位置编码. Defaults to False.
            rel_pos_temporal (bool, optional): 是否使用时间相对位置编码. Defaults to False.
            rel_pos_zero_init (bool, optional): 相对位置编码的初始值是否为 0. Defaults to False.
            residual_pooling (bool, optional): 是否在池化之前对残差连接进行卷积操作. Defaults to False.
            dim_mul_in_att (bool, optional): 是否在注意力层的输入和输出通道数不一致时，使用线性映射变换. Defaults to False.
            separate_qkv (bool, optional): 是否对查询、键、值进行分别卷积. Defaults to False.
        """
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim)
        self.dim_mul_in_att = dim_mul_in_att
        kernel_skip = [s + 1 if s > 1 else s for s in stride_q]
        stride_skip = stride_q
        padding_skip = [int(skip // 2) for skip in kernel_skip]
        att_dim = dim_out if dim_mul_in_att else dim
        self.attn = MultiScaleAttention(
            dim,
            att_dim,
            num_heads=num_heads,
            input_size=input_size,
            qkv_bias=qkv_bias,
            dropout_rate=drop_rate,
            kernel_q=kernel_q,
            kernel_kv=kernel_kv,
            stride_q=stride_q,
            stride_kv=stride_kv,
            norm_layer=norm_layer,
            has_cls_embed=has_cls_embed,
            mode=mode,
            pool_first=pool_first,
            rel_pos_spatial=rel_pos_spatial,
            rel_pos_temporal=rel_pos_temporal,
            rel_pos_zero_init=rel_pos_zero_init,
            residual_pooling=residual_pooling,
            separate_qkv=separate_qkv,
        )
        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        self.norm2 = norm_layer(att_dim)
        mlp_hidden_dim = int(att_dim * mlp_ratio)
        self.has_cls_embed = has_cls_embed
        # TODO: check the use case for up_rate, and merge the following lines
        if up_rate is not None and up_rate > 1:
            mlp_dim_out = dim * up_rate
        else:
            mlp_dim_out = dim_out
        self.mlp = Mlp(
            in_features=att_dim,
            hidden_features=mlp_hidden_dim,
            out_features=mlp_dim_out,
            act_layer=act_layer,
            drop_rate=drop_rate,
        )
        if layer_scale_init_value > 0:
            self.gamma_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True
            )
            self.gamma_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim_out)),
                requires_grad=True,
            )
        else:
            self.gamma_1, self.gamma_2 = None, None

        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

        self.pool_skip = (
            nn.MaxPool3d(
                kernel_skip, stride_skip, padding_skip, ceil_mode=False
            )
            if len(stride_skip) > 0 and numpy.prod(stride_skip) > 1
            else None
        )

    def forward(self, x, thw_shape=None):
        x_norm = self.norm1(x)
        x_block, thw_shape_new = self.attn(x_norm, thw_shape)
        if self.dim_mul_in_att and self.dim != self.dim_out:
            x = self.proj(x_norm)
        x_res, _ = attention_pool(
            x, self.pool_skip, thw_shape, has_cls_embed=self.has_cls_embed
        )
        if self.gamma_1 is not None:
            x = x_res + self.drop_path(self.gamma_1 * x_block)
        else:
            x = x_res + self.drop_path(x_block)
        x_norm = self.norm2(x)
        x_mlp = self.mlp(x_norm)
        if not self.dim_mul_in_att and self.dim != self.dim_out:
            x = self.proj(x_norm)
        if self.gamma_2 is not None:
            x = x + self.drop_path(self.gamma_2 * x_mlp)
        else:
            x = x + self.drop_path(x_mlp)
        if thw_shape:
            return x, thw_shape_new
        else:
            return x
