# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved. All Rights Reserved.


"""MViT models."""

import math
from functools import partial

import torch
import torch.nn as nn
from .attention import MultiScaleBlock
from torch.nn.init import trunc_normal_
import numpy as np


def round_width(width, multiplier, min_width=1, divisor=1):
    if not multiplier:
        return width
    width *= multiplier
    min_width = min_width or divisor
    # if verbose:
    #     logger.info(f"min width {min_width}")
    #     logger.info(f"width {width} divisor {divisor}")
    #     logger.info(f"other {int(width + divisor / 2) // divisor * divisor}")

    width_out = max(min_width, int(width + divisor / 2) // divisor * divisor)
    if width_out < 0.9 * width:
        width_out += divisor
    return int(width_out)


class MViT_(nn.Module):
    """
    Improved Multiscale Vision Transformers for Classification and Detection
    https://arxiv.org/abs/2112.01526

    Multiscale Vision Transformers
    https://arxiv.org/abs/2104.11227
    """

    def __init__(self):
        super().__init__()
        # 输入参数
        in_channels = 1
        # 用于训练的图像裁剪大小, 也就是输入图大小
        input_size = [128, 128]

        # 输出参数
        output_size = 4
        EMBED_DIM = 96

        # 模型参数
        NUM_HEADS = 1
        DEPTH = 16  # Depth of the transformer.
        self.CLS_EMBED_ON = False  # use cls embed
        self.USE_ABS_POS = False  # absolute positional embedding.
        # no decay on positional embedding and cls embedding
        self.ZERO_DECAY_POS_CLS = False

        norm_layer = partial(nn.LayerNorm, eps=1e-6)  # 固定 LayerNorm 的 eps 参数

        # 将原始的2维图像转换成一系列的 1 维 patch embeddings

        PATCH_KERNEL = [7, 7]
        PATCH_STRIDE = [4, 4]
        PATCH_PADDING = [3, 3]
        patch_embed = PatchEmbed(
            dim_in=in_channels,
            dim_out=EMBED_DIM,
            kernel=PATCH_KERNEL,  # Kernel size for patchtification.
            stride=PATCH_STRIDE,  # Stride size for patchtification.
            padding=PATCH_PADDING,  # Padding size for patchtification.
        )

        self.patch_embed = patch_embed

        patch_dims = [
            input_size[0] // PATCH_STRIDE[0],
            input_size[1] // PATCH_STRIDE[1],
        ]
        num_patches = np.prod(patch_dims)

        drop_rate = 0.1

        dpr = [  # 根据 transformer 的深度, 设置每层的 drop 率, 最大为 drop_rate
            x.item() for x in torch.linspace(0, drop_rate, DEPTH)

        ]  # stochastic depth decay rule

        if self.CLS_EMBED_ON:  # 在起始位置, 添加一个 cls 标记, 也就是起始位置编码
            self.cls_token = nn.Parameter(torch.zeros(1, 1, EMBED_DIM))
            pos_embed_dim = num_patches + 1
        else:
            pos_embed_dim = num_patches  # 位置编码的数量

        if self.USE_ABS_POS:  # 建立绝对位置嵌入编码
            self.pos_embed = nn.Parameter(
                torch.zeros(1, pos_embed_dim, EMBED_DIM))

        # 获取模型每一层的结构参数, 如每一层中维度放大倍数, 注意力机制中 QKV 的池化参数
        dim_mul, head_mul, pool_q, pool_kv, stride_q, stride_kv = _prepare_mvit_configs()

        # print(f"dim_mul: {dim_mul}")
        # print(f"head_mul: {head_mul}")
        # print(f"pool_q: {pool_q}")
        # print(f"pool_kv: {pool_kv}")
        # print(f"stride_q: {stride_q}")
        # print(f"stride_kv: {stride_kv}")

        input_size = patch_dims  # 模型输入维度
        self.blocks = nn.ModuleList()
        DIM_MUL_IN_ATT = True  # Dim mul in qkv linear layers of attention block instead of MLP
        for i in range(DEPTH):
            NUM_HEADS = round_width(NUM_HEADS, head_mul[i])
            if DIM_MUL_IN_ATT:
                dim_out = round_width(
                    EMBED_DIM,
                    dim_mul[i],
                    divisor=round_width(NUM_HEADS, head_mul[i]),
                )
            else:
                dim_out = round_width(
                    EMBED_DIM,
                    dim_mul[i + 1],
                    divisor=round_width(NUM_HEADS, head_mul[i + 1]),
                )
            MLP_RATIO = 4.0  # Dimension reduction ratio for the MLP layers.
            QKV_BIAS = True  # If use, use bias term in attention fc layers.
            MODE = "conv"  # Options include `conv`, `max`. 池化方式
            # If True, perform pool before projection in attention.
            POOL_FIRST = False
            # If True, use relative positional embedding for spatial dimentions
            REL_POS_SPATIAL = False
            REL_POS_ZERO_INIT = False  # If True, init rel with zero
            RESIDUAL_POOLING = True  # If True, using Residual Pooling connection

            # print(
            #     f"MultiScaleBlock(dim={EMBED_DIM},dim_out={dim_out},num_heads={NUM_HEADS},input_size={input_size},mlp_ratio={MLP_RATIO},qkv_bias={QKV_BIAS},drop_path={dpr[i]},norm_layer={norm_layer},kernel_q={pool_q[i] if len(pool_q) > i else []},kernel_kv={pool_kv[i] if len(pool_kv) > i else []},stride_q={stride_q[i] if len(stride_q) > i else []},stride_kv={stride_kv[i] if len(stride_kv) > i else []},mode={MODE},has_cls_embed={self.CLS_EMBED_ON},pool_first={POOL_FIRST},rel_pos_spatial={REL_POS_SPATIAL},rel_pos_zero_init={REL_POS_ZERO_INIT},residual_pooling={RESIDUAL_POOLING},dim_mul_in_att={DIM_MUL_IN_ATT},)")
            attention_block = MultiScaleBlock(
                dim=EMBED_DIM,
                dim_out=dim_out,
                num_heads=NUM_HEADS,
                input_size=input_size,
                mlp_ratio=MLP_RATIO,
                qkv_bias=QKV_BIAS,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                kernel_q=pool_q[i] if len(pool_q) > i else [],
                kernel_kv=pool_kv[i] if len(pool_kv) > i else [],
                stride_q=stride_q[i] if len(stride_q) > i else [],
                stride_kv=stride_kv[i] if len(stride_kv) > i else [],
                mode=MODE,
                has_cls_embed=self.CLS_EMBED_ON,
                pool_first=POOL_FIRST,
                rel_pos_spatial=REL_POS_SPATIAL,
                rel_pos_zero_init=REL_POS_ZERO_INIT,
                residual_pooling=RESIDUAL_POOLING,
                dim_mul_in_att=DIM_MUL_IN_ATT,
            )

            self.blocks.append(attention_block)

            if len(stride_q[i]) > 0:  # 每一层, 如果 stride 不为 1 , 则会影响输出图的大小, 因此重新计算
                input_size = [
                    size // stride for size, stride in zip(input_size, stride_q[i])
                ]
            EMBED_DIM = dim_out  # 修改当前输入的 EMBED_DIM 大小为上层输出的维度大小

        self.norm = norm_layer(EMBED_DIM)
        # Dropout rate before final projection in the backbone.
        DROPOUT_RATE = 0.0
        HEAD_ACT = "softmax"  # Activation layer for the output head.
        self.head = TransformerBasicHead(
            EMBED_DIM,
            output_size,
            dropout_rate=DROPOUT_RATE,
            act_func=HEAD_ACT,
        )
        if self.USE_ABS_POS:
            trunc_normal_(self.pos_embed, std=0.02)
        if self.CLS_EMBED_ON:
            trunc_normal_(self.cls_token, std=0.02)

        # 对当前模型内的所有子模型进行递归, 执行参数初始化
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, feature_lens):
        # print(f"{self.__class__.__name__},  Input: {x.shape}")
        # B 1 F S -> B 1 S F
        x = x.permute(0, 1, 3, 2).unsqueeze(1)
        S, F = 1,
        # B 1 S F -> B S*F C
        x, bcsf = self.patch_embed(x)
        S, F = bcsf[-2], bcsf[-1]
        B, N, C = x.shape

        tsf = [S, F]
        for blk in self.blocks:
            x, tsf = blk(x, tsf)

        x = self.norm(x)
        x = x.mean(1)
        x = self.head(x)
        return x, feature_lens


class MViT(nn.Module):
    """
    Improved Multiscale Vision Transformers for Classification and Detection
    https://arxiv.org/abs/2112.01526

    Multiscale Vision Transformers
    https://arxiv.org/abs/2104.11227
    """

    def __init__(self):
        super().__init__()
        # 输入参数
        in_channels = 128
        input_size = [1, 128]

        # 输出参数
        output_size = 4

        # 模型中间参数
        EMBED_DIM = 128
        NUM_HEADS = 1
        DEPTH = 16  # Depth of the transformer.
        self.CLS_EMBED_ON = False  # use cls embed
        self.USE_ABS_POS = False  # absolute positional embedding.
        # no decay on positional embedding and cls embedding
        self.ZERO_DECAY_POS_CLS = False

        norm_layer = partial(nn.LayerNorm, eps=1e-6)  # 固定 LayerNorm 的 eps 参数

        drop_rate = 0.1
        # 根据 transformer 的深度, 设置每层的 drop 率, 最大为 drop_rate
        dpr = [
            x.item() for x in torch.linspace(0, drop_rate, DEPTH)]

        # 获取模型每一层的结构参数, 如每一层中维度放大倍数, 注意力机制中 QKV 的池化参数
        dim_mul, head_mul, pool_q, pool_kv, stride_q, stride_kv = _prepare_mvit_configs()

        self.blocks = nn.ModuleList()
        DIM_MUL_IN_ATT = True  # Dim mul in qkv linear layers of attention block instead of MLP
        for i in range(DEPTH):
            NUM_HEADS = round_width(NUM_HEADS, head_mul[i])
            dim_out = round_width(
                EMBED_DIM,
                dim_mul[i],
                divisor=round_width(NUM_HEADS, head_mul[i]),
            )

            MLP_RATIO = 4.0  # Dimension reduction ratio for the MLP layers.
            QKV_BIAS = True  # If use, use bias term in attention fc layers.
            MODE = "conv"  # Options include `conv`, `max`. 池化方式
            # If True, perform pool before projection in attention.
            POOL_FIRST = False
            # If True, use relative positional embedding for spatial dimentions
            REL_POS_SPATIAL = False
            REL_POS_ZERO_INIT = False  # If True, init rel with zero
            RESIDUAL_POOLING = True  # If True, using Residual Pooling connection

            attention_block = MultiScaleBlock(
                dim=EMBED_DIM,
                dim_out=dim_out,
                num_heads=NUM_HEADS,
                input_size=input_size,
                mlp_ratio=MLP_RATIO,
                qkv_bias=QKV_BIAS,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                kernel_q=pool_q[i] if len(pool_q) > i else [],
                kernel_kv=pool_kv[i] if len(pool_kv) > i else [],
                stride_q=stride_q[i] if len(stride_q) > i else [],
                stride_kv=stride_kv[i] if len(stride_kv) > i else [],
                mode=MODE,
                has_cls_embed=self.CLS_EMBED_ON,
                pool_first=POOL_FIRST,
                rel_pos_spatial=REL_POS_SPATIAL,
                rel_pos_zero_init=REL_POS_ZERO_INIT,
                residual_pooling=RESIDUAL_POOLING,
                dim_mul_in_att=DIM_MUL_IN_ATT,
            )

            self.blocks.append(attention_block)

            if len(stride_q[i]) > 0:  # 每一层, 如果 stride 不为 1 , 则会影响输出图的大小, 因此重新计算
                input_size = [
                    size // stride for size, stride in zip(input_size, stride_q[i])
                ]
            EMBED_DIM = dim_out  # 修改当前输入的 EMBED_DIM 大小为上层输出的维度大小

        self.norm = norm_layer(EMBED_DIM)
        # Dropout rate before final projection in the backbone.
        DROPOUT_RATE = 0.1
        HEAD_ACT = "softmax"  # Activation layer for the output head.
        self.head = TransformerBasicHead(
            EMBED_DIM,
            output_size,
            dropout_rate=DROPOUT_RATE,
            act_func=HEAD_ACT,
        )

        # 对当前模型内的所有子模型进行递归, 执行参数初始化
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, feature_lens):
        # print(f"{self.__class__.__name__},  Input: {x.shape}")
        # B 1 F S -> B S*1 F
        x = x.permute(0, 1, 3, 2).unsqueeze(1)
        S, F = x.shape[1], 1

        tsf = [S, F]
        for blk in self.blocks:
            x, tsf = blk(x, tsf)

        x = self.norm(x)
        x = x.mean(1)
        x = self.head(x)
        return x, feature_lens


def _prepare_mvit_configs():
    """
    Prepare mvit configs for dim_mul and head_mul facotrs, and q and kv pooling
    kernels and strides.
    """
    DEPTH = 16
    dim_mul, head_mul = torch.ones(DEPTH + 1), torch.ones(DEPTH + 1)

    # 第 i 层的 Dimension 的放大倍数。 如果使用 2.0，那么下一个块将维度增加 2 倍。 格式：[depth_i, mul_dim_ratio]
    DIM_MUL = [[1, 2.0], [3, 2.0], [8, 2.0]]
    for i in range(len(DIM_MUL)):
        dim_mul[DIM_MUL[i][0]] = DIM_MUL[i][1]
    # 设置第 i 层的 Head number 的放大倍数。 如果使用 2.0，那么下一个区块会增加 2 倍的 Head number 。 格式：[depth_i, head_mul_ratio]
    HEAD_MUL = [[1, 2.0], [3, 2.0], [8, 2.0]]
    for i in range(len(HEAD_MUL)):
        head_mul[HEAD_MUL[i][0]] = HEAD_MUL[i][1]

    pool_q = [[] for i in range(DEPTH)]  # 注意力机制中 query 的池化层的核大小(窗大小)
    pool_kv = [[] for i in range(DEPTH)]  # 注意力机制中 Key/Value 的池化层的核大小(窗大小)
    stride_q = [[] for i in range(DEPTH)]  # 注意力机制中 query 的池化层的核大小(窗口平移数量)
    stride_kv = [[] for i in range(DEPTH)]  # 注意力机制中 Key/Value 的池化层的步数(窗口平移数量)

    # 统一 Q, K, V 池化层的核大小.
    POOL_KVQ_KERNEL = [3, 3]
    # 指定每层 Pool Q 的 Stride 大小 格式：[depth_i, s1, s2].
    POOL_Q_STRIDE = [[0, 1, 1], [1, 2, 2], [2, 1, 1], [3, 2, 2], [
        4, 1, 1], [5, 1, 1], [6, 1, 1], [7, 1, 1], [8, 2, 2], [9, 1, 1]]
    POOL_KV_STRIDE = None  # 指定每层注意力机制中的 Key/Value 的步数 格式：[depth_i, s1, s2]

    for i in range(len(POOL_Q_STRIDE)):
        stride_q[POOL_Q_STRIDE[i][0]] = POOL_Q_STRIDE[i][1:]
        pool_q[POOL_Q_STRIDE[i][0]] = POOL_KVQ_KERNEL

    # 初始化第一层注意力机制中的 Key/Value 的步数, 以及后续每一层的步数会随着 DIM_MUL 减小. 如果不是 POOL_KV_STRIDE 不空, 也会重写覆盖.
    POOL_KV_STRIDE_ADAPTIVE = [4, 4]
    if POOL_KV_STRIDE_ADAPTIVE is not None:
        _stride_kv = POOL_KV_STRIDE_ADAPTIVE
        POOL_KV_STRIDE = []
        for i in range(DEPTH):
            if len(stride_q[i]) > 0:  # 如果当前层 Pool Q 的 Stride 设置不为空
                _stride_kv = [  # 则 Key/Value 的 Stride 设置
                    max(_stride_kv[d] // stride_q[i][d], 1)
                    for d in range(len(_stride_kv))
                ]
            POOL_KV_STRIDE.append([i] + _stride_kv)

    for i in range(len(POOL_KV_STRIDE)):
        stride_kv[POOL_KV_STRIDE[i][0]
                  ] = POOL_KV_STRIDE[i][1:]
        pool_kv[POOL_KV_STRIDE[i][0]] = POOL_KVQ_KERNEL

    return dim_mul, head_mul, pool_q, pool_kv, stride_q, stride_kv


class PatchEmbed(nn.Module):
    """
    PatchEmbed.
    # B 1 F S -> B FS C
    """

    def __init__(
        self,
        dim_in=3,
        dim_out=768,
        kernel=(7, 7),
        stride=(4, 4),
        padding=(3, 3),
    ):
        super().__init__()

        self.proj = nn.Conv2d(
            dim_in,
            dim_out,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
        )

    def forward(self, x: torch.Tensor):
        # print(f"{self.__class__.__name__},  Input: {x.shape}")

        # B 1 F S -> B C F S
        x = self.proj(x)

        # B C F S -> B FS C
        return x.flatten(2).transpose(1, 2), x.shape


class TransformerBasicHead(nn.Module):
    """
    Basic Transformer Head. No pool.
    """

    def __init__(
        self,
        dim_in,
        num_classes,
        dropout_rate=0.0,
        act_func="softmax",
    ):
        """
        Perform linear projection and activation as head for tranformers.
        Args:
            dim_in (int): the channel dimension of the input to the head.
            num_classes (int): the channel dimensions of the output to the head.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
        """
        super(TransformerBasicHead, self).__init__()
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        self.projection = nn.Linear(dim_in, num_classes, bias=True)

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=1)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation" "function.".format(
                    act_func)
            )

    def forward(self, x):
        # print(f"{self.__class__.__name__},  Input: {x.shape}")
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        x = self.projection(x)

        if not self.training:
            x = self.act(x)
        # print(f"{self.__class__.__name__},  Output: {x.shape}")
        return x


if __name__ == '__main__':
    test = np.random.random((4, 1, 186, 40)).astype(np.float32)
    test = torch.Tensor(test)
    macnn = MViT()
    print(macnn)
    print("")
    macnn(test)
