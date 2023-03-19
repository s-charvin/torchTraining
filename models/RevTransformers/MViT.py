# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved. All Rights Reserved.


"""MViT models."""

import math
from functools import partial

import torch
import torch.nn as nn
from .attention import MultiScaleBlock
from torch.nn.init import trunc_normal_
import numpy as np
import torch.nn.functional as F


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
        in_channels = 128
        sequence_length = 128

        # 输出参数
        output_size = 4

        # 模型中间参数
        EMBED_DIM = in_channels
        NUM_HEADS = 1
        DEPTH = 16  # Depth of the transformer.

        norm_layer = partial(nn.LayerNorm, eps=1e-6)  # 固定 LayerNorm 的 eps 参数

        drop_rate = 0.1
        # 根据 transformer 的深度, 设置每层的 drop 率, 最大为 drop_rate
        dpr = [
            x.item() for x in torch.linspace(0, drop_rate, DEPTH)]

        # 获取模型每一层块的结构参数, 如每一层中维度放大倍数, 注意力机制中 QKV 的池化参数

        # 第 i 层的 Dimension 的放大倍数。 如果使用 2.0，那么下一个块将维度增加 2 倍。 格式：[depth_i, mul_dim_ratio]
        DIM_MUL = [[1, 2.0], [3, 2.0], [8, 2.0]]
        # 设置第 i 层的 Head number 的放大倍数。 如果使用 2.0，那么下一个区块会增加 2 倍的 Head number 。 格式：[depth_i, head_mul_ratio]
        HEAD_MUL = [[1, 2.0], [3, 2.0], [8, 2.0]]
        # 统一 Q, K, V 池化层的核大小.
        POOL_KVQ_KERNEL = [3, 1]
        # 指定每层 Pool Q 的 Stride 大小 格式：[depth_i, s1, s2].
        POOL_Q_STRIDE = [[0, 1, 1], [1, 2, 1], [2, 1, 1], [3, 2, 1], [
            4, 1, 1], [5, 1, 1], [6, 1, 1], [7, 1, 1], [8, 2, 1], [9, 1, 1]]
        # 指定每层注意力机制中的 Key/Value 的步数 格式：[depth_i, s1, s2]
        POOL_KV_STRIDE = None
        # 自动初始化第一层注意力机制中的 Key/Value 的步数, 以及后续每一层的步数会随着 DIM_MUL 减小. 如果不是 POOL_KV_STRIDE 不空, 也会重写覆盖.
        POOL_KV_STRIDE_ADAPTIVE = [3, 1]

        # 初始化每层结构参数的初值
        dim_mul, head_mul = torch.ones(DEPTH + 1), torch.ones(DEPTH + 1)
        # 注意力机制中 query 的池化层的核大小(窗大小)
        pool_q = [[] for i in range(DEPTH)]
        # 注意力机制中 Key/Value 的池化层的核大小(窗大小)
        pool_kv = [[] for i in range(DEPTH)]
        # 注意力机制中 query 的池化层的核大小(窗口平移数量)
        stride_q = [[] for i in range(DEPTH)]
        # 注意力机制中 Key/Value 的池化层的步数(窗口平移数量)
        stride_kv = [[] for i in range(DEPTH)]

        # 解析自定义参数
        for i in range(len(DIM_MUL)):
            dim_mul[DIM_MUL[i][0]] = DIM_MUL[i][1]

        for i in range(len(HEAD_MUL)):
            head_mul[HEAD_MUL[i][0]] = HEAD_MUL[i][1]

        for i in range(len(POOL_Q_STRIDE)):
            stride_q[POOL_Q_STRIDE[i][0]] = POOL_Q_STRIDE[i][1:]
            pool_q[POOL_Q_STRIDE[i][0]] = POOL_KVQ_KERNEL

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

        # 构建计算块

        self.blocks = nn.ModuleList()
        for i in range(DEPTH):
            NUM_HEADS = round_width(NUM_HEADS, head_mul[i])
            dim_out = round_width(
                EMBED_DIM,
                dim_mul[i],
                divisor=round_width(NUM_HEADS, head_mul[i]),
            )

            MLP_RATIO = 4.0  # Dimension reduction ratio for the MLP layers.
            QKV_BIAS = True  # If use, use bias term in attention fc layers.
            MODE = "conv"

            attention_block = MultiScaleBlock(
                dim=EMBED_DIM,
                dim_out=dim_out,
                num_heads=NUM_HEADS,
                mlp_ratio=MLP_RATIO,
                qkv_bias=QKV_BIAS,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                kernel_q=pool_q[i] if len(pool_q) > i else [],
                kernel_kv=pool_kv[i] if len(pool_kv) > i else [],
                stride_q=stride_q[i] if len(stride_q) > i else [],
                stride_kv=stride_kv[i] if len(stride_kv) > i else [],
                mode=MODE,
            )

            self.blocks.append(attention_block)

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

        x = x.permute(0, 1, 3, 2).squeeze(1)

        tsf = [x.shape[1], 1]
        for blk in self.blocks:
            x, tsf = blk(x, tsf)

        x = self.norm(x)
        x = x.mean(1)
        x = self.head(x)
        return x, feature_lens


class Conv2dSame(torch.nn.Conv2d):

    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.size()[-2:]

        pad_h = self.calc_same_pad(
            i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0])
        pad_w = self.calc_same_pad(
            i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1])

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2,
                    pad_h // 2, pad_h - pad_h // 2]
            )
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


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
        sequence_length = 128

        # 输出参数
        output_size = 5

        # 模型中间参数
        EMBED_DIM = 128
        NUM_HEADS = 1
        DEPTH = 8  # Depth of the transformer.

        self.path1 = nn.Sequential(
            Conv2dSame(in_channels=1, out_channels=32, kernel_size=(
                10, 2), stride=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.path2 = nn.Sequential(
            Conv2dSame(in_channels=1, out_channels=32, kernel_size=(
                2, 8), stride=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        # self.path3 = nn.Sequential(
        #     Conv2dSame(in_channels=1, out_channels=32, kernel_size=(
        #         3, 3), stride=(1, 1)),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        # )

        norm_layer = partial(nn.LayerNorm, eps=1e-6)  # 固定 LayerNorm 的 eps 参数

        drop_rate = 0.1
        # 根据 transformer 的深度, 设置每层的 drop 率, 最大为 drop_rate
        dpr = [
            x.item() for x in torch.linspace(0, drop_rate, DEPTH)]

        # 获取模型每一层块的结构参数, 如每一层中维度放大倍数, 注意力机制中 QKV 的池化参数

        # 第 i 层的 Dimension 的放大倍数。 如果使用 2.0，那么下一个块将维度增加 2 倍。 格式：[depth_i, mul_dim_ratio]
        if DEPTH == 16:
            DIM_MUL = [[1, 2.0], [3, 2.0], [8, 2.0]]
            # 设置第 i 层的 Head number 的放大倍数。 如果使用 2.0，那么下一个区块会增加 2 倍的 Head number 。 格式：[depth_i, head_mul_ratio]
            HEAD_MUL = [[1, 2.0], [3, 2.0], [8, 2.0]]
            # 指定每层 Pool Q 的 Stride 大小 格式：[depth_i, s1, s2].
            POOL_Q_STRIDE = [[0, 1, 1], [1, 2, 2], [2, 1, 1], [3, 2, 2], [
                4, 1, 1], [5, 1, 1], [6, 1, 1], [7, 1, 1], [8, 2, 2], [9, 1, 1]]
            # 统一 Q, K, V 池化层的核大小.
        elif DEPTH == 8:
            DIM_MUL = [[1, 2.0], [3, 2.0], [5, 2.0]]
            # 设置第 i 层的 Head number 的放大倍数。 如果使用 2.0，那么下一个区块会增加 2 倍的 Head number 。 格式：[depth_i, head_mul_ratio]
            HEAD_MUL = [[1, 2.0], [3, 2.0], [5, 2.0]]
            # 指定每层 Pool Q 的 Stride 大小 格式：[depth_i, s1, s2].
            POOL_Q_STRIDE = [[0, 1, 1], [1, 2, 2], [2, 1, 1], [3, 2, 2], [
                4, 1, 1], [5, 1, 1], [6, 2, 2], [7, 1, 1]]
            # 统一 Q, K, V 池化层的核大小.
        POOL_KVQ_KERNEL = [3, 3]
        # 指定每层注意力机制中的 Key/Value 的步数 格式：[depth_i, s1, s2]
        POOL_KV_STRIDE = None
        # 自动初始化第一层注意力机制中的 Key/Value 的步数, 以及后续每一层的步数会随着 DIM_MUL 减小. 如果不是 POOL_KV_STRIDE 不空, 也会重写覆盖.
        POOL_KV_STRIDE_ADAPTIVE = [3, 3]

        # 初始化每层结构参数的初值
        dim_mul, head_mul = torch.ones(DEPTH + 1), torch.ones(DEPTH + 1)
        # 注意力机制中 query 的池化层的核大小(窗大小)
        pool_q = [[] for i in range(DEPTH)]
        # 注意力机制中 Key/Value 的池化层的核大小(窗大小)
        pool_kv = [[] for i in range(DEPTH)]
        # 注意力机制中 query 的池化层的核大小(窗口平移数量)
        stride_q = [[] for i in range(DEPTH)]
        # 注意力机制中 Key/Value 的池化层的步数(窗口平移数量)
        stride_kv = [[] for i in range(DEPTH)]

        # 解析自定义参数
        for i in range(len(DIM_MUL)):
            dim_mul[DIM_MUL[i][0]] = DIM_MUL[i][1]

        for i in range(len(HEAD_MUL)):
            head_mul[HEAD_MUL[i][0]] = HEAD_MUL[i][1]

        for i in range(len(POOL_Q_STRIDE)):
            stride_q[POOL_Q_STRIDE[i][0]] = POOL_Q_STRIDE[i][1:]
            pool_q[POOL_Q_STRIDE[i][0]] = POOL_KVQ_KERNEL

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

        # 构建计算块

        self.blocks = nn.ModuleList()
        for i in range(DEPTH):
            NUM_HEADS = round_width(NUM_HEADS, head_mul[i])
            dim_out = round_width(
                EMBED_DIM,
                dim_mul[i],
                divisor=round_width(NUM_HEADS, head_mul[i]),
            )

            MLP_RATIO = 2.0  # Dimension reduction ratio for the MLP layers.
            QKV_BIAS = True  # If use, use bias term in attention fc layers.
            MODE = "conv"

            attention_block = MultiScaleBlock(
                dim=EMBED_DIM,
                dim_out=dim_out,
                num_heads=NUM_HEADS,
                mlp_ratio=MLP_RATIO,
                qkv_bias=QKV_BIAS,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                kernel_q=pool_q[i] if len(pool_q) > i else [],
                kernel_kv=pool_kv[i] if len(pool_kv) > i else [],
                stride_q=stride_q[i] if len(stride_q) > i else [],
                stride_kv=stride_kv[i] if len(stride_kv) > i else [],
                mode=MODE,
            )

            self.blocks.append(attention_block)

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
        # B 1 F W -> B H F W
        out1 = self.path1(x)
        out2 = self.path2(x)
        # out3 = self.path3(x)
        x = torch.cat([out1, out2], dim=1)
        tsf = [x.shape[1], x.shape[3]]
        # B H F W -> B H*W F
        x = x.transpose(1, 2).flatten(2).transpose(1, 2)
        for blk in self.blocks:
            x, tsf = blk(x, tsf)

        x = self.norm(x)
        x = x.mean(1)
        x = self.head(x)
        return x, feature_lens


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
