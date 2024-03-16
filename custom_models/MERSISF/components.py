from typing import Tuple, Type, Any, Callable, Union, List, Optional
import torch
from torch import Tensor, nn
import torchvision
import torch.nn.functional as F
import math
import numpy as np


class EmptyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x


class Conv2dFeatureExtractor(nn.Module):
    """构建连续的2D卷积网络结构主Module
    Args:
        conv_layers (nn.ModuleList): 模型所需的2D卷积网络结构列表
    """

    def __init__(
        self,
        conv_layers: nn.ModuleList,
    ):
        super().__init__()
        self.conv_blocks = conv_layers

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor):
                shape: ``[batch,in_channels, frame, feature]``.
        Returns:
            Tensor:
                输出结果, shape: ``[batch,out_channels, frame, feature]``
        """
        # 检查参数
        if x.ndim != 4:
            raise ValueError(
                f"期望输入的维度为 4D (batch,in_channels, frame, feature), 但得到的输入数据维度为 {list(x.shape)}"
            )
        for block in self.conv_blocks:
            x = block(x)
        return x


def get_Conv2d_extractor(
    in_channels,
    # [channel:int, kernel:list|int, stride:list|int, norm:str|None, pool:str|None, pool_kernel:list|int]
    shapes,
    bias: bool,
) -> Conv2dFeatureExtractor:
    blocks = []
    in_channels = in_channels  # 输入数据的通道

    for i, (
        out_channels,
        kernel_size,
        stride,
        norm_mode,
        pool_mode,
        pool_kernel,
    ) in enumerate(shapes):
        # Norm设置
        assert norm_mode in ["gn", "ln", "bn", "None"]
        assert pool_mode in ["mp", "ap", "gap", "None"]
        if norm_mode == "gn":
            normalization = nn.GroupNorm(
                num_groups=out_channels,
                num_channels=out_channels,
                affine=True,
            )
        elif norm_mode == "ln":
            normalization = nn.LayerNorm(
                normalized_shape=out_channels,
                elementwise_affine=True,
            )
        elif norm_mode == "bn":
            normalization = nn.BatchNorm2d(out_channels)
        else:
            normalization = None

        if pool_mode == "mp":
            poolling = nn.MaxPool2d(pool_kernel)
        elif pool_mode == "ap":
            poolling = nn.AvgPool2d(pool_kernel)
        elif pool_mode == "gap":  # GlobalAveragePooling2D
            pool_kernel = 1
            poolling = nn.AdaptiveAvgPool2d(1)
        else:
            poolling = None
        # 卷积块设置
        blocks.append(
            Conv2dLayerBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                bias=bias,
                layer_norm=normalization,
                layer_poolling=poolling,
                activate_func=nn.functional.relu,
            )
        )

        in_channels = out_channels  # 修改输入通道为 Extractor 的输出通道
    return Conv2dFeatureExtractor(nn.ModuleList(blocks))


def _get_MultiConv2d_extractor(
    in_channels,
    out_channels,
) -> Conv2dFeatureExtractor:
    blocks = []
    in_channels = in_channels  # 输入数据的通道

    for out_channel in out_channels:
        # 卷积块设置
        blocks.append(
            Conv2dMultiBlock(
                in_channels=in_channels,
                out_channels=out_channel,
            )
        )
        in_channels = out_channel  # 修改输入通道为 Extractor 的输出通道
    return Conv2dFeatureExtractor(nn.ModuleList(blocks))


def _get_ResMultiConv2d_extractor(
    in_channels,
    shapes,
    bias: bool,
) -> Conv2dFeatureExtractor:
    blocks = []
    in_channels = in_channels  # 输入数据的通道

    for i, (out_channels, stride, norm_mode, pool_mode, pool_kernel) in enumerate(
        shapes
    ):
        # Norm设置
        assert norm_mode in ["gn", "ln", "bn", "None"]
        assert pool_mode in ["mp", "ap", "gap", "None"]
        if norm_mode == "gn":
            normalization = nn.GroupNorm(
                num_groups=out_channels,
                num_channels=out_channels,
                affine=True,
            )
        elif norm_mode == "ln":
            normalization = nn.LayerNorm(
                normalized_shape=out_channels,
                elementwise_affine=True,
            )
        elif norm_mode == "bn":
            normalization = nn.BatchNorm2d(out_channels)
        else:
            normalization = None

        if pool_mode == "mp":
            poolling = nn.MaxPool2d(pool_kernel)
        elif pool_mode == "ap":
            poolling = nn.AvgPool2d(pool_kernel)
        elif pool_mode == "gap":  # GlobalAveragePooling2D
            pool_kernel = 1
            poolling = nn.AdaptiveAvgPool2d(1)
        else:
            poolling = None
        # 卷积块设置
        blocks.append(
            Conv2dResMultiBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                bias=bias,
                layer_norm=normalization,
                layer_poolling=poolling,
                activate_func=nn.functional.relu,
            )
        )

        in_channels = out_channels  # 修改输入通道为 Extractor 的输出通道

    return Conv2dFeatureExtractor(nn.ModuleList(blocks))


class Conv2dMultiBlock(nn.Module):
    """
    Multi-scale block without short-cut connections
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super(Conv2dMultiBlock, self).__init__()
        assert out_channels % 2 == 0
        self.conv3 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=int(out_channels / 2),
            kernel_size=[3, 3],
            padding=1,
        )
        self.conv5 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=int(out_channels / 2),
            kernel_size=[5, 5],
            padding=2,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        x = torch.cat((x3, x5), 1)
        x = self.bn(x)
        x = F.relu(x)
        return x


class Conv2dResMultiBlock(nn.Module):
    """
    Multi-scale block with short-cut connections
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        bias: bool,
        layer_norm: Optional[nn.Module],
        layer_poolling: Optional[nn.Module],
        activate_func=nn.functional.relu,
    ):
        super(Conv2dResMultiBlock, self).__init__()
        assert out_channels % 2 == 0

        self.layer_norm = layer_norm
        self.activate_func = activate_func
        self.leyer_poolling = layer_poolling

        self.conv3 = Conv2dSame(
            in_channels=in_channels,
            out_channels=int(out_channels / 2),
            kernel_size=[3, 3],
            stride=stride,  # 步长
            bias=bias,  # 是否使用偏置数
            padding_mode="zeros",
        )
        self.conv5 = Conv2dSame(
            in_channels=in_channels,
            out_channels=int(out_channels / 2),
            kernel_size=[5, 5],
            stride=stride,  # 步长
            bias=bias,  # 是否使用偏置数
            padding_mode="zeros",
        )
        self.subconv = None
        if in_channels != int(out_channels / 2):
            self.subconv = nn.Conv2d(
                kernel_size=[1, 1],
                in_channels=in_channels,
                out_channels=int(out_channels / 2),
            )

    def forward(self, x):
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        if self.subconv is not None:
            x = self.subconv(x)
        x3 = x3 + x
        x5 = x5 + x
        x = torch.cat((x3, x5), 1)
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        if self.activate_func is not None:
            x = self.activate_func(x)
        if self.leyer_poolling is not None:
            x = self.leyer_poolling(x)
        return x


class Conv2dLayerBlock(nn.Module):
    """单个卷积块,内部含有 Conv2d_layer, Norm_layer(可选), Activation_layer(可选, 默认GELU)"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride: int,
        bias: bool,
        layer_norm: Optional[nn.Module],
        layer_poolling: Optional[nn.Module],
        activate_func=nn.functional.relu,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

        self.conv = Conv2dSame(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,  # 步长
            bias=bias,  # 是否使用偏置数
            padding_mode="zeros",
        )
        self.layer_norm = layer_norm
        self.activate_func = activate_func
        self.leyer_poolling = layer_poolling

    def forward(
        self,
        x: Tensor,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            x (Tensor): Shape ``[batch, in_channels, in_frames, in_features]``.
        Returns:
            Tensor: Shape ``[batch, out_channels, out_frames, in_features]``.
        """
        x = self.conv(x)
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        if self.activate_func is not None:
            x = self.activate_func(x)
        if self.leyer_poolling is not None:
            x = self.leyer_poolling(x)
        return x


class Conv2dSame(torch.nn.Conv2d):
    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.size()[-2:]

        pad_h = self.calc_same_pad(
            i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0]
        )
        pad_w = self.calc_same_pad(
            i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1]
        )

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
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


class Conv3dFeatureExtractor(nn.Module):
    """构建连续的3D卷积网络结构主 Module
    Args:
        conv_layers (nn.ModuleList): 模型所需的3D卷积网络结构列表
    """

    def __init__(
        self,
        conv_layers: nn.ModuleList,
    ):
        super().__init__()
        self.conv_blocks = conv_layers

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor):
                shape: ``[batch, in_channels, frame, w, h]``.
        Returns:
            Tensor:
                输出结果, shape: ``[batch,out_channels, frame, w, h]``
        """
        # 检查参数
        if x.ndim != 5:
            raise ValueError(
                f"期望输入的维度为 5D (batch, in_channels, frame, w, h), 但得到的输入数据维度为 {list(x.shape)}"
            )
        for block in self.conv_blocks:
            x = block(x)
        return x


def _get_Conv3d_extractor(
    in_channels,
    # [channel:int, kernel:list|int, stride:list|int, norm:str|None, pool:str|None, pool_kernel:list|int]
    shapes,
    bias: bool,
) -> Conv3dFeatureExtractor:
    blocks = []
    in_channels = in_channels  # 输入数据的通道

    for i, (
        out_channels,
        kernel_size,
        stride,
        norm_mode,
        pool_mode,
        pool_kernel,
    ) in enumerate(shapes):
        # Norm设置
        assert norm_mode in ["gn", "ln", "bn", "None"]
        assert pool_mode in ["mp", "ap", "gap", "None"]
        if norm_mode == "gn":
            normalization = nn.GroupNorm(
                num_groups=out_channels,
                num_channels=out_channels,
                affine=True,
            )
        elif norm_mode == "ln":
            normalization = nn.LayerNorm(
                normalized_shape=out_channels,
                elementwise_affine=True,
            )
        elif norm_mode == "bn":
            normalization = nn.BatchNorm3d(out_channels)
        else:
            normalization = None

        if pool_mode == "mp":
            poolling = nn.MaxPool3d(pool_kernel)
        elif pool_mode == "ap":
            poolling = nn.AvgPool3d(pool_kernel)
        elif pool_mode == "gap":  # GlobalAveragePooling2D
            pool_kernel = 1
            poolling = nn.AdaptiveAvgPool3d(1)
        else:
            poolling = None
        # 卷积块设置
        blocks.append(
            Conv3dLayerBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                bias=bias,
                layer_norm=normalization,
                layer_poolling=poolling,
                activate_func=nn.functional.relu,
            )
        )

        in_channels = out_channels  # 修改输入通道为 Extractor 的输出通道
    return Conv3dFeatureExtractor(nn.ModuleList(blocks))


class Conv3dLayerBlock(nn.Module):
    """单个卷积块,内部含有 Conv3d_layer, Norm_layer(可选), Activation_layer(可选, 默认rELU)"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride: int,
        bias: bool,
        layer_norm: Optional[nn.Module],
        layer_poolling: Optional[nn.Module],
        activate_func=nn.functional.relu,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

        self.conv = Conv3dSame(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,  # 步长
            bias=bias,  # 是否使用偏置数
            padding_mode="zeros",
        )
        self.layer_norm = layer_norm
        self.activate_func = activate_func
        self.leyer_poolling = layer_poolling

    def forward(
        self,
        x: Tensor,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            x (Tensor): Shape ``[batch, in_channels, in_frames, in_features]``.
        Returns:
            Tensor: Shape ``[batch, out_channels, out_frames, in_features]``.
        """
        x = self.conv(x)
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        if self.activate_func is not None:
            x = self.activate_func(x)
        if self.leyer_poolling is not None:
            x = self.leyer_poolling(x)
        return x


class Conv3dSame(torch.nn.Conv3d):
    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        id, ih, iw = x.size()[-3:]
        pad_d = self.calc_same_pad(
            i=id, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0]
        )
        pad_h = self.calc_same_pad(
            i=ih, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1]
        )
        pad_w = self.calc_same_pad(
            i=iw, k=self.kernel_size[2], s=self.stride[2], d=self.dilation[2]
        )

        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            a, b = pad_d // 2, pad_d - (pad_d // 2)
            x = F.pad(
                x,
                [
                    pad_w // 2,
                    pad_w - (pad_w // 2),
                    pad_h // 2,
                    pad_h - (pad_h // 2),
                    pad_d // 2,
                    pad_d - (pad_d // 2),
                ],
            )

        return F.conv3d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class BiLSTM_Attention(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        bidirectional: bool = True,
    ):
        super(BiLSTM_Attention, self).__init__()
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            bias=bias,
            batch_first=True,  # 设置为True以支持 batch_first 输入格式
        )

    # lstm_output : [B, n_step, H * d(=2)], F matrix

    def forward(self, X):
        # X : [B, S, F]

        # final_hidden_state, final_cell_state : [n_layer(=1) * d, B, H]
        output, (final_hidden_state, final_cell_state) = self.lstm(X)

        # output : [B, S, H * d]

        # hidden : [B, H * d, n_layer(=1)]
        hidden = final_hidden_state.view(
            -1, self.hidden_size * self.num_directions, self.num_layers
        )

        # [B, S, H * d] * [B, H * d, n_layer(=1)] -> attn_weights : [B, S]
        attn_weights = torch.bmm(output, hidden).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)  # 对每个输入的所有序列点的权重, 即重视程度

        # [B, S, 1] * [B, S, H * d] -> context : [B, H * d]
        context = torch.bmm(soft_attn_weights.unsqueeze(1), output).squeeze(1)

        # model : [B, H * d], attention : [B, S]
        return context, soft_attn_weights

    # def attention_net(self, lstm_output, final_state):
    #     # [num_layers(=1) * d(=1), B, H]
    #     hidden = final_state.view(
    #         -1, self.hidden_size * self.num_directions, self.num_layers
    #     )
    #     # hidden : [B, H * d(=1), 1(=n_layer)]

    #     # [B, len_seq, num_directions * hidden_size] * [B, H * d, 1]
    #     attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)
    #     # attn_weights : [B, len_seq]
    #     soft_attn_weights = F.softmax(attn_weights, 1)  # 对每个输入的所有序列点的权重, 即重视程度

    #     # [B, H * d, len_seq] * [B, len_seq, 1]
    #     # 对每个seq点的特征乘以了一个权重
    #     context = torch.bmm(
    #         lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)
    #     ).squeeze(2)

    #     # context : [B, H * d]
    #     return context, soft_attn_weights.data.numpy()

    # def forward(self, X):
    #     # X : [B, S, F] -> [S, B, F]
    #     X = X.permute(1, 0, 2).contiguous()

    #     # [n_layer * d, B, H]
    #     hidden_state = torch.zeros(
    #         self.num_layers * self.num_directions, len(X), self.hidden_size
    #     )
    #     cell_state = torch.zeros(
    #         self.num_layers * self.num_directions, len(X), self.hidden_size
    #     )

    #     # final_hidden_state, final_cell_state : [n_layer * d, B, H]
    #     output, (final_hidden_state, final_cell_state) = self.lstm(
    #         input, (hidden_state, cell_state)
    #     )

    #     # output : [B, S, d * H]
    #     output = output.permute(1, 0, 2).contiguous()

    #     # [n_layer * d, B, H] -> [B, H * d, n_layer]
    #     hidden = final_hidden_state.view(
    #         -1, self.hidden_size * self.num_directions, self.num_layers
    #     )

    #     # [B, S, d * H] * [B, H * d, n_layer = 1] -> attn_weights : [B, S]
    #     attn_weights = torch.bmm(output, hidden).squeeze(2)
    #     soft_attn_weights = F.softmax(attn_weights, 1)  # 对每个输入的所有序列点的权重, 即重视程度

    #     # [B, H * d, S] * [B, S, 1] -> context : [B, H * d]
    #     # 对每个 seq 点的特征乘以了一个权重
    #     attn_output = torch.bmm(
    #         output.transpose(1, 2), soft_attn_weights.unsqueeze(2)
    #     ).squeeze(2)

    #     # model : [B, num_classes], attention : [B, n_step]
    #     return attn_output, soft_attn_weights.data.numpy()


class AreaAttention(nn.Module):
    """
    Area Attention [1]. This module allows to attend to areas of the memory, where each area
    contains a group of items that are either spatially or temporally adjacent.
    [1] Li, Yang, et al. "Area attention." International Conference on Machine Learning. PMLR 2019
    """

    def __init__(
        self,
        key_query_size: int,
        area_key_mode: str = "mean",
        area_value_mode: str = "sum",
        max_area_height: int = 1,
        max_area_width: int = 1,
        memory_height: int = 1,
        memory_width: int = 1,
        dropout_rate: float = 0.0,
        top_k_areas: int = 0,
    ):
        """
        Initializes the Area Attention module.
        :param key_query_size: size for keys and queries
        :param area_key_mode: mode for computing area keys, one of {"mean", "max", "sample", "concat", "max_concat", "sum", "sample_concat", "sample_sum"}
        :param area_value_mode: mode for computing area values, one of {"mean", "sum"}
        :param max_area_height: max height allowed for an area
        :param max_area_width: max width allowed for an area
        :param memory_height: memory height for arranging features into a grid
        :param memory_width: memory width for arranging features into a grid
        :param dropout_rate: dropout rate
        :param top_k_areas: top key areas used for attention
        """
        super(AreaAttention, self).__init__()
        assert area_key_mode in [
            "mean",
            "max",
            "sample",
            "concat",
            "max_concat",
            "sum",
            "sample_concat",
            "sample_sum",
        ]
        assert area_value_mode in ["mean", "max", "sum"]
        self.area_key_mode = area_key_mode
        self.area_value_mode = area_value_mode
        self.max_area_height = max_area_height
        self.max_area_width = max_area_width
        self.memory_height = memory_height
        self.memory_width = memory_width
        self.dropout = nn.Dropout(p=dropout_rate)
        self.top_k_areas = top_k_areas
        self.area_temperature = np.power(key_query_size, 0.5)
        if self.area_key_mode in [
            "concat",
            "max_concat",
            "sum",
            "sample_concat",
            "sample_sum",
        ]:
            self.area_height_embedding = nn.Embedding(
                max_area_height, key_query_size // 2
            )
            self.area_width_embedding = nn.Embedding(
                max_area_width, key_query_size // 2
            )
            if area_key_mode == "concat":
                area_key_feature_size = 3 * key_query_size
            elif area_key_mode == "max_concat":
                area_key_feature_size = 2 * key_query_size
            elif area_key_mode == "sum":
                area_key_feature_size = key_query_size
            elif area_key_mode == "sample_concat":
                area_key_feature_size = 2 * key_query_size
            else:  # 'sample_sum'
                area_key_feature_size = key_query_size
            self.area_key_embedding = nn.Sequential(
                nn.Linear(area_key_feature_size, key_query_size),
                nn.ReLU(inplace=True),
                nn.Linear(key_query_size, key_query_size),
            )

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the Area Attention module.
        :param q: queries Tensor with shape (B, num_queries, key_query_size)
        :param k: keys Tensor with shape (B, num_keys_values, key_query_size)
        :param v: values Tensor with shape (B, num_keys_values, value_size)
        :returns a Tensor with shape (B, num_queries, value_size)
        """
        k_area = self._compute_area_key(k)
        if self.area_value_mode == "mean":
            v_area, _, _, _, _ = self._compute_area_features(v)
        elif self.area_value_mode == "max":
            v_area, _, _ = self._basic_pool(v, fn=torch.max)
        elif self.area_value_mode == "sum":
            _, _, v_area, _, _ = self._compute_area_features(v)
        else:
            raise ValueError(f"Unsupported area value mode={self.area_value_mode}")
        logits = torch.matmul(q, k_area.transpose(1, 2))
        # logits = logits / self.area_temperature
        weights = logits.softmax(dim=-1)
        if self.top_k_areas > 0:
            top_k = min(weights.size(-1), self.top_k_areas)
            top_weights, _ = weights.topk(top_k)
            min_values, _ = torch.min(top_weights, dim=-1, keepdim=True)
            weights = torch.where(
                weights >= min_values, weights, torch.zeros_like(weights)
            )
            weights /= weights.sum(dim=-1, keepdim=True)
        weights = self.dropout(weights)
        return torch.matmul(weights, v_area)

    def _compute_area_key(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute the key for each area.
        :param features: a Tensor with shape (B, num_features (height * width), feature_size)
        :return: a Tensor with shape (B, num_areas, feature_size)
        """
        area_mean, area_std, _, area_heights, area_widths = self._compute_area_features(
            features
        )
        if self.area_key_mode == "mean":
            return area_mean
        elif self.area_key_mode == "max":
            area_max, _, _ = self._basic_pool(features)
            return area_max
        elif self.area_key_mode == "sample":
            if self.training:
                area_mean += area_std * torch.randn(area_std.size())
            return area_mean

        height_embed = self.area_height_embedding((area_heights[:, :, 0] - 1).long())
        width_embed = self.area_width_embedding((area_widths[:, :, 0] - 1).long())
        size_embed = torch.cat([height_embed, width_embed], dim=-1)
        if self.area_key_mode == "concat":
            area_key_features = torch.cat([area_mean, area_std, size_embed], dim=-1)
        elif self.area_key_mode == "max_concat":
            area_max, _, _ = self._basic_pool(features)
            area_key_features = torch.cat([area_max, size_embed], dim=-1)
        elif self.area_key_mode == "sum":
            area_key_features = size_embed + area_mean + area_std
        elif self.area_key_mode == "sample_concat":
            if self.training:
                area_mean += area_std * torch.randn(area_std.size())
            area_key_features = torch.cat([area_mean, size_embed], dim=-1)
        elif self.area_key_mode == "sample_sum":
            if self.training:
                area_mean += area_std * torch.randn(area_std.size())
            area_key_features = area_mean + size_embed
        else:
            raise ValueError(f"Unsupported area key mode={self.area_key_mode}")
        area_key = self.area_key_embedding(area_key_features)
        return area_key

    def _compute_area_features(
        self, features: torch.Tensor, epsilon: float = 1e-6
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute features for each area.
        :param features: a Tensor with shape (B, num_features (height * width), feature_size)
        :param epsilon: epsilon added to the variance for computing standard deviation
        :return: a tuple with 5 elements:
          - area_mean: a Tensor with shape (B, num_areas, feature_size)
          - area_std: a Tensor with shape (B, num_areas, feature_size)
          - area_sum: a Tensor with shape (B, num_areas, feature_size)
          - area_heights: a Tensor with shape (B, num_areas, 1)
          - area_widths: a Tensor with shape (B, num_areas, 1)
        """
        area_sum, area_heights, area_widths = self._compute_sum_image(features)
        area_squared_sum, _, _ = self._compute_sum_image(features.square())
        area_size = (area_heights * area_widths).float()
        area_mean = area_sum / area_size
        s2_n = area_squared_sum / area_size
        area_variance = s2_n - area_mean.square()
        area_std = (area_variance.abs() + epsilon).sqrt()
        return area_mean, area_std, area_sum, area_heights, area_widths

    def _compute_sum_image(
        self, features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute area sums for features.
        :param features: a Tensor with shape (B, num_features (height * width), feature_size)
        :return: a tuple with 3 elements:
          - sum_image: a Tensor with shape (B, num_areas, feature_size)
          - area_heights: a Tensor with shape (B, num_areas, 1)
          - area_widths: a Tensor with shape (B, num_areas, 1)
        """
        B, num_features, feature_size = features.size()
        features_2d = features.reshape(
            B,
            self.memory_height,
            num_features // self.memory_height,
            feature_size,
        )
        horizontal_integral = features_2d.cumsum(dim=-2)
        vertical_integral = horizontal_integral.cumsum(dim=-3)
        padded_image = F.pad(vertical_integral, pad=[0, 0, 1, 0, 1, 0])
        heights = []
        widths = []
        dst_images = []
        src_images_diag = []
        src_images_h = []
        src_images_v = []
        size = torch.ones_like(padded_image[:, :, :, 0], dtype=torch.int32)
        for area_height in range(self.max_area_height):
            for area_width in range(self.max_area_width):
                dst_images.append(
                    padded_image[:, area_height + 1 :, area_width + 1 :, :].reshape(
                        B, -1, feature_size
                    )
                )
                src_images_diag.append(
                    padded_image[:, : -area_height - 1, : -area_width - 1, :].reshape(
                        B, -1, feature_size
                    )
                )
                src_images_h.append(
                    padded_image[:, area_height + 1 :, : -area_width - 1, :].reshape(
                        B, -1, feature_size
                    )
                )
                src_images_v.append(
                    padded_image[:, : -area_height - 1, area_width + 1 :, :].reshape(
                        B, -1, feature_size
                    )
                )
                heights.append(
                    (
                        size[:, area_height + 1 :, area_width + 1 :] * (area_height + 1)
                    ).reshape(B, -1)
                )
                widths.append(
                    (
                        size[:, area_height + 1 :, area_width + 1 :] * (area_width + 1)
                    ).reshape(B, -1)
                )
        sum_image = torch.sub(
            torch.cat(dst_images, dim=1) + torch.cat(src_images_diag, dim=1),
            torch.cat(src_images_v, dim=1) + torch.cat(src_images_h, dim=1),
        )
        area_heights = torch.cat(heights, dim=1).unsqueeze(dim=2)
        area_widths = torch.cat(widths, dim=1).unsqueeze(dim=2)
        return sum_image, area_heights, area_widths

    def _basic_pool(
        self, features: torch.Tensor, fn=torch.max
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Pool for each area based on a given pooling function (fn).
        :param features: a Tensor with shape (B, num_features (height * width), feature_size).
        :param fn: Torch pooling function
        :return: a tuple with 3 elements:
          - pool_results: a Tensor with shape (B, num_areas, feature_size)
          - area_heights: a Tensor with shape (B, num_areas, 1)
          - area_widths: a Tensor with shape (B, num_areas, 1)
        """
        B, num_features, feature_size = features.size()
        features_2d = features.reshape(
            B, self.memory_height, self.memory_width, feature_size
        )
        heights = []
        widths = []
        pooled_areas = []
        size = torch.ones_like(features_2d[:, :, :, 0], dtype=torch.int32)
        for area_height in range(self.max_area_height):
            for area_width in range(self.max_area_width):
                pooled_areas.append(
                    self._pool_one_shape(
                        features_2d,
                        area_width=area_width + 1,
                        area_height=area_height + 1,
                        fn=fn,
                    )
                )
                heights.append(
                    (size[:, area_height:, area_width:] * (area_height + 1)).reshape(
                        [B, -1]
                    )
                )
                widths.append(
                    (size[:, area_height:, area_width:] * (area_width + 1)).reshape(
                        [B, -1]
                    )
                )
        pooled_areas = torch.cat(pooled_areas, dim=1)
        area_heights = torch.cat(heights, dim=1).unsqueeze(dim=2)
        area_widths = torch.cat(widths, dim=1).unsqueeze(dim=2)
        return pooled_areas, area_heights, area_widths

    def _pool_one_shape(
        self, features_2d: torch.Tensor, area_width: int, area_height: int, fn=torch.max
    ) -> torch.Tensor:
        """
        Pool for an area in features_2d.
        :param features_2d: a Tensor with shape (B, height, width, feature_size)
        :param area_width: max width allowed for an area
        :param area_height: max height allowed for an area
        :param fn: PyTorch pooling function
        :return: a Tensor with shape (B, num_areas, feature_size)
        """
        B, height, width, feature_size = features_2d.size()
        flat_areas = []
        for y_shift in range(area_height):
            image_height = max(height - area_height + 1 + y_shift, 0)
            for x_shift in range(area_width):
                image_width = max(width - area_width + 1 + x_shift, 0)
                area = features_2d[:, y_shift:image_height, x_shift:image_width, :]
                flatten_area = area.reshape(B, -1, feature_size, 1)
                flat_areas.append(flatten_area)
        flat_area = torch.cat(flat_areas, dim=3)
        pooled_area = fn(flat_area, dim=3).values
        return


class MultiConv(nn.Module):
    """
    Multi-scale block without short-cut connections
    """

    def __init__(self, channels=16, **kwargs):
        super(MultiConv, self).__init__()
        self.conv3 = nn.Conv2d(
            kernel_size=(3, 3), in_channels=channels, out_channels=channels, padding=1
        )
        self.conv5 = nn.Conv2d(
            kernel_size=(5, 5), in_channels=channels, out_channels=channels, padding=2
        )
        self.bn = nn.BatchNorm2d(channels * 2)

    def forward(self, x):
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        x = torch.cat((x3, x5), 1)
        x = self.bn(x)
        x = F.relu(x)
        return x


class ResMultiConv(nn.Module):
    """
    Multi-scale block with short-cut connections
    """

    def __init__(self, channels=16, **kwargs):
        super(ResMultiConv, self).__init__()
        self.conv3 = nn.Conv2d(
            kernel_size=(3, 3), in_channels=channels, out_channels=channels, padding=1
        )
        self.conv5 = nn.Conv2d(
            kernel_size=(5, 5), in_channels=channels, out_channels=channels, padding=2
        )
        self.bn = nn.BatchNorm2d(channels * 2)

    def forward(self, x):
        x3 = self.conv3(x) + x
        x5 = self.conv5(x) + x
        x = torch.cat((x3, x5), 1)
        x = self.bn(x)
        x = F.relu(x)
        return x


class ResConv3(nn.Module):
    """
    Resnet with 3x3 kernels
    """

    def __init__(self, channels=16, **kwargs):
        super(ResConv3, self).__init__()
        self.conv3 = nn.Conv2d(
            kernel_size=(3, 3),
            in_channels=channels,
            out_channels=2 * channels,
            padding=1,
        )
        self.bn = nn.BatchNorm2d(channels * 2)

    def forward(self, x):
        x = self.conv3(x) + torch.cat((x, x), 1)
        x = self.bn(x)
        x = F.relu(x)
        return x


class ResConv5(nn.Module):
    """
    Resnet with 5x5 kernels
    """

    def __init__(self, channels=16, **kwargs):
        super(ResConv5, self).__init__()
        self.conv5 = nn.Conv2d(
            kernel_size=(5, 5),
            in_channels=channels,
            out_channels=2 * channels,
            padding=2,
        )
        self.bn = nn.BatchNorm2d(channels * 2)

    def forward(self, x):
        x = self.conv5(x) + torch.cat((x, x), 1)
        x = self.bn(x)
        x = F.relu(x)
        return x


class CNN_Area(nn.Module):
    """Area attention, Mingke Xu, 2020"""

    def __init__(self, height=3, width=3, out_size=4, shape=(26, 63), **kwargs):
        super(CNN_Area, self).__init__()
        self.height = height
        self.width = width
        # self.conv1 = nn.Conv2D(32, (3,3), padding='same', data_format='channels_last',)
        self.conv1a = nn.Conv2d(
            kernel_size=(10, 2), in_channels=1, out_channels=16, padding=(4, 0)
        )
        self.conv1b = nn.Conv2d(
            kernel_size=(2, 8), in_channels=1, out_channels=16, padding=(0, 3)
        )
        self.conv2 = nn.Conv2d(
            kernel_size=(3, 3), in_channels=16, out_channels=32, padding=1
        )
        self.conv3 = nn.Conv2d(
            kernel_size=(3, 3), in_channels=32, out_channels=48, padding=1
        )
        self.conv4 = nn.Conv2d(
            kernel_size=(3, 3), in_channels=48, out_channels=64, padding=1
        )
        self.conv5 = nn.Conv2d(
            kernel_size=(3, 3), in_channels=64, out_channels=80, padding=1
        )
        # self.conv6 = nn.Conv2D(128, (3, 3), padding= )#
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(16)
        self.bn1b = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(48)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(80)
        # self.gap = nn.AdaptiveAvgPool2d(1)

        i = 80 * ((shape[0] - 1) // 2) * ((shape[1] - 1) // 4)
        self.fc = nn.Linear(in_features=i, out_features=4)

    def forward(self, *input):
        x = input[0]
        xa = self.conv1a(x)
        xa = self.bn1a(xa)
        xa = F.relu(xa)
        xb = self.conv1b(x)
        xb = self.bn1b(xb)
        xb = F.relu(xb)
        x = torch.cat((xa, xb), 2)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.maxp(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.maxp(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x


class CNN_AttnPooling(nn.Module):
    """Head atention, Mingke Xu, 2020
    Attention pooling, Pengcheng Li, Interspeech, 2019
    """

    def __init__(self, head=4, attn_hidden=64, shape=(26, 63), **kwargs):
        super(CNN_AttnPooling, self).__init__()
        self.head = head
        self.attn_hidden = attn_hidden
        self.conv1a = nn.Conv2d(
            kernel_size=(10, 2), in_channels=1, out_channels=8, padding=(4, 0)
        )
        self.conv1b = nn.Conv2d(
            kernel_size=(2, 8), in_channels=1, out_channels=8, padding=(0, 3)
        )
        self.conv2 = nn.Conv2d(
            kernel_size=(3, 3), in_channels=16, out_channels=32, padding=1
        )
        self.conv3 = nn.Conv2d(
            kernel_size=(3, 3), in_channels=32, out_channels=48, padding=1
        )
        self.conv4 = nn.Conv2d(
            kernel_size=(3, 3), in_channels=48, out_channels=64, padding=1
        )
        self.conv5 = nn.Conv2d(
            kernel_size=(3, 3), in_channels=64, out_channels=80, padding=1
        )
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(8)
        self.bn1b = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(48)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(80)
        self.top_down = nn.Conv2d(kernel_size=(1, 1), in_channels=80, out_channels=4)
        self.bottom_up = nn.Conv2d(kernel_size=(1, 1), in_channels=80, out_channels=1)
        # i = 80 * ((shape[0]-1)//4) * ((shape[1]-1)//4)
        # self.fc = nn.Linear(in_features=i, out_features=4)

    def forward(self, *input):
        xa = self.conv1a(input[0])
        xa = self.bn1a(xa)

        xa = F.relu(xa)
        xb = self.conv1b(input[0])
        xb = self.bn1b(xb)

        xb = F.relu(xb)
        x = torch.cat((xa, xb), 1)
        x = self.conv2(x)
        x = self.bn2(x)

        x = F.relu(x)
        x = self.maxp(x)
        x = self.conv3(x)
        x = self.bn3(x)

        x = F.relu(x)
        x = self.maxp(x)
        x = self.conv4(x)
        x = self.bn4(x)

        x = F.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)

        x = F.relu(x)

        x1 = self.top_down(x)
        x1 = F.softmax(x1, 1)
        x2 = self.bottom_up(x)

        x = x1 * x2

        # x = x.sum((2,3))
        x = x.mean((2, 3))

        return x


class CNN_GAP(nn.Module):
    """Head atention, Mingke Xu, 2020
    Attention pooling, Pengcheng Li, Interspeech, 2019
    """

    def __init__(self, head=4, attn_hidden=64, shape=(26, 63), **kwargs):
        super(CNN_GAP, self).__init__()
        self.head = head
        self.attn_hidden = attn_hidden
        self.conv1a = nn.Conv2d(
            kernel_size=(10, 2), in_channels=1, out_channels=8, padding=(4, 0)
        )
        self.conv1b = nn.Conv2d(
            kernel_size=(2, 8), in_channels=1, out_channels=8, padding=(0, 3)
        )
        self.conv2 = nn.Conv2d(
            kernel_size=(3, 3), in_channels=16, out_channels=32, padding=1
        )
        self.conv3 = nn.Conv2d(
            kernel_size=(3, 3), in_channels=32, out_channels=48, padding=1
        )
        self.conv4 = nn.Conv2d(
            kernel_size=(3, 3), in_channels=48, out_channels=64, padding=1
        )
        self.conv5 = nn.Conv2d(
            kernel_size=(3, 3), in_channels=64, out_channels=80, padding=1
        )
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(8)
        self.bn1b = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(48)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(80)
        self.gap = nn.AdaptiveAvgPool2d(1)
        i = 80 * ((shape[0] - 1) // 4) * ((shape[1] - 1) // 4)
        self.fc = nn.Linear(in_features=i, out_features=4)

    def forward(self, *input):
        xa = self.conv1a(input[0])
        xa = self.bn1a(xa)

        xa = F.relu(xa)
        xb = self.conv1b(input[0])
        xb = self.bn1b(xb)

        xb = F.relu(xb)
        x = torch.cat((xa, xb), 1)
        x = self.conv2(x)
        x = self.bn2(x)

        x = F.relu(x)
        x = self.maxp(x)
        x = self.conv3(x)
        x = self.bn3(x)

        x = F.relu(x)
        x = self.maxp(x)
        x = self.conv4(x)
        x = self.bn4(x)

        x = F.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)

        x = F.relu(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x


class MHResMultiConv3(nn.Module):
    """
    Multi-Head-Attention with Multiscale blocks
    """

    def __init__(self, head=4, attn_hidden=64, shape=(26, 63), **kwargs):
        super(MHResMultiConv3, self).__init__()
        self.head = head
        self.attn_hidden = attn_hidden
        self.conv1a = nn.Conv2d(
            kernel_size=(3, 1), in_channels=1, out_channels=16, padding=(1, 0)
        )
        self.conv1b = nn.Conv2d(
            kernel_size=(1, 3), in_channels=1, out_channels=16, padding=(0, 1)
        )
        self.conv2 = ResMultiConv(16)
        self.conv3 = ResMultiConv(32)
        self.conv4 = ResMultiConv(64)
        self.conv5 = nn.Conv2d(
            kernel_size=(5, 5), in_channels=128, out_channels=128, padding=2
        )
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(16)
        self.bn1b = nn.BatchNorm2d(16)
        self.bn5 = nn.BatchNorm2d(128)
        i = self.attn_hidden * self.head * (shape[0] // 2) * (shape[1] // 4)
        self.fc = nn.Linear(in_features=i, out_features=4)

        self.dropout = nn.Dropout(0.5)
        self.attention_query = nn.ModuleList()
        self.attention_key = nn.ModuleList()
        self.attention_value = nn.ModuleList()

        for i in range(self.head):
            self.attention_query.append(
                nn.Conv2d(in_channels=128, out_channels=self.attn_hidden, kernel_size=1)
            )
            self.attention_key.append(
                nn.Conv2d(in_channels=128, out_channels=self.attn_hidden, kernel_size=1)
            )
            self.attention_value.append(
                nn.Conv2d(in_channels=128, out_channels=self.attn_hidden, kernel_size=1)
            )

    def forward(self, *input):
        # input[0]: torch.Size([32, 1, 26, 63])
        xa = self.conv1a(input[0])  # (32, 16, 25, 62)
        xa = self.bn1a(xa)  # (32, 16, 25, 62)

        xa = F.relu(xa)
        xb = self.conv1b(input[0])
        xb = self.bn1b(xb)

        xb = F.relu(xb)
        x = torch.cat((xa, xb), 2)  # (32, 16, 50, 62)

        x = self.conv2(x)  # (32, 32, 50, 62)
        x = self.maxp(x)
        x = self.conv3(x)  # (32, 64, 25, 31)
        x = self.maxp(x)
        x = self.conv4(x)  # (32, 128, 12, 15)

        x = self.conv5(x)  # (32, 128, 12, 15)
        x = self.bn5(x)
        x = F.relu(x)

        attn = None
        for i in range(self.head):
            Q = self.attention_query[i](x)
            K = self.attention_key[i](x)
            V = self.attention_value[i](x)
            attention = F.softmax(torch.mul(Q, K), dim=1)
            attention = torch.mul(attention, V)

            if attn is None:
                attn = attention
            else:
                attn = torch.cat((attn, attention), 2)
        x = attn
        x = F.relu(x)

        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])

        x = self.fc(x)
        return x


class AAResMultiConv3(nn.Module):
    """
    Area Attention with Multiscale blocks
    """

    def __init__(self, head=4, attn_hidden=64, shape=(26, 63), **kwargs):
        super(AAResMultiConv3, self).__init__()
        self.head = head
        self.attn_hidden = attn_hidden
        self.conv1a = nn.Conv2d(
            kernel_size=(3, 1), in_channels=1, out_channels=16, padding=(1, 0)
        )
        self.conv1b = nn.Conv2d(
            kernel_size=(1, 3), in_channels=1, out_channels=16, padding=(0, 1)
        )
        self.conv2 = ResMultiConv(16)
        self.conv3 = ResMultiConv(32)
        self.conv4 = ResMultiConv(64)
        self.conv5 = nn.Conv2d(
            kernel_size=(5, 5), in_channels=128, out_channels=128, padding=2
        )
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(16)
        self.bn1b = nn.BatchNorm2d(16)
        self.bn5 = nn.BatchNorm2d(128)

        i = 128 * (shape[0] // 2) * (shape[1] // 4)
        self.fc = nn.Linear(in_features=i, out_features=4)
        # self.dropout = nn.Dropout(0.5)

        self.area_attention = AreaAttention(
            key_query_size=80,
            area_key_mode="mean",
            area_value_mode="sum",
            max_area_height=3,
            max_area_width=3,
            dropout_rate=0.5,
        )

    def forward(self, *input):
        # input[0]: torch.Size([32, 1, 26, 63])
        xa = self.conv1a(input[0])  # (32, 16, 25, 62)
        xa = self.bn1a(xa)  # (32, 16, 25, 62)

        xa = F.relu(xa)
        xb = self.conv1b(input[0])
        xb = self.bn1b(xb)

        xb = F.relu(xb)
        x = torch.cat((xa, xb), 2)  # (32, 16, 50, 62)

        x = self.conv2(x)  # (32, 32, 50, 62)
        x = self.maxp(x)
        x = self.conv3(x)  # (32, 64, 25, 31)
        x = self.maxp(x)
        x = self.conv4(x)  # (32, 128, 12, 15)

        x = self.conv5(x)  # (32, 128, 12, 15)
        x = self.bn5(x)
        x = F.relu(x)

        shape = x.shape
        x = (
            x.contiguous()
            .permute(0, 2, 3, 1)
            .view(shape[0], shape[3] * shape[2], shape[1])
        )
        x = self.area_attention(x, x, x)
        x = F.relu(x)
        x = x.reshape(x.shape[0], -1)

        x = self.fc(x)

        return x


class ResMultiConv3(nn.Module):
    """
    GLAM - gMLP
    """

    def __init__(self, shape=(26, 63), **kwargs):
        super(ResMultiConv3, self).__init__()
        self.conv1a = nn.Conv2d(
            kernel_size=(3, 1), in_channels=1, out_channels=16, padding=(1, 0)
        )
        self.conv1b = nn.Conv2d(
            kernel_size=(1, 3), in_channels=1, out_channels=16, padding=(0, 1)
        )
        self.conv2 = ResMultiConv(16)
        self.conv3 = ResMultiConv(32)
        self.conv4 = ResMultiConv(64)
        self.conv5 = nn.Conv2d(
            kernel_size=(5, 5), in_channels=128, out_channels=128, padding=2
        )
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(16)
        self.bn1b = nn.BatchNorm2d(16)
        self.bn5 = nn.BatchNorm2d(128)
        dim = (shape[0] // 2) * (shape[1] // 4)
        i = 128 * dim
        self.fc = nn.Linear(in_features=i, out_features=4)
        self.dropout = nn.Dropout(0.5)

    def forward(self, *input):
        # input[0]: torch.Size([32, 1, 26, 63])
        xa = self.conv1a(input[0])  # (32, 16, 25, 62)
        xa = self.bn1a(xa)  # (32, 16, 25, 62)

        xa = F.relu(xa)
        xb = self.conv1b(input[0])
        xb = self.bn1b(xb)

        xb = F.relu(xb)
        x = torch.cat((xa, xb), 2)  # (32, 16, 50, 62)

        x = self.conv2(x)  # (32, 32, 50, 62)
        x = self.maxp(x)
        x = self.conv3(x)  # (32, 64, 25, 31)
        x = self.maxp(x)
        x = self.conv4(x)  # (32, 128, 12, 15)

        x = self.conv5(x)  # (32, 128, 12, 15)
        x = self.bn5(x)
        x = F.relu(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x


class ResMultiConv5(nn.Module):
    """
    GLAM5 - gMLP
    """

    def __init__(self, shape=(26, 63), **kwargs):
        super(ResMultiConv5, self).__init__()
        self.conv1a = nn.Conv2d(
            kernel_size=(5, 1), in_channels=1, out_channels=16, padding=(2, 0)
        )
        self.conv1b = nn.Conv2d(
            kernel_size=(1, 5), in_channels=1, out_channels=16, padding=(0, 2)
        )
        self.conv2 = ResMultiConv(16)
        self.conv3 = ResMultiConv(32)
        self.conv4 = ResMultiConv(64)
        self.conv5 = nn.Conv2d(
            kernel_size=(5, 5), in_channels=128, out_channels=128, padding=2
        )
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(16)
        self.bn1b = nn.BatchNorm2d(16)
        self.bn5 = nn.BatchNorm2d(128)
        dim = (shape[0] // 2) * (shape[1] // 4)
        i = 128 * dim
        self.fc = nn.Linear(in_features=i, out_features=4)
        self.dropout = nn.Dropout(0.5)

    def forward(self, *input):
        # input[0]: torch.Size([32, 1, 26, 63])
        xa = self.conv1a(input[0])  # (32, 16, 25, 62)
        xa = self.bn1a(xa)  # (32, 16, 25, 62)

        xa = F.relu(xa)
        xb = self.conv1b(input[0])
        xb = self.bn1b(xb)

        xb = F.relu(xb)
        x = torch.cat((xa, xb), 2)  # (32, 16, 50, 62)

        x = self.conv2(x)  # (32, 32, 50, 62)
        x = self.maxp(x)
        x = self.conv3(x)  # (32, 64, 25, 31)
        x = self.maxp(x)
        x = self.conv4(x)  # (32, 128, 12, 15)

        x = self.conv5(x)  # (32, 128, 12, 15)
        x = self.bn5(x)
        x = F.relu(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x


class ESIAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, max_len=5000):
        super().__init__()
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.multihead_attention = nn.MultiheadAttention(d_model, n_heads)
        # self.norm1 = nn.LayerNorm(d_model)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x, s):
        # Multi-head self-attention
        x = self.positional_encoding(x)
        s = self.positional_encoding(s)
        attn_output, _ = self.multihead_attention(s, x, x)
        x = x + attn_output
        # x = self.norm1(x)
        return x


class TriChannelAttention(nn.Module):
    def __init__(self, feature_dim):
        super(TriChannelAttention, self).__init__()
        self.Q_Encoder = nn.Linear(feature_dim, feature_dim)
        self.E_Encoder = nn.Linear(feature_dim * 2, feature_dim)
        self.V_Encoder = nn.Linear(feature_dim * 2, feature_dim)

    def forward(self, audio_features, video_features, common_features):
        # 转换特征
        query = self.Q_Encoder(common_features)
        key = self.E_Encoder(torch.cat((audio_features, video_features), dim=1))
        value = self.V_Encoder(torch.cat((audio_features, video_features), dim=1))

        # 计算注意力分数
        attention_scores = F.softmax(torch.matmul(query, key.transpose(-2, -1)), dim=-1)

        # 应用注意力机制
        attended_features = torch.matmul(attention_scores, value)
        return attended_features


class BiLSTM_ESIAttention(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super(BiLSTM_ESIAttention, self).__init__()
        self.positional_encoding = PositionalEncoding(hidden_size, max_len = 5000)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=False,
            batch_first=True,
        )
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=n_heads,
            dropout=dropout,
        )

    def forward(self, x, s):
        # LSTM
        x = self.positional_encoding(x)
        s = self.positional_encoding(s)
         
        s_output, _ = self.lstm(s)  # s_output: [B, S, H]
        # Transpose for multihead attention
        x = x.transpose(0, 1)  # [S, B, H]
        s_output = s_output.transpose(0, 1)  # [S, B, H]
        # Multihead attention
        attn_output, attn_weights = self.multihead_attention(
            query=s_output, key=x, value=x
        )  # attn_output: [S, B, H], attn_weights: [B, S, S]
        x = x + attn_output
        x = x.transpose(0, 1)  # [S, B, H]
        return x


def _get_ModulatedDeformConv2d_extractor(
    in_channels,
    # [channel:int, kernel:list|int, stride:list|int, norm:str|None, pool:str|None, pool_kernel:list|int]
    shapes,
    bias: bool,
) -> Conv2dFeatureExtractor:
    blocks = []
    in_channels = in_channels  # 输入数据的通道

    for i, (
        out_channels,
        kernel_size,
        stride,
        norm_mode,
        pool_mode,
        pool_kernel,
    ) in enumerate(shapes):
        # Norm设置
        assert norm_mode in ["gn", "ln", "bn", "None"]
        assert pool_mode in ["mp", "ap", "gap", "None"]
        if norm_mode == "gn":
            normalization = nn.GroupNorm(
                num_groups=out_channels,
                num_channels=out_channels,
                affine=True,
            )
        elif norm_mode == "ln":
            normalization = nn.LayerNorm(
                normalized_shape=out_channels,
                elementwise_affine=True,
            )
        elif norm_mode == "bn":
            normalization = nn.BatchNorm2d(out_channels)
        else:
            normalization = None

        if pool_mode == "mp":
            poolling = nn.MaxPool2d(pool_kernel)
        elif pool_mode == "ap":
            poolling = nn.AvgPool2d(pool_kernel)
        elif pool_mode == "gap":  # GlobalAveragePooling2D
            pool_kernel = 1
            poolling = nn.AdaptiveAvgPool2d(1)
        else:
            poolling = None
        # 卷积块设置
        blocks.append(
            ModulatedDeformConv2dLayerBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                bias=bias,
                layer_norm=normalization,
                layer_poolling=poolling,
                activate_func=nn.functional.relu,
            )
        )

        in_channels = out_channels
    return Conv2dFeatureExtractor(nn.ModuleList(blocks))


class ModulatedDeformConv2dLayerBlock(nn.Module):
    """单个卷积块,内部含有 Conv2d_layer, Norm_layer(可选), Activation_layer(可选, 默认GELU)"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride: int,
        bias: bool,
        layer_norm: Optional[nn.Module],
        layer_poolling: Optional[nn.Module],
        activate_func=nn.functional.relu,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

        self.conv = ModulatedDeformConv2dPack(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,  # 步长
            bias=bias,  # 是否使用偏置数
        )
        self.layer_norm = layer_norm
        self.activate_func = activate_func
        self.leyer_poolling = layer_poolling

    def forward(
        self,
        x: Tensor,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            x (Tensor): Shape ``[batch, in_channels, in_frames, in_features]``.
        Returns:
            Tensor: Shape ``[batch, out_channels, out_frames, in_features]``.
        """
        x = self.conv(x)
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        if self.activate_func is not None:
            x = self.activate_func(x)
        if self.leyer_poolling is not None:
            x = self.leyer_poolling(x)
        return x


class ModulatedDeformConv2dPack(torchvision.ops.DeformConv2d):
    def __init__(self, *args, **kwargs):
        super(ModulatedDeformConv2dPack, self).__init__(*args, **kwargs)
        self.conv_offset2d = nn.Conv2d(
            self.in_channels,
            self.groups * 3 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=True,
        )
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        if hasattr(self, "conv_offset2d"):
            self.conv_offset2d.weight.data.zero_()
            self.conv_offset2d.bias.data.zero_()

    def forward(self, x: torch.Tensor):
        out = self.conv_offset2d(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return torchvision.ops.deform_conv2d(
            x,
            offset,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            mask=mask,
        )


class ModulatedDeformConv2dSamePack(torchvision.ops.DeformConv2d):
    def __init__(self, *args, **kwargs):
        kwargs.pop("padding_mode")
        super(ModulatedDeformConv2dSamePack, self).__init__(*args, **kwargs)
        self.conv_offset2d = nn.Conv2d(
            self.in_channels,
            self.groups * 3 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=True,
        )
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        if hasattr(self, "conv_offset2d"):
            self.conv_offset2d.weight.data.zero_()
            self.conv_offset2d.bias.data.zero_()

    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor):
        ih, iw = x.size()[-2:]

        pad_h = self.calc_same_pad(
            i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0]
        )
        pad_w = self.calc_same_pad(
            i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1]
        )

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            ).contiguous()

        out = self.conv_offset2d(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return torchvision.ops.deform_conv2d(
            x,
            offset,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            mask=mask,
        )


class DeformConv2dPack(torchvision.ops.DeformConv2d):
    def __init__(self, *args, **kwargs):
        super(DeformConv2dPack, self).__init__(*args, **kwargs)
        self.conv_offset2d = nn.Conv2d(
            self.in_channels,
            self.groups * 2 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=True,
        )
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        if hasattr(self, "conv_offset2d"):
            self.conv_offset2d.weight.data.zero_()
            self.conv_offset2d.bias.data.zero_()

    def forward(self, x: torch.Tensor):
        offset = self.conv_offset2d(x)
        return torchvision.ops.deform_conv2d(
            x,
            offset,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )


# Core code, for ease of understanding, we mark the dimensions of input and output next to the code
class DSC(object):
    def __init__(self, input_shape, kernel_size, extend_scope, morph, device):
        self.num_points = kernel_size
        self.width = input_shape[2]
        self.height = input_shape[3]
        self.morph = morph
        self.device = device
        self.extend_scope = extend_scope  # offset (-1 ~ 1) * extend_scope

        # define feature map shape
        """
        B: Batch size  C: Channel  W: Width  H: Height
        """
        self.num_batch = input_shape[0]
        self.num_channels = input_shape[1]

    """
    input: offset [B,2*K,W,H]  K: Kernel size (2*K: 2D image, deformation contains <x_offset> and <y_offset>)
    output_x: [B,1,W,K*H]   coordinate map
    output_y: [B,1,K*W,H]   coordinate map
    """

    def _coordinate_map_3D(self, offset, if_offset):
        # offset
        y_offset, x_offset = torch.split(offset, self.num_points, dim=1)

        y_center = torch.arange(0, self.width).repeat([self.height])
        y_center = y_center.reshape(self.height, self.width)
        y_center = y_center.permute(1, 0)
        y_center = y_center.reshape([-1, self.width, self.height])
        y_center = y_center.repeat([self.num_points, 1, 1]).float()
        y_center = y_center.unsqueeze(0)

        x_center = torch.arange(0, self.height).repeat([self.width])
        x_center = x_center.reshape(self.width, self.height)
        x_center = x_center.permute(0, 1)
        x_center = x_center.reshape([-1, self.width, self.height])
        x_center = x_center.repeat([self.num_points, 1, 1]).float()
        x_center = x_center.unsqueeze(0)

        if self.morph == 0:
            """
            Initialize the kernel and flatten the kernel
                y: only need 0
                x: -num_points//2 ~ num_points//2 (Determined by the kernel size)
                !!! The related PPT will be submitted later, and the PPT will contain the whole changes of each step
            """
            y = torch.linspace(0, 0, 1)
            x = torch.linspace(
                -int(self.num_points // 2),
                int(self.num_points // 2),
                int(self.num_points),
            )

            y, x = torch.meshgrid(y, x)
            y_spread = y.reshape(-1, 1)
            x_spread = x.reshape(-1, 1)

            y_grid = y_spread.repeat([1, self.width * self.height])
            y_grid = y_grid.reshape([self.num_points, self.width, self.height])
            y_grid = y_grid.unsqueeze(0)  # [B*K*K, W,H]

            x_grid = x_spread.repeat([1, self.width * self.height])
            x_grid = x_grid.reshape([self.num_points, self.width, self.height])
            x_grid = x_grid.unsqueeze(0)  # [B*K*K, W,H]

            y_new = y_center + y_grid
            x_new = x_center + x_grid

            y_new = y_new.repeat(self.num_batch, 1, 1, 1).to(self.device)
            x_new = x_new.repeat(self.num_batch, 1, 1, 1).to(self.device)

            y_offset_new = y_offset.detach().clone()

            if if_offset:
                y_offset = y_offset.permute(1, 0, 2, 3)
                y_offset_new = y_offset_new.permute(1, 0, 2, 3)
                center = int(self.num_points // 2)

                # The center position remains unchanged and the rest of the positions begin to swing
                # This part is quite simple. The main idea is that "offset is an iterative process"
                y_offset_new[center] = 0
                for index in range(1, center):
                    y_offset_new[center + index] = (
                        y_offset_new[center + index - 1] + y_offset[center + index]
                    )
                    y_offset_new[center - index] = (
                        y_offset_new[center - index + 1] + y_offset[center - index]
                    )
                y_offset_new = y_offset_new.permute(1, 0, 2, 3).to(self.device)
                y_new = y_new.add(y_offset_new.mul(self.extend_scope))

            y_new = y_new.reshape(
                [self.num_batch, self.num_points, 1, self.width, self.height]
            )
            y_new = y_new.permute(0, 3, 1, 4, 2)
            y_new = y_new.reshape(
                [self.num_batch, self.num_points * self.width, 1 * self.height]
            )
            x_new = x_new.reshape(
                [self.num_batch, self.num_points, 1, self.width, self.height]
            )
            x_new = x_new.permute(0, 3, 1, 4, 2)
            x_new = x_new.reshape(
                [self.num_batch, self.num_points * self.width, 1 * self.height]
            )
            return y_new, x_new

        else:
            """
            Initialize the kernel and flatten the kernel
                y: -num_points//2 ~ num_points//2 (Determined by the kernel size)
                x: only need 0
            """
            y = torch.linspace(
                -int(self.num_points // 2),
                int(self.num_points // 2),
                int(self.num_points),
            )
            x = torch.linspace(0, 0, 1)

            y, x = torch.meshgrid(y, x)
            y_spread = y.reshape(-1, 1)
            x_spread = x.reshape(-1, 1)

            y_grid = y_spread.repeat([1, self.width * self.height])
            y_grid = y_grid.reshape([self.num_points, self.width, self.height])
            y_grid = y_grid.unsqueeze(0)

            x_grid = x_spread.repeat([1, self.width * self.height])
            x_grid = x_grid.reshape([self.num_points, self.width, self.height])
            x_grid = x_grid.unsqueeze(0)

            y_new = y_center + y_grid
            x_new = x_center + x_grid

            y_new = y_new.repeat(self.num_batch, 1, 1, 1)
            x_new = x_new.repeat(self.num_batch, 1, 1, 1)

            y_new = y_new.to(self.device)
            x_new = x_new.to(self.device)
            x_offset_new = x_offset.detach().clone()

            if if_offset:
                x_offset = x_offset.permute(1, 0, 2, 3)
                x_offset_new = x_offset_new.permute(1, 0, 2, 3)
                center = int(self.num_points // 2)
                x_offset_new[center] = 0
                for index in range(1, center):
                    x_offset_new[center + index] = (
                        x_offset_new[center + index - 1] + x_offset[center + index]
                    )
                    x_offset_new[center - index] = (
                        x_offset_new[center - index + 1] + x_offset[center - index]
                    )
                x_offset_new = x_offset_new.permute(1, 0, 2, 3).to(self.device)
                x_new = x_new.add(x_offset_new.mul(self.extend_scope))

            y_new = y_new.reshape(
                [self.num_batch, 1, self.num_points, self.width, self.height]
            )
            y_new = y_new.permute(0, 3, 1, 4, 2)
            y_new = y_new.reshape(
                [self.num_batch, 1 * self.width, self.num_points * self.height]
            )
            x_new = x_new.reshape(
                [self.num_batch, 1, self.num_points, self.width, self.height]
            )
            x_new = x_new.permute(0, 3, 1, 4, 2)
            x_new = x_new.reshape(
                [self.num_batch, 1 * self.width, self.num_points * self.height]
            )
            return y_new, x_new

    """
    input: input feature map [N,C,D,W,H]；coordinate map [N,K*D,K*W,K*H] 
    output: [N,1,K*D,K*W,K*H]  deformed feature map
    """

    def _bilinear_interpolate_3D(self, input_feature, y, x):
        y = y.reshape([-1]).float()
        x = x.reshape([-1]).float()

        zero = torch.zeros([]).int()
        max_y = self.width - 1
        max_x = self.height - 1

        # find 8 grid locations
        y0 = torch.floor(y).int()
        y1 = y0 + 1
        x0 = torch.floor(x).int()
        x1 = x0 + 1

        # clip out coordinates exceeding feature map volume
        y0 = torch.clamp(y0, zero, max_y)
        y1 = torch.clamp(y1, zero, max_y)
        x0 = torch.clamp(x0, zero, max_x)
        x1 = torch.clamp(x1, zero, max_x)

        input_feature_flat = input_feature.flatten()
        input_feature_flat = input_feature_flat.reshape(
            self.num_batch, self.num_channels, self.width, self.height
        )
        input_feature_flat = input_feature_flat.permute(0, 2, 3, 1)
        input_feature_flat = input_feature_flat.reshape(-1, self.num_channels)
        dimension = self.height * self.width

        base = torch.arange(self.num_batch) * dimension
        base = base.reshape([-1, 1]).float()

        repeat = torch.ones([self.num_points * self.width * self.height]).unsqueeze(0)
        repeat = repeat.float()

        base = torch.matmul(base, repeat)
        base = base.reshape([-1])

        base = base.to(self.device)

        base_y0 = base + y0 * self.height
        base_y1 = base + y1 * self.height

        # top rectangle of the neighbourhood volume
        index_a0 = base_y0 - base + x0
        index_c0 = base_y0 - base + x1

        # bottom rectangle of the neighbourhood volume
        index_a1 = base_y1 - base + x0
        index_c1 = base_y1 - base + x1

        # get 8 grid values
        value_a0 = input_feature_flat[index_a0.type(torch.int64)].to(self.device)
        value_c0 = input_feature_flat[index_c0.type(torch.int64)].to(self.device)
        value_a1 = input_feature_flat[index_a1.type(torch.int64)].to(self.device)
        value_c1 = input_feature_flat[index_c1.type(torch.int64)].to(self.device)

        # find 8 grid locations
        y0 = torch.floor(y).int()
        y1 = y0 + 1
        x0 = torch.floor(x).int()
        x1 = x0 + 1

        # clip out coordinates exceeding feature map volume
        y0 = torch.clamp(y0, zero, max_y + 1)
        y1 = torch.clamp(y1, zero, max_y + 1)
        x0 = torch.clamp(x0, zero, max_x + 1)
        x1 = torch.clamp(x1, zero, max_x + 1)

        x0_float = x0.float()
        x1_float = x1.float()
        y0_float = y0.float()
        y1_float = y1.float()

        vol_a0 = ((y1_float - y) * (x1_float - x)).unsqueeze(-1).to(self.device)
        vol_c0 = ((y1_float - y) * (x - x0_float)).unsqueeze(-1).to(self.device)
        vol_a1 = ((y - y0_float) * (x1_float - x)).unsqueeze(-1).to(self.device)
        vol_c1 = ((y - y0_float) * (x - x0_float)).unsqueeze(-1).to(self.device)

        outputs = (
            value_a0 * vol_a0
            + value_c0 * vol_c0
            + value_a1 * vol_a1
            + value_c1 * vol_c1
        )

        if self.morph == 0:
            outputs = outputs.reshape(
                [
                    self.num_batch,
                    self.num_points * self.width,
                    1 * self.height,
                    self.num_channels,
                ]
            )
            outputs = outputs.permute(0, 3, 1, 2)
        else:
            outputs = outputs.reshape(
                [
                    self.num_batch,
                    1 * self.width,
                    self.num_points * self.height,
                    self.num_channels,
                ]
            )
            outputs = outputs.permute(0, 3, 1, 2)
        return outputs

    def deform_conv(self, input, offset, if_offset):
        y, x = self._coordinate_map_3D(offset, if_offset)
        deformed_feature = self._bilinear_interpolate_3D(input, y, x)
        return deformed_feature


class DSConv(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, extend_scope, morph, if_offset
    ):
        """
        The Dynamic Snake Convolution
        :param in_ch: input channel
        :param out_ch: output channel
        :param kernel_size: the size of kernel
        :param extend_scope: the range to expand (default 1 for this method)
        :param morph: the morphology of the convolution kernel is mainly divided into two types
                        along the x-axis (0) and the y-axis (1) (see the paper for details)
        :param if_offset: whether deformation is required, if it is False, it is the standard convolution kernel
        """
        super(DSConv, self).__init__()
        # use the <offset_conv> to learn the deformable offset
        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size, 3, padding=1)
        self.bn = nn.BatchNorm2d(2 * kernel_size)
        self.kernel_size = kernel_size

        # two types of the DSConv (along x-axis and y-axis)
        self.dsc_conv_x = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            stride=(kernel_size, 1),
            padding=0,
        )
        self.dsc_conv_y = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, kernel_size),
            stride=(1, kernel_size),
            padding=0,
        )

        self.gn = nn.GroupNorm(out_channels // 4, out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.extend_scope = extend_scope
        self.morph = morph
        self.if_offset = if_offset

    def forward(self, f):
        offset = self.offset_conv(f)
        offset = self.bn(offset)
        # We need a range of deformation between -1 and 1 to mimic the snake's swing
        offset = torch.tanh(offset)
        input_shape = f.shape
        device = f.device
        dsc = DSC(input_shape, self.kernel_size, self.extend_scope, self.morph, device)
        deformed_feature = dsc.deform_conv(f, offset, self.if_offset)
        if self.morph == 0:
            x = self.dsc_conv_x(deformed_feature)
            x = self.gn(x)
            x = self.relu(x)
            return x
        else:
            x = self.dsc_conv_y(deformed_feature)
            x = self.gn(x)
            x = self.relu(x)
            return x


def conv2d3x3(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
    with_dcn=False,
    with_dsc=False,
) -> nn.Conv2d:
    """含 padding 的 3x3 卷积"""
    if with_dcn:
        return ModulatedDeformConv2dPack(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            groups=groups,  # 组卷积
            bias=False,
            dilation=dilation,  # 膨胀(空洞)卷积
        )
    if with_dsc:
        return DSConv(
            in_planes,
            out_planes,
            kernel_size=3,
            extend_scope=1,
            morph=0,
            if_offset=True,
        )
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,  # 组卷积
        bias=False,
        dilation=dilation,  # 膨胀(空洞)卷积
    )


def conv2d1x1(
    in_planes: int, out_planes: int, stride: int = 1, with_dcn=False
) -> nn.Conv2d:
    """1x1 卷积"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,  # conv2 的 stride
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,  # 固定值
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        with_dcn=False,
        with_dsc=False,
    ) -> None:
        super().__init__()
        assert not (with_dcn and with_dsc), "可变形卷积和蛇形卷积只能使用一个"
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock 仅支持 groups=1 或 base_width=64")
        if dilation > 1:
            raise NotImplementedError("BasicBlock 不支持 Dilation > 1")
        # 当stride != 1时, self.conv1 和 self.downsample 都会对输入进行下采样操作, 维度发生变化
        self.conv1 = conv2d3x3(
            inplanes, planes, stride, with_dcn=with_dcn, with_dsc=with_dsc
        )
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv2d3x3(planes, planes, with_dcn=with_dcn, with_dsc=with_dsc)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        with_dcn=False,
        with_dsc=False,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        assert not (with_dcn and with_dsc), "可变形卷积和蛇形卷积只能使用一个"

        width = int(planes * (base_width / 64.0)) * groups
        # 当stride != 1时, self.conv2 和 self.downsample 都会对输入进行下采样操作, 维度发生变化
        self.conv1 = conv2d1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv2d3x3(
            width, width, stride, groups, dilation, with_dcn=with_dcn, with_dsc=with_dsc
        )
        self.bn2 = norm_layer(width)
        self.conv3 = conv2d1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        res = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            res = self.downsample(x)

        out += res
        out = self.relu(out)

        return out
