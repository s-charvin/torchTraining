from typing import Type, Any, Callable, Union, List, Optional
import torch
from torch import Tensor, nn
from typing import Tuple, Optional

from ..emotion_baseline_attention_lstm.components import TestLSTMCell, AttLSTMCell
from torch.nn import init
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """多头注意力机制 (Multihead Self Attention module )

    Args:
        embed_dim (int): Total dimension of the model.
        num_heads (int): The number of heads.
        dropout (float, optional):
            Dropout probabiliry on attn_output_weights. Default: ``0.0``
    """

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dropout: float = 0.0,):
        super().__init__()
        head_dim = embed_dim // num_heads  # 计算每个 Head 对应的 Q, K, V 的输出特征维度
        if head_dim * num_heads != embed_dim:
            raise ValueError(
                f"`输入嵌入向量为度 embed_dim({embed_dim})`不能整除 `num_heads ({num_heads})`")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = head_dim

        self.scaling = self.head_dim ** -0.5

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(
            self,
            x: Tensor,
            attention_mask: Optional[Tensor] = None,) -> Tensor:
        """
        Args:
            x (Tensor): shape: ``[batch_size, sequence_length, embed_dim]``.
            attention_mask (Tensor or None, optional):
                shape: ``[batch_size, 1, sequence_length, sequence_length]``

        Returns:
            Tensor: The resulting tensor. shape: ``[batch, sequence_length, embed_dim]``
        """
        # 检查参数
        if x.ndim != 3 or x.shape[2] != self.embed_dim:
            raise ValueError(
                f"期望输入维度大小为 (batch, sequence, embed_dim=={self.embed_dim}), 但得到的却是 {x.shape}."
            )
        batch_size, length, embed_dim = x.size()
        if attention_mask is not None:
            shape_ = (batch_size, 1, length, length)
            if attention_mask.size() != shape_:
                raise ValueError(
                    f"期望的 attention mask 维度大小为 {shape_}, 但得到的却是 {attention_mask.size()}."
                )

        shape = (batch_size, length, self.num_heads, self.head_dim)
        # [batch_size, num_heads, sequence_length, head_dim], 这里 num_heads*head_dim =embed_dim
        q = self.q_proj(x).view(*shape).transpose(2, 1)  # B, nH, L, Hd
        # [batch_size, num_heads, head_dim, sequence_length]
        k = self.k_proj(x).view(*shape).permute(0, 2, 3, 1)  # B, nH, Hd, L
        # [batch_size, num_heads, sequence_length, head_dim]
        v = self.v_proj(x).view(*shape).transpose(2, 1)  # B, nH, L, Hd

        # 求平均, 进行特征缩放
        weights = self.scaling * (q @ k)  # B, nH, L, L , 注意 @ = torch.matmul
        if attention_mask is not None:
            weights += attention_mask

        weights = torch.nn.functional.softmax(weights, dim=-1)
        weights = torch.nn.functional.dropout(
            weights, p=self.dropout, training=self.training)
        # [batch_size, num_heads, sequence_length, head_dim]
        output = weights @ v  # B, nH, L, Hd
        # [batch_size, sequence_length, embed_dim]
        output = output.transpose(2, 1).reshape(batch_size, length, embed_dim)

        output = self.out_proj(output)
        return output


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

        self.layer_norm = layer_norm
        self.conv = nn.Conv2d(  # 卷积核是一维的，卷积操作沿着输入[batch, in_channels, in_frame]第二维在第三维上进行。
            in_channels=in_channels,
            out_channels=out_channels,  # 输出通道数,即使用几个卷积核
            kernel_size=kernel_size,
            stride=stride,  # 步长
            bias=bias,  # 是否使用偏置数
        )
        self.activate_func = activate_func
        self.leyer_poolling = layer_poolling

    def forward(self, x: Tensor,) -> Tensor:
        """
        Args:
            x (Tensor): Shape ``[batch, in_channels, H, W]``.
        Returns:
            Tensor: Shape ``[batch, out_channels, out_frames]``.
        """
        x = self.conv(x)
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        if self.activate_func is not None:
            x = self.activate_func(x)
        if self.leyer_poolling is not None:
            x = self.leyer_poolling(x)
        return x


class Conv1dLayerBlock(nn.Module):
    """单个卷积块,内部含有 Conv1d_layer, Norm_layer(可选), Activation_layer(可选, 默认GELU)"""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size,
            stride: int,
            bias: bool,
            layer_norm: Optional[nn.Module],
            layer_poolling: Optional[nn.Module],
            activate_func=nn.functional.relu,):

        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

        self.layer_norm = layer_norm
        self.conv = nn.Conv1d(  # 卷积核是一维的，卷积操作沿着输入[batch, in_channels, in_frame]第二维在第三维上进行。
            in_channels=in_channels,
            out_channels=out_channels,  # 输出通道数,即使用几个卷积核
            kernel_size=kernel_size,  # 竖着扫的范围
            stride=stride,  # 步长
            bias=bias,  # 是否使用偏置数
        )
        self.activate_func = activate_func
        self.leyer_poolling = layer_poolling

    def forward(
            self,
            x: Tensor,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            x (Tensor): Shape ``[batch, in_channels, in_frames]``.
        Returns:
            Tensor: Shape ``[batch, out_channels, out_frames]``.
            Optional[Tensor]: Shape ``[batch, ]``.
        """
        x = self.conv(x)
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        if self.activate_func is not None:
            x = self.activate_func(x)
        if self.leyer_poolling is not None:
            x = self.leyer_poolling(x)
        return x


class Conv2dFeatureExtractor(nn.Module):
    """
    Args:
        conv_layers (nn.ModuleList): 模型所需的网络结构列表
    """

    def __init__(self, conv_layers: nn.ModuleList,):
        super().__init__()
        self.conv_blocks = conv_layers

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor):
                shape: ``[batch, time]``.
        Returns:
            Tensor:
                输出结果, shape: ``[batch, frame, feature]``
        """
        # 检查参数
        if x.ndim == 3:
            x = x.unsqueeze(1)  # 调整输入数据维度为(batch, channel==1, H, W)
        if x.ndim != 4:
            raise ValueError(
                f"期望输入的维度为 4D (batch, channel, H, W), 但得到的输入数据维度为 {list(x.shape)}")

        for block in self.conv_blocks:
            x = block(x)
        return x


class Conv1dFeatureExtractor(nn.Module):
    """
    Args:
        conv_layers (nn.ModuleList): 模型所需的网络结构列表
    """

    def __init__(self, conv_layers: nn.ModuleList,):
        super().__init__()
        self.conv_blocks = conv_layers

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor):
                shape: ``[batch, frame, feature]``.
        Returns:
            Tensor:
                输出结果, shape: ``[batch, frame, feature]``
        """
        # 检查参数
        if x.ndim != 3:
            raise ValueError(
                f"期望输入的维度为 3D (batch, seq, H), 但得到的输入数据维度为 {list(x.shape)}")
        x = x.permute(0, 2, 1)  # [batch, feature, frame]
        for block in self.conv_blocks:
            x = block(x)
        return x.permute(0, 2, 1)


def _get_Conv1d_extractor(norm_mode,
                          pool_mode,
                          in_channels: int,
                          shapes,
                          bias: bool,) -> Conv1dFeatureExtractor:

    assert norm_mode in ["group_norm", "layer_norm", "batch_norm", None]
    assert pool_mode in ["max_pool", None]
    blocks = []
    in_channels = in_channels  # 输入数据的通道

    for i, (out_channels, kernel_size, stride) in enumerate(shapes):

        # Norm设置
        normalization = None
        if norm_mode == "group_norm" and i == 0:  # 只在第一个卷积块使用GroupNorm

            normalization = nn.GroupNorm(
                num_groups=out_channels,
                num_channels=out_channels,
                affine=True,
            )
        elif norm_mode == "layer_norm":  # 所有卷积块都有一个layer_norm层
            normalization = nn.LayerNorm(
                normalized_shape=out_channels,
                elementwise_affine=True,
            )
        elif norm_mode == "batch_norm":   # 所有卷积块都有一个batch_norm层
            normalization = nn.BatchNorm1d(out_channels)
        else:
            normalization = None

        if pool_mode == "max_pool":
            poolling = nn.MaxPool1d(2)
        else:
            poolling = None
        # 卷积块设置
        blocks.append(
            Conv1dLayerBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                bias=bias,
                layer_norm=normalization,
                layer_poolling=poolling,
                activate_func=nn.functional.relu
            )
        )

        in_channels = out_channels  # 修改输入通道为 Extractor 的输出通道
    return Conv1dFeatureExtractor(nn.ModuleList(blocks))


def _get_Conv2d_extractor(norm_mode,
                          pool_mode,
                          in_channels: int,
                          shapes,
                          bias: bool,) -> Conv2dFeatureExtractor:

    assert norm_mode in ["group_norm", "layer_norm", "batch_norm", None]
    assert pool_mode in ["max_pool", "avg_pool", None]
    blocks = []
    in_channels = in_channels  # 输入数据的通道

    for i, (out_channels, kernel_size, stride) in enumerate(shapes):

        # Norm设置
        normalization = None
        if norm_mode == "group_norm" and i == 0:  # 只在第一个卷积块使用GroupNorm

            normalization = nn.GroupNorm(
                num_groups=out_channels,
                num_channels=out_channels,
                affine=True,
            )
        elif norm_mode == "layer_norm":  # 所有卷积块都有一个layer_norm层
            normalization = nn.LayerNorm(
                normalized_shape=out_channels,
                elementwise_affine=True,
            )
        elif norm_mode == "batch_norm":   # 所有卷积块都有一个batch_norm层
            normalization = nn.BatchNorm2d(out_channels)
        else:
            normalization = None

        if pool_mode == "max_pool":
            poolling = nn.MaxPool2d(2)
        elif pool_mode == "avg_pool":
            if i != 3:
                poolling = nn.AvgPool2d(2)
            else:
                poolling = None
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
                activate_func=nn.functional.relu
            )
        )

        in_channels = out_channels  # 修改输入通道为 Extractor 的输出通道
    return Conv2dFeatureExtractor(nn.ModuleList(blocks))
