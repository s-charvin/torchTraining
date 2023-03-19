from typing import Type, Any, Callable, Union, List, Optional
import torch
from torch import Tensor, nn
from typing import Tuple, Optional
import torch.nn.functional as F
import math


class Conv2dFeatureExtractor(nn.Module):
    """构建连续的2D卷积网络结构主Module
    Args:
        conv_layers (nn.ModuleList): 模型所需的2D卷积网络结构列表
    """

    def __init__(self, conv_layers: nn.ModuleList,):
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
                f"期望输入的维度为 4D (batch,in_channels, frame, feature), 但得到的输入数据维度为 {list(x.shape)}")
        for block in self.conv_blocks:
            x = block(x)
        return x


def _get_Conv2d_extractor(
        in_channels,
        # [channel:int, kernel:list|int, stride:list|int, norm:str|None, pool:str|None, pool_kernel:list|int]
        shapes,
        bias: bool,) -> Conv2dFeatureExtractor:

    blocks = []
    in_channels = in_channels  # 输入数据的通道

    for i, (out_channels, kernel_size, stride, norm_mode, pool_mode, pool_kernel) in enumerate(shapes):

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
                activate_func=nn.functional.relu
            )
        )

        in_channels = out_channels  # 修改输入通道为 Extractor 的输出通道
    return Conv2dFeatureExtractor(nn.ModuleList(blocks))


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
            activate_func=nn.functional.relu,):

        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

        self.conv = Conv2dSame(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,  # 步长
            bias=bias,  # 是否使用偏置数
            padding_mode='zeros'
        )
        self.layer_norm = layer_norm
        self.activate_func = activate_func
        self.leyer_poolling = layer_poolling

    def forward(
            self,
            x: Tensor,) -> Tuple[Tensor, Optional[Tensor]]:
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
            i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0])
        pad_w = self.calc_same_pad(
            i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1])

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2,
                    pad_h // 2, pad_h - pad_h // 2]
            ).contiguous()
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        ).contiguous()


class BiLSTM_Attention(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int = 1,
                 bias: bool = True,
                 bidirectional: bool = False,):
        super(BiLSTM_Attention, self).__init__()
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, bidirectional=bidirectional, bias=bias)

    # lstm_output : [batch_size, n_step, hidden_size * num_directions(=2)], F matrix

    def attention_net(self, lstm_output, final_state):
        # [num_layers(=1) * num_directions(=1), batch_size, hidden_size]
        hidden = final_state.view(-1, self.hidden_size *
                                  self.num_directions, self.num_layers)
        # hidden : [batch_size, hidden_size * num_directions(=1), 1(=n_layer)]

        # [batch_size, len_seq, num_directions * hidden_size] * [batch_size, hidden_size * num_directions, 1]
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(
            2)
        # attn_weights : [batch_size, len_seq]
        soft_attn_weights = F.softmax(attn_weights, 1)  # 对每个输入的所有序列点的权重, 即重视程度

        # [batch_size, hidden_size * num_directions, len_seq] * [batch_size, len_seq, 1]
        # 对每个seq点的特征乘以了一个权重
        context = torch.bmm(lstm_output.transpose(
            1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)

        # context : [batch_size, hidden_size * num_directions]
        return context, soft_attn_weights.data.numpy()

    def forward(self, X):
        # X : [batch_size, len_seq, input_size]
        X = X.permute(1, 0, 2)
        # X : [len_seq, batch_size, input_size]

        # [num_layers * num_directions, batch_size, hidden_size]
        hidden_state = torch.zeros(
            self.num_layers*self.num_directions, len(X), self.hidden_size)
        cell_state = torch.zeros(
            self.num_layers*self.num_directions, len(X), self.hidden_size)

        # final_hidden_state, final_cell_state : [num_layers * num_directions, batch_size, hidden_size]
        output, (final_hidden_state, final_cell_state) = self.lstm(
            input, (hidden_state, cell_state))

        output = output.permute(1, 0, 2)
        # output : [batch_size, len_seq, num_directions * hidden_size]
        attn_output, attention = self.attention_net(output, final_hidden_state)
        # model : [batch_size, num_classes], attention : [batch_size, n_step]
        return attn_output, attention
