from typing import Tuple, Type, Any, Callable, Union, List, Optional
import torch
from torch import Tensor, nn
import torchvision
import torch.nn.functional as F
import math
import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F


class Transformer_Attention(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        nhead: int = 1,
        dim_feedforward: int = 2048,
    ):
        super(Transformer_Attention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.nhead = nhead

        # 位置编码 (可根据需要进行修改或增强)
        self.positional_encoding = nn.Parameter(torch.randn(1, hidden_size))

        # Transformer Encoder Layer
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=nhead, dim_feedforward=dim_feedforward
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        # 线性层将输入尺寸转换为 hidden_size
        self.input_linear = nn.Linear(input_size, hidden_size)

    def forward(self, X):
        # X : [B, S, F]

        # 将输入转换为 hidden_size
        X = self.input_linear(X)

        # 将位置编码添加到输入
        S = X.size(1)
        X = X + self.positional_encoding[:, :S]

        # Transformer Encoder
        # 需要调整维度为 [S, B, H]
        X = X.permute(1, 0, 2)
        output = self.transformer_encoder(X)
        output = output.permute(1, 0, 2)  # 调整回 [B, S, H]

        # 使用自注意力的最后一层的输出作为 context
        context = output[:, -1, :]

        # 这里没有显式的注意力权重，如果需要，可以从 Transformer 的内部结构提取
        # 或使用额外的注意力机制来获取这些权重

        # model : [B, H], attention (暂时未提供) : [B, S]
        return context, None  # 由于没有显式的注意力权重，这里返回 None


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
        kwargs.pop("padding_mode", None)
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

    def _coordinate_map_3D(self, offset, if_offset):
        """
        input: offset [B,2*K,W,H]  K: Kernel size (2*K: 2D image, deformation contains <x_offset> and <y_offset>)
        output_x: [B,1,W,K*H]   coordinate map
        output_y: [B,1,K*W,H]   coordinate map
        """
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

    def _bilinear_interpolate_3D(self, input_feature, y, x):
        """
        input: input feature map [N,C,D,W,H]；coordinate map [N,K*D,K*W,K*H]
        output: [N,1,K*D,K*W,K*H]  deformed feature map
        """
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
        self,
        in_channels,
        out_channels,
        kernel_size,
        extend_scope=1,
        morph=0,
        if_offset=True,
    ):
        """参数介绍
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param kernel_size: 卷积核大小
        :param extend_scope: 扩展的范围(默认为1)
        :param morph: 卷积核的形态主要分为两种类型, 沿着x轴(0)和y轴(1)方向(详见论文)
        :param if_offset: 控制是否使用可变形卷积， False 时为标准卷积
        """
        super(DSConv, self).__init__()
        # 使用 offset_conv 学习可变形偏移
        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size, 3, padding=1)
        self.bn = nn.BatchNorm2d(2 * kernel_size)
        self.kernel_size = kernel_size

        # 两种类型的 DSConv (沿着 x 轴和 y 轴)
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
        self.if_offset = if_offset  #

    def forward(self, f):
        offset = self.offset_conv(f)
        offset = self.bn(offset)

        # 这里需要一个 -1 到 1 的变形范围来模拟蛇的摆动
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


class EncoderPath1(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.branch0 = nn.Sequential(
            Conv2dSame(1, 16, kernel_size=(1, 1), stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.branch1 = nn.Sequential(
            Conv2dSame(1, 8, kernel_size=(1, 1), stride=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            Conv2dSame(8, 16, kernel_size=(1, 3), stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            Conv2dSame(16, 16, kernel_size=(11, 1), stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        self.branch2 = nn.Sequential(
            nn.AvgPool2d(2),
            Conv2dSame(1, 16, kernel_size=(1, 1), stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x = torch.cat([x0, x1, x2], dim=1)
        return x


class EncoderPath2(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.branch0 = nn.Sequential(
            Conv2dSame(1, 16, kernel_size=(1, 1), stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.branch2 = nn.Sequential(
            nn.AvgPool2d(2),
            Conv2dSame(1, 16, kernel_size=(1, 1), stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.branch1 = nn.Sequential(
            Conv2dSame(1, 8, kernel_size=(1, 1), stride=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            Conv2dSame(8, 16, kernel_size=(3, 1), stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            Conv2dSame(16, 16, kernel_size=(1, 9), stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x = torch.cat([x0, x1, x2], dim=1)
        return x


class EncoderPath3(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.branch0 = nn.Sequential(
            Conv2dSame(1, 16, kernel_size=(1, 1), stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.branch2 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            Conv2dSame(1, 16, kernel_size=(1, 1), stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.branch1 = nn.Sequential(
            Conv2dSame(1, 8, kernel_size=(1, 1), stride=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            Conv2dSame(8, 16, kernel_size=(3, 1), stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            Conv2dSame(16, 16, kernel_size=(1, 3), stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            Conv2dSame(16, 16, kernel_size=(3, 1), stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            Conv2dSame(16, 16, kernel_size=(1, 3), stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x = torch.cat([x0, x1, x2], dim=1)
        return x


class EncoderExtractorBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.branch0 = nn.Sequential(
            Conv2dSame(48, 48, kernel_size=(1, 1), stride=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
        )

        self.branch1 = nn.Sequential(
            Conv2dSame(48, 48, kernel_size=(1, 1), stride=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            Conv2dSame(48, 48, kernel_size=(3, 1), stride=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            Conv2dSame(48, 48, kernel_size=(1, 3), stride=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
        )

        self.branch2 = nn.Sequential(
            Conv2dSame(48, 48, kernel_size=(1, 1), stride=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            Conv2dSame(48, 48, kernel_size=(5, 1), stride=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            Conv2dSame(48, 48, kernel_size=(1, 5), stride=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
        )

        self.conv_extractor = nn.Sequential(
            Conv2dSame(144, 48, kernel_size=(1, 1), stride=1),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x = torch.cat([x0, x1, x2], dim=1)
        x = self.conv_extractor(x)
        return x


class CLANet(nn.Module):
    def __init__(self, seq_len=189) -> None:
        super().__init__()

        self.path1 = EncoderPath1()
        self.path2 = EncoderPath2()
        self.path3 = EncoderPath3()
        self.conv_extractor1 = EncoderExtractorBlock(48, 48)
        self.conv_extractor2 = EncoderExtractorBlock(48, 48)
        self.conv_extractor3 = EncoderExtractorBlock(48, 48)
        self.conv_extractor = nn.Sequential(
            Conv2dSame(144, 48, kernel_size=(1, 1), stride=1),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(0.5)
        self.seq_len = seq_len

    def forward(self, x: Tensor, feature_lens=None) -> Tensor:
        x = x.unsqueeze(1).squeeze(-1)
        B, S, F = x.shape

        segments = torch.split(x, self.seq_len, dim=1)  # 沿序列维度分割
        processed_segments = []
        for segment in segments:
            out1 = self.path1(segment)
            out2 = self.path2(segment)
            out3 = self.path3(segment)
            out = torch.cat([out1, out2, out3], dim=1)
            out = self.conv_extractor(out)
            processed_segments.append(out)

        x = torch.cat(processed_segments, dim=1)
        x = self.dropout(x.squeeze(2).squeeze(2))
        x = self.last_linear(x)

        return x, None
