from typing import Type, Any, Callable, Union, List, Optional
import torch
from torch import Tensor, nn
from typing import Tuple, Optional
import torch.nn.functional as F
import math
import torchvision

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

def _get_ModulatedDeformConv2d_extractor(
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
            ModulatedDeformConv2dLayerBlock(
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

def _get_ResMultiConv2d_extractor(
        in_channels,
        shapes,
        bias: bool,) -> Conv2dFeatureExtractor:
    blocks = []
    in_channels = in_channels  # 输入数据的通道

    for i, (out_channels, stride, norm_mode, pool_mode, pool_kernel) in enumerate(shapes):

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
            activate_func=nn.functional.relu,):

        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

        self.conv = ModulatedDeformConv2dSamePack(
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

class Conv2dResMultiBlock(nn.Module):
    '''
    Multi-scale block with short-cut connections
    '''

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int,
                 bias: bool,
                 layer_norm: Optional[nn.Module],
                 layer_poolling: Optional[nn.Module],
                 activate_func=nn.functional.relu,
                groups: int = 1, 
                dilation: int = 1, 
                with_dcn = False
                 ):
        

        
        super().__init__()
        assert (out_channels % 2 == 0)

        self.layer_norm = layer_norm
        self.activate_func = activate_func
        self.leyer_poolling = layer_poolling

        if with_dcn:
            conv2dSame = ModulatedDeformConv2dSamePack
        else:
            conv2dSame = Conv2dSame

        self.conv3 = conv2dSame(
            in_channels=in_channels,
            out_channels=int(out_channels/2),
            kernel_size=[3, 3],
            stride=stride,  # 步长
            bias=bias,  # 是否使用偏置数
            padding_mode='zeros',
            groups = groups,
            dilation = dilation
        )
        self.conv5 = conv2dSame(
            in_channels=in_channels,
            out_channels=int(out_channels/2),
            kernel_size=[5, 5],
            stride=stride,  # 步长
            bias=bias,  # 是否使用偏置数
            padding_mode='zeros',
            groups = groups,
            dilation = dilation,
        )
        self.subconv = None
        if in_channels != int(out_channels/2):
            self.subconv = nn.Conv2d(
                kernel_size=[1, 1], in_channels=in_channels, out_channels=int(out_channels/2), stride=stride, bias=bias, )

    def forward(self, x):

        x3 = self.conv3(x)
        x5 = self.conv5(x)
        if self.subconv is not None:
            x = self.subconv(x)
        x3 = x3+x
        x5 = x5+x
        x = torch.cat((x3, x5), 1)
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
            bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        if hasattr(self, 'conv_offset2d'):
            self.conv_offset2d.weight.data.zero_()
            self.conv_offset2d.bias.data.zero_()
        
    def forward(self, x: torch.Tensor):
        out = self.conv_offset2d(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return torchvision.ops.deform_conv2d(x, offset, self.weight, self.bias, stride=self.stride,
                             padding=self.padding, dilation=self.dilation, mask=mask)

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
            bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        if hasattr(self, 'conv_offset2d'):
            self.conv_offset2d.weight.data.zero_()
            self.conv_offset2d.bias.data.zero_()

    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)


    def forward(self, x: torch.Tensor):
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

        out = self.conv_offset2d(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return torchvision.ops.deform_conv2d(x, offset, self.weight, self.bias, stride=self.stride,
                             padding=self.padding, dilation=self.dilation, mask=mask)

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
            bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        if hasattr(self, 'conv_offset2d'):
            self.conv_offset2d.weight.data.zero_()
            self.conv_offset2d.bias.data.zero_()
        
    def forward(self, x: torch.Tensor):
        offset = self.conv_offset2d(x)
        return torchvision.ops.deform_conv2d(x, offset, self.weight, self.bias, stride=self.stride,
                             padding=self.padding, dilation=self.dilation)
    

def conv2d3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1, with_dcn = False) -> nn.Conv2d:
    """含 padding 的 3x3 卷积 """
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


def conv2d1x1(in_planes: int, out_planes: int, stride: int = 1, with_dcn = False) -> nn.Conv2d:
    """1x1 卷积 """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv2dMultiScale(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1, with_dcn = False):
    return Conv2dResMultiBlock(
        in_planes,
            out_planes,
            stride=stride,
            bias=False,
            groups=groups,  # 组卷积
            dilation=dilation,  # 膨胀(空洞)卷积
                layer_norm = None,
                layer_poolling = None,
                activate_func = None,
             with_dcn = with_dcn

            )


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
        with_dcn = True,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                "BasicBlock 仅支持 groups=1 或 base_width=64")
        if dilation > 1:
            raise NotImplementedError(
                "BasicBlock 不支持 Dilation > 1")
        # 当stride != 1时, self.conv1 和 self.downsample 都会对输入进行下采样操作, 维度发生变化
        self.conv1 = conv2d3x3(inplanes, planes, stride, with_dcn = with_dcn)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv2d3x3(planes, planes, with_dcn = with_dcn)
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
        with_dcn = True,
    ) -> None:

        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # 当stride != 1时, self.conv2 和 self.downsample 都会对输入进行下采样操作, 维度发生变化
        self.conv1 = conv2d1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        # self.conv2 = conv2d3x3(width, width, stride, groups, dilation, with_dcn = with_dcn)
        self.conv2 = conv2dMultiScale(width, width, stride, groups, dilation, with_dcn = with_dcn)
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








