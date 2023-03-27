from typing import Type, Any, Callable, Union, List, Optional

import torch
from torch import Tensor, nn
from torch.nn import Module


def conv2d3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """含 padding 的 3x3 卷积 """
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


def conv2d1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 卷积 """
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
        self.conv1 = conv2d3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv2d3x3(planes, planes)
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
    ) -> None:

        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # 当stride != 1时, self.conv2 和 self.downsample 都会对输入进行下采样操作, 维度发生变化
        self.conv1 = conv2d1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv2d3x3(width, width, stride, groups, dilation)
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
