
from . import components
from typing import Type, Any, Callable, Union, List, Optional
import torch
from torch import Tensor, nn


class ResNet(nn.Module):
    """ ResNet 基本模型, 可通过给定参数(block, layers等)定制模型. 例如:
        resnet18 = ResNet(components.Bottleneck, [2, 2, 2, 2])
        resnet34 = ResNet(components.BasicBlock, [3, 4, 6, 3])
        resnet50 = ResNet(components.Bottleneck, [3, 4, 6, 3])
        resnet101 = ResNet(components.Bottleneck, [3, 4, 23, 3])
        resnet152 = ResNet(components.Bottleneck, [3, 8, 36, 3])
        resnext50_32x4d = ResNet(components.Bottleneck, [3, 4, 6, 3], groups = 32, width_per_group = 4)
        resnext101_32x8d = ResNet(components.Bottleneck, [3, 4, 23, 3], groups = 32, width_per_group = 8)
        wide_resnet50_2 = ResNet(components.Bottleneck, [3, 4, 6, 3], width_per_group = 64 * 2)
        wide_resnet101_2 = ResNet(components.Bottleneck, [3, 4, 23, 3], width_per_group = 64 * 2)

    Args:
        block (Type[Union[components.BasicBlock, components.Bottleneck]]): 可调 block 类型
        layers (List[int]): 每个 layer 中 block 的数目
        num_classes (int): 分类数量, 也是输出数量
        zero_init_residual (bool, optional): _description_. Defaults to False.
        groups (int, optional): 卷积分组数. Defaults to 1.
        width_per_group (int, optional): 卷积过程中每组的宽度. Defaults to 64.
        replace_stride_with_dilation (Optional[List[bool]], optional): # 使用膨胀(空洞)卷积替换普通组卷积. 
            Defaults to None.
            含有三个 bool 元素的列表, 分别对应是否替换 layer[1-3] 的卷积
        norm_layer (Optional[Callable[..., nn.Module]], optional): norm 层. Defaults to nn.BatchNorm2d.
    """

    def __init__(
            self,
            in_channels=3,
            num_classes=1000,
            blocktype="Bottleneck",  # "BasicBlock", "Bottleneck"
            layers=[3, 4, 6, 3],
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d
    ) -> None:

        super().__init__()

        if blocktype == "Bottleneck":
            block = components.Bottleneck
        elif blocktype == "BasicBlock":
            block = components.BasicBlock

        self.norm_layer = norm_layer
        self.inplanes = 64   # 初始特征通道数
        self.dilation = 1
        zero_init_residual = zero_init_residual
        self.groups = groups
        self.base_width = width_per_group
        self.num_outputs = num_classes

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        self.conv1 = nn.Conv2d(in_channels, self.inplanes,
                               kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self.norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_block(block, 64, layers[0])
        self.layer2 = self._make_block(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_block(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_block(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.last_linear = nn.Linear(512 * block.expansion, self.num_outputs)

        self.init_parameters(zero_init_residual)

    def init_parameters(self, zero_init_residual):
        # Conv2d 和 Norm 参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # 初始化每个残差分支中的最后一个 BN 层，以便残差分支从零开始，并且每个残差块的行为就像一个identity
        # 根据 https://arxiv.org/abs/1706.02677 的数据，通过这种操作, 可以改进0.2~0.3个百分点

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, components.Bottleneck):
                    # type: ignore[arg-type]
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, components.BasicBlock):
                    # type: ignore[arg-type]
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_block(
            self,
            block: Type[Union[components.BasicBlock, components.Bottleneck]],
            planes: int,
            blocks: int,
            stride: int = 1,
            dilate: bool = False,) -> nn.Sequential:
        """构建动态结构
        Args:
            block (Type[Union[components.BasicBlock, components.Bottleneck]]): 
            planes (int): _description_
            blocks (int): block数量
            stride (int, optional): 卷积步长. Defaults to 1.
            dilate (bool, optional): 是否使用空洞卷积. Defaults to False.
        Returns:
            nn.Sequential: 
        """
        norm_layer = self.norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                components.conv2d1x1(
                    self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        # 静态运算结构
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 动态可调运算结构
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 输出结构
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.last_linear(x)
        return x



# 基线模型
class DCResNet(nn.Module):
    """ 
        ResNet 基本模型, 可通过给定参数(block, layers等)定制模型. 例如:
        resnet18 = ResNet(components.Bottleneck, [2, 2, 2, 2])
        resnet34 = ResNet(components.BasicBlock, [3, 4, 6, 3])
        resnet50 = ResNet(components.Bottleneck, [3, 4, 6, 3])
        resnet101 = ResNet(components.Bottleneck, [3, 4, 23, 3])
        resnet152 = ResNet(components.Bottleneck, [3, 8, 36, 3])
        resnext50_32x4d = ResNet(components.Bottleneck, [3, 4, 6, 3], groups = 32, width_per_group = 4)
        resnext101_32x8d = ResNet(components.Bottleneck, [3, 4, 23, 3], groups = 32, width_per_group = 8)
        wide_resnet50_2 = ResNet(components.Bottleneck, [3, 4, 6, 3], width_per_group = 64 * 2)
        wide_resnet101_2 = ResNet(components.Bottleneck, [3, 4, 23, 3], width_per_group = 64 * 2)

    Args:
        block (Type[Union[components.BasicBlock, components.Bottleneck]]): 可调 block 类型
        layers (List[int]): 每个 layer 中 block 的数目
        num_classes (int): 分类数量, 也是输出数量
        zero_init_residual (bool, optional): _description_. Defaults to False.
        groups (int, optional): 卷积分组数. Defaults to 1.
        width_per_group (int, optional): 卷积过程中每组的宽度. Defaults to 64.
        replace_stride_with_dilation (Optional[List[bool]], optional): # 使用膨胀(空洞)卷积替换普通组卷积. 
            Defaults to None.
            含有三个 bool 元素的列表, 分别对应是否替换 layer[1-3] 的卷积
        norm_layer (Optional[Callable[..., nn.Module]], optional): norm 层. Defaults to nn.BatchNorm2d.
    """

    def __init__(
            self,
            in_channels=3,
            num_classes=1000,
            blocktype="Bottleneck",  # "BasicBlock", "Bottleneck"
            layers=[3, 4, 6, 3],
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
            with_dcn = True,
    ) -> None:

        super().__init__()

        if blocktype == "Bottleneck":
            block = components.Bottleneck
        elif blocktype == "BasicBlock":
            block = components.BasicBlock

        self.norm_layer = norm_layer
        self.inplanes = 64   # 初始特征通道数
        self.dilation = 1
        zero_init_residual = zero_init_residual
        self.groups = groups
        self.base_width = width_per_group
        self.num_outputs = num_classes

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        self.conv1 = nn.Conv2d(in_channels, self.inplanes,
                               kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self.norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_block(block, 64, layers[0])
        self.layer2 = self._make_block(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0], with_dcn = with_dcn)
        self.layer3 = self._make_block(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1], with_dcn = with_dcn)
        self.layer4 = self._make_block(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2], with_dcn = with_dcn)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.last_linear = nn.Linear(512 * block.expansion, self.num_outputs)

        self.init_parameters(zero_init_residual)

    def init_parameters(self, zero_init_residual):
        # Conv2d 和 Norm 参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # 初始化每个残差分支中的最后一个 BN 层，以便残差分支从零开始，并且每个残差块的行为就像一个identity
        # 根据 https://arxiv.org/abs/1706.02677 的数据，通过这种操作, 可以改进0.2~0.3个百分点

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, components.Bottleneck):
                    # type: ignore[arg-type]
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, components.BasicBlock):
                    # type: ignore[arg-type]
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_block(
            self,
            block: Type[Union[components.BasicBlock, components.Bottleneck]],
            planes: int,
            blocks: int,
            stride: int = 1,
            dilate: bool = False,
            with_dcn = False,
            ) -> nn.Sequential:
        """构建动态结构
        Args:
            block (Type[Union[components.BasicBlock, components.Bottleneck]]): 
            planes (int): _description_
            blocks (int): block数量
            stride (int, optional): 卷积步长. Defaults to 1.
            dilate (bool, optional): 是否使用空洞卷积. Defaults to False.
        Returns:
            nn.Sequential: 
        """
        norm_layer = self.norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                components.conv2d1x1(
                    self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer, with_dcn = with_dcn
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    with_dcn = with_dcn
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: Tensor, feature_lens=None) -> Tensor:
        # 静态运算结构
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 动态可调运算结构
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 输出结构
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.last_linear(x)
        return x