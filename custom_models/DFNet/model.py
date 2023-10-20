from . import components
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from typing import Type, Any, Callable, Union, List, Optional

class LightResMultiSerNet2(nn.Module):
    """
    paper: Light-SERNet: A lightweight fully convolutional neural network for speech emotion recognition
    """

    def __init__(self) -> None:
        super().__init__()
        self.path1 = nn.Sequential(
            components.Conv2dSame(in_channels=1, out_channels=32, kernel_size=(
                11, 1), stride=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2)
        )
        self.path2 = nn.Sequential(
            components.Conv2dSame(in_channels=1, out_channels=32, kernel_size=(
                1, 9), stride=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2)
        )
        self.path3 = nn.Sequential(
            components.Conv2dSame(in_channels=1, out_channels=32, kernel_size=(
                3, 3), stride=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2)
        )

        self.conv_extractor = components._get_ResMultiConv2d_extractor(
            in_channels=32*3,
            shapes=[  # [channel:int, kernel:list|int, stride:list|int, norm:str|None, pool:str|None, pool_kernel:list|int]
                [32, 1, "bn", "ap", [2, 2]],
                [64, 1, "bn", "ap", [2, 2]],
                [256, 1, "bn", "ap", [2, 1]],
                [512, 1, "bn", "ap", [2, 1]],
            ],
            bias=False
        )
        self.gap = components.Conv2dLayerBlock(in_channels=512, out_channels=320, kernel_size=[
            1, 1], stride=1, bias=False, layer_norm=nn.BatchNorm2d(320), layer_poolling=nn.AdaptiveAvgPool2d(1), activate_func=nn.functional.relu)
        self.dropout = nn.Dropout(0.3)
        self.last_linear = nn.Linear(in_features=320, out_features=4)

    def forward(self, x: Tensor, feature_lens=None) -> Tensor:
        x = x.permute(0, 3, 1, 2)  # [B, C, Seq, F]
        out1 = self.path1(x)
        out2 = self.path2(x)
        out3 = self.path3(x)

        x = torch.cat([out1, out2, out3], dim=1).contiguous()

        x = self.conv_extractor(x)
        x = self.gap(x)
        x = self.dropout(x.squeeze(2).squeeze(2))
        x = self.last_linear(x)
        return x, None

class LightTestSerNet(nn.Module):
    """
    paper: Light-SERNet: A lightweight fully convolutional neural network for speech emotion recognition
    """

    def __init__(self) -> None:
        super().__init__()
        self.path11 = nn.Sequential(
            components.Conv2dSame(in_channels=1, out_channels=16*1, kernel_size=(
                3, 1), stride=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2)
        )
        self.path21 = nn.Sequential(
            components.Conv2dSame(in_channels=1, out_channels=16*1, kernel_size=(
                1, 3), stride=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2)
        )
        self.path31 = nn.Sequential(
            components.Conv2dSame(in_channels=1, out_channels=16*1, kernel_size=(
                3, 3), stride=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2)
        )

        self.path12 = nn.Sequential(
            components.Conv2dSame(in_channels=16*3, out_channels=16*3, kernel_size=(
                7, 1), stride=(1, 1)),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2)
        )
        self.path22 = nn.Sequential(
            components.Conv2dSame(in_channels=16*3, out_channels=16*3, kernel_size=(
                1, 7), stride=(1, 1)),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2)
        )
        self.path32 = nn.Sequential(
            components.Conv2dSame(in_channels=16*3, out_channels=16*3, kernel_size=(
                7, 7), stride=(1, 1)),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2)
        )

        self.path13 = nn.Sequential(
            components.Conv2dSame(in_channels=16*9, out_channels=16*3, kernel_size=(
                11, 1), stride=(1, 1)),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2)
        )
        self.path23 = nn.Sequential(
            components.Conv2dSame(in_channels=16*9, out_channels=16*3, kernel_size=(
                1, 9), stride=(1, 1)),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2)
        )
        self.path33 = nn.Sequential(
            components.Conv2dSame(in_channels=16*9, out_channels=16*3, kernel_size=(
                11, 9), stride=(1, 1)),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2)
        )

        self.conv_extractor = components._get_ModulatedDeformConv2d_extractor(
            in_channels=48*3,
            shapes=[  # [channel:int, kernel:list|int, stride:list|int, norm:str|None, pool:str|None, pool_kernel:list|int]
                [64, [3, 3], 1, "bn", "ap", [2, 2]],
                [96, [3, 3], 1, "bn", "ap", [2, 2]],
                [128, [3, 3], 1, "bn", "ap", [2, 1]],
                [160, [3, 3], 1, "bn", "ap", [2, 1]],
                [320, [1, 1], 1, "bn", "gap", 1]
            ],
            bias=False
        )
        self.dropout = nn.Dropout(0.3)
        self.last_linear = nn.Linear(in_features=320, out_features=4)

    def forward(self, x: Tensor, feature_lens=None) -> Tensor:
        x = x.permute(0, 3, 1, 2)  # [B, C, Seq, F]
        out1 = self.path11(x)
        out2 = self.path21(x)
        out3 = self.path31(x)
        x = torch.cat([out1, out2, out3], dim=1).contiguous()

        out1 = self.path12(x)
        out2 = self.path22(x)
        out3 = self.path32(x)
        x = torch.cat([out1, out2, out3], dim=1).contiguous()

        out1 = self.path13(x)
        out2 = self.path23(x)
        out3 = self.path33(x)
        x = torch.cat([out1, out2, out3], dim=1).contiguous()

        x = self.conv_extractor(x)
        x = self.dropout(x.squeeze(2).squeeze(2))
        x = self.last_linear(x)

        return x, None


# 基线模型
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
            norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
            with_dcn = False,
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


class DCRNet(nn.Module):
    """ DCRNet 基本模型, 可通过给定参数(block, layers等)定制模型. 例如:
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
            in_channels=1,
            num_classes=4,
            layers=[3, 4, 6, 3],
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
            with_dcn = True
    ) -> None:

        super().__init__()

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


        self.layer1 = self._make_block(components.Bottleneck, 96, layers[0])

        self.layer2 = self._make_block(
            components.Bottleneck, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0], with_dcn = with_dcn)
        self.layer3 = self._make_block(
            components.Bottleneck, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1], with_dcn = with_dcn)
        self.layer4 = self._make_block(
            components.Bottleneck, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2], with_dcn = with_dcn)

        self.adapool = nn.AdaptiveAvgPool2d((1, 1))

        self.last_linear = nn.Linear(512 * components.Bottleneck.expansion, self.num_outputs)

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
            with_dcn = False
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


    def forward(self, x: Tensor, feature_lens=None ) -> Tensor:

        x = x.permute(0, 3, 1, 2)  # [B, C, Seq, F]
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
        x = self.adapool(x)
        x = torch.flatten(x, 1)
        x = self.last_linear(x)
        return x, None
