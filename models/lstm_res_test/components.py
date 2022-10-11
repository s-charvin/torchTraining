from typing import Type, Any, Callable, Union, List, Optional
import torch
from torch import Tensor, nn
from typing import Tuple, Optional

from ..emotion_baseline_attention_lstm.components import TestLSTMCell, AttLSTMCell
from ..resnet import components as resnet_components
from ..resnext import components as resnext_components
from torch.nn import init
import torch.nn.functional as F


class DiyLSTM(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int = 1,
                 bias: bool = True,
                 bidirectional: bool = False,) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1

        self.forward_layer = nn.ModuleList(
            [
                AttLSTMCell(input_size=(input_size if layer == 0 else hidden_size * num_directions),
                            hidden_size=hidden_size, bias=bias)
                for layer in range(num_layers)
            ]
        )

        if self.bidirectional:
            self.backward_layer = nn.ModuleList(
                [
                    AttLSTMCell(input_size=(input_size if layer == 0 else hidden_size * num_directions),
                                hidden_size=hidden_size, bias=bias)
                    for layer in range(num_layers)
                ]
            )

    def check_hidden_size(self, hx: Tensor, expected_hidden_size: Tuple[int, int, int],
                          msg: str = 'Expected hidden size {}, got {}') -> None:
        if hx.size() != expected_hidden_size:
            raise RuntimeError(msg.format(
                expected_hidden_size, list(hx.size())))

    def check_forward_args(self, input: Tensor, hx: Tensor, cx: Tensor):

        # check_input

        expected_input_dim = 3
        if input.dim() != expected_input_dim:
            raise RuntimeError(
                'input must have {} dimensions, got {}'.format(
                    expected_input_dim, input.dim()))
        if self.input_size != input.size(-1):
            raise RuntimeError(
                'input.size(-1) must be equal to input_size. Expected {}, got {}'.format(
                    self.input_size, input.size(-1)))
        # check_hidden_size
        mini_batch = input.size(1)
        num_directions = 2 if self.bidirectional else 1
        expected_hidden_size = (
            self.num_layers * num_directions, mini_batch, self.hidden_size)
        self.check_hidden_size(hx, expected_hidden_size,
                               'Expected hx size {}, got {}')
        # check_cell_size
        expected_cell_size = (
            self.num_layers * num_directions, mini_batch, self.hidden_size)
        self.check_hidden_size(cx, expected_cell_size,
                               'Expected cx size {}, got {}')

    def copy_parameters(self, rnn_old):
        for param in rnn_old.named_parameters():
            name_ = param[0].split("_")
            layer = int(name_[2].replace("l", ""))
            sub_name = "_".join(name_[:2])
            if len(name_) > 3:
                self.backward_layer[layer].register_parameter(
                    sub_name, param[1])
            else:
                self.forward_layer[layer].register_parameter(
                    sub_name, param[1])

    def forward(self, input: Tensor, hidden: Optional[Tuple[Tensor, Tensor]] = None):
        # [seq_len, batch_size, hidden_size]
        # [num_layers * num_directions, batch, hidden_size]
        seq_len, batch_size, _ = input.size()

        num_directions = 2 if self.bidirectional else 1
        if hidden is None:

            hx = input.new_zeros(self.num_layers * num_directions,
                                 batch_size, self.hidden_size, requires_grad=False)
            # hx = torch.zeros(self.num_layers * num_directions,
            #                  batch_size, self.hidden_size,
            #                  dtype=input.dtype, device=input.device)
            cx = input.new_zeros(self.num_layers * num_directions,
                                 batch_size, self.hidden_size, requires_grad=False)
        else:
            hx, cx = hidden

        self.check_forward_args(input, hx, cx)

        ht = []  # [seq_len, num_layers*num_directions] , 保存
        for i in range(seq_len):
            ht.append([None] * (self.num_layers * num_directions))
        ct = []  # [seq_len, num_layers*num_directions]
        for i in range(seq_len):
            ct.append([None] * (self.num_layers * num_directions))

        if self.bidirectional:
            hy = []
            cy = []
            xs = input

            # [(f_layer, b_layer), ... ], 数量是层数
            for l, (f_layer, b_layer) in enumerate(zip(self.forward_layer, self.backward_layer)):  # l是第几层
                # 获取初始输入的第 l 层的前向和后向 (h, c) 向量所在位置
                f_l, b_l = 2 * l, 2 * l + 1  # [(0,1) (2,3), (4,5) ... ]
                # 获取初始输入的第 l 层的前向和后向 (h, c) 向量
                f_h, f_c, b_h, b_c = hx[f_l], cx[f_l], hx[b_l], cx[b_l]
                # [(正向输入, 反向输入), ... ], 数量是序列时间点长度
                for t, (f_x, b_x) in enumerate(zip(xs, reversed(xs))):
                    # forward
                    # 输入前向输入的时间 t 点得到的 (h, c)
                    # ht_, ct_: [batch, hidden_size]
                    ht_, ct_ = f_layer(f_x, (f_h, f_c))
                    # t 时间点, 前向向量所在位置更新值
                    # ht: --> [seq_len, num_layers*num_directions, batch, hidden_size]
                    ht[t][f_l] = ht_
                    # ct: --> [seq_len, num_layers*num_directions, batch, hidden_size]
                    ct[t][f_l] = ct_
                    # 在每次计算完后更新计算后的前向向量
                    # f_h, f_c: [batch, hidden_size]
                    f_h, f_c = ht[t][f_l], ct[t][f_l]

                    # backward
                    t = seq_len - 1 - t  # 反向时间
                    # ht_, ct_: [batch, hidden_size]
                    ht_, ct_ = b_layer(b_x, (b_h, b_c))
                    # t 时间点, 后向向量所在位置更新值
                    # ht: --> [seq_len, num_layers*num_directions, batch, hidden_size]
                    ht[t][b_l] = ht_
                    # ct: --> [seq_len, num_layers*num_directions, batch, hidden_size]
                    ct[t][b_l] = ct_
                    # 在每次计算完后更新计算后的后向向量
                    # b_h, b_c: [batch, hidden_size]
                    b_h, b_c = ht[t][b_l], ct[t][b_l]

                # 第l层，所有时间点，所有方向的特征
                # xs: [seq_len, batch, hidden_size * num_directions]
                xs = [
                    torch.cat((h[f_l], h[b_l]), dim=1) for h in ht
                ]

                # 第l层, 所有时间点, 所有方向, 特征向量
                # ht_temp: [seq_len, num_directions, batch, hidden_size]
                ht_temp = torch.stack(
                    [
                        # [num_layers*num_directions, batch, hidden_size]
                        torch.stack([h[f_l], h[b_l]]) for h in ht
                    ]
                )
                # ct_temp: [seq_len, num_directions, batch, hidden_size]
                ct_temp = torch.stack(
                    [
                        torch.stack([c[f_l], c[b_l]]) for c in ct
                    ]
                )

                if len(hy) == 0:  # 初始
                    hy = ht_temp[-1]
                else:
                    # [num_layers*num_directions, batch, hidden_size]
                    hy = torch.cat((hy, ht_temp[-1]), dim=0)
                if len(cy) == 0:
                    cy = ct_temp[-1]
                else:
                    cy = torch.cat((cy, ct_temp[-1]), dim=0)
            # [seq_len, batch, hidden_size * num_directions]
            y = torch.stack(xs)

        else:
            h, c = hx, cx  # [num_layers, batch, hidden_size]

            for t, x in enumerate(input):  # [0-seq_len]
                # [0, num_layers]
                for l, layer in enumerate(self.forward_layer):
                    ht_, ct_ = layer(x, (h[l], c[l]))  # [batch, hidden_size]
                    # [seq_len, num_layers]-->[seq_len, num_layers, batch, hidden_size]
                    ht[t][l] = ht_
                    ct[t][l] = ct_
                    x = ht[t][l]  # [batch, hidden_size]
                ht[t] = torch.stack(ht[t])
                ct[t] = torch.stack(ct[t])
                h, c = ht[t], ct[t]  # [num_layers, batch, hidden_size]
            # y: [seq_len, batch, hidden_size] 最后一层，所有时间点，所有方向的特征
            y = torch.stack(
                [
                    h[-1] for h in ht

                ]
            )
            #
            hy = torch.stack(list(ht[-1]))  # [num_layers, batch, hidden_size]
            cy = torch.stack(list(ct[-1]))
        return y, (hy, cy)


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
            config,) -> None:

        super().__init__()

        if config["lstm_res"]["resnet"]["block"] == "Bottleneck":
            block = resnet_components.Bottleneck
        elif config["lstm_res"]["resnet"]["block"] == "BasicBlock":
            block = resnet_components.BasicBlock

        if config["lstm_res"]["resnet"]["norm_layer"] == "BN":
            self.norm_layer = nn.BatchNorm2d
        else:
            raise ValueError("除了 BN 外,其他 Nomal 暂未定义")

        self.inplanes = config["lstm_res"]["resnet"]["inplanes"]  # 初始特征通道数
        layers = config["lstm_res"]["resnet"]["layers"]
        self.groups = config["lstm_res"]["resnet"]["groups"]
        self.base_width = config["lstm_res"]["resnet"]["width_per_group"]
        aux_num = config["lstm_res"]["resnext"]["aux_num"]
        if aux_num == "None":
            self.num_outputs = config["model"]["num_outputs"]
        else:
            self.num_outputs = aux_num

        replace_stride_with_dilation = config["lstm_res"]["resnet"]["replace_stride_with_dilation"]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.dilation = config["lstm_res"]["resnet"]["dilation"]
        zero_init_residual = config["lstm_res"]["resnet"]["zero_init_residual"]

        self.conv1 = nn.Conv2d(
            config["model"]["input_channel"], self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
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

        self.fc = nn.Linear(512 * block.expansion, self.num_outputs)

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
                if isinstance(m, resnet_components.Bottleneck):
                    # type: ignore[arg-type]
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, resnet_components.BasicBlock):
                    # type: ignore[arg-type]
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_block(
            self,
            block: Type[Union[resnet_components.BasicBlock, resnet_components.Bottleneck]],
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
                resnet_components.conv2d1x1(
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
        if len(x.size()) == 3:
            x = x.unsqueeze(1)  # x
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
        x = self.fc(x)
        return x


class ResNeXt(nn.Module):
    """
    Xie, S., Girshick, R., Dollár, P., Tu, Z., & He, K. (2016). 
    Aggregated residual transformations for deep neural networks. 
    https://arxiv.org/pdf/1611.05431.pdf
    """

    def __init__(self, config):
        """ 
        Args:
            cardinality: 卷积分组数.
            depth: 网络层数(深度), 含预处理层,分类层,Block等所有层数.
            nlabels: 输出分类数
            base_width: 每组的通道数.
            widen_factor: 通道加宽因子, 用来调整通道维度
        """
        super(ResNeXt, self).__init__()

        self.input_channel = config["model"]["input_channel"]

        self.cardinality = config["lstm_res"]["resnext"]["cardinality"]
        self.base_width = config["lstm_res"]["resnext"]["base_width"]
        self.widen_factor = config["lstm_res"]["resnext"]["widen_factor"]

        aux_num = config["lstm_res"]["resnext"]["aux_num"]
        if aux_num == "None":
            self.num_outputs = config["model"]["num_outputs"]
        else:
            self.num_outputs = aux_num

        self.depth = config["lstm_res"]["resnext"]["depth"]
        self.block_depth = (self.depth - 2) // 9

        output_size = config["lstm_res"]["resnext"]["output_size"]
        self.stages = [output_size[0], output_size[1] * self.widen_factor, output_size[2] *
                       self.widen_factor, output_size[3] * self.widen_factor]

        self.conv_1_3x3 = nn.Conv2d(
            self.input_channel, 64, 3, 1, 1, bias=False)
        self.bn_1 = nn.BatchNorm2d(64)

        self.stage_1 = self.block('stage_1', self.stages[0], self.stages[1], 1)
        self.stage_2 = self.block('stage_2', self.stages[1], self.stages[2], 2)
        self.stage_3 = self.block('stage_3', self.stages[2], self.stages[3], 2)

        self.classifier = nn.Sequential(
            # x = F.avg_pool2d(x, (8, 8), 1),
            # x = x.view(x.shape[0], -1, self.stages[3]),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.stages[3], self.num_outputs),
        )

        # 初始化参数
        init.kaiming_normal_(self.classifier.weight)
        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal_(
                        self.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    self.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0

    def block(self, name, in_channels, out_channels, pool_stride=2):
        """ 堆叠 n 个 bottleneck modules.
        Args:
            name: block的名称.
            in_channels: 输入通道数
            out_channels: 输出通道数
            pool_stride: factor to reduce the spatial dimensionality in the first bottleneck of the block.
        Returns: Sequential
        """
        block = nn.Sequential()
        for bottleneck in range(self.block_depth):
            name_ = '%s_bottleneck_%d' % (name, bottleneck)
            if bottleneck == 0:
                block.add_module(name_, resnext_components.ResNeXtBottleneck_C(in_channels, out_channels, pool_stride, self.cardinality,
                                                                               self.base_width, self.widen_factor))
            else:
                block.add_module(name_,
                                 resnext_components.ResNeXtBottleneck_C(out_channels, out_channels, 1, self.cardinality, self.base_width,
                                                                        self.widen_factor))
        return block

    def forward(self, x):
        if len(x.size()) == 3:
            x = x.unsqueeze(1)  # x
        x = self.conv_1_3x3.forward(x)
        x = F.relu(self.bn_1.forward(x), inplace=True)
        x = self.stage_1.forward(x)
        x = self.stage_2.forward(x)
        x = self.stage_3.forward(x)

        return self.classifier(x)
