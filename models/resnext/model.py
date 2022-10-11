import torch
from . import components
from torch import nn
from torch.nn import init
import torch.nn.functional as F


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

        self.cardinality = config["resnext"]["cardinality"]
        self.base_width = config["resnext"]["base_width"]
        self.widen_factor = config["resnext"]["widen_factor"]

        aux_num = config["resnext"]["aux_num"]
        if aux_num == "None":
            self.num_outputs = config["model"]["num_outputs"]
        else:
            self.num_outputs = aux_num

        self.depth = config["resnext"]["depth"]
        self.block_depth = (self.depth - 2) // 9

        output_size = config["resnext"]["output_size"]
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
                block.add_module(name_, components.ResNeXtBottleneck_C(in_channels, out_channels, pool_stride, self.cardinality,
                                                                       self.base_width, self.widen_factor))
            else:
                block.add_module(name_,
                                 components.ResNeXtBottleneck_C(out_channels, out_channels, 1, self.cardinality, self.base_width,
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
