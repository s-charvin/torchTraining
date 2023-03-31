from . import components
from typing import Type, Any, Callable, Union, List, Optional
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from custom_models import *


class PretrainedBaseModel(nn.Module):
    def __init__(self, in_channels=3, output_size: int = 4, base_model: str = 'resnet50', before_dropout=0.5, before_softmax=False):
        super(PretrainedBaseModel, self).__init__()
        self.in_channels = in_channels
        assert (before_dropout >= 0 and output_size >=
                0), 'before_dropout 和 num_class 必须大于等于 0.'
        assert (not (output_size == 0 and before_softmax)
                ), '当不进行分类时, 不推荐使用概率输出'
        self.before_dropout = before_dropout
        self.before_softmax = before_softmax
        self._prepare_base_model(base_model)  # 构建基础模型
        self._change_model_out(output_size)
        self._change_model_in()

    def _prepare_base_model(self, base_model):
        # 加载已经训练好的模型
        import custom_models.pretrainedmodels as pretrainedmodels
        if base_model in pretrainedmodels.model_names:
            self.base_model = getattr(pretrainedmodels, base_model)()
            self.base_model.last_layer_name = 'last_linear'
            return True
        else:
            raise ValueError(f"不支持的基本模型: {base_model}")

    def _change_model_out(self, num_class):
        # 调整模型输出层
        linear_layer = getattr(
            self.base_model, self.base_model.last_layer_name)
        if not isinstance(linear_layer, nn.Linear):
            raise ModuleNotFoundError(
                f"需要基础模型的最后输出层应为 Linear 层, 而不是 {linear_layer} 层.")

        feature_dim = linear_layer.in_features  # 记录原基本模型的最终全连接层输入维度
        std = 0.001
        if num_class == 0 or num_class == feature_dim:
            self.new_fc = None
            setattr(self.base_model, self.base_model.last_layer_name,
                    EmptyModel)  # 修改最终的全连接层为空, 直接返回最终隐藏层的向量

        elif self.before_dropout == 0:  # 替换最终的全连接层为指定的输出维度, 重新训练最后一层
            setattr(self.base_model, self.base_model.last_layer_name,
                    nn.Linear(feature_dim, num_class))
            self.new_fc = None
            torch.nn.init.normal(getattr(self.base_model,
                                         self.base_model.last_layer_name).weight, 0, std)
            torch.nn.init.constant(getattr(self.base_model,
                                           self.base_model.last_layer_name).bias, 0)
        else:  # 替换最终的全连接层为指定的输出维度, 同时在前面添加一个 Dropout
            setattr(self.base_model, self.base_model.last_layer_name,
                    nn.Dropout(p=self.before_dropout))  # 修改原基本模型的最终全连接层为 Dropout
            self.new_fc = nn.Linear(feature_dim, num_class)
            torch.nn.init.normal_(self.new_fc.weight, 0, std)
            torch.nn.init.constant_(self.new_fc.bias, 0)
        # 初始化基本模型的最终全连接层参数

    def _change_model_in(self):
        # 调整模型输入层
        modules = list(self.base_model.modules())  # 获取子模型列表
        first_conv_idx = list(
            filter(  # 用于过滤可迭代序列，过滤掉序列中不符合条件的元素，返回由符合条件元素组成的新可迭代序列。
                # 判断第参数 x 个 module 是否是卷积层
                lambda x: isinstance(modules[x], nn.Conv2d),
                # 要被过滤的可迭代序列
                list(range(len(modules)))
            )
        )[0]  # 获得第一个卷积层位置
        conv_layer = modules[first_conv_idx]  # 获得第一个卷积层
        # 获得第一个卷积层的包裹层(这里假设了输入只有最初的单通道卷积层处理)
        container = modules[first_conv_idx - 1]

        # 获取第一个卷积层的参数
        params = [x.clone() for x in conv_layer.parameters()]
        # 通过 new_in_c 计算出新的卷积核参数维度 (out_c,new_in_c,h,w)
        kernel_size = params[0].size()  # 获取原卷积核矩阵参数维度 (out_c,in_c,h,w)
        new_kernel_size = kernel_size[:1] + \
            (self.in_channels, ) + kernel_size[2:]

        # 构建新卷积核
        new_kernels = params[0].data.mean(
            dim=1, keepdim=True).expand(new_kernel_size).contiguous()  # 利用原卷积矩阵参数 in_c 维度的平均值填充新通道 new_in_c 维度

        new_conv = nn.Conv2d(self.in_channels, conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)  # 建立一个新卷积
        new_conv.weight.data = new_kernels  # 给新卷积设定参数初始值
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # 如有必要增加偏置参数
        # 获取需要被替换的第一个卷积的名字
        layer_name = list(container.state_dict().keys())[0][:-7]
        setattr(container, layer_name, new_conv)  # 替换卷积

    def forward(self, x):
        # size: [B,C,H,W]
        base_out = self.base_model(x)  # [B,C,H,W]->[B,F]
        if self.before_dropout > 0:
            base_out = self.new_fc(base_out)
        if self.before_softmax:
            base_out = F.softmax(base_out, dim=1)
        return base_out


class CooperationSVCNet(nn.Module):
    def __init__(self, num_classes, model_names, input_size, seq_len, last_hidden_dim, seq_step) -> None:
        super().__init__()
        for i, name in enumerate(model_names):

            # setattr(self, f"coopnet{i}", AudioSVCNet_Step(
            #     num_classes=num_classes[i], model_name=name, seq_len=seq_len[i], seq_step=seq_step[i], last_hidden_dim=last_hidden_dim[i], input_size=input_size[i]))

            setattr(self, f"coopnet{i}", AudioNet(
                num_classes=num_classes[i], model_name=name,  last_hidden_dim=last_hidden_dim[i], input_size=input_size[i], pretrained=False))

    def forward(self, x: Tensor, **args) -> Tensor:
        raise "CooperationNet 只是一个对多个模型的包装, 而非实际要训练的网络. 请使用内部的 coopnet{id} 进行训练"
        return x
