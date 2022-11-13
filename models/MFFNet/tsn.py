from typing import Type, Any, Callable, Union, List, Optional
import numpy as np
import torch
from torch import Tensor, nn
from typing import Tuple, Optional
import torch.nn.functional as F
import math
import torchvision


class Identity(torch.nn.Module):
    def forward(self, input):
        return input


class SegmentConsensus(torch.autograd.Function):

    def __init__(self, consensus_type, dim=1):
        self.consensus_type = consensus_type
        self.dim = dim
        self.shape = None

    def forward(self, input_tensor):
        # [batch, num_segment, feature]
        self.shape = input_tensor.size()
        if self.consensus_type == 'avg':
            output = input_tensor.mean(dim=self.dim, keepdim=True)
        elif self.consensus_type == 'identity':
            output = input_tensor
        else:
            output = None

        return output

    def backward(self, grad_output):
        if self.consensus_type == 'avg':
            grad_in = grad_output.expand(
                self.shape) / float(self.shape[self.dim])
        elif self.consensus_type == 'identity':
            grad_in = grad_output
        else:
            grad_in = None

        return grad_in


class ConsensusModule(torch.nn.Module):

    def __init__(self, consensus_type, dim=1):
        super(ConsensusModule, self).__init__()
        self.consensus_type = consensus_type if consensus_type != 'rnn' else 'identity'
        self.dim = dim

    def forward(self, input):
        return SegmentConsensus(self.consensus_type, self.dim)(input)


class TSN(nn.Module):

    def __init__(self, num_class: int = 4, num_segments: int = 3, modality: str = "RGB",
                 base_model: str = 'resnet50', new_length: int = None,
                 consensus_type: str = 'avg', before_softmax: bool = True,
                 dropout: float = 0.5, crop_num: int = 1, partial_bn: bool = True):
        """ 视频动作识别领域的 TSN 网络实现

        paper: Temporal Segment Networks: Towards Good Practices for Deep Action Recognition

        Ref: https://github.com/yjxiong/tsn-pytorch

        Args:
            num_class (int): 最终分类数量. Defaults to 4.
            num_segments (int): 视频分段数. Defaults to 3.
            modality (int): 视频使用的输入模式(RGB,Flow,RGBDiff,扭曲光流). Defaults to 'RGB'.
            base_model (str, optional): 基础模型架构(resnet*, vgg*, inception*). Defaults to 'resnet50'.
            new_length (int, optional): _description_. Defaults to None.
            consensus_type (str, optional): 最终共识融合模式. Defaults to 'avg'.
            before_softmax (bool, optional): 最终的共识结果是否转换为概率值. Defaults to True.
            dropout (float, optional): 在最终全连接输入前 Dropout 率. Defaults to 0.5.
            partial_bn (bool, optional): 是否将除第一层外的其他 BN 层的 mean 和 variance 全部固定. Defaults to True.

        Raises:
            ValueError: _description_
        """
        super(TSN, self).__init__()
        self.modality = modality
        self.num_segments = num_segments
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.crop_num = crop_num
        self.consensus_type = consensus_type
        if not before_softmax and consensus_type != 'avg':
            raise ValueError("Softmax 之后只能使用 avg 共识函数")

        if new_length is None:
            self.new_length = 1 if modality == "RGB" else 5
        else:
            self.new_length = new_length

        self._prepare_base_model(base_model)  # 构建基础模型

        feature_dim = self._prepare_tsn(num_class)

        if self.modality == 'Flow':
            print("# Converting the ImageNet model to a flow init model")
            self.base_model = self._construct_flow_model(self.base_model)
            print("# Done. Flow model ready...")
        elif self.modality == 'RGBDiff':
            print("# Converting the ImageNet model to RGB+Diff init model")
            self.base_model = self._construct_diff_model(self.base_model)
            print("# Done. RGBDiff model ready.")

        self.consensus = ConsensusModule(consensus_type)

        if not self.before_softmax:
            self.softmax = nn.Softmax()  # 将最终的共识输入转换为概率值.

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)  # 将除第一层外的其他 BN 层的 mean 和 variance 全部固定

    def _prepare_base_model(self, base_model):

        if 'resnet' in base_model or 'vgg' in base_model:
            self.base_model = getattr(
                torchvision.models, base_model)(True)  # 加载经过预训练的网络结构
            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            if self.modality == 'Flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]
            elif self.modality == 'RGBDiff':
                self.input_mean = [0.485, 0.456, 0.406] + \
                    [0] * 3 * self.new_length
                self.input_std = self.input_std + \
                    [np.mean(self.input_std) * 2] * 3 * self.new_length
        elif base_model == 'bninception':
            import pretrainedmodels
            self.base_model = getattr(pretrainedmodels, base_model)()
            self.base_model.last_layer_name = 'last_linear'
            self.input_size = 224
            self.input_mean = [104, 117, 128]
            self.input_std = [1]

            if self.modality == 'Flow':
                self.input_mean = [128]
            elif self.modality == 'RGBDiff':
                self.input_mean = self.input_mean * (1 + self.new_length)

        elif 'inception' in base_model:
            import pretrainedmodels
            self.base_model = getattr(pretrainedmodels, base_model)()
            self.base_model.last_layer_name = 'last_linear'
            self.input_size = 299
            self.input_mean = [0.5]
            self.input_std = [0.5]
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def _prepare_tsn(self, num_class):

        feature_dim = getattr(
            self.base_model, self.base_model.last_layer_name).in_features  # 记录原原基本模型的最终全连接层输入参数
        if self.dropout == 0:
            setattr(self.base_model, self.base_model.last_layer_name,
                    nn.Linear(feature_dim, num_class))  # 修改原基本模型的最终全连接层参数(主要是修改输出参数和重新训练)
            self.new_fc = None
        else:
            setattr(self.base_model, self.base_model.last_layer_name,
                    nn.Dropout(p=self.dropout))  # 修改原基本模型的最终全连接层为 Dropout
            self.new_fc = nn.Linear(feature_dim, num_class)  # 同时添加一个新全连接层

        # 初始化基本模型的最终全连接层参数
        std = 0.001
        if self.new_fc is None:

            torch.nn.init.normal(getattr(self.base_model,
                                         self.base_model.last_layer_name).weight, 0, std)
            torch.nn.init.constant(getattr(self.base_model,
                                           self.base_model.last_layer_name).bias, 0)
        else:
            torch.nn.init.normal(self.new_fc.weight, 0, std)
            torch.nn.init.constant(self.new_fc.bias, 0)
        return feature_dim

    def _construct_flow_model(self, base_model):
        # 修改所使用基本模型架构中的第一个卷积层, 使其适应本模型的输入. 这样就可以利用原训练参数了
        # Torch models are usually defined in a hierarchical way.
        # 通过 nn.modules.children() 可以以 DFS manner 的方式返回 所有 sub module.
        modules = list(self.base_model.modules())  # 获取子模型
        first_conv_idx = list(
            filter(  # 用于过滤可迭代序列，过滤掉序列中不符合条件的元素，返回由符合条件元素组成的新可迭代序列。
                # 判断第参数 x 个 module 是否是卷积层
                lambda x: isinstance(modules[x], nn.Conv2d),
                # 要被过滤的可迭代序列
                list(range(len(modules)))
            )
        )[0]  # 获得第一个卷积层
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # 修改第一个卷积核的参数
        params = [x.clone() for x in conv_layer.parameters()]  # 获取第一个卷积层的参数
        kernel_size = params[0].size()  # 获取原卷积核参数维度 (out_c,in_c,h,w)
        new_kernel_size = kernel_size[:1] + \
            (2 * self.new_length, ) + \
            kernel_size[2:
                        ]  # 通过 new_length 计算出新的卷积核参数维度 (out_c,2*new_length,h,w)
        new_kernels = params[0].data.mean(
            dim=1, keepdim=True).expand(new_kernel_size).contiguous()  # 利用原卷积核 in_c 维度的平均值填充新通道 new_in_c 维度

        new_conv = nn.Conv2d(2 * self.new_length, conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)  # 建立一个新卷积
        new_conv.weight.data = new_kernels  # 给新卷积设定初始值
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # 如有必要增加偏置参数
        # 获取需要被替换的第一个卷积的名字
        layer_name = list(container.state_dict().keys())[
            0][:-7]

        setattr(container, layer_name, new_conv)  # 替换卷积
        return base_model

    def _construct_diff_model(self, base_model, keep_rgb=False):
        # 修改所使用基本模型架构中的第一个卷积层, 使其适应本模型的输入. 这样就可以利用原训练参数了
        # Torch models are usually defined in a hierarchical way.
        # 通过 nn.modules.children() 可以以 DFS manner 的方式返回 所有 sub module.
        modules = list(self.base_model.modules())  # 获取子模型
        first_conv_idx = list(
            filter(  # 用于过滤可迭代序列，过滤掉序列中不符合条件的元素，返回由符合条件元素组成的新可迭代序列。
                # 判断第参数 x 个 module 是否是卷积层
                lambda x: isinstance(modules[x], nn.Conv2d),
                # 要被过滤的可迭代序列
                list(range(len(modules)))
            )
        )[0]  # 获得第一个卷积层
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # 修改第一个卷积核的参数
        params = [x.clone() for x in conv_layer.parameters()]  # 获取第一个卷积层的参数
        kernel_size = params[0].size()  # 获取原卷积核参数维度 (out_c,in_c,h,w)
        if not keep_rgb:  # 仅使用 diff 模式
            new_kernel_size = kernel_size[:1] + \
                (3 * self.new_length,) + \
                kernel_size[2:
                            ]  # 通过 new_length 计算出新的卷积核参数维度 (out_c,3*new_length,h,w)
            new_kernels = params[0].data.mean(
                dim=1, keepdim=True).expand(new_kernel_size).contiguous()  # 利用原卷积核 in_c 维度的平均值填充新通道 new_in_c 维度
        else:  # 如果想要同时使用 RGB 模式
            new_kernel_size = kernel_size[:1] + \
                (3 * self.new_length,) + \
                kernel_size[2:
                            ]
            new_kernels = torch.cat((params[0].data, params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()),
                                    1)  # 利用原卷积核 in_c 维度的平均值填充新通道 in_c+new_in_c 维度
            new_kernel_size = kernel_size[:1] + \
                (3 + 3 * self.new_length,) + \
                kernel_size[2:
                            ]  # 通过 new_length 计算出新的卷积核参数维度 (out_c,3+3*new_length,h,w)

        new_conv = nn.Conv2d(new_kernel_size[1], conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)  # 建立一个新卷积
        new_conv.weight.data = new_kernels  # 给新卷积设定初始值
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # 如有必要增加偏置参数
        # 获取需要被替换的第一个卷积的名字
        layer_name = list(container.state_dict().keys())[0][:-7]

        # 替换卷积
        setattr(container, layer_name, new_conv)
        return base_model

    def train(self, mode=True):
        """
        重写默认的 train() 函数以冻结 BN 参数
        :return:
        """
        super(TSN, self).train(mode)
        count = 0
        if self._enable_pbn:  # 仅在开启 partial_bn 的时候启用
            print("# Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()

                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_optim_policies(self):
        # 自定义优化策略, 仅学习基本模型中自定义的第一个卷积和
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        bn = []

        conv_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])

            elif isinstance(m, torch.nn.BatchNorm1d):
                bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_cnt += 1
                # 因为后面的 BN 都被 frozen 了, 因此只优化第一个 BN
                if (not self._enable_pbn) or (bn_cnt == 1):
                    bn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError(
                        "New atomic module type: {}. Need to give it a learning policy".format(type(m)))
        return [
            {'params': first_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
        ]

    def _get_diff(self, input, keep_rgb=False):
        # 获取图像间的差分图像
        # input: [batch, num_segment*(new_length + 1), channel, width, height]
        input_c = 3 if self.modality in ["RGB", "RGBDiff"] else 2  # 设置输入通道数

        input_view = input.view(  # 将输入视频分段 -> [batch, num_segments, new_length + 1, channel, width, height]
            (-1, self.num_segments, self.new_length + 1, input_c,) + input.size()[2:])
        if keep_rgb:
            new_data = input_view.clone()
        else:
            # [batch, num_segments, new_length, channel, width, height]
            new_data = input_view[:, :, 1:, :, :, :].clone()

        for x in reversed(list(range(1, self.new_length + 1))):  # 倒数计算差分
            if keep_rgb:
                new_data[:, :, x, :, :, :] = input_view[:, :, x,  # 后一张图像减去前一张图像
                                                        :, :, :] - input_view[:, :, x - 1, :, :, :]
            else:
                new_data[:, :, x - 1, :, :, :] = input_view[:, :,
                                                            x, :, :, :] - input_view[:, :, x - 1, :, :, :]
        return new_data

    def forward(self, input):
        # input: [batch, num_segment*(new_length + 1), channel, width, height]

        # sample
        # # 计算视频序列分割成 num_segments 数量后的平均间隔
        # average_duration = (num_frames - new_length + 1) // num_segments
        # # 计算每一段的平均起始点
        # np.multiply(list(range(num_segments)), average_duration)
        # # 在每个起始点的基础上, 添加一个小于 average_duration 的随机数字
        #  + randint(average_duration, size=num_segments)
        # # 如果 num_frames 太小,
        # offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))

        # extract_optical_flow
        sample_len = (3 if self.modality == "RGB" else 2) * self.new_length

        if self.modality == 'RGBDiff':
            sample_len = 3 * self.new_length
            input = self._get_diff(input)
            # [batch, num_segment, new_length(+1),channel, width, height]
        # [batch * num_segment, new_length(+1)*channel, width, height]
        base_out = self.base_model(input.view(
            (-1, sample_len) + input.size()[-2:]))
        # [batch * num_segment, feature]

        if self.dropout > 0:
            base_out = self.new_fc(base_out)

        if not self.before_softmax:
            base_out = self.softmax(base_out)
        if self.reshape:
            base_out = base_out.view(
                (-1, self.num_segments) + base_out.size()[1:])
            # [batch, num_segment, num_class]

        output = self.consensus(base_out)
        # [batch, 1, num_class]
        return output.squeeze(1)
