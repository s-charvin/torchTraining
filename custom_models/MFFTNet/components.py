from typing import Tuple, Type, Any, Callable, Union, List, Optional
import torch
from torch import Tensor, nn
import torchvision
import torch.nn.functional as F
import math
import numpy as np


class EmptyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x



class PretrainedBaseModel():
    def __init__(self, in_channels=3, out_features: int = 4, model_name: str = 'resnet50', input_size=[224, 224], num_frames=1, before_dropout=0.5, before_softmax=False):
        self.in_channels = in_channels
        self.before_dropout = before_dropout
        self.before_softmax = before_softmax
        self.input_size = input_size
        self.num_frames = num_frames

        assert (before_dropout >= 0 and out_features >=
                0), 'before_dropout 和 num_class 必须大于等于 0.'
        assert (not (out_features == 0 and before_softmax)
                ), '当不进行分类时, 不推荐使用概率输出'
        self._prepare_base_model(model_name,)  # 构建基础模型
        self._change_model_out(out_features)
        self._change_model_in()

    def _prepare_base_model(self, model_name):
        # 加载已经训练好的模型
        import custom_models.pretrainedmodels as pretrainedmodels
        if model_name in pretrainedmodels.model_names:
            self.base_model = getattr(pretrainedmodels, model_name)()
            self.base_model.last_layer_name = 'last_linear'

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
                lambda x: isinstance(modules[x], nn.Conv2d) or isinstance(
                    modules[x], nn.Conv3d),
                # 要被过滤的可迭代序列
                list(range(len(modules)))
            )
        )[0]  # 获得第一个卷积层位置
        conv_layer = modules[first_conv_idx]  # 获得第一个卷积层
        nn_conv = nn.Conv2d if conv_layer._get_name() == "Conv2d" else nn.Conv3d
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

        new_conv = nn_conv(self.in_channels, conv_layer.out_channels,
                           conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                           bias=True if len(params) == 2 else False)  # 建立一个新卷积
        new_conv.weight.data = new_kernels  # 给新卷积设定参数初始值
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # 如有必要增加偏置参数
        # 获取需要被替换的第一个卷积的名字
        layer_name = list(container.state_dict().keys())[0][:-7]
        setattr(container, layer_name, new_conv)  # 替换卷积

    def __call__(self):
        return self.base_model



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
            )
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


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

        in_channels = out_channels
    return Conv2dFeatureExtractor(nn.ModuleList(blocks))


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


class ModulatedDeformConv2dPack(torchvision.ops.DeformConv2d):
    def __init__(self, *args, **kwargs):
        super(ModulatedDeformConv2dPack, self).__init__(*args, **kwargs)
        self.conv_offset2d = nn.Conv2d(
            self.in_channels,
            self.deformable_groups * 3 *
            self.kernel_size[0] * self.kernel_size[1],
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


def _get_ResMultiModulatedDeformConv2d_extractor(
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
            ModulatedDeformConv2dResMultiBlock(
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


class ModulatedDeformConv2dResMultiBlock(nn.Module):
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
                 activate_func=nn.functional.relu,):
        super(ModulatedDeformConv2dResMultiBlock, self).__init__()
        assert (out_channels % 2 == 0)

        self.layer_norm = layer_norm
        self.activate_func = activate_func
        self.leyer_poolling = layer_poolling

        self.conv3 = ModulatedDeformConv2dPack(
            in_channels=in_channels,
            out_channels=int(out_channels/2),
            kernel_size=[3, 3],
            stride=stride,  # 步长
            bias=bias,  # 是否使用偏置数
        )
        self.conv5 = ModulatedDeformConv2dPack(
            in_channels=in_channels,
            out_channels=int(out_channels/2),
            kernel_size=[5, 5],
            stride=stride,  # 步长
            bias=bias,  # 是否使用偏置数
        )
        self.subconv = None
        if in_channels != int(out_channels/2):
            self.subconv = nn.Conv2d(
                kernel_size=[1, 1], in_channels=in_channels, out_channels=int(out_channels/2))

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


class AreaAttention(nn.Module):
    """
    Area Attention [1]. This module allows to attend to areas of the memory, where each area
    contains a group of items that are either spatially or temporally adjacent.
    [1] Li, Yang, et al. "Area attention." International Conference on Machine Learning. PMLR 2019
    """

    def __init__(
            self, key_query_size: int, area_key_mode: str = 'mean', area_value_mode: str = 'sum',
            max_area_height: int = 1, max_area_width: int = 1, memory_height: int = 1,
            memory_width: int = 1, dropout_rate: float = 0.0, top_k_areas: int = 0
    ):
        """
        Initializes the Area Attention module.
        :param key_query_size: size for keys and queries
        :param area_key_mode: mode for computing area keys, one of {"mean", "max", "sample", "concat", "max_concat", "sum", "sample_concat", "sample_sum"}
        :param area_value_mode: mode for computing area values, one of {"mean", "sum"}
        :param max_area_height: max height allowed for an area
        :param max_area_width: max width allowed for an area
        :param memory_height: memory height for arranging features into a grid
        :param memory_width: memory width for arranging features into a grid
        :param dropout_rate: dropout rate
        :param top_k_areas: top key areas used for attention
        """
        super(AreaAttention, self).__init__()
        assert area_key_mode in ['mean', 'max', 'sample', 'concat',
                                 'max_concat', 'sum', 'sample_concat', 'sample_sum']
        assert area_value_mode in ['mean', 'max', 'sum']
        self.area_key_mode = area_key_mode
        self.area_value_mode = area_value_mode
        self.max_area_height = max_area_height
        self.max_area_width = max_area_width
        self.memory_height = memory_height
        self.memory_width = memory_width
        self.dropout = nn.Dropout(p=dropout_rate)
        self.top_k_areas = top_k_areas
        self.area_temperature = np.power(key_query_size, 0.5)
        if self.area_key_mode in ['concat', 'max_concat', 'sum', 'sample_concat', 'sample_sum']:
            self.area_height_embedding = nn.Embedding(
                max_area_height, key_query_size // 2)
            self.area_width_embedding = nn.Embedding(
                max_area_width, key_query_size // 2)
            if area_key_mode == 'concat':
                area_key_feature_size = 3 * key_query_size
            elif area_key_mode == 'max_concat':
                area_key_feature_size = 2 * key_query_size
            elif area_key_mode == 'sum':
                area_key_feature_size = key_query_size
            elif area_key_mode == 'sample_concat':
                area_key_feature_size = 2 * key_query_size
            else:  # 'sample_sum'
                area_key_feature_size = key_query_size
            self.area_key_embedding = nn.Sequential(
                nn.Linear(area_key_feature_size, key_query_size),
                nn.ReLU(inplace=True),
                nn.Linear(key_query_size, key_query_size)
            )

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Area Attention module.
        :param q: queries Tensor with shape (batch_size, num_queries, key_query_size)
        :param k: keys Tensor with shape (batch_size, num_keys_values, key_query_size)
        :param v: values Tensor with shape (batch_size, num_keys_values, value_size)
        :returns a Tensor with shape (batch_size, num_queries, value_size)
        """
        k_area = self._compute_area_key(k)
        if self.area_value_mode == 'mean':
            v_area, _, _, _, _ = self._compute_area_features(v)
        elif self.area_value_mode == 'max':
            v_area, _, _ = self._basic_pool(v, fn=torch.max)
        elif self.area_value_mode == 'sum':
            _, _, v_area, _, _ = self._compute_area_features(v)
        else:
            raise ValueError(
                f"Unsupported area value mode={self.area_value_mode}")
        logits = torch.matmul(q, k_area.transpose(1, 2))
        # logits = logits / self.area_temperature
        weights = logits.softmax(dim=-1)
        if self.top_k_areas > 0:
            top_k = min(weights.size(-1), self.top_k_areas)
            top_weights, _ = weights.topk(top_k)
            min_values, _ = torch.min(top_weights, dim=-1, keepdim=True)
            weights = torch.where(weights >= min_values,
                                  weights, torch.zeros_like(weights))
            weights /= weights.sum(dim=-1, keepdim=True)
        weights = self.dropout(weights)
        return torch.matmul(weights, v_area)

    def _compute_area_key(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute the key for each area.
        :param features: a Tensor with shape (batch_size, num_features (height * width), feature_size)
        :return: a Tensor with shape (batch_size, num_areas, feature_size)
        """
        area_mean, area_std, _, area_heights, area_widths = self._compute_area_features(
            features)
        if self.area_key_mode == 'mean':
            return area_mean
        elif self.area_key_mode == 'max':
            area_max, _, _ = self._basic_pool(features)
            return area_max
        elif self.area_key_mode == 'sample':
            if self.training:
                area_mean += (area_std * torch.randn(area_std.size()))
            return area_mean

        height_embed = self.area_height_embedding(
            (area_heights[:, :, 0] - 1).long())
        width_embed = self.area_width_embedding(
            (area_widths[:, :, 0] - 1).long())
        size_embed = torch.cat([height_embed, width_embed], dim=-1)
        if self.area_key_mode == 'concat':
            area_key_features = torch.cat(
                [area_mean, area_std, size_embed], dim=-1)
        elif self.area_key_mode == 'max_concat':
            area_max, _, _ = self._basic_pool(features)
            area_key_features = torch.cat([area_max, size_embed], dim=-1)
        elif self.area_key_mode == 'sum':
            area_key_features = size_embed + area_mean + area_std
        elif self.area_key_mode == 'sample_concat':
            if self.training:
                area_mean += (area_std * torch.randn(area_std.size()))
            area_key_features = torch.cat([area_mean, size_embed], dim=-1)
        elif self.area_key_mode == 'sample_sum':
            if self.training:
                area_mean += (area_std * torch.randn(area_std.size()))
            area_key_features = area_mean + size_embed
        else:
            raise ValueError(f"Unsupported area key mode={self.area_key_mode}")
        area_key = self.area_key_embedding(area_key_features)
        return area_key

    def _compute_area_features(
            self, features: torch.Tensor, epsilon: float = 1e-6
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute features for each area.
        :param features: a Tensor with shape (batch_size, num_features (height * width), feature_size)
        :param epsilon: epsilon added to the variance for computing standard deviation
        :return: a tuple with 5 elements:
          - area_mean: a Tensor with shape (batch_size, num_areas, feature_size)
          - area_std: a Tensor with shape (batch_size, num_areas, feature_size)
          - area_sum: a Tensor with shape (batch_size, num_areas, feature_size)
          - area_heights: a Tensor with shape (batch_size, num_areas, 1)
          - area_widths: a Tensor with shape (batch_size, num_areas, 1)
        """
        area_sum, area_heights, area_widths = self._compute_sum_image(features)
        area_squared_sum, _, _ = self._compute_sum_image(features.square())
        area_size = (area_heights * area_widths).float()
        area_mean = area_sum / area_size
        s2_n = area_squared_sum / area_size
        area_variance = s2_n - area_mean.square()
        area_std = (area_variance.abs() + epsilon).sqrt()
        return area_mean, area_std, area_sum, area_heights, area_widths

    def _compute_sum_image(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute area sums for features.
        :param features: a Tensor with shape (batch_size, num_features (height * width), feature_size)
        :return: a tuple with 3 elements:
          - sum_image: a Tensor with shape (batch_size, num_areas, feature_size)
          - area_heights: a Tensor with shape (batch_size, num_areas, 1)
          - area_widths: a Tensor with shape (batch_size, num_areas, 1)
        """
        batch_size, num_features, feature_size = features.size()
        features_2d = features.reshape(
            batch_size, self.memory_height, num_features//self.memory_height, feature_size)
        horizontal_integral = features_2d.cumsum(dim=-2)
        vertical_integral = horizontal_integral.cumsum(dim=-3)
        padded_image = F.pad(vertical_integral, pad=[0, 0, 1, 0, 1, 0])
        heights = []
        widths = []
        dst_images = []
        src_images_diag = []
        src_images_h = []
        src_images_v = []
        size = torch.ones_like(padded_image[:, :, :, 0], dtype=torch.int32)
        for area_height in range(self.max_area_height):
            for area_width in range(self.max_area_width):
                dst_images.append(
                    padded_image[:, area_height + 1:, area_width + 1:, :].reshape(batch_size, -1, feature_size))
                src_images_diag.append(
                    padded_image[:, :-area_height - 1, :-area_width - 1, :].reshape(batch_size, -1, feature_size))
                src_images_h.append(
                    padded_image[:, area_height + 1:, :-area_width - 1, :].reshape(batch_size, -1, feature_size))
                src_images_v.append(
                    padded_image[:, :-area_height - 1, area_width + 1:, :].reshape(batch_size, -1, feature_size))
                heights.append((size[:, area_height + 1:, area_width + 1:]
                               * (area_height + 1)).reshape(batch_size, -1))
                widths.append((size[:, area_height + 1:, area_width + 1:]
                              * (area_width + 1)).reshape(batch_size, -1))
        sum_image = torch.sub(
            torch.cat(dst_images, dim=1) + torch.cat(src_images_diag, dim=1),
            torch.cat(src_images_v, dim=1) + torch.cat(src_images_h, dim=1)
        )
        area_heights = torch.cat(heights, dim=1).unsqueeze(dim=2)
        area_widths = torch.cat(widths, dim=1).unsqueeze(dim=2)
        return sum_image, area_heights, area_widths

    def _basic_pool(self, features: torch.Tensor, fn=torch.max) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Pool for each area based on a given pooling function (fn).
        :param features: a Tensor with shape (batch_size, num_features (height * width), feature_size).
        :param fn: Torch pooling function
        :return: a tuple with 3 elements:
          - pool_results: a Tensor with shape (batch_size, num_areas, feature_size)
          - area_heights: a Tensor with shape (batch_size, num_areas, 1)
          - area_widths: a Tensor with shape (batch_size, num_areas, 1)
        """
        batch_size, num_features, feature_size = features.size()
        features_2d = features.reshape(
            batch_size, self.memory_height, self.memory_width, feature_size)
        heights = []
        widths = []
        pooled_areas = []
        size = torch.ones_like(features_2d[:, :, :, 0], dtype=torch.int32)
        for area_height in range(self.max_area_height):
            for area_width in range(self.max_area_width):
                pooled_areas.append(self._pool_one_shape(
                    features_2d, area_width=area_width + 1, area_height=area_height + 1, fn=fn))
                heights.append((size[:, area_height:, area_width:]
                               * (area_height + 1)).reshape([batch_size, -1]))
                widths.append((size[:, area_height:, area_width:]
                              * (area_width + 1)).reshape([batch_size, -1]))
        pooled_areas = torch.cat(pooled_areas, dim=1)
        area_heights = torch.cat(heights, dim=1).unsqueeze(dim=2)
        area_widths = torch.cat(widths, dim=1).unsqueeze(dim=2)
        return pooled_areas, area_heights, area_widths

    def _pool_one_shape(self, features_2d: torch.Tensor, area_width: int, area_height: int, fn=torch.max) -> torch.Tensor:
        """
        Pool for an area in features_2d.
        :param features_2d: a Tensor with shape (batch_size, height, width, feature_size)
        :param area_width: max width allowed for an area
        :param area_height: max height allowed for an area
        :param fn: PyTorch pooling function
        :return: a Tensor with shape (batch_size, num_areas, feature_size)
        """
        batch_size, height, width, feature_size = features_2d.size()
        flat_areas = []
        for y_shift in range(area_height):
            image_height = max(height - area_height + 1 + y_shift, 0)
            for x_shift in range(area_width):
                image_width = max(width - area_width + 1 + x_shift, 0)
                area = features_2d[:, y_shift:image_height,
                                   x_shift:image_width, :]
                flatten_area = area.reshape(batch_size, -1, feature_size, 1)
                flat_areas.append(flatten_area)
        flat_area = torch.cat(flat_areas, dim=3)
        pooled_area = fn(flat_area, dim=3).values
        return


class ResMultiConv(nn.Module):
    '''
    Multi-scale block with short-cut connections
    '''

    def __init__(self, channels=16, **kwargs):
        super(ResMultiConv, self).__init__()
        self.conv3 = nn.Conv2d(kernel_size=(
            3, 3), in_channels=channels, out_channels=channels, padding=1)
        self.conv5 = nn.Conv2d(kernel_size=(
            5, 5), in_channels=channels, out_channels=channels, padding=2)
        self.bn = nn.BatchNorm2d(channels*2)

    def forward(self, x):
        x3 = self.conv3(x) + x
        x5 = self.conv5(x) + x
        x = torch.cat((x3, x5), 1)
        x = self.bn(x)
        x = F.relu(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        a = self.pe[:, :x.size(1)]
        x = x + self.pe[:, :x.size(1)]
        return x


class ESIAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, max_len=5000):
        super().__init__()
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.multihead_attention = nn.MultiheadAttention(d_model, n_heads)
        # self.norm1 = nn.LayerNorm(d_model)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x, s):
        # Multi-head self-attention
        x = self.positional_encoding(x)
        s = self.positional_encoding(s)
        attn_output, _ = self.multihead_attention(s, x, x)
        x = x + attn_output
        # x = self.norm1(x)
        return x
