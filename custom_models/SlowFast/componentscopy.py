from codecs import ascii_encode
from typing import Type, Any, Callable, Union, List, Optional, Tuple, Iterable
import torch
from torch import Tensor, nn
from torch.nn import Parameter
import torch.nn.functional as F
import math


class Conv2d_same(torch.nn.Conv2d):
    """ 等同于"same"模式的卷积, 因为引入了合适的展平操作,
        其得到的输出维度和输入维度相等.
    """

    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        # 计算输入维度应该填充的维度大小
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.size()[-2:]  # 获取输入维度

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
    """ 单个2D卷积块, 可自定义norm ,activate, pool层
        并且支持"same"模式, 这在老版本torch是不支持的
        Args:
            in_channels (int):
            out_channels (int):
            kernel_size (_type_):
            stride (int):
            bias (bool):
            layer_norm (Optional[nn.Module]):
            layer_poolling (Optional[nn.Module]):
            activate_func (Optional[nn.Module]): . Defaults to Optional[nn.Module].
            padding (int, optional): . Defaults to 0.
        example:
            net =  Conv2dLayerBlock(
                in_channel = 64,
                out_channel = 128,
                kernel_size = [3,3],
                stride = [1,1],
                bias= False,
                padding= 0,
                padding_mode = "zeros",
                dilation = 1,
                groups = 1,
                layer_norm = None,
                layer_poolling = None,
                activate_func= None)
    """

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size,
        stride: int,
        bias: bool = False,
        padding=0,  # 最新版可用"same"模式, 老版本没有, 替代品为自定义的Conv2d_same
        padding_mode="zeros",
        dilation=1,
        groups=1,
        layer_norm: Optional[nn.Module] = None,
        layer_poolling: Optional[nn.Module] = None,
        activate_func: Optional[nn.Module] = None,

    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        if padding == "same":
            self.conv = Conv2d_same(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=kernel_size,
                stride=stride,  # 步长
                bias=bias,  # 是否使用偏置数
                padding_mode=padding_mode,
                dilation=dilation,
                groups=groups,
            )
        else:
            self.conv = nn.Conv2d(
                in_channels=in_channel,
                out_channels=out_channel,  # 输出通道数,即使用几个卷积核
                kernel_size=kernel_size,
                stride=stride,  # 步长
                bias=bias,  # 是否使用偏置数
                padding=padding,
                padding_mode=padding_mode,
                dilation=dilation,
                groups=groups,
            )
        self.layer_norm = layer_norm
        self.activate_func = activate_func
        self.leyer_poolling = layer_poolling

    def forward(self, x: Tensor,) -> Tensor:
        """
        Args:
            x (Tensor): Shape ``[batch, in_channels, H, W]``.
        Returns:
            Tensor: Shape ``[batch, out_channels, H, W]``.
        """
        if x.ndim != 4:
            raise ValueError(
                f"期望输入的维度为 4D (batch, channel, H, W), 但得到的输入数据维度为 {list(x.shape)}")
        x = self.conv(x)
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        if self.activate_func is not None:
            x = self.activate_func(x)
        if self.leyer_poolling is not None:
            x = self.leyer_poolling(x)
        return x


class Conv2dFeatureExtractor(nn.Module):
    """构建多层连续的2D卷积网络结构
    参数:
        in_channel (int):
        shape (list):
    示例:
        net = Conv2dFeatureExtractor(
            in_channel = 1,
            shape =
            out_channel, kernel, stride, norm_type,norm_para,
            pool_type,pool_para,act_type,act_para

            [
                [64, {"kernel_size":[3, 3], "stride":[1, 1]},
                      "bn",{"num_features":64},
                      "mp",{"kernel_size":3},
                      "relu",{}],
                [96, {"kernel_size":[3, 3], "stride":[1, 1]}, "ln",{"normalized_shape"=[]}, "ap",{"kernel_size":3}, "gelu",{}],
                [128, {"kernel_size":[3, 3], "stride":[1, 1]}, "gn",{
                    "num_groups":1}, "ap",{"kernel_size":3}, "softmax",{}],
                [320, {"kernel_size":[3, 3], "stride":[1, 1]},
                    "bn",{}, "gap",{}, "",{}]
            ]
        )
    """

    def __init__(self,
                 in_channel,
                 shape,
                 ):

        super().__init__()
        self.conv_blocks = self._construct(in_channel, shape)

    def _construct(self, in_channel, shapes):
        blocks = []

        for i, (out_channel, conv_para, norm_mode, norm_para, pool_mode, pool_para, act_mode, act_para) in enumerate(shapes):
            # Norm设置
            assert (norm_mode in ["gn", "ln", "bn", ""]), "不支持的Norm类型"
            if norm_mode == "gn":
                assert ("num_groups" in norm_para), "GroupNorm 必需指定num_groups参数"
                normalization = nn.GroupNorm(
                    num_groups=norm_para["num_groups"],
                    num_channels=out_channel,
                    eps=norm_para["eps"] if "eps" in norm_para else 0.00001,
                    affine=norm_para["affine"] if "affine" in norm_para else True,
                )
            elif norm_mode == "ln":
                assert (
                    "normalized_shape" in norm_para), "LayerNorm 必需指定 normalized_shape 参数"
                normalization = nn.LayerNorm(
                    normalized_shape=norm_para["normalized_shape"],
                    eps=norm_para["eps"] if "eps" in norm_para else 0.00001,
                    elementwise_affine=norm_para["elementwise_affine"] if "elementwise_affine" in norm_para else True,

                )
            elif norm_mode == "bn":
                normalization = nn.BatchNorm2d(
                    num_features=out_channel,
                    eps=norm_para["eps"] if "eps" in norm_para else 0.00001,
                    momentum=norm_para["momentum"] if "momentum" in norm_para else 0.1,
                    affine=norm_para["affine"] if "affine" in norm_para else True,
                    track_running_stats=norm_para["track_running_stats"] if "track_running_stats" in norm_para else True,
                )
            else:
                normalization = None

            # Pool设置
            assert (pool_mode in ["mp", "ap", "gap", ""]), "不支持的Pool类型"

            if pool_mode == "mp":
                assert ("kernel_size" in pool_para), "MaxPool2d 必需指定 kernel_size 参数"
                poolling = nn.MaxPool2d(
                    kernel_size=pool_para["kernel_size"],
                    stride=pool_para["stride"] if "stride" in pool_para else None,
                    padding=pool_para["padding"] if "padding" in pool_para else 0,
                    dilation=pool_para["dilation"] if "dilation" in pool_para else 1,
                    return_indices=pool_para["return_indices"] if "return_indices" in pool_para else False,
                    ceil_mode=pool_para["ceil_mode"] if "ceil_mode" in pool_para else False,
                )

            elif pool_mode == "ap":
                assert ("kernel_size" in pool_para), "MaxPool2d 必需指定 kernel_size 参数"
                poolling = nn.AvgPool2d(
                    kernel_size=pool_para["kernel_size"],
                    stride=pool_para["stride"] if "stride" in pool_para else None,
                    padding=pool_para["padding"] if "padding" in pool_para else 0,
                    ceil_mode=pool_para["ceil_mode"] if "ceil_mode" in pool_para else False,
                    count_include_pad=pool_para["count_include_pad"] if "count_include_pad" in pool_para else True,
                    divisor_override=pool_para["divisor_override"] if "divisor_override" in pool_para else None,
                )
            elif pool_mode == "gap":  # GlobalAveragePooling2D
                pool_kernel = 1
                poolling = nn.AdaptiveAvgPool2d(1)
            else:
                poolling = None

            # Activate
            assert (act_mode in ["relu", "rrelu", "relu6", "sigmoid", "tanh", "glu", "gelu", "leakyreLU", "prelu", "softmax", ""]
                    ), "不支持的 Activate Function 类型"

            if act_mode == "relu":
                activate_func = nn.ReLU(
                    inplace=act_para["inplace"] if "inplace" in act_para else False)
            elif act_mode == "rrelu":
                activate_func = nn.RReLU(
                    inplace=act_para["inplace"] if "inplace" in act_para else False,
                    lower=act_para["lower"] if "lower" in act_para else 1 / 8,
                    upper=act_para["upper"] if "upper" in act_para else 1 / 3
                )
            elif act_mode == "relu6":
                activate_func = nn.ReLU6(
                    inplace=act_para["inplace"] if "inplace" in act_para else False)
            elif act_mode == "sigmoid":
                activate_func = nn.Sigmoid()
            elif act_mode == "tanh":
                activate_func = nn.Tanh()
            elif act_mode == "glu":
                activate_func = nn.GLU(
                    dim=act_para["dim"] if "dim" in act_para else -1)
            elif act_mode == "gelu":
                activate_func = nn.GELU()
            elif act_mode == "leakyreLU":
                activate_func = nn.LeakyReLU(
                    inplace=act_para["inplace"] if "inplace" in act_para else False,
                    negative_slope=act_para["negative_slope"] if "negative_slope" in act_para else 0.01,
                )
            elif act_mode == "prelu":
                activate_func = nn.PReLU(
                    num_parameters=act_para["num_parameters"] if "num_parameters" in act_para else 1,
                    init=act_para["init"] if "init" in act_para else 0.25)
            elif act_mode == "softmax":
                activate_func = nn.Softmax(
                    dim=act_para["dim"] if "dim" in act_para else None)
            else:
                activate_func = None

            # 卷积块设置
            assert ("kernel_size" in conv_para), "卷积参数必须设定kernel_size"
            blocks.append(
                Conv2dLayerBlock(
                    in_channel=in_channel,
                    out_channel=out_channel,
                    kernel_size=conv_para["kernel_size"],
                    stride=conv_para["stride"] if "stride" in conv_para else 1,
                    bias=conv_para["bias"] if "bias" in conv_para else True,
                    padding=conv_para["padding"] if "padding" in conv_para else 0,
                    dilation=conv_para["dilation"] if "dilation" in conv_para else 1,
                    groups=conv_para["groups"] if "groups" in conv_para else 1,
                    layer_norm=normalization,
                    layer_poolling=poolling,
                    activate_func=activate_func,

                )
            )

            in_channel = out_channel  # 修改下一个block的输入通道为当前block的输出通道

        return nn.ModuleList(blocks)

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
                f"期望输入的维度为 4D (batch, channel, H, W), 但得到的输入数据维度为 {list(x.shape)}")
        for block in self.conv_blocks:
            x = block(x)
        return x


class MultiStream_Conv2d_extractor(nn.Module):
    """支持多路径输入的多层2D卷积网络结构.

    示例:
        net = MultiStream_Conv2d_extractor(
            in_channel = [1,1],
            shape = # out_channel, kernel, stride, norm, pool, pool_kernel
            [
                [
                    [64, {"kernel_size":[3, 3], "stride":[1, 1]},
                        "bn",{"num_features":64},
                        "mp",{"kernel_size":3},
                        "relu",{}],
                    [96, {"kernel_size":[3, 3], "stride":[1, 1]}, "ln",{"normalized_shape"=[]}, "ap",{"kernel_size":3}, "gelu",{}],
                    [128, {"kernel_size":[3, 3], "stride":[1, 1]}, "gn",{
                        "num_groups":1}, "ap",{"kernel_size":3}, "softmax",{}],
                    [320, {"kernel_size":[3, 3], "stride":[1, 1]},
                        "bn",{}, "gap",{}, "",{}]
                ],

                [
                    [64, {"kernel_size":[3, 3], "stride":[1, 1]},
                        "bn",{"num_features":64},
                        "mp",{"kernel_size":3},
                        "relu",{}],
                    [96, {"kernel_size":[3, 3], "stride":[1, 1]}, "ln",{"normalized_shape"=[]}, "ap",{"kernel_size":3}, "gelu",{}],
                    [128, {"kernel_size":[3, 3], "stride":[1, 1]}, "gn",{
                        "num_groups":1}, "ap",{"kernel_size":3}, "softmax",{}],
                    [320, {"kernel_size":[3, 3], "stride":[1, 1]},
                        "bn",{}, "gap",{}, "",{}]
                ]
            ]
        )
    """

    def __init__(
        self,
        in_channels,
        shapes,
    ):
        super(MultiStream_Conv2d_extractor, self).__init__()
        # 计算输入Stream数量
        self.num_pathways = len(shapes)
        # Construct the stem layer.
        self._construct(in_channels, shapes)

    def _construct(self, in_channels, shapes):
        for pathway in range(self.num_pathways):
            stem = Conv2dFeatureExtractor(
                in_channels[pathway],
                shapes[pathway],
            )
            # 动态添加模型
            self.add_module("stream{}_stem".format(pathway), stem)

    def forward(self, x):
        assert (len(x) == self.num_pathways
                ), f"输入数据包含路径数量不正确, 请确认是否为: {self.num_pathways}"
        assert (len(x[0].shape) == 4 and len(x[1].shape)
                ), f"期望输入的维度为({self.num_pathways},B, C, H, W), 但得到的输入数据维度为 {[list(x[0].shape),list(x[1].shape)]}"

        for pathway in range(len(x)):
            m = getattr(self, f"stream{pathway}_stem")
            x[pathway] = m(x[pathway])
        return x


class FuseS1ToS0(nn.Module):
    """
    融合路径1的信息到路径0中, 返回原1向量和融合后的0向量
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        kernel,
        stride,

    ):
        """
        Args:
            dim_in (_type_): Stream1 的输入通道维度.
            dim_out (_type_): Stream0 的输入通道维度.
            kernel (_type_): 融合计算的卷积核大小
            stride (_type_): Stream1 与 Stream0 的特征维度比值
            padding (_type_): kernel // 2 方便得出的特征图特征维度以及通道维度均相等
        """
        super(FuseS1ToS0, self).__init__()
        kernel = [kernel]*2 if isinstance(kernel, int) else kernel
        self.conv_f2s = Conv2dLayerBlock(
            in_channel=dim_in,
            out_channel=dim_out,
            kernel_size=kernel,
            stride=stride,
            bias=False,
            padding=[int(kernel[0]//2), int(kernel[1]//2)],
            layer_norm=nn.BatchNorm2d(num_features=dim_out),
            activate_func=nn.ReLU())

    def forward(self, x):
        x_0 = x[0]
        x_1 = x[1]
        fuse = self.conv_f2s(x_1)
        x_0_fuse = torch.cat([x_0, fuse], 1)
        return [x_0_fuse, x_1]


class ResBlock(nn.Module):
    """ 给定Block函数, 创建一个 Residual block.
        其中, 若维度发生变换必须要指定projection_func.
    paper: https://arxiv.org/abs/1512.03385
    """

    def __init__(
        self,
        trans_func,
        activate_func=None,
        projection_func=None,
    ):
        super(ResBlock, self).__init__()
        self.projection_func = projection_func
        self.branch = trans_func
        self.activate_func = activate_func

    def forward(self, x):
        if self.projection_func != None:
            a = self.projection_func(x)
            b = self.branch(x)
            x = a+b
        else:
            x = x + self.branch(x)
        if self.activate_func != None:
            x = self.activate_func(x)
        return x


class MultiStream_ResStage(nn.Module):
    """给定单或多Stream中block的数量, 建立 ResStage
    """

    def __init__(
        self,
        num_blocks,
        in_channels,
        shapes,

    ):
        super(MultiStream_ResStage, self).__init__()
        # 获取每层 Resblock 的数量
        self.num_blocks = num_blocks
        # 获取数据Stream的数量
        self.num_pathways = len(self.num_blocks)
        self._construct(in_channels, shapes)

    def _construct(self, in_channels, shapes):

        # 对每个数据Stream建立计算路径
        for pathway in range(self.num_pathways):
            # 对每个路径中含有的block进行单独设置
            in_channel = in_channels[pathway]
            for i in range(self.num_blocks[pathway]):
                # Retrieve the transformation function.
                # Construct the block.
                shape = shapes[pathway]

                trans_func = Conv2dFeatureExtractor(
                    in_channel=in_channel,
                    shape=shape)
                if in_channel != shape[0][0] or shape[1][1]["stride"] != [1, 1] or shape[1][1]["stride"] != 1:
                    projection_func = Conv2dFeatureExtractor(
                        in_channel=in_channel,
                        shape=[[shape[-1][0], {"kernel_size": [1, 1], "stride":shape[1][1]["stride"], "padding":0, "bias":False},
                               "bn", {"num_features": shape[0][0]},
                                "", [],
                                "", {}
                                ]]
                    )
                res_block = ResBlock(
                    trans_func=trans_func,
                    projection_func=projection_func,
                    activate_func=nn.ReLU(),)
                self.add_module("stream{}_res{}".format(
                    pathway, i), res_block)
                in_channel = shape[-1][0]

    def forward(self, inputs):
        output = []
        for pathway in range(self.num_pathways):
            x = inputs[pathway]
            for i in range(self.num_blocks[pathway]):
                m = getattr(self, "stream{}_res{}".format(pathway, i))
                x = m(x)
            output.append(x)

        return output


class BasicHead(nn.Module):
    """
    """

    def __init__(
        self,
        dim_in,
        num_classes,
        act_func,
        dropout_rate=0.0,
        training=False,
    ):

        super(BasicHead, self).__init__()
        self.num_pathways = len(dim_in)
        self.training = training
        # 添加自适应池化层, 减少数据维度到仅剩通道层
        for pathway in range(self.num_pathways):
            ga_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.add_module("pathway{}_gap".format(pathway), ga_pool)
        # 添加dropout层
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)

        self.projection = nn.Linear(sum(dim_in), num_classes, bias=True)

        self.num_classes = num_classes
        # Softmax for evaluation and testing.
        assert (act_func in ["softmax", "sigmoid"]), f"不支持{act_func}激活函数"
        if act_func == "softmax":
            self.act = nn.Softmax(dim=3)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()

    def forward(self, inputs):
        pool_out = []

        for pathway in range(self.num_pathways):
            m = getattr(self, "pathway{}_gap".format(pathway))
            pool_out.append(m(inputs[pathway]))
        # 合并Stream通道
        x = torch.cat(pool_out, 1)
        # (N, C, T, H) -> (N, T, H, C). 将通道维度放在最后面方便全连接计算
        x = x.permute((0, 2, 3, 1))
        # Perform dropout.
        if hasattr(self, "dropout"):
            x = self.dropout(x)

        x = self.projection(x)

        # Performs fully convlutional inference.
        if not self.training:
            x = self.act(x)
            x = x.mean([1, 2])  # 避免输入数据特征维度与训练数据不同
        x = x.view(x.shape[0], -1)
        return x
