import torch
from torch import Tensor, nn
import torch.nn.functional as F
import numpy as np
from .components import *

# audio
from .lightsernet import LightSerNet, LightResMultiSerNet_Encoder
from .glamnet import GLAM
from .aacnnnet import AACNN
from .macnnnet import MACNN
from .mdease import MDEASE

# video
from .resnet import ResNet, DCResNet
from .densenet import DenseNet
from .res2net import res2net50_14w_8s
from .convnext import convnext_tiny
from .i3d import InceptionI3d
from .x3d import X3D
from .slowfast import SlowFast
from .uniformerv2 import uniformerv2_b16


class PretrainedBaseModel:
    def __init__(
        self,
        in_channels=3,
        out_features: int = 4,
        model_name: str = "resnet50",
        input_size=[224, 224],
        num_frames=1,
        before_dropout=0.5,
        before_softmax=False,
    ):
        self.in_channels = in_channels
        self.before_dropout = before_dropout
        self.before_softmax = before_softmax
        self.input_size = input_size
        self.num_frames = num_frames

        assert (
            before_dropout >= 0 and out_features >= 0
        ), "before_dropout 和 num_class 必须大于等于 0."
        assert not (out_features == 0 and before_softmax), "当不进行分类时, 不推荐使用概率输出"
        self._prepare_base_model(
            model_name,
        )  # 构建基础模型
        self._change_model_out(out_features)
        self._change_model_in()

    def _prepare_base_model(self, model_name):
        # 加载已经训练好的模型
        import custom_models.pretrainedmodels as pretrainedmodels

        if model_name in pretrainedmodels.model_names:
            self.base_model = getattr(pretrainedmodels, model_name)()
            self.base_model.last_layer_name = "last_linear"
            return True
        elif model_name == "rev_mvit_b_conv2d":
            self.base_model = rev_mvit_b_conv2d(
                pretrained=True, crop_size=self.input_size, input_channel_num=[3, 3]
            )
            self.base_model.last_layer_name = "last_linear"
            return True
        elif model_name == "rev_mvit_b_16x4_conv3d":
            self.base_model = rev_mvit_b_16x4_conv3d(
                pretrained=True,
                crop_size=self.input_size,
                input_channel_num=[3, 3],
                num_frames=self.num_frames,
            )
            self.base_model.last_layer_name = "last_linear"
            return True
        else:
            raise ValueError(f"不支持的基本模型: {model_name}")

    def _change_model_out(self, num_class):
        # 调整模型输出层
        linear_layer = getattr(self.base_model, self.base_model.last_layer_name)
        if not isinstance(linear_layer, nn.Linear):
            raise ModuleNotFoundError(f"需要基础模型的最后输出层应为 Linear 层, 而不是 {linear_layer} 层.")

        feature_dim = linear_layer.in_features  # 记录原基本模型的最终全连接层输入维度
        std = 0.001
        if num_class == 0 or num_class == feature_dim:
            self.new_fc = None
            setattr(
                self.base_model, self.base_model.last_layer_name, EmptyModel
            )  # 修改最终的全连接层为空, 直接返回最终隐藏层的向量

        elif self.before_dropout == 0:  # 替换最终的全连接层为指定的输出维度, 重新训练最后一层
            setattr(
                self.base_model,
                self.base_model.last_layer_name,
                nn.Linear(feature_dim, num_class),
            )
            self.new_fc = None
            torch.nn.init.normal(
                getattr(self.base_model, self.base_model.last_layer_name).weight, 0, std
            )
            torch.nn.init.constant(
                getattr(self.base_model, self.base_model.last_layer_name).bias, 0
            )
        else:  # 替换最终的全连接层为指定的输出维度, 同时在前面添加一个 Dropout
            setattr(
                self.base_model,
                self.base_model.last_layer_name,
                nn.Dropout(p=self.before_dropout),
            )  # 修改原基本模型的最终全连接层为 Dropout
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
                lambda x: isinstance(modules[x], nn.Conv2d)
                or isinstance(modules[x], nn.Conv3d),
                # 要被过滤的可迭代序列
                list(range(len(modules))),
            )
        )[
            0
        ]  # 获得第一个卷积层位置
        conv_layer = modules[first_conv_idx]  # 获得第一个卷积层
        nn_conv = nn.Conv2d if conv_layer._get_name() == "Conv2d" else nn.Conv3d
        # 获得第一个卷积层的包裹层(这里假设了输入只有最初的单通道卷积层处理)
        container = modules[first_conv_idx - 1]

        # 获取第一个卷积层的参数
        params = [x.clone() for x in conv_layer.parameters()]
        # 通过 new_in_c 计算出新的卷积核参数维度 (out_c,new_in_c,h,w)
        kernel_size = params[0].size()  # 获取原卷积核矩阵参数维度 (out_c,in_c,h,w)
        new_kernel_size = kernel_size[:1] + (self.in_channels,) + kernel_size[2:]

        # 构建新卷积核
        new_kernels = (
            params[0]
            .data.mean(dim=1, keepdim=True)
            .expand(new_kernel_size)
            .contiguous()
        )  # 利用原卷积矩阵参数 in_c 维度的平均值填充新通道 new_in_c 维度

        new_conv = nn_conv(
            self.in_channels,
            conv_layer.out_channels,
            conv_layer.kernel_size,
            conv_layer.stride,
            conv_layer.padding,
            bias=True if len(params) == 2 else False,
        )  # 建立一个新卷积
        new_conv.weight.data = new_kernels  # 给新卷积设定参数初始值
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # 如有必要增加偏置参数
        # 获取需要被替换的第一个卷积的名字
        layer_name = list(container.state_dict().keys())[0][:-7]
        setattr(container, layer_name, new_conv)  # 替换卷积

    def __call__(self):
        return self.base_model


class VideoImageEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.model(x)
        x = x.view(B, T, -1)
        return x.mean(dim=1)


def getModel(
    model_name,
    out_features,
    in_channels=1,
    input_size=[126, 40],
    num_frames=1,
    pretrained=False,
):
    """"""
    if pretrained:
        try:
            return PretrainedBaseModel(
                in_channels=in_channels,
                out_features=out_features,
                model_name=model_name,
                input_size=input_size,
                num_frames=num_frames,
                before_dropout=0,
                before_softmax=False,
            )()
        except ValueError as e:
            if "不支持的基本模型" in e.args[0]:
                pass

    if model_name == "lightsernet":
        model = LightSerNet(in_channels=in_channels, out_features=out_features)

        model.last_linear = EmptyModel()
        return model  # 320
    elif model_name == "md-ease":
        model = MDEASE(in_channels=in_channels, out_features=out_features)
        model.last_linear = EmptyModel()
        return model  # 320
    elif model_name == "glamnet":
        # 此网络需要提供输入图维度
        model = GLAM(shape=input_size, out_features=out_features)
        # model.last_linear = EmptyModel()
        return model  # 128 * shape[0]//2 * shape[1]//4
    elif model_name == "aacnn":
        # 此网络需要提供输入图维度
        model = AACNN(shape=input_size, out_features=out_features)
        model.last_linear = EmptyModel()
        return model  # 80 * ((shape[0] - 1) // 2) * ((shape[1] - 1) // 4)
    elif model_name == "macnn":
        # 此网络需要提供输入图维度
        model = MACNN(out_features=out_features)
        model.last_linear = EmptyModel()
        return model  # 256

    elif model_name == "resnet50":
        model = ResNet(in_channels=in_channels, num_classes=out_features)
        model.last_linear = EmptyModel()
        return model  # 512 * block.expansion = 2048
    elif model_name == "densenet161":
        model = DenseNet(
            in_channels=in_channels,
            num_classes=out_features,
            growth_rate=48,
            block_config=[6, 12, 36, 24],
            num_init_features=96,
        )
        model.last_linear = EmptyModel()
        return model  # 96 + 6 * 48 + 12 * 48 + 36 * 48 + 24 * 48 = 3840
    elif model_name == "res2net50":
        model = res2net50_14w_8s(in_channels=in_channels, num_classes=out_features)
        model.fc = EmptyModel()
        return model  # 512 * block.expansion
    elif model_name == "convnext":
        model = convnext_tiny(in_channels=in_channels, num_classes=out_features)
        model.head = EmptyModel()
        return model  # 768
    elif model_name == "i3d":
        model = InceptionI3d(num_classes=out_features, in_channels=in_channels)
        model.replace_logits(384 + 384 + 128 + 128)
        return model  # 384 + 384 + 128 + 128
    elif model_name == "x3d":
        model = X3D(
            in_channels=in_channels,
            num_classes=out_features,
            crop_size=input_size,
            num_frames=num_frames,
            depth_factor=2.2,
        )
        model.head.projection = EmptyModel()
        return model  # 2048
    elif model_name == "slowfast":
        model = SlowFast(
            in_channels=[in_channels, in_channels],
            num_classes=out_features,
            crop_size=input_size,
            num_frames=num_frames,
            alpha=5,
            beta_inv=8,
            fusion_conv_channel_ratio=2,
            fusion_kernel_sz=7,
            nonlocal_location=[[[], []], [[], []], [[], []], [[], []]],
            nonlocal_group=[[1, 1], [1, 1], [1, 1], [1, 1]],
        )
        model.head.projection = EmptyModel()
        return model  # 64 * 32 + (64 * 32 // 8) = 2048 + 256
    elif model_name == "uniformerv2":
        model = uniformerv2_b16(
            input_size=input_size,
            num_classes=out_features,
            backbone_drop_path_rate=0.2,
            drop_path_rate=0.4,
            dw_reduction=1.5,
            no_lmhra=True,
            temporal_downsample=False,
            num_frames=num_frames,
        )
        model.transformer.proj = EmptyModel()
        return model
    else:
        raise ValueError(f"不支持的基本模型: {model_name}")


def getVideoModel(
    model_name,
    out_features,
    in_channels=3,
    num_frames=30,
    input_size=[126, 40],
    pretrained=False,
):
    if model_name in ["resnet50", "densenet161", "res2net50", "convnext"]:
        model = VideoImageEncoder(
            getModel(
                model_name,
                out_features,
                in_channels=in_channels,
                input_size=input_size,
                pretrained=pretrained,
            )
        )
        return model
    return getModel(
        model_name,
        out_features,
        in_channels=in_channels,
        input_size=input_size,
        num_frames=num_frames,
        pretrained=pretrained,
    )


# 纯音频基线模型
class MER_AudioNet(nn.Module):
    def __init__(
        self,
        num_classes=4,
        model_name="lightsernet",
        last_hidden_dim=320,
        input_size=[126, 40],
        pretrained=False,
        enable_classifier=True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.enable_classifier = enable_classifier
        self.model_name = model_name
        self.last_hidden_dim = last_hidden_dim
        self.audio_feature_extractor = getModel(
            model_name,
            out_features=num_classes,
            in_channels=1,
            input_size=input_size,
            pretrained=pretrained,
        )

        self.audio_classifier = self.audio_feature_extractor.last_linear
        self.audio_feature_extractor.last_linear = EmptyModel()

    def forward(self, af: Tensor, af_len=None) -> Tensor:
        """
        Args:
            vf (Tensor): size:[batch, channel, seq, F]
            vf_len (Sequence, optional): size:[batch]. Defaults to None.
        Returns:
            Tensor: size:[batch, out]
        """
        af = af.permute(0, 3, 1, 2)
        af_fea = self.audio_feature_extractor(af)
        # 分类
        if self.enable_classifier:
            af_out = self.audio_classifier(af_fea)
            return af_out, None
        else:
            return af_fea


# 纯视频基线模型
class MER_VideoNet(nn.Module):
    """ """

    def __init__(
        self,
        num_classes=4,
        model_name="resnet50",
        last_hidden_dim=320,
        enable_classifier=True,
        input_size=[150, 150],
        num_frames=30,
        pretrained=False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.model_name = model_name
        self.enable_classifier = enable_classifier
        self.last_hidden_dim = last_hidden_dim
        self.video_feature_extractor = getVideoModel(
            self.model_name,
            out_features=self.last_hidden_dim,
            in_channels=3,
            num_frames=num_frames,
            input_size=input_size,
            pretrained=pretrained,
        )
        self.video_classifier = nn.Linear(
            in_features=self.last_hidden_dim, out_features=num_classes
        )

    def forward(self, vf: Tensor, vf_len=None) -> Tensor:
        """
        Args:
            vf (Tensor): size:[B, Seq, W, H, C]
            vf_len (Sequence, optional): size:[batch]. Defaults to None.
        Returns:
            Tensor: size:[batch, out]
        """
        seq_length = vf.shape[1]
        #  [B, T, W, H, C]
        vf = vf.permute(0, 1, 4, 2, 3).contiguous()
        # [B, T, C, W, H]
        vf_fea = self.video_feature_extractor(vf)

        if self.enable_classifier:
            # [batch * seq, 320]
            vf_out = self.video_classifier(vf_fea)
            return vf_out, None
        else:
            return vf_fea.view((-1, seq_length) + vf_fea.size()[1:]).mean(dim=1)


# 分段融合取平均, 音视频双任务独立
class MER_SISF_Conv2D_SVC(nn.Module):
    """
    paper: MER_SISF: Multimodal interframe feature fusion network
    """

    def __init__(
        self,
        num_classes=4,
        model_name=["lightsernet", "resnet50"],
        seq_len=[189, 30],
        last_hidden_dim=[320, 320],
        input_size=[[126, 40], [150, 150]],
    ) -> None:
        super().__init__()

        self.af_seq_len = seq_len[0]
        self.vf_seq_len = seq_len[1]
        self.af_model_name = model_name[0]
        self.vf_model_name = model_name[1]
        self.af_last_hidden_dim = last_hidden_dim[0]
        self.vf_last_hidden_dim = last_hidden_dim[1]

        self.af_input_size = input_size[0]
        self.vf_input_size = input_size[1]

        if self.af_model_name == "lightsernet":
            assert (
                self.af_last_hidden_dim == 320
            ), "由原论文可知 情绪识别中的 LightSerNet 的隐藏层向量最好限定为 320"

        self.audio_feature_extractor = getModel(
            self.af_model_name,
            out_features=num_classes,
            in_channels=1,
            input_size=input_size,
        )
        self.af_last_hidden_dim = self.audio_feature_extractor.last_linear.in_features
        self.audio_classifier = self.audio_feature_extractor.last_linear
        self.audio_feature_extractor.last_linear = EmptyModel()

        self.video_feature_extractor = getVideoModel(
            self.vf_model_name,
            out_features=num_classes,
            in_channels=3,
            num_frames=30,
            input_size=self.vf_input_size,
            pretrained=True,
        )
        self.vf_last_hidden_dim = self.video_feature_extractor.last_linear.in_features
        self.video_classifier = self.video_feature_extractor.last_linear
        self.video_feature_extractor.last_linear = EmptyModel()

    def forward(self, af: Tensor, vf: Tensor, af_len=None, vf_len=None) -> Tensor:
        """
        Args:
            af (Tensor): size:[B, C,Seq, F]
            vf (Tensor): size:[B, C, Seq, W, H]
            af_len (Sequence, optional): size:[B]. Defaults to None.
            vf_len (Sequence, optional): size:[B]. Defaults to None.

        Returns:
            Tensor: size:[batch, out]
        """
        af = af.permute(0, 3, 1, 2)
        vf = vf.permute(0, 4, 1, 2, 3)

        # 处理语音数据
        # [B, C, Seq, F]
        af = torch.split(af, self.af_seq_len, dim=2)  # 分割数据段

        # af_seg_num * [B, C, af_seq_len, F]

        # 避免最后一个数据段长度太短
        af_floor_ = False
        if af[-1].shape[2] != af[0].shape[2]:
            af = af[:-1]
            af_floor_ = True  # 控制计算有效长度时是否需要去掉最后一段数据
        # 记录分段数量, af_seg_num = ceil or floor(seq/af_seq_len)
        af_seg_num = len(af)
        af = torch.stack(af).permute(1, 0, 2, 3, 4).contiguous()
        # [B, af_seg_num, C, af_seq_len, F]

        # [B * af_seg_num, C, af_seq_len, F]
        af_fea = self.audio_feature_extractor(af.view((-1,) + af.size()[2:]))
        # [B * af_seg_num, F]
        af_fea = af_fea.view((-1, af_seg_num) + af_fea.size()[1:])
        # [B, af_seg_num, F]
        if af_len is not None:  # 如果提供了原数据的有效 Seq 长度
            batch_size, max_len, _ = af_fea.shape
            # 计算每一段实际的有效数据段数量
            af_len = (
                torch.floor(af_len / self.af_seq_len)
                if af_floor_
                else torch.ceil(af_len / self.af_seq_len)
            )
            af_mask = (
                torch.arange(max_len, device=af_len.device).expand(batch_size, max_len)
                >= af_len[:, None]
            )
            af_fea[af_mask] = 0.0

        # 处理视频数据
        # [B, C, Seq, W, H]
        vf = torch.split(vf, self.vf_seq_len, dim=2)  # 分割数据段

        # vf_seg_len * [B, C, vf_seq_len, W, H]

        # 避免最后一个数据段长度太短
        vf_floor_ = False
        if vf[-1].shape[2] != vf[0].shape[2]:
            vf = vf[:-1]
            vf_floor_ = True  # 控制计算有效长度时是否需要去掉最后一段数据
        vf_seg_len = len(vf)  # 记录分段数量, vf_seg_len = ceil(seq/vf_seg_len)
        vf = torch.stack(vf).permute(1, 0, 3, 2, 4, 5).contiguous()
        # [B, vf_seg_len, vf_seq_len, C, W, H]
        # [B * vf_seg_len * vf_seq_len, C, W, H]
        vf_fea = self.video_feature_extractor(vf.view((-1,) + vf.size()[-3:]))
        # [B * vf_seg_len * vf_seq_len, F]
        vf_fea = vf_fea.view((-1, vf_seg_len, self.vf_seq_len) + vf_fea.size()[1:])
        # [B, vf_seg_len, vf_seq_len, F]
        vf_fea = vf_fea.mean(dim=2)
        # [B, vf_seg_len, F]
        if vf_len is not None:
            batch_size, max_len, _ = vf_fea.shape
            vf_len = (
                torch.floor(vf_len / self.vf_seq_len)
                if vf_floor_
                else torch.ceil(vf_len / self.vf_seq_len)
            )
            vf_mask = (
                torch.arange(max_len, device=vf_len.device).expand(batch_size, max_len)
                >= vf_len[:, None]
            )
            vf_fea[vf_mask] = 0.0

        af_fea = af_fea.mean(dim=1)
        vf_fea = vf_fea.mean(dim=1)
        # 分类
        af_out = self.audio_classifier(af_fea)
        vf_out = self.video_classifier(vf_fea)

        return F.softmax(af_out, dim=1), F.softmax(vf_out, dim=1)


# 分段融合取平均, 音视频以及融合任务进行三任务学习
class MER_SISF_Conv2D_SVC_InterFusion(nn.Module):
    """
    paper: MER_SISF: Multimodal interframe feature fusion network
    """

    def __init__(
        self,
        num_classes=4,
        model_name=["lightsernet", "resnet50"],
        seq_len=[189, 30],
        last_hidden_dim=[320, 320],
        input_size=[[126, 40], [150, 150]],
        enable_classifier=True,
    ) -> None:
        super().__init__()
        self.af_seq_len = seq_len[0]
        self.vf_seq_len = seq_len[1]
        self.af_model_name = model_name[0]
        self.vf_model_name = model_name[1]
        self.af_last_hidden_dim = last_hidden_dim[0]
        self.vf_last_hidden_dim = last_hidden_dim[1]
        self.af_input_size = input_size[0]
        self.vf_input_size = input_size[1]
        self.enable_classifier = enable_classifier

        if self.af_model_name == "lightsernet":
            assert (
                self.af_last_hidden_dim == 320
            ), "由原论文可知 情绪识别中的 LightSerNet 的隐藏层向量最好限定为 320"

        self.audio_feature_extractor = getModel(
            self.af_model_name,
            out_features=num_classes,
            in_channels=1,
            input_size=self.af_input_size,
        )
        self.af_last_hidden_dim = self.audio_feature_extractor.last_linear.in_features

        self.audio_feature_extractor.last_linear = EmptyModel()

        self.video_feature_extractor = getVideoModel(
            self.vf_model_name,
            out_features=num_classes,
            in_channels=3,
            num_frames=30,
            input_size=self.vf_input_size,
            pretrained=True,
        )
        self.vf_last_hidden_dim = self.video_feature_extractor.last_linear.in_features
        self.video_feature_extractor.last_linear = EmptyModel()

        mid_hidden_dim = int((self.af_last_hidden_dim + self.vf_last_hidden_dim) / 2)

        self.fusion_feature = nn.Linear(
            in_features=self.af_last_hidden_dim + self.vf_last_hidden_dim,
            out_features=mid_hidden_dim,
        )

        self.audio_classifier = nn.Linear(
            in_features=self.af_last_hidden_dim + mid_hidden_dim,
            out_features=num_classes,
        )

        self.video_classifier = nn.Linear(
            in_features=self.vf_last_hidden_dim + mid_hidden_dim,
            out_features=num_classes,
        )
        self.fusion_classifier = nn.Linear(
            in_features=self.af_last_hidden_dim + self.vf_last_hidden_dim,
            out_features=num_classes,
        )

    def forward(self, af: Tensor, vf: Tensor, af_len=None, vf_len=None) -> Tensor:
        """
        Args:
            af (Tensor): size:[B, C,Seq, F]
            vf (Tensor): size:[B, C, Seq, W, H]
            af_len (Sequence, optional): size:[B]. Defaults to None.
            vf_len (Sequence, optional): size:[B]. Defaults to None.

        Returns:
            Tensor: size:[batch, out]
        """
        af = af.permute(0, 3, 1, 2)  # [B, C, Seq, F]
        vf = vf.permute(0, 4, 1, 2, 3)  # [B, C, Seq, F]

        # 处理语音数据

        af = torch.split(af, self.af_seq_len, dim=2)  # 分割数据段

        # af_seg_num * [B, C, af_seq_len, F]

        # 避免最后一个数据段长度太短
        af_floor_ = False
        if af[-1].shape[2] != af[0].shape[2]:
            af = af[:-1]
            af_floor_ = True  # 控制计算有效长度时是否需要去掉最后一段数据
        # 记录分段数量, af_seg_num = ceil or floor(seq/af_seq_len)
        af_seg_num = len(af)
        af = torch.stack(af).permute(1, 0, 2, 3, 4).contiguous()
        # [B, af_seg_num, C, af_seq_len, F]

        # [B * af_seg_num, C, af_seq_len, F]

        af_fea = self.audio_feature_extractor(af.view((-1,) + af.size()[2:]))
        # [B * af_seg_num, F]
        af_fea = af_fea.view((-1, af_seg_num) + af_fea.size()[1:])
        # [B, af_seg_num, F]
        if af_len is not None:  # 如果提供了原数据的有效 Seq 长度
            batch_size, max_len, _ = af_fea.shape
            # 计算每一段实际的有效数据段数量
            af_len = (
                torch.floor(af_len / self.af_seq_len)
                if af_floor_
                else torch.ceil(af_len / self.af_seq_len)
            )
            af_mask = (
                torch.arange(max_len, device=af_len.device).expand(batch_size, max_len)
                >= af_len[:, None]
            )
            af_fea[af_mask] = 0.0

        # 处理视频数据
        # [B, C, Seq, W, H]
        vf = torch.split(vf, self.vf_seq_len, dim=2)  # 分割数据段

        # vf_seg_len * [B, C, vf_seq_len, W, H]

        # 避免最后一个数据段长度太短
        vf_floor_ = False
        if vf[-1].shape[2] != vf[0].shape[2]:
            vf = vf[:-1]
            vf_floor_ = True  # 控制计算有效长度时是否需要去掉最后一段数据
        vf_seg_len = len(vf)  # 记录分段数量, vf_seg_len = ceil(seq/vf_seg_len)
        vf = torch.stack(vf).permute(1, 0, 3, 2, 4, 5).contiguous()
        # [B, vf_seg_len, vf_seq_len, C, W, H]

        # # 二维

        # [B * vf_seg_len * vf_seq_len, C, W, H]
        vf_fea = self.video_feature_extractor(vf.view((-1,) + vf.size()[-3:]))
        # [B * vf_seg_len * vf_seq_len, F]
        vf_fea = vf_fea.view((-1, vf_seg_len, self.vf_seq_len) + vf_fea.size()[1:])
        # [B, vf_seg_len, vf_seq_len, F]
        vf_fea = vf_fea.mean(dim=2)
        # [B, vf_seg_len, F]

        # 三维
        # vf_fea = self.video_feature_extractor(
        #     vf.view((-1, ) + vf.size()[-4:]).permute(0, 2, 1, 3, 4))
        # # [B* vf_seg_len, F]
        # vf_fea = vf_fea.view(
        #     (-1, vf_seg_len) + vf_fea.size()[1:])
        # # [B, vf_seg_len, F]

        if vf_len is not None:
            batch_size, max_len, _ = vf_fea.shape
            vf_len = (
                torch.floor(vf_len / self.vf_seq_len)
                if vf_floor_
                else torch.ceil(vf_len / self.vf_seq_len)
            )
            vf_mask = (
                torch.arange(max_len, device=vf_len.device).expand(batch_size, max_len)
                >= vf_len[:, None]
            )
            vf_fea[vf_mask] = 0.0

        # 中间融合

        seg_len = min(vf_seg_len, af_seg_num)
        fusion_fea = torch.cat([vf_fea[:, :seg_len, :], af_fea[:, :seg_len, :]], dim=-1)
        # [B, seg_len, 2*F]

        common_feature = self.fusion_feature(fusion_fea.detach())
        common_feature = common_feature.mean(dim=1)

        af_fea = af_fea.mean(dim=1)
        vf_fea = vf_fea.mean(dim=1)
        fusion_fea = fusion_fea.mean(dim=1)

        # [B, 2*F]
        af_fea = torch.cat([af_fea, common_feature], dim=1)
        vf_fea = torch.cat([vf_fea, common_feature], dim=1)
        # 分类
        if self.enable_classifier:
            af_out = self.audio_classifier(af_fea)
            vf_out = self.video_classifier(vf_fea)
            av_out = self.fusion_classifier(fusion_fea)
            return (
                F.softmax(af_out, dim=1),
                F.softmax(vf_out, dim=1),
                F.softmax(av_out, dim=1),
                [None, None],
            )
        else:
            return af_fea, vf_fea, fusion_fea


# 音视频编码后提取通过相似性信息进行特征补充
class MER_SISF_Conv2D_InterFusion_Joint(nn.Module):
    """
    paper: MER_SISF: Multimodal interframe feature fusion network
    """

    def __init__(
        self,
        num_classes=4,
        model_name=["lightsernet", "resnet50"],
        seq_len=[189, 30],
        last_hidden_dim=[320, 320],
        input_size=[[126, 40], [150, 150]],
        enable_classifier=True,
    ) -> None:
        super().__init__()
        self.af_seq_len = seq_len[0]
        self.vf_seq_len = seq_len[1]
        self.af_model_name = model_name[0]
        self.vf_model_name = model_name[1]
        self.af_last_hidden_dim = last_hidden_dim[0]
        self.vf_last_hidden_dim = last_hidden_dim[1]
        self.af_input_size = input_size[0]
        self.vf_input_size = input_size[1]
        self.enable_classifier = enable_classifier

        if self.af_model_name == "lightsernet":
            assert (
                self.af_last_hidden_dim == 320
            ), "由原论文可知 情绪识别中的 LightSerNet 的隐藏层向量最好限定为 320"

        self.audio_feature_extractor = getModel(
            self.af_model_name,
            out_features=num_classes,
            in_channels=1,
            input_size=self.af_input_size,
        )
        self.af_last_hidden_dim = self.audio_feature_extractor.last_linear.in_features

        self.audio_feature_extractor.last_linear = EmptyModel()

        self.video_feature_extractor = getVideoModel(
            self.vf_model_name,
            out_features=num_classes,
            in_channels=3,
            num_frames=30,
            input_size=self.vf_input_size,
            pretrained=True,
        )
        self.vf_last_hidden_dim = self.video_feature_extractor.last_linear.in_features
        self.video_feature_extractor.last_linear = EmptyModel()

        mid_hidden_dim = int((self.af_last_hidden_dim + self.vf_last_hidden_dim) / 2)

        self.fusion_feature = nn.Linear(
            in_features=self.af_last_hidden_dim + self.vf_last_hidden_dim,
            out_features=mid_hidden_dim,
        )

        self.classifier = nn.Linear(
            in_features=self.af_last_hidden_dim
            + mid_hidden_dim
            + self.vf_last_hidden_dim,
            out_features=num_classes,
        )

    def forward(self, af: Tensor, vf: Tensor, af_len=None, vf_len=None) -> Tensor:
        """
        Args:
            af (Tensor): size:[B, Seq, F, C]
            vf (Tensor): size:[B, Seq, W, H, C]
            af_len (Sequence, optional): size:[B]. Defaults to None.
            vf_len (Sequence, optional): size:[B]. Defaults to None.

        Returns:
            Tensor: size:[batch, out]
        """
        af = af.permute(0, 3, 1, 2)
        # [B, C, Seq, F]
        af_fea = self.audio_feature_extractor(af)
        # [B, F]

        # 处理视频数据
        # [B, Seq, W, H, C]
        seq_length = vf.shape[1]
        vf = vf.permute(0, 1, 4, 2, 3).contiguous()
        # [B, Seq, C, W, H]
        vf_fea = self.video_feature_extractor(vf.view((-1,) + vf.size()[-3:]))
        # [B * Seq, F]
        vf_fea = vf_fea.view((-1, seq_length) + vf_fea.size()[1:])
        # [B, Seq, F]
        vf_fea = vf_fea.mean(dim=1)
        # [B, F]
        # 中间融合
        fusion_fea = torch.cat([vf_fea, af_fea], dim=-1)
        # [B, 2*F]
        common_feature = self.fusion_feature(fusion_fea.detach())

        # [B, 2*F]
        joint_fea = torch.cat([af_fea, common_feature, vf_fea], dim=1)
        # 分类
        if self.enable_classifier:
            out = self.classifier(joint_fea)
            return out, None
        else:
            return (joint_fea,)


# 分段融合取平均, 音视频编码后提取通过相似性信息进行特征补充
class MER_SISF_Conv2D_SVC_InterFusion_Joint(nn.Module):
    """
    paper: MER_SISF: Multimodal interframe feature fusion network
    """

    def __init__(
        self,
        num_classes=4,
        model_name=["lightsernet", "resnet50"],
        seq_len=[189, 30],
        last_hidden_dim=[320, 320],
        input_size=[[126, 40], [150, 150]],
        enable_classifier=True,
    ) -> None:
        super().__init__()
        self.af_seq_len = seq_len[0]
        self.vf_seq_len = seq_len[1]
        self.af_model_name = model_name[0]
        self.vf_model_name = model_name[1]
        self.af_last_hidden_dim = last_hidden_dim[0]
        self.vf_last_hidden_dim = last_hidden_dim[1]
        self.af_input_size = input_size[0]
        self.vf_input_size = input_size[1]
        self.enable_classifier = enable_classifier

        if self.af_model_name == "lightsernet":
            assert (
                self.af_last_hidden_dim == 320
            ), "由原论文可知 情绪识别中的 LightSerNet 的隐藏层向量最好限定为 320"

        self.audio_feature_extractor = getModel(
            self.af_model_name,
            out_features=num_classes,
            in_channels=1,
            input_size=self.af_input_size,
        )
        self.af_last_hidden_dim = self.audio_feature_extractor.last_linear.in_features

        self.audio_feature_extractor.last_linear = EmptyModel()

        self.video_feature_extractor = getVideoModel(
            self.vf_model_name,
            out_features=num_classes,
            in_channels=3,
            num_frames=30,
            input_size=self.vf_input_size,
            pretrained=True,
        )
        self.vf_last_hidden_dim = self.video_feature_extractor.last_linear.in_features
        self.video_feature_extractor.last_linear = EmptyModel()

        mid_hidden_dim = int((self.af_last_hidden_dim + self.vf_last_hidden_dim) / 2)

        self.fusion_feature = nn.Linear(
            in_features=self.af_last_hidden_dim + self.vf_last_hidden_dim,
            out_features=mid_hidden_dim,
        )

        self.classifier = nn.Linear(
            in_features=self.af_last_hidden_dim
            + mid_hidden_dim
            + self.vf_last_hidden_dim,
            out_features=num_classes,
        )

    def forward(self, af: Tensor, vf: Tensor, af_len=None, vf_len=None) -> Tensor:
        """
        Args:
            af (Tensor): size:[B, C,Seq, F]
            vf (Tensor): size:[B, C, Seq, W, H]
            af_len (Sequence, optional): size:[B]. Defaults to None.
            vf_len (Sequence, optional): size:[B]. Defaults to None.

        Returns:
            Tensor: size:[batch, out]
        """
        af = af.permute(0, 3, 1, 2)  # [B, C, Seq, F]
        vf = vf.permute(0, 4, 1, 2, 3)  # [B, C, Seq, F]

        # 处理语音数据

        af = torch.split(af, self.af_seq_len, dim=2)  # 分割数据段

        # af_seg_num * [B, C, af_seq_len, F]

        # 避免最后一个数据段长度太短
        af_floor_ = False
        if af[-1].shape[2] != af[0].shape[2]:
            af = af[:-1]
            af_floor_ = True  # 控制计算有效长度时是否需要去掉最后一段数据
        # 记录分段数量, af_seg_num = ceil or floor(seq/af_seq_len)
        af_seg_num = len(af)
        af = torch.stack(af).permute(1, 0, 2, 3, 4).contiguous()
        # [B, af_seg_num, C, af_seq_len, F]

        # [B * af_seg_num, C, af_seq_len, F]

        af_fea = self.audio_feature_extractor(af.view((-1,) + af.size()[2:]))
        # [B * af_seg_num, F]
        af_fea = af_fea.view((-1, af_seg_num) + af_fea.size()[1:])
        # [B, af_seg_num, F]
        if af_len is not None:  # 如果提供了原数据的有效 Seq 长度
            batch_size, max_len, _ = af_fea.shape
            # 计算每一段实际的有效数据段数量
            af_len = (
                torch.floor(af_len / self.af_seq_len)
                if af_floor_
                else torch.ceil(af_len / self.af_seq_len)
            )
            af_mask = (
                torch.arange(max_len, device=af_len.device).expand(batch_size, max_len)
                >= af_len[:, None]
            )
            af_fea[af_mask] = 0.0

        # 处理视频数据
        # [B, C, Seq, W, H]
        vf = torch.split(vf, self.vf_seq_len, dim=2)  # 分割数据段

        # vf_seg_len * [B, C, vf_seq_len, W, H]

        # 避免最后一个数据段长度太短
        vf_floor_ = False
        if vf[-1].shape[2] != vf[0].shape[2]:
            vf = vf[:-1]
            vf_floor_ = True  # 控制计算有效长度时是否需要去掉最后一段数据
        vf_seg_len = len(vf)  # 记录分段数量, vf_seg_len = ceil(seq/vf_seg_len)
        vf = torch.stack(vf).permute(1, 0, 3, 2, 4, 5).contiguous()
        # [B, vf_seg_len, vf_seq_len, C, W, H]

        # # 二维

        # [B * vf_seg_len * vf_seq_len, C, W, H]
        vf_fea = self.video_feature_extractor(vf.view((-1,) + vf.size()[-3:]))
        # [B * vf_seg_len * vf_seq_len, F]
        vf_fea = vf_fea.view((-1, vf_seg_len, self.vf_seq_len) + vf_fea.size()[1:])
        # [B, vf_seg_len, vf_seq_len, F]
        vf_fea = vf_fea.mean(dim=2)
        # [B, vf_seg_len, F]

        # 三维
        # vf_fea = self.video_feature_extractor(
        #     vf.view((-1, ) + vf.size()[-4:]).permute(0, 2, 1, 3, 4))
        # # [B* vf_seg_len, F]
        # vf_fea = vf_fea.view(
        #     (-1, vf_seg_len) + vf_fea.size()[1:])
        # # [B, vf_seg_len, F]

        if vf_len is not None:
            batch_size, max_len, _ = vf_fea.shape
            vf_len = (
                torch.floor(vf_len / self.vf_seq_len)
                if vf_floor_
                else torch.ceil(vf_len / self.vf_seq_len)
            )
            vf_mask = (
                torch.arange(max_len, device=vf_len.device).expand(batch_size, max_len)
                >= vf_len[:, None]
            )
            vf_fea[vf_mask] = 0.0

        # 中间融合

        seg_len = min(vf_seg_len, af_seg_num)
        fusion_fea = torch.cat([vf_fea[:, :seg_len, :], af_fea[:, :seg_len, :]], dim=-1)
        # [B, seg_len, 2*F]

        common_feature = self.fusion_feature(fusion_fea.detach())
        common_feature = common_feature.mean(dim=1)

        af_fea = af_fea.mean(dim=1)
        vf_fea = vf_fea.mean(dim=1)

        # [B, 2*F]
        joint_fea = torch.cat([af_fea, common_feature, vf_fea], dim=1)
        # 分类
        if self.enable_classifier:
            out = self.classifier(joint_fea)
            return F.softmax(out, dim=1), None
        else:
            return joint_fea, af_fea, vf_fea


# 分段融合取平均, 音视频编码后提取通过相似性信息进行特征补充, 并通过注意力机制进行段间加权
class MER_SISF_Conv2D_SVC_InterFusion_Joint_Attention(nn.Module):
    """
    paper: MER_SISF: Multimodal interframe feature fusion network
    """

    def __init__(
        self,
        num_classes=4,
        model_name=["lightsernet", "resnet50"],
        seq_len=[189, 30],
        last_hidden_dim=[320, 320],
        input_size=[[126, 40], [150, 150]],
        enable_classifier=True,
    ) -> None:
        super().__init__()
        self.af_seq_len = seq_len[0]
        self.vf_seq_len = seq_len[1]
        self.af_model_name = model_name[0]
        self.vf_model_name = model_name[1]
        self.af_last_hidden_dim = last_hidden_dim[0]
        self.vf_last_hidden_dim = last_hidden_dim[1]
        assert self.af_last_hidden_dim == self.vf_last_hidden_dim, "两个模态的最后隐藏层维度必须相同"
        self.af_input_size = input_size[0]
        self.vf_input_size = input_size[1]
        self.enable_classifier = enable_classifier

        if self.af_model_name == "lightsernet":
            assert (
                self.af_last_hidden_dim == 320
            ), "由原论文可知 情绪识别中的 LightSerNet 的隐藏层向量最好限定为 320"

        self.audio_feature_extractor = LightResMultiSerNet_Encoder()
        self.video_feature_extractor = getVideoModel(
            self.vf_model_name,
            out_features=self.vf_last_hidden_dim,
            in_channels=3,
            num_frames=30,
            input_size=self.vf_input_size,
            pretrained=False,
        )

        self.af_esaAttention = ESIAttention(self.af_last_hidden_dim, 4)
        self.vf_esaAttention = ESIAttention(self.vf_last_hidden_dim, 4)
        mid_hidden_dim = int((self.af_last_hidden_dim + self.vf_last_hidden_dim) / 2)
        self.fusion_feature = nn.Linear(
            in_features=self.af_last_hidden_dim + self.vf_last_hidden_dim,
            out_features=mid_hidden_dim,
        )

        self.classifier = nn.Linear(
            in_features=self.af_last_hidden_dim
            + mid_hidden_dim
            + self.vf_last_hidden_dim,
            out_features=num_classes,
        )

    def forward(self, af: Tensor, vf: Tensor, af_len=None, vf_len=None) -> Tensor:
        """
        Args:
            af (Tensor): size:[B, C,Seq, F]
            vf (Tensor): size:[B, C, Seq, W, H]
            af_len (Sequence, optional): size:[B]. Defaults to None.
            vf_len (Sequence, optional): size:[B]. Defaults to None.

        Returns:
            Tensor: size:[batch, out]
        """
        af = af.permute(0, 3, 1, 2)  # [B, C, Seq, F]
        vf = vf.permute(0, 4, 1, 2, 3)  # [B, C, Seq, F]

        # 处理语音数据

        af = torch.split(af, self.af_seq_len, dim=2)  # 分割数据段

        # af_seg_num * [B, C, af_seq_len, F]

        # 避免最后一个数据段长度太短
        af_floor_ = False
        if af[-1].shape[2] != af[0].shape[2]:
            af = af[:-1]
            af_floor_ = True  # 控制计算有效长度时是否需要去掉最后一段数据
        # 记录分段数量, af_seg_num = ceil or floor(seq/af_seq_len)
        af_seg_num = len(af)
        af = torch.stack(af).permute(1, 0, 2, 3, 4).contiguous()
        # [B, af_seg_num, C, af_seq_len, F]

        # [B * af_seg_num, C, af_seq_len, F]

        af_fea = self.audio_feature_extractor(af.view((-1,) + af.size()[2:]))
        # [B * af_seg_num, F]
        af_fea = af_fea.view((-1, af_seg_num) + af_fea.size()[1:])
        # [B, af_seg_num, F]
        if af_len is not None:  # 如果提供了原数据的有效 Seq 长度
            batch_size, max_len, _ = af_fea.shape
            # 计算每一段实际的有效数据段数量
            af_len = (
                torch.floor(af_len / self.af_seq_len)
                if af_floor_
                else torch.ceil(af_len / self.af_seq_len)
            )
            af_mask = (
                torch.arange(max_len, device=af_len.device).expand(batch_size, max_len)
                >= af_len[:, None]
            )
            af_fea[af_mask] = 0.0

        # 处理视频数据
        # [B, C, Seq, W, H]
        vf = torch.split(vf, self.vf_seq_len, dim=2)  # 分割数据段

        # vf_seg_len * [B, C, vf_seq_len, W, H]

        # 避免最后一个数据段长度太短
        vf_floor_ = False
        if vf[-1].shape[2] != vf[0].shape[2]:
            vf = vf[:-1]
            vf_floor_ = True  # 控制计算有效长度时是否需要去掉最后一段数据
        vf_seg_len = len(vf)  # 记录分段数量, vf_seg_len = ceil(seq/vf_seg_len)
        vf = torch.stack(vf).permute(1, 0, 3, 2, 4, 5).contiguous()
        # [B, vf_seg_len, vf_seq_len, C, W, H]

        # # 二维

        # [B * vf_seg_len * vf_seq_len, C, W, H]
        vf_fea = self.video_feature_extractor(vf.view((-1,) + vf.size()[-3:]))
        # [B * vf_seg_len * vf_seq_len, F]
        vf_fea = vf_fea.view((-1, vf_seg_len, self.vf_seq_len) + vf_fea.size()[1:])
        # [B, vf_seg_len, vf_seq_len, F]
        vf_fea = vf_fea.mean(dim=2)
        # [B, vf_seg_len, F]

        # 三维
        # vf_fea = self.video_feature_extractor(
        #     vf.view((-1, ) + vf.size()[-4:]).permute(0, 2, 1, 3, 4))
        # # [B* vf_seg_len, F]
        # vf_fea = vf_fea.view(
        #     (-1, vf_seg_len) + vf_fea.size()[1:])
        # # [B, vf_seg_len, F]

        if vf_len is not None:
            batch_size, max_len, _ = vf_fea.shape
            vf_len = (
                torch.floor(vf_len / self.vf_seq_len)
                if vf_floor_
                else torch.ceil(vf_len / self.vf_seq_len)
            )
            vf_mask = (
                torch.arange(max_len, device=vf_len.device).expand(batch_size, max_len)
                >= vf_len[:, None]
            )
            vf_fea[vf_mask] = 0.0

        # 中间融合

        seg_len = min(vf_seg_len, af_seg_num)
        fusion_fea = torch.cat([vf_fea[:, :seg_len, :], af_fea[:, :seg_len, :]], dim=-1)
        # [B, seg_len, 2*F]
        common_feature = self.fusion_feature(fusion_fea.detach())
        af_fea = self.af_esaAttention(af_fea, common_feature)
        vf_fea = self.vf_esaAttention(vf_fea, common_feature)

        common_feature = common_feature.mean(dim=1)
        af_fea = af_fea.mean(dim=1)
        vf_fea = vf_fea.mean(dim=1)

        # [B, 2*F]
        joint_fea = torch.cat([af_fea, common_feature, vf_fea], dim=1)
        # 分类
        if self.enable_classifier:
            out = self.classifier(joint_fea)
            return out, None
        else:
            return (joint_fea,)

    def normal(self, x):
        std, mean = torch.std_mean(x, -1, unbiased=False)
        return (x - mean[:, None]) / (std[:, None] + 1e-5)


class MER_SISF_Conv2D_InterFusion_Joint_Attention(nn.Module):
    """
    paper: MER_SISF: Multimodal interframe feature fusion network
    """

    def __init__(
        self,
        num_classes=4,
        model_name=["lightsernet", "resnet50"],
        seq_len=[189, 30],
        last_hidden_dim=[320, 320],
        input_size=[[126, 40], [150, 150]],
        enable_classifier=True,
    ) -> None:
        super().__init__()
        self.af_seq_len = seq_len[0]
        self.vf_seq_len = seq_len[1]
        self.af_model_name = model_name[0]
        self.vf_model_name = model_name[1]
        self.af_last_hidden_dim = last_hidden_dim[0]
        self.vf_last_hidden_dim = last_hidden_dim[1]

        self.af_input_size = input_size[0]
        self.vf_input_size = input_size[1]
        self.enable_classifier = enable_classifier

        if self.af_model_name == "lightsernet":
            assert (
                self.af_last_hidden_dim == 320
            ), "由原论文可知 情绪识别中的 LightSerNet 的隐藏层向量最好限定为 320"

        self.audio_feature_extractor = getModel(
            self.af_model_name,
            out_features=num_classes,
            in_channels=1,
            input_size=self.af_input_size,
        )
        self.af_last_hidden_dim = self.audio_feature_extractor.last_linear.in_features

        self.audio_feature_extractor.last_linear = EmptyModel()

        self.video_feature_extractor = getVideoModel(
            self.vf_model_name,
            out_features=num_classes,
            in_channels=3,
            num_frames=30,
            input_size=self.vf_input_size,
            pretrained=True,
        )
        self.vf_last_hidden_dim = self.video_feature_extractor.last_linear.in_features
        self.video_feature_extractor.last_linear = EmptyModel()

        self.af_esaAttention = ESIAttention(self.af_last_hidden_dim, 4)
        self.vf_esaAttention = ESIAttention(self.vf_last_hidden_dim, 4)
        mid_hidden_dim = int((self.af_last_hidden_dim + self.vf_last_hidden_dim) / 2)
        self.fusion_feature = nn.Linear(
            in_features=self.af_last_hidden_dim + self.vf_last_hidden_dim,
            out_features=mid_hidden_dim,
        )

        self.classifier = nn.Linear(
            in_features=self.af_last_hidden_dim
            + mid_hidden_dim
            + self.vf_last_hidden_dim,
            out_features=num_classes,
        )

    def forward(self, af: Tensor, vf: Tensor, af_len=None, vf_len=None) -> Tensor:
        """
        Args:
            af (Tensor): size:[B, Seq, F, C]
            vf (Tensor): size:[B, Seq, W, H, C]
            af_len (Sequence, optional): size:[B]. Defaults to None.
            vf_len (Sequence, optional): size:[B]. Defaults to None.

        Returns:
            Tensor: size:[batch, out]
        """
        af = af.permute(0, 3, 1, 2)  # [B, C, Seq, F]
        vf = vf.permute(0, 1, 4, 2, 3)  # [B, Seq, C, W, H]

        # 处理语音数据
        # [B, C, Seq, F]
        af_fea = self.audio_feature_extractor(af)
        # [B, F]

        # 处理视频数据
        seq_length = vf.shape[1]
        vf = vf.contiguous()

        vf_fea = self.video_feature_extractor(vf.view((-1,) + vf.size()[-3:]))
        # [B * Seq, F]

        vf_fea = vf_fea.view((-1, seq_length) + vf_fea.size()[1:])
        # [B, Seq, F]
        vf_fea = vf_fea.mean(dim=1)
        # [B, F]

        # 中间融合

        fusion_fea = torch.cat([vf_fea, af_fea], dim=-1)
        # [B, seg_len, 2*F]
        common_feature = self.fusion_feature(fusion_fea.detach())
        af_fea = self.af_esaAttention(af_fea, common_feature)
        vf_fea = self.vf_esaAttention(vf_fea, common_feature)

        common_feature = common_feature.mean(dim=1)
        af_fea = af_fea.mean(dim=1)
        vf_fea = vf_fea.mean(dim=1)

        # [B, 2*F]
        joint_fea = torch.cat([af_fea, common_feature, vf_fea], dim=1)
        # 分类
        if self.enable_classifier:
            out = self.classifier(joint_fea)
            return out, None
        else:
            return (joint_fea,)

    def normal(self, x):
        std, mean = torch.std_mean(x, -1, unbiased=False)
        return (x - mean[:, None]) / (std[:, None] + 1e-5)


# 分段融合取平均, 音视频编码后提取通过相似性信息进行特征补充, 并通过注意力机制进行段间加权
class MER_SISF_Conv2D_SVC_InterFusion_Joint_Attention_SLoss(nn.Module):
    """
    paper: MER_SISF: Multimodal interframe feature fusion network
    """

    def __init__(
        self,
        num_classes=4,
        model_name=["lightsernet", "resnet50"],
        seq_len=[189, 30],
        last_hidden_dim=[320, 320],
        input_size=[[126, 40], [150, 150]],
        enable_classifier=True,
    ) -> None:
        super().__init__()
        self.af_seq_len = seq_len[0]
        self.vf_seq_len = seq_len[1]
        self.af_model_name = model_name[0]
        self.vf_model_name = model_name[1]
        self.af_last_hidden_dim = last_hidden_dim[0]
        self.vf_last_hidden_dim = last_hidden_dim[1]
        assert self.af_last_hidden_dim == self.vf_last_hidden_dim, "两个模态的最后隐藏层维度必须相同"
        self.af_input_size = input_size[0]
        self.vf_input_size = input_size[1]
        self.enable_classifier = enable_classifier

        if self.af_model_name == "lightsernet":
            assert (
                self.af_last_hidden_dim == 320
            ), "由原论文可知 情绪识别中的 LightSerNet 的隐藏层向量最好限定为 320"

        self.audio_feature_extractor = LightResMultiSerNet_Encoder()
        self.video_feature_extractor = getVideoModel(
            self.vf_model_name,
            out_features=self.vf_last_hidden_dim,
            in_channels=3,
            num_frames=10,
            input_size=self.vf_input_size,
            pretrained=True,
        )

        self.af_esaAttention = ESIAttention(self.af_last_hidden_dim, 4)
        self.vf_esaAttention = ESIAttention(self.vf_last_hidden_dim, 4)

        # self.af_esaAttention = BiLSTM_ESIAttention(
        #     input_size=self.af_last_hidden_dim, hidden_size=self.af_last_hidden_dim
        # )
        # self.vf_esaAttention = BiLSTM_ESIAttention(
        #     input_size=self.vf_last_hidden_dim, hidden_size=self.vf_last_hidden_dim
        # )

        mid_hidden_dim = int((self.af_last_hidden_dim + self.vf_last_hidden_dim) / 2)
        self.fusion_feature = nn.Linear(
            in_features=self.af_last_hidden_dim + self.vf_last_hidden_dim,
            out_features=mid_hidden_dim,
        )
        self.af_feature2prob = nn.Linear(
            in_features=mid_hidden_dim, out_features=mid_hidden_dim
        )
        self.vf_feature2prob = nn.Linear(
            in_features=mid_hidden_dim, out_features=mid_hidden_dim
        )
        self.common_feature2prob = nn.Linear(
            in_features=mid_hidden_dim, out_features=mid_hidden_dim
        )

        self.classifier = nn.Linear(
            in_features=self.af_last_hidden_dim
            + mid_hidden_dim
            + self.vf_last_hidden_dim,
            out_features=num_classes,
        )

    def forward(self, af: Tensor, vf: Tensor, af_len=None, vf_len=None) -> Tensor:
        """
        Args:
            af (Tensor): size:[B, C,Seq, F]
            vf (Tensor): size:[B, C, Seq, W, H]
            af_len (Sequence, optional): size:[B]. Defaults to None.
            vf_len (Sequence, optional): size:[B]. Defaults to None.

        Returns:
            Tensor: size:[batch, out]
        """
        af = af.permute(0, 3, 1, 2)  # [B, C, Seq, F]
        vf = vf.permute(0, 4, 1, 2, 3)  # [B, C, Seq, F]

        # 处理语音数据

        af = torch.split(af, self.af_seq_len, dim=2)  # 分割数据段

        # af_seg_num * [B, C, af_seq_len, F]

        # 避免最后一个数据段长度太短
        af_floor_ = False
        if af[-1].shape[2] != af[0].shape[2]:
            af = af[:-1]
            af_floor_ = True  # 控制计算有效长度时是否需要去掉最后一段数据
        # 记录分段数量, af_seg_num = ceil or floor(seq/af_seq_len)
        af_seg_num = len(af)
        af = torch.stack(af).permute(1, 0, 2, 3, 4).contiguous()
        # [B, af_seg_num, C, af_seq_len, F]

        # [B * af_seg_num, C, af_seq_len, F]

        af_fea = self.audio_feature_extractor(af.view((-1,) + af.size()[2:]))
        # [B * af_seg_num, F]
        af_fea = af_fea.view((-1, af_seg_num) + af_fea.size()[1:])
        # [B, af_seg_num, F]
        if af_len is not None:  # 如果提供了原数据的有效 Seq 长度
            batch_size, max_len, _ = af_fea.shape
            # 计算每一段实际的有效数据段数量
            af_len = (
                torch.floor(af_len / self.af_seq_len)
                if af_floor_
                else torch.ceil(af_len / self.af_seq_len)
            )
            af_mask = (
                torch.arange(max_len, device=af_len.device).expand(batch_size, max_len)
                >= af_len[:, None]
            )
            af_fea[af_mask] = 0.0

        # 处理视频数据
        # [B, C, Seq, W, H]
        vf = torch.split(vf, self.vf_seq_len, dim=2)  # 分割数据段

        # vf_seg_len * [B, C, vf_seq_len, W, H]

        # 避免最后一个数据段长度太短
        vf_floor_ = False
        if vf[-1].shape[2] != vf[0].shape[2]:
            vf = vf[:-1]
            vf_floor_ = True  # 控制计算有效长度时是否需要去掉最后一段数据
        vf_seg_len = len(vf)  # 记录分段数量, vf_seg_len = ceil(seq/vf_seg_len)

        # vf: [B, vf_seg_len, vf_seq_len, C, W, H]
        vf = torch.stack(vf).permute(1, 0, 3, 2, 4, 5).contiguous()

        # in: [B * vf_seg_len, vf_seq_len, C, W, H], out: [B * vf_seg_len, F]
        vf_fea = self.video_feature_extractor(vf.view((-1,) + vf.size()[-4:]))

        # vf_fea: [B * vf_seg_len, F] -> [B, vf_seg_len, F]
        vf_fea = vf_fea.view((-1, vf_seg_len) + vf_fea.size()[1:])
        
        # 二维

        ################################
        # # [B * vf_seg_len * vf_seq_len, C, W, H]
        # vf_fea = self.video_feature_extractor(vf.view((-1,) + vf.size()[-3:]))
        # # [B * vf_seg_len * vf_seq_len, F]
        # vf_fea = vf_fea.view((-1, vf_seg_len, self.vf_seq_len) + vf_fea.size()[1:])
        # # [B, vf_seg_len, vf_seq_len, F]
        # vf_fea = vf_fea.mean(dim=2)
        # # [B, vf_seg_len, F]
        ################################

        if vf_len is not None:
            batch_size, max_len, _ = vf_fea.shape
            vf_len = (
                torch.floor(vf_len / self.vf_seq_len)
                if vf_floor_
                else torch.ceil(vf_len / self.vf_seq_len)
            )
            vf_mask = (
                torch.arange(max_len, device=vf_len.device).expand(batch_size, max_len)
                >= vf_len[:, None]
            )
            vf_fea[vf_mask] = 0.0

        # 中间融合

        seg_len = min(vf_seg_len, af_seg_num)
        fusion_fea = torch.cat([vf_fea[:, :seg_len, :], af_fea[:, :seg_len, :]], dim=-1)
        # [B, seg_len, 2*F]
        common_feature = self.fusion_feature(fusion_fea.detach())

        af_fea = self.af_esaAttention(af_fea, common_feature)
        vf_fea = self.vf_esaAttention(vf_fea, common_feature)

        common_feature = common_feature.mean(dim=1)
        af_fea = af_fea.mean(dim=1)
        vf_fea = vf_fea.mean(dim=1)
        ##############################
        # distance_af = F.pairwise_distance(common_feature, af_fea)
        # distance_vf = F.pairwise_distance(common_feature, vf_fea)
        # dis_loss = torch.mean(distance_af, distance_vf)

        # # similarity loss, 鼓励提取相似性特征

        ##############################
        af_p = F.softmax(self.af_feature2prob(af_fea))
        vf_p = F.softmax(self.vf_feature2prob(vf_fea))
        common_p = F.softmax(self.common_feature2prob(common_feature))
        ac_klSim = F.kl_div(common_p.log(), af_p, reduction="batchmean")
        vc_klSim = F.kl_div(common_p.log(), vf_p, reduction="batchmean")
        sim_loss = (ac_klSim + vc_klSim) / 2
        ##############################
        # epsilon = 1e-10
        # mean_common = common_feature.mean(dim=0)
        # std_common = common_feature.std(dim=0)

        # mean_af = af_fea.mean(dim=0)
        # std_af = af_fea.std(dim=0)

        # mean_vf = vf_fea.mean(dim=0)
        # std_vf = vf_fea.std(dim=0)
        # common_feature_gaussian = (common_feature - mean_common) / (
        #     std_common + epsilon
        # )
        # af_fea_gaussian = (af_fea - mean_af) / (std_af + epsilon)
        # vf_fea_gaussian = (vf_fea - mean_vf) / (std_vf + epsilon)

        # cos_similarity_common_af = F.cosine_similarity(
        #     common_feature_gaussian, af_fea_gaussian, dim=1
        # )
        # cos_similarity_common_vf = F.cosine_similarity(
        #     common_feature_gaussian, vf_fea_gaussian, dim=1
        # )
        # sim_loss = torch.mean((cos_similarity_common_af + cos_similarity_common_vf)/2 ** 2)

        #####
        # [B, 2*F]
        joint_fea = torch.cat([af_fea, common_feature, vf_fea], dim=1)
        # 分类
        if self.enable_classifier:
            out = self.classifier(joint_fea)
            return out, sim_loss, torch.tensor(0.0)
        else:
            return joint_fea

    def normal(self, x):
        std, mean = torch.std_mean(x, -1, unbiased=False)
        return (x - mean[:, None]) / (std[:, None] + 1e-5)


# 分段融合取平均, 音视频编码后提取通过相似性信息进行特征补充, 并通过注意力机制进行段间加权
class MER_SISF_Conv2D_SVC_InterFusion_Joint_Attention_SLoss_Luck(nn.Module):
    """
    paper: MER_SISF: Multimodal interframe feature fusion network
    """

    def __init__(
        self,
        num_classes=4,
        model_name=["lightsernet", "resnet50"],
        seq_len=[189, 30],
        last_hidden_dim=[320, 320],
        input_size=[[126, 40], [150, 150]],
        enable_classifier=True,
    ) -> None:
        super().__init__()
        self.af_seq_len = seq_len[0]
        self.vf_seq_len = seq_len[1]
        self.af_model_name = model_name[0]
        self.vf_model_name = model_name[1]
        self.af_last_hidden_dim = last_hidden_dim[0]
        self.vf_last_hidden_dim = last_hidden_dim[1]
        assert self.af_last_hidden_dim == self.vf_last_hidden_dim, "两个模态的最后隐藏层维度必须相同"
        self.af_input_size = input_size[0]
        self.vf_input_size = input_size[1]
        self.enable_classifier = enable_classifier

        if self.af_model_name == "lightsernet":
            assert (
                self.af_last_hidden_dim == 320
            ), "由原论文可知 情绪识别中的 LightSerNet 的隐藏层向量最好限定为 320"

        self.audio_feature_extractor = LightResMultiSerNet_Encoder()
        self.video_feature_extractor = PretrainedBaseModel(
                in_channels=3,
                out_features=self.vf_last_hidden_dim,
                model_name=self.vf_model_name,
                input_size=self.vf_input_size,
                num_frames=10,
                before_dropout=0,
                before_softmax=False,
            )()

        self.af_esaAttention = ESIAttention(self.af_last_hidden_dim, 4)
        self.vf_esaAttention = ESIAttention(self.vf_last_hidden_dim, 4)

        mid_hidden_dim = int((self.af_last_hidden_dim + self.vf_last_hidden_dim) / 2)
        self.fusion_feature = nn.Linear(
            in_features=self.af_last_hidden_dim + self.vf_last_hidden_dim,
            out_features=mid_hidden_dim,
        )
        self.af_feature2prob = nn.Linear(
            in_features=mid_hidden_dim, out_features=mid_hidden_dim
        )
        self.vf_feature2prob = nn.Linear(
            in_features=mid_hidden_dim, out_features=mid_hidden_dim
        )
        self.common_feature2prob = nn.Linear(
            in_features=mid_hidden_dim, out_features=mid_hidden_dim
        )

        self.classifier = nn.Linear(
            in_features=self.af_last_hidden_dim
            + mid_hidden_dim
            + self.vf_last_hidden_dim,
            out_features=num_classes,
        )

    def forward(self, af: Tensor, vf: Tensor, af_len=None, vf_len=None) -> Tensor:
        """
        Args:
            af (Tensor): size:[B, C,Seq, F]
            vf (Tensor): size:[B, C, Seq, W, H]
            af_len (Sequence, optional): size:[B]. Defaults to None.
            vf_len (Sequence, optional): size:[B]. Defaults to None.

        Returns:
            Tensor: size:[batch, out]
        """
        af = af.permute(0, 3, 1, 2)  # [B, C, Seq, F]
        vf = vf.permute(0, 4, 1, 2, 3)  # [B, C, Seq, F]

        # 处理语音数据

        af = torch.split(af, self.af_seq_len, dim=2)  # 分割数据段

        # af_seg_num * [B, C, af_seq_len, F]

        # 避免最后一个数据段长度太短
        af_floor_ = False
        if af[-1].shape[2] != af[0].shape[2]:
            af = af[:-1]
            af_floor_ = True  # 控制计算有效长度时是否需要去掉最后一段数据
        # 记录分段数量, af_seg_num = ceil or floor(seq/af_seq_len)
        af_seg_num = len(af)
        af = torch.stack(af).permute(1, 0, 2, 3, 4).contiguous()
        # [B, af_seg_num, C, af_seq_len, F]

        # [B * af_seg_num, C, af_seq_len, F]

        af_fea = self.audio_feature_extractor(af.view((-1,) + af.size()[2:]))
        # [B * af_seg_num, F]
        af_fea = af_fea.view((-1, af_seg_num) + af_fea.size()[1:])
        # [B, af_seg_num, F]
        if af_len is not None:  # 如果提供了原数据的有效 Seq 长度
            batch_size, max_len, _ = af_fea.shape
            # 计算每一段实际的有效数据段数量
            af_len = (
                torch.floor(af_len / self.af_seq_len)
                if af_floor_
                else torch.ceil(af_len / self.af_seq_len)
            )
            af_mask = (
                torch.arange(max_len, device=af_len.device).expand(batch_size, max_len)
                >= af_len[:, None]
            )
            af_fea[af_mask] = 0.0

        # 处理视频数据
        # [B, C, Seq, W, H]
        vf = torch.split(vf, self.vf_seq_len, dim=2)  # 分割数据段

        # vf_seg_len * [B, C, vf_seq_len, W, H]

        # 避免最后一个数据段长度太短
        vf_floor_ = False
        if vf[-1].shape[2] != vf[0].shape[2]:
            vf = vf[:-1]
            vf_floor_ = True  # 控制计算有效长度时是否需要去掉最后一段数据
        vf_seg_len = len(vf)  # 记录分段数量, vf_seg_len = ceil(seq/vf_seg_len)

        # vf: [B, vf_seg_len, vf_seq_len, C, W, H]
        vf = torch.stack(vf).permute(1, 0, 3, 2, 4, 5).contiguous()

        # 二维
        # [B * vf_seg_len * vf_seq_len, C, W, H]
        vf_fea = self.video_feature_extractor(vf.view((-1,) + vf.size()[-3:]))
        # [B * vf_seg_len * vf_seq_len, F]
        vf_fea = vf_fea.view((-1, vf_seg_len, self.vf_seq_len) + vf_fea.size()[1:])
        # [B, vf_seg_len, vf_seq_len, F]
        vf_fea = vf_fea.mean(dim=2)
        # [B, vf_seg_len, F]

        if vf_len is not None:
            batch_size, max_len, _ = vf_fea.shape
            vf_len = (
                torch.floor(vf_len / self.vf_seq_len)
                if vf_floor_
                else torch.ceil(vf_len / self.vf_seq_len)
            )
            vf_mask = (
                torch.arange(max_len, device=vf_len.device).expand(batch_size, max_len)
                >= vf_len[:, None]
            )
            vf_fea[vf_mask] = 0.0

        # 中间融合

        seg_len = min(vf_seg_len, af_seg_num)
        fusion_fea = torch.cat([vf_fea[:, :seg_len, :], af_fea[:, :seg_len, :]], dim=-1)
        # [B, seg_len, 2*F]
        common_feature = self.fusion_feature(fusion_fea.detach())

        af_fea = self.af_esaAttention(af_fea, common_feature)
        vf_fea = self.vf_esaAttention(vf_fea, common_feature)

        common_feature = common_feature.mean(dim=1)
        af_fea = af_fea.mean(dim=1)
        vf_fea = vf_fea.mean(dim=1)

        af_p = F.softmax(self.af_feature2prob(af_fea))
        vf_p = F.softmax(self.vf_feature2prob(vf_fea))
        common_p = F.softmax(self.common_feature2prob(common_feature))
        ac_klSim = F.kl_div(common_p.log(), af_p, reduction="batchmean")
        vc_klSim = F.kl_div(common_p.log(), vf_p, reduction="batchmean")
        sim_loss = (ac_klSim + vc_klSim) / 2

        # [B, 2*F]
        joint_fea = torch.cat([af_fea, common_feature, vf_fea], dim=1)
        # 分类
        if self.enable_classifier:
            out = self.classifier(joint_fea)
            return out, sim_loss, torch.tensor(0.0)
        else:
            return joint_fea

    def normal(self, x):
        std, mean = torch.std_mean(x, -1, unbiased=False)
        return (x - mean[:, None]) / (std[:, None] + 1e-5)
