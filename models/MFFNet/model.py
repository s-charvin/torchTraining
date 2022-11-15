from tkinter.tix import Tree
import torch
from torch import Tensor, nn
import torch.nn.functional as F


from components import *
from .lightserconv3dnet import LightSerConv3dNet
from .lightsernet import LightSerNet


class PretrainedBaseModel(nn.Module):
    def __init__(self, in_channels=3, num_class: int = 4, base_model: str = 'resnet50', before_dropout=0.5, before_softmax=False):
        self.in_channels = in_channels
        assert (before_dropout >= 0 and num_class >= 0), ''
        assert (num_class == 0 and before_softmax)
        self.before_dropout = before_dropout
        self.before_softmax = before_softmax
        self._prepare_base_model(base_model)  # 构建基础模型
        feature_dim = self._change_model_out(num_class)
        self.base_model = self._construct__model(self.base_model)

    def _prepare_base_model(self, base_model):
        # 加载已经训练好的模型
        import pretrainedmodels
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
            torch.nn.init.normal(self.new_fc.weight, 0, std)
            torch.nn.init.constant(self.new_fc.bias, 0)
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

    def forward(self, input):
        # size: [B,C,H,W]
        base_out = self.base_model(input)  # [B,C,H,W]->[B,F]
        if self.dropout > 0:
            base_out = self.new_fc(base_out)
        if self.before_softmax:
            base_out = self.softmax(base_out)
        return base_out.squeeze()


class MIFFNet_Conv3D(nn.Module):
    """
    paper: MIFFNet: Multimodal interframe feature fusion network
    """

    def __init__(self) -> None:
        super().__init__()
        self.audio_feature_extractor = LightSerNet()
        self.video_feature_extractor = LightSerConv3dNet()
        self.audio_classifier = nn.Linear(in_features=320, out_features=4)
        self.video_classifier = nn.Linear(in_features=320, out_features=4)

    def forward(self, af: Tensor, vf: Tensor, af_len=None, vf_len=None) -> Tensor:
        """
        Args:
            af (Tensor): size:[batch, channel,seq, fea]
            vf (Tensor): size:[batch, channel,seq, w, h]
            af_len (Sequence, optional): size:[batch]. Defaults to None.
            vf_len (Sequence, optional): size:[batch]. Defaults to None.

        Returns:
            Tensor: size:[batch, out]
        """
        # 分割数据
        af_seqlen = 188*3
        vf_seqlen = 10*3
        # [ceil(seq/af_seqlen), batch, 1, af_seqlen, fea]
        af = torch.split(af, af_seqlen, dim=2)
        # [ceil(seq/vf_seqlen),batch, 3, vf_seqlen,  w, h]
        vf = torch.split(vf, vf_seqlen, dim=2)

        # 避免数据块 seq 长度太短
        af_floor_ = False
        vf_floor_ = False
        if af[-1].shape[2] < 50:
            af = af[:-1]  # 避免语音块 seq 长度太短
            af_floor_ = True

        if vf[-1].shape[2] < 3:
            vf = vf[:-1]  # 避免语音块 seq 长度太短
            vf_floor_ = True

        # 处理语音数据
        af_fea = []
        for i in af:
            af_fea.append(self.audio_feature_extractor(i))
        af_fea = torch.stack(af_fea).permute(
            1, 0, 2)  # [batch, ceil(seq/af_seqlen), fea]
        if af_len is not None:
            batch_size, max_len, _ = af_fea.shape
            af_len = torch.floor(
                af_len/af_seqlen) if af_floor_ else torch.ceil(af_len/af_seqlen)
            af_mask = torch.arange(max_len, device=af_len.device
                                   ).expand(batch_size, max_len) >= af_len[:, None]
            af_fea[af_mask] = 0.0
        # [batch, ceil(seq/af_seqlen), fea] -> (batch,ceil(seq/af_seqlen), 1)
        af_fea = af_fea.mean(dim=1)  # [batch, fea]
        af_fea = self.audio_classifier(af_fea)

        # 处理视频数据
        vf_fea = []
        for i in vf:
            vf_fea.append(self.video_feature_extractor(i))
        vf_fea = torch.stack(vf_fea).permute(
            1, 0, 2)  # [batch, ceil(seq/vf_seqlen), fea]
        if vf_len is not None:
            batch_size, max_len, _ = vf_fea.shape
            vf_len = torch.floor(
                vf_len/vf_seqlen) if vf_floor_ else torch.ceil(vf_len/vf_seqlen)
            vf_mask = torch.arange(max_len, device=vf_len.device
                                   ).expand(batch_size, max_len) >= vf_len[:, None]
            vf_fea[vf_mask] = 0.0

        vf_fea = vf_fea.mean(dim=1)  # [batch, fea]
        vf_fea = self.video_classifier(vf_fea)

        return F.softmax(af_fea, dim=1), F.softmax(vf_fea, dim=1)


class MIFFNet_Conv3D_GRU(nn.Module):
    """
    paper: MIFFNet: Multimodal interframe feature fusion network
    """

    def __init__(self) -> None:
        super().__init__()
        self.audio_feature_extractor = LightSerNet()
        self.video_feature_extractor = LightSerConv3dNet()
        self.audio_classifier = nn.Linear(in_features=320, out_features=4)
        self.video_classifier = nn.Linear(in_features=320, out_features=4)
        self.audio_emotional_GRU = nn.GRU(input_size=320, hidden_size=64,
                                          num_layers=1, batch_first=True)
        self.audio_weight = nn.Linear(64, 1)
        self.video_emotional_GRU = nn.GRU(input_size=320, hidden_size=64,
                                          num_layers=1, batch_first=True)
        self.video_weight = nn.Linear(64, 1)

    def forward(self, af: Tensor, vf: Tensor, af_len=None, vf_len=None) -> Tensor:
        """
        Args:
            af (Tensor): size:[batch, channel,seq, fea]
            vf (Tensor): size:[batch, channel,seq, w, h]
            af_len (Sequence, optional): size:[batch]. Defaults to None.
            vf_len (Sequence, optional): size:[batch]. Defaults to None.

        Returns:
            Tensor: size:[batch, out]
        """
        # 分割数据
        af_seqlen = 188*3
        vf_seqlen = 10*3
        # [ceil(seq/af_seqlen), batch, 1, af_seqlen, fea]
        af = torch.split(af, af_seqlen, dim=2)
        # [ceil(seq/vf_seqlen),batch, 3, vf_seqlen,  w, h]
        vf = torch.split(vf, vf_seqlen, dim=2)

        # 避免数据块 seq 长度太短
        af_floor_ = False
        vf_floor_ = False
        if af[-1].shape[2] < 50:
            af = af[:-1]  # 避免语音块 seq 长度太短
            af_floor_ = True

        if vf[-1].shape[2] < 3:
            vf = vf[:-1]  # 避免语音块 seq 长度太短
            vf_floor_ = True

        # 处理语音数据
        af_fea = []
        for i in af:
            af_fea.append(self.audio_feature_extractor(i))
        af_fea = torch.stack(af_fea).permute(
            1, 0, 2)  # [batch, ceil(seq/af_seqlen), fea]
        if af_len is not None:
            batch_size, max_len, _ = af_fea.shape
            af_len = torch.floor(
                af_len/af_seqlen) if af_floor_ else torch.ceil(af_len/af_seqlen)
            af_mask = torch.arange(max_len, device=af_len.device
                                   ).expand(batch_size, max_len) >= af_len[:, None]
            af_fea[af_mask] = 0.0
        # [batch, ceil(seq/af_seqlen), fea] -> (batch,ceil(seq/af_seqlen), 1)
        audio_weight = self.audio_emotional_GRU(af_fea)[0]
        audio_weight = self.audio_weight(audio_weight)
        audio_weight = F.softmax(audio_weight, dim=1)
        af_fea = af_fea * audio_weight
        af_fea = af_fea.mean(dim=1)  # [batch, fea]
        af_fea = self.audio_classifier(af_fea)

        # 处理视频数据
        vf_fea = []
        for i in vf:
            vf_fea.append(self.video_feature_extractor(i))
        vf_fea = torch.stack(vf_fea).permute(
            1, 0, 2)  # [batch, ceil(seq/vf_seqlen), fea]
        if vf_len is not None:
            batch_size, max_len, _ = vf_fea.shape
            vf_len = torch.floor(
                vf_len/vf_seqlen) if vf_floor_ else torch.ceil(vf_len/vf_seqlen)
            vf_mask = torch.arange(max_len, device=vf_len.device
                                   ).expand(batch_size, max_len) >= vf_len[:, None]
            vf_fea[vf_mask] = 0.0

        video_weight = self.video_emotional_GRU(vf_fea)[0]
        video_weight = self.video_weight(video_weight)
        video_weight = F.softmax(video_weight, dim=1)
        vf_fea = vf_fea * video_weight
        vf_fea = vf_fea.mean(dim=1)  # [batch, fea]
        vf_fea = self.video_classifier(vf_fea)

        return F.softmax(af_fea, dim=1), F.softmax(vf_fea, dim=1)


class MIFFNet_Conv2D(nn.Module):
    """
    paper: MIFFNet: Multimodal interframe feature fusion network
    """

    def __init__(self) -> None:
        super().__init__()
        self.audio_feature_extractor = LightSerNet()
        self.audio_classifier = nn.Linear(in_features=320, out_features=4)

        self.video_feature_extractor = PretrainedBaseModel(
            in_channels=3, num_class=320, base_model='resnet50', before_dropout=0, before_softmax=False)()
        self.video_classifier = nn.Linear(in_features=320, out_features=4)

    def forward(self, af: Tensor, vf: Tensor, af_len=None, vf_len=None) -> Tensor:
        """
        Args:
            af (Tensor): size:[batch, channel,seq, feature]
            vf (Tensor): size:[batch, channel,seq, w, h]
            af_len (Sequence, optional): size:[batch]. Defaults to None.
            vf_len (Sequence, optional): size:[batch]. Defaults to None.

        Returns:
            Tensor: size:[batch, out]
        """
        # 分割数据
        af_seqlen = 188*3
        # [ batch, 1, seq, fea]
        af = torch.split(af, af_seqlen, dim=2)
        # ceil(seq/af_seqlen) * [batch, 1, af_seqlen, fea]
        af = torch.stack(af).permute(1, 0, 2, 3, 4)
        # [batch, ceil(seq/af_seqlen), 1, af_seqlen, fea]
        af_seq_length = af.shape[1]
        # 避免数据块 seq 长度太短
        af_floor_ = False
        if af[-1].shape[2] < 50:
            af = af[:-1]  # 避免语音块 seq 长度太短
            af_floor_ = True
        # 处理语音数据
        af_fea = self.audio_feature_extractor(af.view((-1,)+af.size()[1:]))
        # [batch * ceil(seq/af_seqlen), feature]
        af_fea = af_fea.view((-1, af_seq_length) + vf_fea.size()[1:])
        # [batch, ceil(seq/af_seqlen), feature]
        if af_len is not None:
            batch_size, max_len, _ = af_fea.shape
            af_len = torch.floor(
                af_len/af_seqlen) if af_floor_ else torch.ceil(af_len/af_seqlen)
            af_mask = torch.arange(max_len, device=af_len.device
                                   ).expand(batch_size, max_len) >= af_len[:, None]
            af_fea[af_mask] = 0.0
        af_fea = af_fea.mean(dim=1)  # [batch, feature]
        af_fea = self.audio_classifier(af_fea)

        # 处理视频数据
        vf_seq_length = vf.shape[2]
        vf_fea = self.base_model(vf.permute(
            0, 2, 1, 3, 4).view((-1, self.in_channels) + input.size()[-3:]))
        # [batch * seq, feature]
        vf_fea = vf_fea.view((-1, vf_seq_length) + vf_fea.size()[1:])
        # [batch , seq, feature]
        if vf_len is not None:
            batch_size, max_len, _ = vf_fea.shape
            vf_mask = torch.arange(max_len, device=vf_len.device
                                   ).expand(batch_size, max_len) >= vf_len[:, None]
            vf_fea[vf_mask] = 0.0
        vf_fea = vf_fea.mean(dim=1)  # [batch, feature]
        vf_fea = self.video_classifier(vf_fea)
        return F.softmax(af_fea, dim=1), F.softmax(vf_fea, dim=1)


class MIFFNet_Conv2D_GRU(nn.Module):
    """
    paper: MIFFNet: Multimodal interframe feature fusion network
    """

    def __init__(self) -> None:
        super().__init__()
        self.audio_feature_extractor = LightSerNet()
        self.audio_classifier = nn.Linear(in_features=320, out_features=4)
        self.audio_emotional_GRU = nn.GRU(input_size=320, hidden_size=64,
                                          num_layers=1, batch_first=True)
        self.calc_audio_weight = nn.Linear(64, 1)

        self.video_feature_extractor = PretrainedBaseModel(
            in_channels=3, num_class=320, base_model='resnet50', before_dropout=0, before_softmax=False)()
        self.video_classifier = nn.Linear(in_features=320, out_features=4)
        self.video_emotional_GRU = nn.GRU(input_size=320, hidden_size=64,
                                          num_layers=1, batch_first=True)
        self.calc_video_weight = nn.Linear(64, 1)

    def forward(self, af: Tensor, vf: Tensor, af_len=None, vf_len=None) -> Tensor:
        """
        Args:
            af (Tensor): size:[batch, channel,seq, fea]
            vf (Tensor): size:[batch, channel,seq, w, h]
            af_len (Sequence, optional): size:[batch]. Defaults to None.
            vf_len (Sequence, optional): size:[batch]. Defaults to None.

        Returns:
            Tensor: size:[batch, out]
        """
        # 分割数据
        af_seqlen = 188*3
        # [ batch, 1, seq, fea]
        af = torch.split(af, af_seqlen, dim=2)
        # ceil(seq/af_seqlen) * [batch, 1, af_seqlen, fea]
        af = torch.stack(af).permute(1, 0, 2, 3, 4)
        # [batch, ceil(seq/af_seqlen), 1, af_seqlen, fea]
        af_seq_length = af.shape[1]

        # 避免数据块 seq 长度太短
        af_floor_ = False
        if af[-1].shape[2] < 50:
            af = af[:-1]  # 避免语音块 seq 长度太短
            af_floor_ = True

        # 处理语音数据
        af_fea = self.audio_feature_extractor(af.view((-1,)+af.size()[1:]))
        # [batch * ceil(seq/af_seqlen), feature]
        af_fea = af_fea.view((-1, af_seq_length) + vf_fea.size()[1:])
        # [batch, ceil(seq/af_seqlen), feature]
        if af_len is not None:
            batch_size, max_len, _ = af_fea.shape
            af_len = torch.floor(
                af_len/af_seqlen) if af_floor_ else torch.ceil(af_len/af_seqlen)
            af_mask = torch.arange(max_len, device=af_len.device
                                   ).expand(batch_size, max_len) >= af_len[:, None]
            af_fea[af_mask] = 0.0
        # [batch, ceil(seq/af_seqlen), feature]
        audio_weight = self.audio_emotional_GRU(af_fea)[0]
        audio_weight = self.calc_audio_weight(audio_weight)
        audio_weight = F.softmax(audio_weight, dim=1)
        # [batch, ceil(seq/af_seqlen), feature] * [batch, ceil(seq/af_seqlen), 1]
        af_fea = af_fea * audio_weight
        af_fea = af_fea.mean(dim=1)
        # [batch, feature]
        af_fea = self.audio_classifier(af_fea)

        # 处理视频数据
        seq_length = vf.shape[2]
        vf_fea = self.base_model(vf.permute(
            0, 2, 1, 3, 4).view((-1, self.in_channels) + input.size()[-3:]))
        # [batch * seq, feature]
        vf_fea = vf_fea.view((-1, seq_length) + vf_fea.size()[1:])
        # [batch , seq, feature]
        if vf_len is not None:
            batch_size, max_len, _ = vf_fea.shape
            vf_mask = torch.arange(max_len, device=vf_len.device
                                   ).expand(batch_size, max_len) >= vf_len[:, None]
            vf_fea[vf_mask] = 0.0

        video_weight = self.video_emotional_GRU(vf_fea)[0]
        video_weight = self.calc_video_weight(video_weight)
        video_weight = F.softmax(video_weight, dim=1)
        vf_fea = vf_fea * video_weight
        vf_fea = vf_fea.mean(dim=1)  # [batch, fea]
        vf_fea = self.video_classifier(vf_fea)
        return F.softmax(af_fea, dim=1), F.softmax(vf_fea, dim=1)


class MIFFNet_Conv2D_GRU_InterFusion(nn.Module):
    """
    paper: MIFFNet: Multimodal interframe feature fusion network
    """

    def __init__(self, num_class=4, af_seq_len=188*3, vf_seq_len=10*3) -> None:
        super().__init__()
        self.af_seq_len = af_seq_len
        self.vf_seq_len = vf_seq_len

        self.audio_feature_extractor = LightSerNet()
        self.video_feature_extractor = PretrainedBaseModel(
            in_channels=3, num_class=320, base_model='resnet50', before_dropout=0, before_softmax=False)()

        self.audio_emotional_GRU = nn.GRU(input_size=320, hidden_size=64,
                                          num_layers=1, batch_first=True)
        self.calc_audio_weight = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())
        self.video_emotional_GRU = nn.GRU(input_size=320, hidden_size=64,
                                          num_layers=1, batch_first=True)
        self.calc_video_weight = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())

        self.fusion_feature = nn.Linear(
            in_features=320*2, out_features=320)

        self.audio_classifier = nn.Linear(
            in_features=320*2, out_features=num_class)
        self.video_classifier = nn.Linear(
            in_features=320*2, out_features=num_class)
        self.fusion_classifier = nn.Linear(
            in_features=320*2, out_features=num_class)

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
        # 处理语音数据
        # [B, C, Seq, F]
        af = torch.split(af, self.af_seq_len, dim=2)  # 分割数据段
        af_seg_len = len(af)  # 记录分段数量, af_seg_len = ceil(seq/af_seq_len)
        # af_seg_len * [B, C, af_seq_len, F]

        # 避免最后一个数据段长度太短
        af_floor_ = False
        if len(af) > 1 and af[-1].shape[2] < 50:
            af = af[:-1]
            af_floor_ = True  # 控制计算有效长度时是否需要去掉最后一段数据

        af = torch.stack(af).permute(1, 0, 2, 3, 4)
        # [B, af_seg_len, C, af_seq_len, F]

        # [B * af_seg_len, C, af_seq_len, F]
        af_fea = self.audio_feature_extractor(
            af.view((-1,)+af.size()[2:]))
        # [B * af_seg_len, F]
        af_fea = af_fea.view((-1, af_seg_len) + vf_fea.size()[1:])
        # [B, af_seg_len, F]
        if af_len is not None:  # 如果提供了原数据的有效 Seq 长度
            batch_size, max_len, _ = af_fea.shape
            # 计算每一段实际的有效数据段数量
            af_len = torch.floor(
                af_len/self.af_seq_len) if af_floor_ else torch.ceil(af_len/self.af_seq_len)
            af_mask = torch.arange(max_len, device=af_len.device
                                   ).expand(batch_size, max_len) >= af_len[:, None]
            af_fea[af_mask] = 0.0
        audio_weight = self.audio_emotional_GRU(af_fea)[0]
        # [B, af_seg_len, F_]
        audio_weight = self.calc_audio_weight(audio_weight)
        # [B, af_seg_len, F] * [B, af_seg_len, 1]
        af_fea = af_fea * audio_weight
        # [B, af_seg_len, F]

        # 处理视频数据
        # [B, C, Seq, W, H]
        vf = torch.split(vf, self.vf_seq_len, dim=2)  # 分割数据段
        vf_seg_len = len(af)  # 记录分段数量, vf_seg_len = ceil(seq/vf_seg_len)
        # vf_seg_len * [B, C, vf_seq_len, W, H]

        # 避免最后一个数据段长度太短
        vf_floor_ = False
        if len(vf) > 1 and vf[-1].shape[2] < 3:
            vf = vf[:-1]
            vf_floor_ = True  # 控制计算有效长度时是否需要去掉最后一段数据
        vf = torch.stack(vf).permute(1, 0, 3, 2, 4, 5)
        # [B, vf_seg_len, vf_seq_len, C, W, H]
        # [B * vf_seg_len * vf_seq_len, C, W, H]
        vf_fea = self.base_model(
            vf.view((-1, ) + vf.size()[-3:]))
        # [B * vf_seg_len * vf_seq_len, F]
        vf_fea = vf_fea.view(
            (-1, vf_seg_len, self.vf_seq_len) + vf_fea.size()[1:])
        # [B, vf_seg_len, vf_seq_len, F]
        vf_fea = vf_fea.mean(dim=2)
        # [B, vf_seg_len, F]
        if vf_len is not None:
            batch_size, max_len, _ = vf_fea.shape
            vf_len = torch.floor(
                vf_len/self.vf_seq_len) if vf_floor_ else torch.ceil(vf_len/self.vf_seq_len)
            vf_mask = torch.arange(max_len, device=vf_len.device
                                   ).expand(batch_size, max_len) >= vf_len[:, None]
            vf_fea[vf_mask] = 0.0

        video_weight = self.video_emotional_GRU(vf_fea)[0]
        # [B, vf_seg_len, F_]
        video_weight = self.calc_video_weight(video_weight)
        # [B, vf_seg_len, F] * [B, vf_seg_len, 1]
        vf_fea = vf_fea * video_weight
        # [B, vf_seg_len, F]

        # 中间融合
        vf_seg_len
        seg_len = min(vf_seg_len, af_seg_len)
        fusion_fea = torch.cat(
            [vf_fea[:, :seg_len, :], af_fea[:, :seg_len, :]], dim=-1)
        # [B, seg_len, 2*F]

        common_feature = self.fusion_feature(fusion_fea)
        common_feature = common_feature.mean(dim=1)

        af_fea = af_fea.mean(dim=1)
        vf_fea = vf_fea.mean(dim=1)
        af_fea = torch.cat([af_fea, common_feature], dim=1)
        vf_fea = torch.cat([vf_fea, common_feature], dim=1)
        # [B, 2*F]

        # 分类
        af_out = self.audio_classifier(af_fea)
        vf_out = self.video_classifier(vf_fea)
        av_out = self.video_classifier(fusion_fea)
        return F.softmax(af_out, dim=1), F.softmax(vf_out, dim=1), F.softmax(av_out, dim=1)


class AudioIFFNet(nn.Module):
    """
    paper: AIFFNet: Audio interframe feature fusion network
    """

    def __init__(self) -> None:
        super().__init__()
        self.feature_extractor = LightSerNet()
        self.audio_classifier = nn.Linear(in_features=320, out_features=4)

    def forward(self, af: Tensor, af_len=None) -> Tensor:
        """
        Args:
            af (Tensor): size:[batch, 1, seq, fea]
            af_len (Sequence, optional): size:[batch]. Defaults to None.

        Returns:
            Tensor: size:[batch, out]
        """
        audio_fea = []
        af = torch.split(af, 438, dim=2)  # [ceil(seq/438), batch, 1, 438, fea]
        floor_ = False
        if af[-1].shape[2] < 50:
            af = af[:-1]  # 避免语音块 seq 长度太短
            floor_ = True
        for i in af:
            audio_fea.append(self.feature_extractor(i))
        audio_fea = torch.stack(audio_fea).permute(
            1, 0, 2)  # [batch, ceil(seq/188), fea]
        if af_len is not None:
            batch_size, max_len, _ = audio_fea.shape
            af_len = torch.floor(
                af_len/188) if floor_ else torch.ceil(af_len/188)
            mask = torch.arange(max_len, device=af_len.device
                                ).expand(batch_size, max_len) >= af_len[:, None]
            audio_fea[mask] = 0.0

        audio_fea = audio_fea.mean(dim=1)  # [batch, fea]
        audio_fea = self.audio_classifier(audio_fea)
        return F.softmax(audio_fea, dim=1)


class VideoIFFNet(nn.Module):
    """
    paper: VIFFNet: Video interframe feature fusion network
    """

    def __init__(self, in_channels=3) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.base_model = PretrainedBaseModel(
            in_channels=5, num_class=4, base_model='resnet50', before_dropout=0, before_softmax=True)

    def forward(self,  vf: Tensor, vf_len=None) -> Tensor:
        """
        Args:
            vf (Tensor): size:[batch, channel, seq, w, h]
            vf_len (Sequence, optional): size:[batch]. Defaults to None.
        Returns:
            Tensor: size:[batch, out]
        """
        seq_length = vf.shape[2]
        base_out = self.base_model(vf.permute(
            0, 2, 1, 3, 4).view((-1, self.in_channels) + input.size()[-3:]))
        # [batch * seq, feature]
        base_out = base_out.view((-1, seq_length) + base_out.size()[1:])
        base_out = base_out.mean(dim=1)
        return base_out
