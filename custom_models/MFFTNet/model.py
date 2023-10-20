
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import numpy as np

from .components import Conv2dSame, _get_ResMultiModulatedDeformConv2d_extractor, ESIAttention, EmptyModel, Conv2dLayerBlock, PretrainedBaseModel


class AudioEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.path1 = nn.Sequential(
            Conv2dSame(in_channels=1, out_channels=32, kernel_size=(
                11, 1), stride=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2)
        )
        self.path2 = nn.Sequential(
            Conv2dSame(in_channels=1, out_channels=32, kernel_size=(
                1, 9), stride=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2)
        )
        self.path3 = nn.Sequential(
            Conv2dSame(in_channels=1, out_channels=32, kernel_size=(
                3, 3), stride=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2)
        )

        self.conv_extractor = _get_ResMultiModulatedDeformConv2d_extractor(
            in_channels=32*3,
            out_channels=[64, 96,  128, 160]
        )

        self.gap = Conv2dLayerBlock(in_channels=160,
                                    out_channels=320,
                                    kernel_size=[1, 1],
                                    stride=1,
                                    bias=False,
                                    layer_norm=nn.BatchNorm2d(320),
                                    layer_poolling=nn.AdaptiveAvgPool2d(1),
                                    activate_func=nn.functional.relu)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: Tensor, feature_lens=None) -> Tensor:
        x = x.permute(0, 3, 1, 2)  # [B, C, Seq, F]
        out1 = self.path1(x)
        out2 = self.path2(x)
        out3 = self.path3(x)

        x = torch.cat([out1, out2, out3], dim=1).contiguous()

        x = self.conv_extractor(x)

        x = self.gap(x)
        x = self.dropout(x.squeeze(2).squeeze(2))
        return x, None


class VideoEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.extractor = PretrainedBaseModel(
            in_channels=3, out_features=320, model_name="resnet50", input_size=[150, 150], num_frames=30, before_dropout=0, before_softmax=False)()

    def forward(self, x: Tensor, feature_lens=None) -> Tensor:
        return self.extractor(x)


class AudioNet(nn.Module):
    def __init__(self, num_classes=4, model_name="lightsernet", last_hidden_dim=320, input_size=[126, 40], pretrained=False, enable_classifier=True) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.enable_classifier = enable_classifier
        self.model_name = model_name
        self.last_hidden_dim = last_hidden_dim
        self.audio_feature_extractor = AudioEncoder()

        self.audio_classifier = self.audio_feature_extractor.last_linear

        self.audio_feature_extractor.last_linear = EmptyModel()

    def forward(self,  af: Tensor, af_len=None) -> Tensor:
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


class VideoNet_Conv2D(nn.Module):
    def __init__(self, num_classes=4, enable_classifier=True) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.model_name = "resnet50"
        self.enable_classifier = enable_classifier
        self.last_hidden_dim = 320
        self.video_feature_extractor = VideoEncoder()
        self.video_classifier = nn.Linear(
            in_features=self.last_hidden_dim, out_features=num_classes)

    def forward(self,  vf: Tensor, vf_len=None) -> Tensor:
        """
        Args:
            vf (Tensor): size:[B, Seq, W, H, C]
            vf_len (Sequence, optional): size:[batch]. Defaults to None.
        Returns:
            Tensor: size:[batch, out]
        """
        seq_length = vf.shape[1]
        #  [B, Seq, W, H, C]
        vf = vf.permute(0, 1, 4, 2, 3).contiguous()
        # [B, Seq, C, W, H]
        vf_fea = self.video_feature_extractor(
            vf.view((-1, ) + vf.size()[-3:]))

        if self.enable_classifier:
            # [batch * seq, 320]
            vf_out = self.video_classifier(vf_fea)

            vf_out = vf_out.view((-1, seq_length) + vf_out.size()[1:])
            # [batch, seq, num_class]
            vf_out = vf_out.mean(dim=1)
            return F.softmax(vf_out, dim=1), None
        else:
            return vf_fea.view((-1, seq_length) + vf_fea.size()[1:]).mean(dim=1)


class MIFFTNet_Conv2D_SVC_InterFusion_Joint_Attention(nn.Module):
    """
    paper: MIFFNet: Multimodal interframe feature fusion network
    """

    def __init__(self, num_classes=4, seq_len=[189, 30], last_hidden_dim=[320, 320], input_size=[[126, 40], [150, 150]], enable_classifier=True) -> None:
        super().__init__()
        self.af_seq_len = seq_len[0]
        self.vf_seq_len = seq_len[1]

        self.af_last_hidden_dim = 320
        self.vf_last_hidden_dim = 320
        assert self.af_last_hidden_dim == self.vf_last_hidden_dim, "两个模态的最后隐藏层维度必须相同"
        self.af_input_size = input_size[0]
        self.vf_input_size = input_size[1]
        self.enable_classifier = enable_classifier

        self.audio_feature_extractor = AudioEncoder()
        self.video_feature_extractor = VideoEncoder()

        self.af_esaAttention = ESIAttention(
            self.af_last_hidden_dim, 4)
        self.vf_esaAttention = ESIAttention(
            self.vf_last_hidden_dim, 4)
        mid_hidden_dim = int(
            (self.af_last_hidden_dim+self.vf_last_hidden_dim)/2)
        self.fusion_feature = nn.Linear(
            in_features=self.af_last_hidden_dim+self.vf_last_hidden_dim, out_features=mid_hidden_dim)

        self.classifier = nn.Linear(
            in_features=self.af_last_hidden_dim+mid_hidden_dim+self.vf_last_hidden_dim, out_features=num_classes)

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

        af_fea = self.audio_feature_extractor(af.view((-1,)+af.size()[2:]))
        # [B * af_seg_num, F]
        af_fea = af_fea.view((-1, af_seg_num) + af_fea.size()[1:])
        # [B, af_seg_num, F]
        if af_len is not None:  # 如果提供了原数据的有效 Seq 长度
            batch_size, max_len, _ = af_fea.shape
            # 计算每一段实际的有效数据段数量
            af_len = torch.floor(
                af_len/self.af_seq_len) if af_floor_ else torch.ceil(af_len/self.af_seq_len)
            af_mask = torch.arange(max_len, device=af_len.device
                                   ).expand(batch_size, max_len) >= af_len[:, None]
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
        vf_fea = self.video_feature_extractor(
            vf.view((-1, ) + vf.size()[-3:]))
        # [B * vf_seg_len * vf_seq_len, F]
        vf_fea = vf_fea.view(
            (-1, vf_seg_len, self.vf_seq_len) + vf_fea.size()[1:])
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
            vf_len = torch.floor(
                vf_len/self.vf_seq_len) if vf_floor_ else torch.ceil(vf_len/self.vf_seq_len)
            vf_mask = torch.arange(max_len, device=vf_len.device
                                   ).expand(batch_size, max_len) >= vf_len[:, None]
            vf_fea[vf_mask] = 0.0

        # 中间融合

        seg_len = min(vf_seg_len, af_seg_num)
        fusion_fea = torch.cat(
            [vf_fea[:, :seg_len, :], af_fea[:, :seg_len, :]], dim=-1)
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
            return joint_fea,

    def normal(self, x):
        std, mean = torch.std_mean(x, -1, unbiased=False)
        return (x - mean[:, None]) / (std[:, None]+1e-5)
