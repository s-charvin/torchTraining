import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .lightserconv3dnet import LightSerConv3dNet
from .lightsernet import LightSerNet


class MIFFNet(nn.Module):
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

    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=1024, hidden_size=1024,
                            num_layers=2, bidirectional=True, bias=True)

    def forward(self, af: Tensor, vf: Tensor, af_len=None, vf_len=None) -> Tensor:
        """
        Args:
            af (Tensor): size:[seq, batch, channel, fea]
            vf (Tensor): size:[seq, batch, channel, w, h]
            af_len (Sequence, optional): size:[batch]. Defaults to None.
            vf_len (Sequence, optional): size:[batch]. Defaults to None.

        Returns:
            Tensor: size:[batch, out]
        """
        return af, vf
