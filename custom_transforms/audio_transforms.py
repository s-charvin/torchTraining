
try:
    import opensmile
except:
    Warning("opensmile 库加载失败, 这会导致无法提取 LLD 特征.")
from typing import Optional
from torch import Tensor
import torch
import torch.nn.functional as F
from torchaudio.transforms import (
    Spectrogram,
    GriffinLim,
    AmplitudeToDB,
    MelScale,
    InverseMelScale,
    MelSpectrogram,
    MFCC,
    MuLawEncoding,
    MuLawDecoding,
    Resample,
    ComplexNorm,
    TimeStretch,
    Fade,
    FrequencyMasking,
    TimeMasking,
    SlidingWindowCmn,
    Vad,
    SpectralCentroid)

from torchaudio.compliance.kaldi import fbank
from torchaudio.functional import (
    amplitude_to_DB,
    angle,
    complex_norm,
    compute_deltas,
    compute_kaldi_pitch,
    create_dct,
    create_fb_matrix,
    DB_to_amplitude,
    detect_pitch_frequency,
    griffinlim,
    magphase,
    mask_along_axis,
    mask_along_axis_iid,
    mu_law_encoding,
    mu_law_decoding,
    phase_vocoder,
    sliding_window_cmn,
    spectrogram,
    spectral_centroid,
    apply_codec,
)


class WaveCopyPad(torch.nn.Module):
    r""" 根据提供的语音采样点数量, 填充或剪切输入语音.
    Args:
        samplingNum (int, optional):  -1, 不做处理. Defaults to -1.
        mode (str, optional): "left": 左填充, "right": 右填充, "biside": 两侧填充. Defaults to "right".
    """

    def __init__(self, sampleNum: int = -1) -> None:
        super(WaveCopyPad, self).__init__()
        self.sampleNum = sampleNum
        if self.sampleNum < -1:
            raise ValueError("采样点数量必须为正数或-1")

    def transform(self, waveform):
        if self.sampleNum == -1:
            return waveform
        length = waveform.shape[-1]
        if length <= self.sampleNum:
            copy_num = (self.sampleNum-length)//length
            copy_d = self.sampleNum - copy_num * length - length
            for i in range(copy_num):
                waveform = torch.cat(
                    (waveform, waveform[:, :].clone()), 1).clone()
            waveform = torch.cat(
                (waveform, waveform[:, :copy_d].clone()), 1).clone()
        elif length > self.sampleNum:
            waveform = waveform[:, :self.sampleNum].clone()
        return waveform

    def forward(self, datadict: dict) -> Tensor:
        r"""
        Args:
            datadict (dict): 字典格式存储的数据.

        Returns:
            datadict: .
        """
        index = range(len(datadict["audio"]))
        for i in index:
            datadict["audio"][i] = self.transform(
                datadict["audio"][i])  # 处理当前行的语音数据
        return datadict


class WaveZeroPad(torch.nn.Module):
    r""" 根据提供的语音采样点数量, 填充或剪切输入语音.
    Args:
        samplingNum (int, optional):  -1, 不做处理. Defaults to -1.
        mode (str, optional): "left": 左填充, "right": 右填充, "biside": 两侧填充. Defaults to "right".
    """

    def __init__(self, sampleNum: int = -1, mode: str = "right") -> None:
        super(WaveZeroPad, self).__init__()
        self.samplingNum = sampleNum
        self.mode = mode
        if self.samplingNum < -1:
            raise ValueError("采样点数量必须为正数或-1")

    def transform(self, waveform):
        if self.samplingNum == -1:
            return waveform
        length = waveform.shape[-1]
        if length <= self.samplingNum:
            if self.mode == "left":
                waveform = F.pad(
                    waveform, (self.samplingNum-length, 0), 'constant', 0)
            elif self.mode == "right":
                waveform = F.pad(
                    waveform, (0, self.samplingNum-length), 'constant', 0)
            elif self.mode == "biside":
                left_pad = int((self.samplingNum-length)/2.0)
                right_pad = self.samplingNum-length-left_pad
                waveform = F.pad(
                    waveform, (left_pad, right_pad), 'constant', 0)
        elif length > self.samplingNum:
            if self.mode == "left":
                waveform = waveform[:, -self.samplingNum:].clone()
            elif self.mode == "right":
                waveform = waveform[:, :self.samplingNum].clone()
            elif self.mode == "biside":
                left_cut = int((length-self.samplingNum)/2.0)
                right_cut = length-self.samplingNum-left_cut
                waveform = waveform[:, left_cut:-right_cut].clone()
        return waveform

    def forward(self, datadict: dict) -> Tensor:
        r"""
        Args:
            datadict (dict): 字典格式存储的数据.

        Returns:
            datadict: .
        """
        index = range(len(datadict["audio"]))
        for i in index:
            datadict["audio"][i] = self.transform(
                datadict["audio"][i])  # 处理当前行的语音数据
        return datadict


class EmoBase_2010(torch.nn.Module):
    r"""
    """

    def __init__(self, sample_rate, feature_level) -> None:
        super(EmoBase_2010, self).__init__()
        self.sample_rate = sample_rate
        self.feature_level = feature_level  # 'lld', 'lld_de', 'func'

    def transform(self, waveform):
        smile = opensmile.Smile(
            feature_set="C:/Users/19115/Desktop/emobase2010.conf",
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors,)
        frame = smile.process_signal(waveform.numpy(), self.sample_rate)
        # return: [hannels, features, time].
        return torch.from_numpy(smile.to_numpy(frame))

    def forward(self, datadict: dict) -> Tensor:
        r"""
        Args:
            datadict (dict): 字典格式存储的数据.

        Returns:
            datadict: .
        """
        index = range(len(datadict["audio"]))
        for i in index:
            datadict["audio"][i] = self.transform(
                datadict["audio"][i])  # 处理当前行的语音数据
        return datadict


class Fbank(torch.nn.Module):
    r""" 
    """

    def __init__(self,) -> None:
        super(Fbank, self).__init__()

    def transform(self, waveform):
        return fbank(waveform)

    def forward(self, datadict: dict) -> Tensor:
        r"""
        Args:
            datadict (dict): 字典格式存储的数据.

        Returns:
            datadict: .
        """
        index = range(len(datadict["audio"]))
        for i in index:
            datadict["audio"][i] = self.transform(
                datadict["audio"][i])  # 处理当前行的语音数据
        return datadict


class AmplitudeToDB(AmplitudeToDB):
    r""" 
    """

    def transform(self, spec):
        return super(AmplitudeToDB, self).forward(spec)

    def forward(self, datadict: dict) -> Tensor:
        r"""
        Args:
            datadict (dict): 字典格式存储的数据.

        Returns:
            datadict: .
        """
        index = range(len(datadict["audio"]))
        for i in index:
            datadict["audio"][i] = self.transform(
                datadict["audio"][i])  # 处理当前行的语音数据
        return datadict


class MelSpectrogram(MelSpectrogram):
    r""" 
    """

    def transform(self, waveform):
        return super(MelSpectrogram, self).forward(waveform)

    def forward(self, datadict: dict) -> Tensor:
        r"""
        Args:
            datadict (dict): 字典格式存储的数据.

        Returns:
            datadict: .
        """
        index = range(len(datadict["audio"]))
        for i in index:
            datadict["audio"][i] = self.transform(
                datadict["audio"][i])  # 处理当前行的语音数据
        return datadict


class MFCC(MFCC):
    r""" 
    """

    def transform(self, waveform):
        return super(MFCC, self).forward(waveform)

    def forward(self, datadict: dict) -> Tensor:
        r"""
        Args:
            datadict (dict): 字典格式存储的数据.

        Returns:
            datadict: .
        """
        index = range(len(datadict["audio"]))
        for i in index:
            datadict["audio"][i] = self.transform(
                datadict["audio"][i])  # 处理当前行的语音数据
        return datadict


if __name__ == '__main__':
    datadict = {
        "path": [1, 2, 3, 4, 5, 6],
        "video": [torch.randn((150, 120, 120, 3)), torch.randn((150, 120, 120, 3)), torch.randn((150, 120, 120, 3)), torch.randn((150, 120, 120, 3)), torch.randn((150, 120, 120, 3)), torch.randn((150, 120, 120, 3))],
        "audio": [torch.randn((1, 192000)), torch.randn((1, 192000)), torch.randn((1, 192000)), torch.randn((1, 36005)), torch.randn(
            (1, 192000)), torch.randn((1, 192000))],
        "sample_rate": 16000,
        "text": ["asdasdasdasd", "asdasdasdasd", "asdasdasdasd", "asdasdasdasd", "asdasdasdasd", "asdasdasdasd"],
        "labelid": [1, 2, 1, 3, 2, 2],
        "val/act/dom": [(3.5, 2.5, 1.5), (3.5, 2.5, 1.5), (3.5, 2.5, 1.5), (3.5, 2.5, 1.5), (3.5, 2.5, 1.5), (3.5, 2.5, 1.5), ],
        "gender": ["F", "M", "F", "F", "M", "M", ],
        "label": [1, 2, 1, 3, 2, 2]}
    # audioSplit = VtlpAug(sample_rate=16000, frame_rate=29.97, durations=7.0)

    audioAug = globals()['MFCC'](**{
        "sample_rate": 16000,
        "n_mfcc": 40,
        "log_mels": True,
        "melkwargs": {
            "n_fft": 1024,
            "win_length": 1024,
            "hop_length": 256,
            "f_min": 80.,
            "f_max": 7600.,
            "pad": 0,
            "n_mels": 128,
            "power": 2.,
            "normalized": True,
            "center":  True,
            "pad_mode": "reflect",
            "onesided": True, },
    })

    # audioAug = WaveCopyPad(sampleNum=48000)
    dataframe = audioAug(datadict)
    print(dataframe)
