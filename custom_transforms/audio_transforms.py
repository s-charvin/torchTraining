
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

from spafe.features.mfcc import imfcc
from spafe.features.lfcc import lfcc
from spafe.utils.preprocessing import SlidingWindow

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

class IMFCC(torch.nn.Module):
    def __init__(self, 
                 sample_rate=16000, 
                 n_mfcc=40, 
                 n_fft=400, 
                 window_fn="hanning",
                 win_length = 1024,
                 hop_length = 256,
                 f_min=0,
                 f_max=None, 
                 n_mels=128, 
                 pre_emph=True, 
                 pre_emph_coeff=0.97,
                 scale='constant', 
                 dct_type=2, 
                 use_energy=False, 
                 lifter=0, 
                 normalize="mvn", 
                 fbanks=None, 
                 conversion_approach='Oshaghnessy'):
        super(IMFCC, self).__init__()
        self.fs = sample_rate
        self.num_ceps = n_mfcc
        self.pre_emph = pre_emph
        self.pre_emph_coeff = pre_emph_coeff
        self.window = SlidingWindow(win_length/sample_rate, hop_length/sample_rate, window_fn)
        self.nfilts = n_mels
        self.nfft = n_fft
        self.low_freq = f_min
        self.high_freq = f_max
        self.scale = scale
        self.dct_type = dct_type
        self.use_energy = use_energy
        self.lifter = lifter
        self.normalize = normalize
        self.fbanks = fbanks
        self.conversion_approach = conversion_approach

    def transform(self, waveform:Tensor):
        """Transform waveform using IMFCC feature extraction."""
        # [1, F]
        result = imfcc(sig=waveform[0].numpy(), fs=self.fs, num_ceps=self.num_ceps, pre_emph=self.pre_emph, 
                     pre_emph_coeff=self.pre_emph_coeff, window=self.window, nfilts=self.nfilts, 
                     nfft=self.nfft, low_freq=self.low_freq, high_freq=self.high_freq, scale=self.scale, 
                     dct_type=self.dct_type, use_energy=self.use_energy, lifter=self.lifter, 
                     normalize=self.normalize, fbanks=self.fbanks, 
                     conversion_approach=self.conversion_approach)
        result = torch.from_numpy(result).T
        return result.unsqueeze(dim=0).float()

    def forward(self, datadict):
        """
        Process audio data in the dictionary format using IMFCC feature extraction.
        
        Args:
            datadict (dict): Dictionary containing audio data.
        
        Returns:
            dict: Dictionary with the processed audio data.
        """
        index = range(len(datadict["audio"]))
        for i in index:
            datadict["audio"][i] = self.transform(datadict["audio"][i])
        return datadict

class LFCC(torch.nn.Module):
    def __init__(self,
                 sample_rate=16000, 
                 n_mfcc=40,
                 n_fft=400,
                 window_fn="hanning",
                 win_length = 1024,
                 hop_length = 256,
                 f_min=0,
                 f_max=None, 
                 n_mels=128, 
                 pre_emph=True, 
                 pre_emph_coeff=0.97,
                 scale='constant', 
                 dct_type=2, 
                 use_energy=False, 
                 lifter=0, 
                 normalize="mvn", 
                 fbanks=None):
        super(LFCC, self).__init__()
        self.fs = sample_rate
        self.num_ceps = n_mfcc
        self.pre_emph = pre_emph
        self.pre_emph_coeff = pre_emph_coeff
        self.window = SlidingWindow(win_length/sample_rate, hop_length/sample_rate, window_fn)
        self.nfilts = n_mels
        self.nfft = n_fft
        self.low_freq = f_min
        self.high_freq = f_max
        self.scale = scale
        self.dct_type = dct_type
        self.use_energy = use_energy
        self.lifter = lifter
        self.normalize = normalize
        self.fbanks = fbanks

    def transform(self, waveform:Tensor):
        """Transform waveform using IMFCC feature extraction."""
        # [1, F]
        result = lfcc(sig=waveform[0].numpy(), fs=self.fs, num_ceps=self.num_ceps, pre_emph=self.pre_emph, 
                     pre_emph_coeff=self.pre_emph_coeff, window=self.window, nfilts=self.nfilts, 
                     nfft=self.nfft, low_freq=self.low_freq, high_freq=self.high_freq, scale=self.scale, 
                     dct_type=self.dct_type, use_energy=self.use_energy, lifter=self.lifter, 
                     normalize=self.normalize, fbanks=self.fbanks)
        result = torch.from_numpy(result).T
        return result.unsqueeze(dim=0).float()

    def forward(self, datadict):
        """
        Process audio data in the dictionary format using IMFCC feature extraction.
        
        Args:
            datadict (dict): Dictionary containing audio data.
        
        Returns:
            dict: Dictionary with the processed audio data.
        """
        index = range(len(datadict["audio"]))
        for i in index:
            datadict["audio"][i] = self.transform(datadict["audio"][i])
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

    audioAug1 = globals()['MFCC'](**{
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
    
    audioAug2 = globals()['LFCC'](**{
        "sample_rate": 16000,
        "n_mfcc": 40,
        "n_fft": 1024,
        "win_length": 1024,
        "hop_length": 256,
        "f_min": 80.,
        "f_max": 7600.,
        "n_mels": 128,
        "pre_emph": True,
        "pre_emph_coeff":  0.97,
        },)

    # # audioAug = WaveCopyPad(sampleNum=48000)
    # dataframe = audioAug1(datadict)
    # print(dataframe)
    
    
    # audioAug = WaveCopyPad(sampleNum=48000)
    dataframe = audioAug2(datadict)
    print(dataframe)