
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


class Wave_pad_cut(torch.nn.Module):
    r""" 根据提供的语音采样点数量, 填充或剪切输入语音.
    Args:
        samplingNum (int, optional):  -1, 不做处理. Defaults to -1.
        mode (str, optional): "left": 左填充, "right": 右填充, "biside": 两侧填充. Defaults to "right".
    """

    def __init__(self, samplingNum: int = -1, mode: str = "right") -> None:
        super(Wave_pad_cut, self).__init__()
        self.samplingNum = samplingNum
        self.mode = mode
        if self.samplingNum < -1:
            raise ValueError("采样点数量必须为正数或-1")

    def forward(self, waveform: Tensor) -> Tensor:
        r"""

        Args:
            waveform (Tensor): Audio waveform of dimension of `(c, time)`.

        Returns:
            Tensor: Waveform of dimension of `(c, time)`.
        """

        if self.samplingNum == -1:
            return waveform
        length = waveform.shape[-1]

        if length <= self.samplingNum:
            if self.mode == "left":
                waveform = F.pad(
                    waveform, (self.samplingNum-length, 0), 'constant', 0)
            if self.mode == "right":
                waveform = F.pad(
                    waveform, (0, self.samplingNum-length), 'constant', 0)
            if self.mode == "biside":
                left_pad = int((self.samplingNum-length)/2.0)
                right_pad = self.samplingNum-length-left_pad
                waveform = F.pad(
                    waveform, (left_pad, right_pad), 'constant', 0)
        elif length > self.samplingNum:
            if self.mode == "left":
                waveform = waveform[:, -self.samplingNum:]
            if self.mode == "right":
                waveform = waveform[:, :self.samplingNum]
            if self.mode == "biside":
                left_cut = int((length-self.samplingNum)/2.0)
                right_cut = length-self.samplingNum-left_cut
                waveform = waveform[:, left_cut:-right_cut].copy()
        return waveform


# class ComParE_2016(torch.nn.Module):
#     r"""
#     """

#     def __init__(self, sampling_rate) -> None:
#         super(Wave_pad_cut, self).__init__()
#         self.sampling_rate = sampling_rate

#     def forward(self, waveform: Tensor) -> Tensor:
#         r"""
#         Args:
#             waveform (Tensor): Audio waveform of dimension of `([channels, samples])`.
#         Returns:
#             Tensor: features of dimension of `([channels, features])`.
#         """
#         smile = opensmile.Smile(
#             feature_set=opensmile.FeatureSet.eGeMAPSv02,
#             feature_level=opensmile.FeatureLevel.Functionals,)
#         smile.process_signal(waveform.numpy(), self.sampling_rate)
#         return torch.from_numpy(smile.to_numpy())


if __name__ == '__main__':
    splitMFCC = MFCC(sample_rate=16000, n_mfcc=40, log_mels=True, melkwargs={
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
                    "onesided": True})
    wav = torch.randn((1, 16000))
    print(wav.shape)
    wav = splitMFCC(wav)
    print(wav.shape)
