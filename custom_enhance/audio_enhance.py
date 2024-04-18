
from torch import Tensor
import torch
import torch.nn.functional as F
import nlpaug.augmenter.audio as naa
import logging

class AudioSplit(torch.nn.Module):
    r"""输入字典格式存储的数据, 对其中 "auido" 中的每个语音进行处理, 语音单个数据格式为 [c, samples]

    Args:
        sample_rate (int, optional): Sample rate of audio signal. (Default: ``16000``)
    """

    def __init__(self,
                 sampleNum) -> None:
        super(AudioSplit, self).__init__()

        self.sampleNum = int(sampleNum)

    def forward(self, datadict: dict) -> Tensor:
        r"""
        Args:
            datadict (dict): 字典格式存储的数据.
        Returns:
            datadict: .
        """
        columns = datadict.keys()
        index = range(len(datadict["audio"]))
        newdatadict = {c: [] for c in columns}

        for i in index:
            waveform = datadict["audio"][i]  # 当前行的语音数据
            datadict["audio"][i] = None
            if waveform.shape[-1] > self.sampleNum:
                samples = int(
                    (waveform.shape[-1] // self.sampleNum) * self.sampleNum)
                waveform = waveform[:, :samples]
            waveform = torch.split(waveform, self.sampleNum, dim=1)
            for wav in waveform:
                for col in columns:
                    if col == "audio":
                        newdatadict[col].append(wav)
                    else:
                        try:
                            newdatadict[col].append(datadict[col][i])
                        except:
                            newdatadict[col].append(datadict[col])
        return newdatadict


class AVSplit(torch.nn.Module):
    r"""输入字典格式存储的数据, 对其中 "auido","video" 中的每个模态进行处理, 按照指定时长进行截取

    Args:
        sample_rate (int, optional): Sample rate of audio signal. (Default: ``16000``)
    """

    def __init__(self,
                 sampleNum,
                 video_sample) -> None:
        super(AVSplit, self).__init__()
        self.audio_samples = int(sampleNum)
        self.video_samples = int(video_sample)

    def forward(self, datadict: dict) -> Tensor:
        r"""
        Args:
            datadict (dict): 字典格式存储的数据.
        Returns:
            dataframe: .
        """
        columns = datadict.keys()
        index = range(len(datadict["path"]))
        newdatadict = {c: [] for c in columns}

        for i in index:
            waveform = datadict["audio"][i]  # 当前行的语音数据 [c,seq]
            video = datadict["video"][i]  # 当前行的视频数据 [seq,h,w,c] RGB
            datadict["video"][i] = None
            datadict["audio"][i] = None

            if waveform.shape[-1] > self.audio_samples:
                samples = int(  # 处理比指定时间段长的数据, 将多余的数据截取掉
                    (waveform.shape[-1] // self.audio_samples) * self.audio_samples)
                waveform = waveform[:, :samples]
            else:  # 填充比指定时间段短的数据
                waveform = F.pad(
                    waveform, (0, self.audio_samples-waveform.shape[-1]))
            waveform = torch.split(waveform, self.audio_samples, dim=1)

            if video.shape[0] > self.video_samples:
                samples = int(  # 处理比指定时间段长的数据, 将多余的数据截取掉
                    (video.shape[0] // self.video_samples) * self.video_samples)
                video = video[:samples, :, :, :]
            else:  # 填充比指定时间段短的数据
                video = F.pad(
                    video, (0, 0, 0, 0, 0, 0, 0, self.video_samples-video.shape[0]))
            video = torch.split(video, self.video_samples, dim=0)

            for wav, vid in zip(waveform, video):
                for col in columns:
                    if col == "audio":
                        newdatadict[col].append(wav)
                    elif col == "video":
                        newdatadict[col].append(vid)
                    else:
                        try:
                            newdatadict[col].append(datadict[col][i])
                        except:
                            newdatadict[col].append(datadict[col])
        del datadict
        return newdatadict


if __name__ == '__main__':
    datadict = {
        "path": [1, 2, 3, 4, 5, 6],
        "video": [torch.randn((150, 120, 120, 3)), torch.randn((150, 120, 120, 3)), torch.randn((150, 120, 120, 3)), torch.randn((150, 120, 120, 3)), torch.randn((150, 120, 120, 3)), torch.randn((150, 120, 120, 3))],
        "audio": [torch.randn((1, 192000)), torch.randn((1, 192000)), torch.randn((1, 192000)), torch.randn((1, 192000)), torch.randn(
            (1, 192000)), torch.randn((1, 192000))],
        "sample_rate": 16000,
        "text": ["asdasdasdasd", "asdasdasdasd", "asdasdasdasd", "asdasdasdasd", "asdasdasdasd", "asdasdasdasd"],
        "labelid": [1, 2, 1, 3, 2, 2],
        "val/act/dom": [(3.5, 2.5, 1.5), (3.5, 2.5, 1.5), (3.5, 2.5, 1.5), (3.5, 2.5, 1.5), (3.5, 2.5, 1.5), (3.5, 2.5, 1.5), ],
        "gender": ["F", "M", "F", "F", "M", "M", ],
        "label": [1, 2, 1, 3, 2, 2]}
    # audioSplit = VtlpAug(sample_rate=16000, frame_rate=29.97, durations=7.0)

    audioAug = globals()['VtlpAug'](**{'sample_rate': 16000, 'n': 7})
    dataframe = audioAug(datadict)
    logging.info(dataframe)
