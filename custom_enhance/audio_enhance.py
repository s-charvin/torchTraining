
from typing import Optional
from unittest import result
from pandas import DataFrame
import pandas as pd
from torch import Tensor
import torch
import torch.nn.functional as F


class AudioSplit(torch.nn.Module):
    r"""输入字典格式存储的数据, 对其中 "auido" 中的每个语音进行处理, 语音单个数据格式为 [c, samples]

    Args:
        sample_rate (int, optional): Sample rate of audio signal. (Default: ``16000``)
    """

    def __init__(self,
                 sample_rate,
                 durations: float = 3.) -> None:
        super(AudioSplit, self).__init__()
        self.samples = int(sample_rate * durations)

    def forward(self, datadict: dict) -> Tensor:
        r"""
        Args:
            datadict (dict): 字典格式存储的数据.
        Returns:
            dataframe: .
        """
        columns = datadict.keys()
        index = range(len(datadict["audio"]))
        newdatadict = {c: [] for c in columns}

        for i in index:
            waveform = datadict["audio"][i]  # 当前行的语音数据
            if waveform.shape[-1] > self.samples:
                samples = int(
                    (waveform.shape[-1] // self.samples) * self.samples)
                waveform = waveform[:, :samples]
            else:
                waveform = F.pad(
                    waveform, (0, self.samples-waveform.shape[-1]))
            # waveform = F.pad( # 将语音数据 pad
            #     waveform, (0, int(((waveform.shape[-1] // self.samples) + 1) * self.samples)-waveform.shape[-1]))
            waveform = torch.split(waveform, self.samples, dim=1)

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


if __name__ == '__main__':
    datadict = {"id": [1, 2, 3, 4, 5, 6], "video": [torch.randn((250, 120, 120, 3)), torch.randn((250, 120, 120, 3)), torch.randn((250, 120, 120, 3)), torch.randn((250, 120, 120, 3)), torch.randn((250, 120, 120, 3)), torch.randn((250, 120, 120, 3))], "audio": [torch.randn((1, 225630)), torch.randn((1, 335642)), torch.randn((1, 122658)), torch.randn((1, 99520)), torch.randn(
        (1, 116220)), torch.randn((1, 110255))], "sample_rate": 16000, "text": ["asdasdasdasd", "asdasdasdasd", "asdasdasdasd", "asdasdasdasd", "asdasdasdasd", "asdasdasdasd"], "labelid": [1, 2, 1, 3, 2, 2], "val/act/dom": [(3.5, 2.5, 1.5), (3.5, 2.5, 1.5), (3.5, 2.5, 1.5), (3.5, 2.5, 1.5), (3.5, 2.5, 1.5), (3.5, 2.5, 1.5), ], "gender": ["F", "M", "F", "F", "M", "M", ], "label": [1, 2, 1, 3, 2, 2]}
    audioSplit = AudioSplit(sample_rate=16000)
    dataframe = audioSplit(datadict)
    print(dataframe)
