# 官方库
import os
from typing import Dict, Union, Optional
from pathlib import Path
import logging
# 第三方库
import librosa
import pandas as pd
import torch
import soundfile as sf
# 自定库
from custom_transforms import *
from custom_enhance import *
from custom_datasets.subset import MediaDataset
from tqdm import tqdm


class EMODB(MediaDataset):
    """Create a Dataset for EMODB.
        Args:
        Returns:
            tuple: ``(waveform, sample_rate, transcript, val, act, dom, speaker)``

    """

    def __init__(self,
                 root: Union[str, Path],
                 info: str = "",
                 filter: dict = {"replace": {}, "dropna": {'emotion': [
                     "other", "xxx"]}, "contains": "", "query": "", "sort_values": ""},
                 transform: dict = {},
                 enhance: dict = {},
                 savePath: str = "",
                 ):
        super(EMODB, self).__init__(
            root=root, info=info, filter=filter,
            transform=transform, enhance=enhance, mode="a",
            savePath=savePath,)
        self.dataset_name = "EMODB"

        if os.path.isfile(root):
            self.load_data()
            if filter:
                self.build_filter()
        elif os.path.isdir(root):
            self.build_datadict()
            self.build_enhance()
            self.build_transform()
            self.save_data()
            if filter:
                self.build_filter()
        else:
            raise ValueError("数据库地址不对")

    def _load_audio(self, wav_path) -> torch.Tensor:
        y, sr = librosa.load(wav_path, sr=None)
        y = torch.from_numpy(y).unsqueeze(0)
        return y

    def build_datadict(self):
        """
        读取 EMODB 语音数据集, 获取文件名、 情感标签、 文本、 维度值列表。
        """
        logging.info("构建初始数据")
        # 文件情感标签与序号的对应关系
        tag_to_emotion = {  # 文件标签与情感的对应关系
            "W": "angry",  # angry
            "L": "bored",  # boreded
            "E": "disgusted",  # disgusted
            "A": "fearful",  # fearful
            "F": "happy",  # happy
            "T": "sad",  # sad
            "N": "neutral"  # neutral
        }
        textcode = {  # 文件标签与文本的对应关系
            "a01": "Der Lappen liegt auf dem Eisschrank.",
            "a02": "Das will sie am Mittwoch abgeben.",
            "a04": "Heute abend könnte ich es ihm sagen.",
            "a05": "Das schwarze Stück Papier befindet sich da oben neben dem Holzstück.",
            "a07": "In sieben Stunden wird es soweit sein.",
            "b01": "Was sind denn das für Tüten, die da unter dem Tisch stehen?",
            "b02": "Sie haben es gerade hochgetragen und jetzt gehen sie wieder runter.",
            "b03": "An den Wochenenden bin ich jetzt immer nach Hause gefahren und habe Agnes besucht.",
            "b09": "Ich will das eben wegbringen und dann mit Karl was trinken gehen.",
            "b10": "Die wird auf dem Platz sein, wo wir sie immer hinlegen."
        }
        wavpath = os.path.join(self.root, "wav")
        self._pathList = []
        _labelList = []
        self._periodList = []
        _textList = []
        _genderList = []
        _sample_rate = []
        # 获取指定目录下所有.wav文件列表
        files = [l for l in os.listdir(wavpath) if ('.wav' in l)]
        for file in files:  # 遍历所有文件
            try:
                emotion = tag_to_emotion[os.path.basename(file)[
                    5]]  # 从文件名中获取情感标签
                text = textcode[os.path.basename(file)[2:5]]  # 从文件名中获取情感标签
            except KeyError:
                continue
            self._pathList.append(os.path.join(wavpath, file))
            _labelList.append(emotion)
            _textList.append(text)
            # 获取说话人性别
            if os.path.basename(file)[0:2] in ["03", "10", "12", "15", "11"]:
                _genderList.append("M")
            elif os.path.basename(file)[0:2] in ["08", "09", "13", "14", "16"]:
                _genderList.append("F")
            else:
                _genderList.append("")
            params = sf.info(os.path.join(wavpath, file))
            _sample_rate.append(params.samplerate)  # 采样频率
            self._periodList.append(params.duration)

        self.datadict = {
            "path": self._pathList,
            "duration": self._periodList,
            "text": _textList,
            "gender": _genderList,
            "label": _labelList,
            "sample_rate": _sample_rate
        }

        logging.info("构建语音数据")
        audiolist = [None]*len(self._pathList)
        for i in tqdm(range(len(self.datadict["path"])), desc="音频数据处理中: "):
            audiolist[i] = self._load_audio(self.datadict["path"][i])

        self.datadict["audio"] = audiolist
        self.datadict["audio_sampleNum"] = [a.shape[0]
                                            for a in self.datadict["audio"]]

    def __getitem__(self, n: int) -> Dict[str, Optional[Union[int, torch.Tensor, int, str, int, str, str]]]:
        """加载 EMODB 数据库的第 n 个样本数据
        Args:
            n (int): 要加载的样本的索引
        Returns:
            Dict[str, Optional[Union[int, torch.Tensor, int, str, int, str, str]]]:
        """
        return {
            "id": n,
            "audio": self.datadict["audio"][n],
            "sample_rate": self.datadict["sample_rate"][n],
            "text": self.datadict["text"][n],
            "labelid": self.datadict["labelid"][n],
            "gender": self.datadict["gender"][n],
            "label": self.datadict["label"][n]
        }

    def __len__(self) -> int:
        """为 EMODB 数据库类型重写 len 默认行为.
        Returns:
            int: EMODB 数据库数据数量
        """
        return len(self.datadict["path"])


if __name__ == '__main__':
    a = EMODB("/home/user0/SCW/deeplearning/data/EMODB")
    feature = a.__getitem__(100)
    pass
