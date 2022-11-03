# 官方库
import os
from typing import Dict, Union, Optional
from pathlib import Path
# 第三方库
import librosa
import pandas as pd
import torch
import soundfile as sf
# 自定库
from custom_transforms import *
from custom_enhance import *
from .subset import CustomDataset


class EMODB(CustomDataset):
    """Create a Dataset for EMODB.
        Args:
            root (Union[str, Path]): 官方数据库解压后的路径.
            filter(dict)
            threads (int, optional): 视频数据处理进程数. Defaults to 0.

        Returns:
            tuple: ``(waveform, sample_rate, transcript, val, act, dom, speaker)``

    """

    def __init__(self,
                 root: Union[str, Path],
                 name: str = "",
                 filter: dict = {"replace": {}, "dropna": {'emotion': ["other", "xxx"]}, "contains": "", "query": "", "sort_values": ""},
                 transform: dict = None,
                 enhance: dict = None,
                 ):
        super(EMODB, self).__init__(root, name, filter, transform, enhance)
        root = os.fspath(root)
        self.dataset_name = "EMODB"
        self.root = root
        self.name = name
        if os.path.isfile(root):
            self.load_data()
            if filter:
                self.filter(filter)
        elif os.path.isdir(root):
            self.make_datadict()
            self.make_enhance(enhance)
            self.make_transform(transform)
            self.save_frame(transform, enhance)
            if filter:
                self.filter(filter)

    def make_enhance(self, enhance):
        self.audio_enhance = None
        self.text_enhance = None

        processors = {}
        for key, processor in enhance.items():
            if processor:
                pl = []
                for name, param in processor.items():

                    if param:
                        pl.append(globals()[name](**param))
                    else:
                        pl.append(globals()[name]())
                processors[key] = Compose(pl)

        if processors:
            if "audio" in processors:
                self.datadict = processors["audio"](self.datadict)
            if "text" in processors:
                self.datadict = processors["text"](self.datadict)

    def make_transform(self, transform):
        self.audio_transform = None
        self.text_transform = None

        processors = {}
        for key, processor in transform.items():
            if processor:
                pl = []
                for name, param in processor.items():

                    if param:
                        pl.append(globals()[name](**param))
                    else:
                        pl.append(globals()[name]())
                processors[key] = Compose(pl)

        if processors:
            if "audio" in processors:
                self.audio_transform = processors["audio"]
            if "text" in processors:
                self.text_transform = processors["text"]

        if self.audio_transform:
            for i in range(len(self.datadict["audio"])):
                self.datadict["audio"][i] = self.audio_transform(
                    self.datadict["audio"][i])
        if self.text_transform:
            for i in range(len(self.datadict["text"])):
                self.datadict["text"][i] = self.text_transform(
                    self.datadict["text"][i])

    def save_frame(self, transform, enhance):
        namelist = []
        for k, v in enhance.items():
            if v:
                namelist.append(k+"_"+"_".join([i for i in v.keys()]))
        for k, v in transform.items():
            if v:
                namelist.append(k+"_"+"_".join([i for i in v.keys()]))
        torch.save(self.datadict, os.path.join(
            self.root, "-".join([self.dataset_name, self.name, *namelist, ".npy"])))

    def load_data(self):
        self.datadict = torch.load(self.root)

    def _load_audio(self, wav_path) -> torch.Tensor:
        y, sr = librosa.load(wav_path, sr=None)
        y = torch.from_numpy(y).unsqueeze(0)
        return y

    def filter(self, filter: dict):
        # 根据提供的字典替换值, key 决定要替换的列, value 是 字符串字典, 决定被替换的值
        self.datadict = pd.DataFrame(self.datadict)
        if "replace" in filter:
            self.datadict.replace(
                None if filter["replace"] else filter["replace"], inplace=True)
            # 根据提供的字典删除指定值所在行, key 决定要替换的列, value 是被替换的值列表
        if "dropna" in filter:
            for k, v in filter["dropna"].items():
                self.datadict = self.datadict[~self.datadict[k].isin(v)]
            self.datadict.dropna(axis=0, how='any', thresh=None,
                                 subset=None, inplace=True)
        if "contains" in filter:
            # 根据提供的字典删除包含指定值的所在行, key 决定要替换的列, value 是被替换的值列表
            for k, v in filter["contains"].items():
                self.datadict = self.datadict[self.datadict[k].str.contains(
                    v, case=True)]
        if "query" in filter:
            if filter["query"]:
                self.datadict.query(filter["query"], inplace=True)
        if "sort_values" in filter:
            self.datadict.sort_values(
                by=filter["sort_values"], inplace=True, ascending=False)
        self.datadict.reset_index(drop=True, inplace=True)
        self.datadict = self.datadict.to_dict(orient="list")
        self.datadict["labelid"] = label2id(self.datadict["label"])

    def make_datadict(self):
        """
        读取 EMODB 语音数据集, 获取文件名、 情感标签、 文本、 维度值列表。
        """
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

        self.datadict["audio"] = [self._load_audio(
            self.datadict["path"][n]) for n in range(len(self.datadict["path"]))]

    def _load_EMODB_item(self, n: int) -> Dict[str, Optional[Union[int, torch.Tensor, int, str, int, str, str]]]:

        return {
            "id": n,
            "audio": self.datadict["audio"][n],
            "sample_rate": self.datadict["sample_rate"][n],
            "text": self.datadict["text"][n],
            "labelid": self.datadict["labelid"][n],
            "gender": self.datadict["gender"][n],
            "label": self.datadict["label"][n]
        }

    def __getitem__(self, n: int) -> Dict[str, Optional[Union[int, torch.Tensor, int, str, int, str, str]]]:
        """加载 EMODB 数据库的第 n 个样本数据
        Args:
            n (int): 要加载的样本的索引
        Returns:
            Dict[str, Optional[Union[int, torch.Tensor, int, str, int, str, str]]]:
        """
        return self._load_EMODB_item(n)

    def __len__(self) -> int:
        """为 EMODB 数据库类型重写 len 默认行为.
        Returns:
            int: EMODB 数据库数据数量
        """
        return len(self.datadict["path"])


if __name__ == '__main__':
    a = EMODB("/home/visitors2/SCW/deeplearning/data/EMODB")
    feature = a.__getitem__(100)
    pass
