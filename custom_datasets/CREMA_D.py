import os
from pathlib import Path
import torch
from torch.utils.data import Dataset
import logging
import pandas as pd
from typing import Union
from tqdm import tqdm
import multiprocessing as mp
import librosa
import audioread.ffdec
import numpy as np
import decord
import pandas as pd
import torch

from custom_datasets.subset import *
from utils.filefolderTool import create_folder
from utils.video_featureExtract_extractFace import capture_face_video, capture_video

class CREMA_D(MediaDataset):
    """_summary_

    Args:
        MediaDataset (_type_): _description_
    """
    def __init__(self,
                 root: Union[str, Path],
                 info: str = "",
                 filter: dict = {},
                 transform: dict = {},
                 enhance: dict = {},
                 savePath: str = "",
                 mode: str = "#",
                 resample: bool = True,
                 ):
        super(CREMA_D, self).__init__(
            root=root, info=info, filter=filter,
            transform=transform, enhance=enhance, mode=mode,
            savePath=savePath,
        )
        self.dataset_name = "CREMA-D"
        self.resample = resample

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

    def build_datadict(self):
        emotion_map = {
            "A": "anger",
            "D": "disgust",
            "F": "fear",
            "H": "happiness",
            "S": "sadness",
            "N": "neutral",
            "X": "unknown",
        }
        sentence_map = {
            "IEO": "It's eleven o'clock",
            "TIE": "That is exactly what happened",
            "IOM": "I'm on my way to the meeting",
            "IWW": "I wonder what this is about",
            "TAI": "The airplane is almost full",
            "MTI": "Maybe tomorrow it will be cold",
            "IWL": "I would like a new alarm clock",
            "ITH": "I think I have a doctor's appointment",
            "DFA": "Don't forget a jacket",
            "ITS": "I think I've seen this before",
            "TSI": "The surface is slick",
            "WSI": "We'll stop in a couple of minutes",
        }

        audio_dir = Path(self.root) / "AudioMP3"
        audio_paths = list(audio_dir.glob("*.mp3"))

        self.datadict = {
            "path": [str(p) for p in audio_paths],
            "label": [emotion_map[p.stem[9]] for p in audio_paths],
            "speaker": [p.stem[:4] for p in audio_paths],
            "sentence_code": [p.stem[5:8] for p in audio_paths],
            "sentence": [sentence_map[p.stem[5:8]] for p in audio_paths],
            "level": [p.stem[-2:] for p in audio_paths],
        }

        if self.resample:
            resample_dir = Path(self.root) / "resampled"
            resample_dir.mkdir(exist_ok=True)
            # Implement resampling logic here if needed

    def __getitem__(self, n: int):
        """加载 CREMA-D 数据库的第 n 个样本数据
        Args:
            n (int): 要加载的样本的索引
        Returns:
            Dict[str, Optional[Union[int, torch.Tensor, str]]]
        """
        out = {
            "id": n,
            "label": self.datadict["label"][n],
            "speaker": self.datadict["speaker"][n],
            "sentence_code": self.datadict["sentence_code"][n],
            "sentence": self.datadict["sentence"][n],
            "level": self.datadict["level"][n],
        }
        # Load audio data if needed
        return out

    def __len__(self) -> int:
        """为 CREMA-D 数据库类型重写 len 默认行为.
        Returns:
            int: CREMA-D 数据库数据数量
        """
        return len(self.datadict["path"])
