import torch
from torch.utils.data.dataset import Dataset
from custom_datasets.subset import *
import numpy as np
import random

class RandomAudioVideoDataset(MediaDataset):
    def __init__(self, 
                 root, 
                 num_samples = 1000, 
                 audio_size = (1, 186, 40), 
                 video_size = (30, 3, 150, 150),
                 classes = ["happy", "sad", "angry", "neutral"],
                info: str = "",
                 filter: dict = {"replace": {}, "dropna": {}, "contains": "", "query": "", "sort_values": ""},
                 transform: dict = {},
                 enhance: dict = {},
                 savePath: str = "",
                 ):
        self.num_samples = num_samples
        self.audio_size = audio_size
        self.video_size = video_size
        self.classes = classes
        
        super(RandomAudioVideoDataset, self).__init__(
            root=root, info="", filter={"replace": {}, "dropna": {}, "contains": "", "query": "", "sort_values": ""},
            transform={}, enhance={}, mode="av",
            savePath="",)
        
        self.dataset_name = "RandomAudioVideoDataset"
        self.datadict = {
            "gender": [random.choice(["male", "female"]) for _ in range(num_samples)],
            "label": [random.choice(self.classes) for _ in range(num_samples)]
        }

    def build_datadict(self):
        pass

    def __len__(self):
        return self.num_samples


    def __getitem__(self, n: int):
        """加载 IEMOCAP 数据库的第 n 个样本数据
        Args:
            n (int): 要加载的样本的索引
        Returns:
            Dict[str, Optional[Union[int, torch.Tensor, torch.Tensor, int, str, str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor], str]]]
        """
        out = {"id": n,
               "label": self.datadict["label"][n],
               }
        if "v" in self.mode:
            out["video"] = torch.randn(*self.video_size)
        if "a" in self.mode:
            out["audio"] = torch.randn(*self.audio_size)
            out["sample_rate"] = 16000
        return out
    