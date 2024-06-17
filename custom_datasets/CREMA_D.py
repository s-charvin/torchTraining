import os
from pathlib import Path
import torch
import logging
from typing import Union
from tqdm import tqdm
import multiprocessing as mp
import librosa
import audioread.ffdec
import decord
import pandas as pd
import torch

from custom_datasets.subset import *
import audioread.ffdec

class CREMA_D(MediaDataset):
    """Create a Dataset for CREMA-D dataset.
    Args:
        root (Union[str, Path]): Path to the root directory of the CREMA-D dataset.
        info (str): Path to the info file of the dataset.
        filter (dict): Filter to apply on the dataset.
        transform (dict): Transforms to apply on the dataset.
        enhance (dict): Enhancements to apply on the dataset.
        savePath (str): Path to save the dataset.
        mode (str): Mode to use for the dataset.
    """
    def __init__(self,
                 root: Union[str, Path],
                 info: str = "",
                 filter: dict = {"replace": {}, "dropna": {'emotion': [
                     "other", "xxx"]}, "contains": "", "query": "", "sort_values": ""},
                 transform: dict = {},
                 enhance: dict = {},
                 savePath: str = "",
                 mode: str = "avt",
                 videomode: str = "crop",  # "face"
                 cascPATH: Union[str, Path] = None,
                 threads: int = 0,
                 ):
        super(CREMA_D, self).__init__(
            root=root, info=info, filter=filter,
            transform=transform, enhance=enhance, mode=mode,
            savePath=savePath,
        )
        self.dataset_name = "CREMA-D"
        self.threads = threads
        self.cascPATH = cascPATH
        if self.mode in ["v", "av"] and cascPATH == "":
            raise("# 使用视频数据需要提供 cascPATH ")
        self.videomode = videomode
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
            "A": "angry",
            "D": "disgusted",
            "F": "fearful",
            "H": "happy",
            "S": "sad",
            "N": "neutral",
            "X": "xxx",
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
        video_dir = Path(self.root) / "VideoFlash"
        video_paths = [video_dir / (p.stem + ".mp4") for p in audio_paths]

        self.datadict = {
            "audio_path": [str(p) for p in audio_paths],
            "video_path": [str(p) for p in video_paths],
            "label": [emotion_map[p.stem[9]] for p in audio_paths],
            "speaker": [p.stem[:4] for p in audio_paths],
            "sentence_code": [p.stem[5:8] for p in audio_paths],
            "text": [sentence_map[p.stem[5:8]] for p in audio_paths],
            "level": [p.stem[-2:] for p in audio_paths],
        }
        
        if self.mode in ["a", "av",  "at", "avt"]:
            logging.info("构建语音数据")
            audiolist = [None]*len(audio_paths)
            if self.threads > 0:
                audiolist = self.sequence_multithread_processing(
                    len(audio_paths), self.thread_load_audio)
            else:
                for i in tqdm(range(len(audio_paths)), desc="音频数据处理中: "):
                    audiolist[i] = self._load_audio(self.datadict["audio_path"][i])
            self.datadict["audio"] = [torch.from_numpy(i) for i in audiolist]
            self.datadict["audio_sampleNum"] = [a.shape[1]
                                                for a in self.datadict["audio"]]
        if self.mode in ["v", "av", "vt", "avt"]:
            logging.info("构建视频数据")
            videolist = [None]*len(video_paths)
            for i in tqdm(range(len(video_paths)), desc="视频数据处理中: "):
                videolist[i] = self._load_video(self.datadict["video_path"][i])
            self.datadict["video"] = [torch.from_numpy(i) for i in videolist]
            self.datadict["video_sampleNum"] = [v.shape[0]
                                               for v in self.datadict["video"]]
    def _load_audio(self, path) -> torch.Tensor:
        if audioread.ffdec.available():
            path = audioread.ffdec.FFmpegAudioFile(path)
        y, sr = librosa.load(path, sr=None)
        return y[None, :]  # 1xseq
    
    def _load_video(self, path) -> torch.Tensor:
        with open(path, "rb") as fh:
            video_file = io.BytesIO(fh.read())
        try:
            _av_reader = decord.VideoReader(
                uri=video_file,
                ctx=decord.cpu(0),
                width=width,
                height=height,
                num_threads=self.threads,
                fault_tol=-1,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to open video {path} with Decord. {e}")
        self._fps = _av_reader.get_avg_fps()
        self._duration = float(len(_av_reader)) / float(self._fps)
        frame_idxs = list(range(0, len(_av_reader)))
        try:
            video = _av_reader.get_batch(frame_idxs)
        except Exception as e:
            raise(f"Failed to decode video with Decord: {path}. {e}")
        out = video.asnumpy()
        return out
    
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
            "level": self.datadict["level"][n],
        }
        if "v" in self.mode:
            out["video"] = self.datadict["video"][n]
        if "a" in self.mode:
            out["audio"] = self.datadict["audio"][n]
            out["sample_rate"] = self.datadict["sample_rate"][n]
        if "t" in self.mode:
            out["text"] = self.datadict["text"][n]
        return out

    def __len__(self) -> int:
        """为 CREMA-D 数据库类型重写 len 默认行为.
        Returns:
            int: CREMA-D 数据库数据数量
        """
        return len(self.datadict["path"])

    def sequence_multithread_processing(self, length, func):

        # 构建数据处理完成度监视进程和数据队列存储监视进程
        manage = mp.Manager()
        pbar_queue = manage.Queue()
        data_queue = manage.Queue()
        p_pbar = mp.Process(target=self.pbar_listener,
                            args=(pbar_queue, length))
        sharedata = mp.Manager().list([None])  # 建立共享数据
        p_data = mp.Process(target=self.data_listener,
                            args=(length, sharedata, data_queue))
        p_pbar.start()
        p_data.start()

        # 建立任务执行进程
        processList = []  # 任务处理进程池

        # 任务参数分割
        step = int(length / self.threads) + 1  # 每份的长度
        index = [i for i in range(length)]
        split_lists = [index[x:x+step] for x in range(0, length, step)]
        for ind in split_lists:
            t = mp.Process(target=func, args=(
                data_queue, ind, pbar_queue))
            processList.append(t)
        for t in processList:
            t.start()
        for t in processList:
            t.join()
        pbar_queue.put(-1)  # 停止监听
        data_queue.put(-1)  # 停止监听
        p_pbar.join()
        p_data.join()
        sequence = sharedata[0]
        return sequence
    
    def thread_load_audio(self, data_queue, index, pbar_queue=None):
        for i in index:
            data_queue.put({i: self._load_audio(self.datadict["path"][i])})
            if pbar_queue:
                pbar_queue.put(1)
                
if __name__ == "__main__":
    dataset = CREMA_D(root="/sdb/visitors2/SCW/data/CREMA-D", mode="avt", threads=4)
    print(len(dataset))
    print(dataset[0]["audio"].shape)
    print(dataset[0]["video"].shape)
    print(dataset[0]["text"])
    print(dataset[0]["label"])
    print(dataset[0]["speaker"])
    print(dataset[0]["sentence_code"])
    print(dataset[0]["level"])