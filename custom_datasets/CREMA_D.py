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
                 filter: dict = {"replace": {}, "dropna": {}, "contains": {}, "query": {}, "sort_values": ""},
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
        pass

    def build_datadict(self):
        emotion_map = {
            "A": "angry", #
            "D": "disgusted",
            "F": "fearful",
            "H": "happy", #
            "S": "sad", #
            "N": "neutral", #
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

        audio_dir = Path(self.root) / "AudioWAV"
        audio_paths = list(audio_dir.glob("*.wav"))
        video_dir = Path(self.root) / "VideoFlash"
        video_paths = [video_dir / (p.stem + ".flv") for p in audio_paths]
        
        if len(video_paths) != len(video_paths):
            raise Exception("音视频数据数量有误")

        self.datadict = {
            "path": [{"audio": str(audio), "video": str(video)} for audio, video in zip(audio_paths, video_paths)],
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
                    audiolist[i] = self._load_audio(self.datadict["path"][i]["audio"])
            self.datadict["audio"] = [torch.from_numpy(i) for i in audiolist]
            self.datadict["audio_sampleNum"] = [a.shape[1]
                                                for a in self.datadict["audio"]]
        if self.mode in ["v", "av", "vt", "avt"]:
            logging.info("构建视频数据")
            videolist = [None]*len(video_paths)
            for i in tqdm(range(len(video_paths)), desc="视频数据处理中: "):
                videolist[i] = self._load_video(self.datadict["path"][i]["video"])
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
                width=-1,
                height=-1,
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
        # Process each frame: resize and crop
        processed_frames = []
        for frame in out:
            resized_frame = self._resize_frame(frame, scale_factor=0.5)
            cropped_frame = self._crop_center(resized_frame, crop_size=(150, 150))
            processed_frames.append(cropped_frame)
        # Stack frames to create a video tensor
        processed_video = np.stack(processed_frames)
        return processed_video
    
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
            # out["sample_rate"] = self.datadict["sample_rate"][n]
        if "t" in self.mode:
            out["text"] = self.datadict["text"][n]
        return out

    def __len__(self) -> int:
        """为 CREMA-D 数据库类型重写 len 默认行为.
        Returns:
            int: CREMA-D 数据库数据数量
        """
        return len(self.datadict["path"])

    def pbar_listener(self, pbar_queue, total):
        pbar = tqdm(total=total)
        pbar.set_description('处理中: ')
        while True:
            if not pbar_queue.empty():
                k = pbar_queue.get()
                if k == 1:
                    pbar.update(1)
                else:
                    break
        pbar.close()
        
    def data_listener(self, length, sharedata, data_queue):
        datalist = [None]*length
        while True:
            if not data_queue.empty():
                data = data_queue.get()
                if isinstance(data, dict):
                    for k, v in data.items():
                        datalist[k] = v
                else:
                    break
        sharedata[0] = datalist
        
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
            data_queue.put({i: self._load_audio(self.datadict["path"][i]["audio"])})
            if pbar_queue:
                pbar_queue.put(1)
                
    def _resize_frame(self, frame, scale_factor):
        """Resize the frame by a given scale factor."""
        height, width = frame.shape[:2]
        new_height = int(height * scale_factor)
        new_width = int(width * scale_factor)
        return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

    def _crop_center(self, frame, crop_size):
        """Crop the center of the frame to the given size."""
        height, width = frame.shape[:2]
        crop_width, crop_height = crop_size

        center_x, center_y = width // 2, height // 2
        start_x = max(center_x - crop_width // 2, 0)
        start_y = max(center_y - crop_height // 2, 0)
        end_x = min(center_x + crop_width // 2, width)
        end_y = min(center_y + crop_height // 2, height)

        return frame[start_y:end_y, start_x:end_x]
    
    
    def _save_video(self, frames, output_path, fps):
        # Define the codec and create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        out = cv2.VideoWriter(output_path, fourcc, fps, (150, 150))

        for frame in frames:
            # Write the frame to the output video
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame)

        # Release the VideoWriter
        out.release()