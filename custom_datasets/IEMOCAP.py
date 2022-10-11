# 官方库
import os
import sys
import io
import re
from typing import Dict, Tuple, Union, Optional
from pathlib import Path
# 第三方库
import librosa
import numpy as np
import decord
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
import torchaudio
# 自定库
from utils.extractFace import capture_face_video
from utils.filefolderTool import create_folder


class CustomDataset(Dataset):
    def __init__(self,
                 root: Union[str, Path],
                 filter: dict,
                 ):
        super(Dataset, self).__init__()


class IEMOCAP(CustomDataset):
    """Create a Dataset for IEMOCAP.
        Args:
            root (Union[str, Path]): 官方数据库解压后的路径.
            filter(dict)
            video (bool, optional): 是否使用视频片段数据, 如果使用, 在视频数据不存在的情况下会进行一次视频数据处理(非常耗时). Defaults to False.
            cascPATH (Union[str, Path], optional): 视频数据处理所用到的 opencv 配置文件地址(需要与 video 同时提供). Defaults to None.
            threads (int, optional): 视频数据处理进程数. Defaults to 0.

        Returns:
            tuple: ``(waveform, sample_rate, transcript, video, val, act, dom, speaker)``

    """

    def __init__(self,
                 root: Union[str, Path],
                 filter: dict = {"replace": {}, "dropna": {'emotion': ["other", "xxx"]}, "contains": "", "query": "", "sort_values": ""},
                 transform: dict = None,
                 video: bool = False,
                 cascPATH: Union[str, Path] = None,
                 threads: int = 0,
                 ):
        super(IEMOCAP, self).__init__(root, filter)
        root = os.fspath(root)
        self.dataset_name = "IEMOCAP"
        self.root = root
        self.make_DataFrame()
        if video and cascPATH:
            self.check_video_file(cascPATH, threads)
        self.make_transform(transform)
        if filter:
            self.filter(filter)

        elif video or cascPATH:
            raise("# video 和 cascPATH 必须同时提供")

    def make_transform(self, transform):
        self.audio_transform = None
        self.video_transform = None
        self.text_transform = None
        if transform:
            if "audio" in transform:
                self.audio_transform = transform["audio"]
            if "video" in transform:
                self.video_transform = transform["video"]
            if "text" in transform:
                self.text_transform = transform["text"]

    def check_video_file(self, cascPATH, threads):
        print("# 检测视频文件夹是否存在...")
        fileExists = list(set([os.path.exists(os.path.join(
            self.root, f"Session{sess}/sentences/video/")) for sess in range(1, 6)]))
        if len(fileExists) != 1 or fileExists[0]:
            print("# 检测视频文件是否损坏...")
            self.check_videoFile(cascPATH, threads)
        elif not fileExists[0]:
            self._creat_video_process(self._pathList, cascPATH, threads)

    def check_videoFile(self, cascPATH, threads) -> bool:
        # 验证处理后或已有的视频文件数量完整性...
        no_exists_filelist = []
        corrupted_filelist = []

        for filename in self._pathList:
            filename_ = filename.split("_")
            sess = int(filename_[0][3:-1])
            path = os.path.join(
                self.root, f"Session{sess}/sentences/video/"+"_".join(filename_[:-1])+"/"+filename+".avi")
            if not os.path.exists(path):
                no_exists_filelist.append(filename)
            else:
                with open(path, "rb") as fh:
                    video_file = io.BytesIO(fh.read())
                try:
                    self._av_reader = decord.VideoReader(
                        uri=video_file,
                        ctx=decord.cpu(0),
                        width=-1,
                        height=-1,
                        num_threads=0,
                        fault_tol=-1,
                    )
                except Exception as e:
                    corrupted_filelist.append(filename)

        if len(no_exists_filelist) > 0 or len(corrupted_filelist) > 0:
            print("# 丢失文件:")
            [print("# \t"+i) for i in no_exists_filelist]
            print("# 损坏文件:")
            [print("# \t"+i) for i in corrupted_filelist]
            print("# 尝试重新处理文件...")
            self._creat_video_process(
                no_exists_filelist+corrupted_filelist, cascPATH, threads)
            self.check_video_file(cascPATH, threads)
        return False

    def _creat_video_process(self, filelist, cascPATH, threads):
        print("# 自动提取脸部表情视频片段...")
        [create_folder(os.path.join(
            self.root, f"Session{sess}/sentences/video/")) for sess in range(1, 6)]
        # 开始处理视频文件
        if threads > 0:
            from multiprocessing import Process
            pool = []  # 进程池
            length = len(filelist)
            step = int(length / threads) + 1  # 每份的长度
            for i in range(0, length, step):
                p = Process(target=self._creat_video, args=(
                    filelist[i: i + step], cascPATH))
                pool.append(p)
            for p in pool:
                p.start()
            for p in pool:
                p.join()
        else:
            self._creat_video(filelist, cascPATH)
        print("# 视频文件处理成功")

    def _creat_video(self, filelist, cascPATH):
        # 给定文件名列表, 处理视频文件函数
        for ind, filename in enumerate(filelist):  # Ses01F_impro01_F000
            # F(L:female;R:male),M(L:male;R:female)
            mapRule = {"FF": "L", "FM": "R", "MM": "L", "MF": "R"}
            filename_ = filename.split("_")
            sess = int(filename_[0][3:-1])
            speaker = mapRule[filename_[0][-1]+filename_[-1][0]]
            # 建立当前 sess 存放视频的文件夹
            videoPath = os.path.join(
                self.root, f"Session{sess}/sentences/video/"+"_".join(filename_[:-1]))
            if not os.path.exists(videoPath):
                os.makedirs(videoPath)
            video_clip_path = os.path.join(videoPath, filename+".avi")
            origin_video_path = os.path.join(
                self.root, f'Session{sess}/dialog/avi/DivX/'+"_".join(filename_[:-1])+".avi")
            print(f"# Processing: {filename}.avi  path: {video_clip_path}")
            capture_face_video(cascPATH, origin_video_path,
                               video_clip_path, *self._periodList[ind], speaker)

    def _get_speaker(self, filename) -> str:
        # 给定文件名获取此文件对应说话人性别
        if "_F" in filename:
            return("F")
        elif "_M" in filename:
            return("M")
        else:
            return("")

    def _load_audio(self, filename) -> torch.Tensor:
        # 指定文件名加载语音文件数据
        filename_ = filename.split("_")
        sess = int(filename_[0][3:-1])
        wav_path = os.path.join(
            self.root, f"Session{sess}/sentences/wav/"+"_".join(filename_[:-1])+"/"+filename+".wav")
        y, sr = librosa.load(wav_path)
        y = torch.from_numpy(y).unsqueeze(0)
        if self.audio_transform:
            y = self.audio_transform(y)
        return y

    def _load_video(self, filename) -> torch.Tensor:
        # 指定文件名加载视频文件数据
        filename_ = filename.split("_")
        sess = int(filename_[0][3:-1])
        # 视频文件路径
        video_path = os.path.join(
            self.root, f"Session{sess}/sentences/video/"+"_".join(filename_[:-1])+"/"+filename+".avi")
        return self.process_video(video_path)

    def process_video(self,
                      video_path,
                      width: int = -1,
                      height: int = -1,
                      num_threads: int = 0,
                      fault_tol: int = -1,
                      ) -> torch.Tensor:
        """解码视频得到对应的 Tensor 类型数据

        Args:
            video_path (_type_): 视频文件所在路径
            width (int, optional): 视频期望的输出像素宽度, 如果指定了'-1'则不变. Defaults to -1.
            height (int, optional): 视频期望的输出像素高度, 如果指定了'-1'则不变. Defaults to -1.
            num_threads (int, optional): 解码视频所使用的线程数量, 如果指定了 '0', 则为 auto. Defaults to 0.
            fault_tol (int, optional): 
                损坏和恢复帧的阈值。这是为了在例如视频的50%的帧不能被解码并且重复的帧被返回时防止静默容错。您可能会发现容错功能在许多情况下很有用，但不适用于训练模型。
                    如果 `fault_tol` < 0, 则什么都不做.
                    如果 0 < `fault_tol` < 1.0, 并且 N > `fault_tol * len(video)`, 引发 `DECORDLimitReachedError` 类型错误.
                    如果 1 < `fault_tol`, 并且 N > `fault_tol`, 引发 `DECORDLimitReachedError` 类型错误.
            Defaults to -1.
        Returns:
            torch.Tensor
        Refer:
            Link: https://github.com/facebookresearch/pytorchvideo/blob/b2b453bdcd7afcabc3804fba89e0889e27ec9535/pytorchvideo/data/encoded_video_decord.py
        """
        with open(video_path, "rb") as fh:
            video_file = io.BytesIO(fh.read())
        try:
            self._av_reader = decord.VideoReader(
                uri=video_file,
                ctx=decord.cpu(0),
                width=width,
                height=height,
                num_threads=num_threads,
                fault_tol=fault_tol,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to open video {video_path} with Decord. {e}")
        self._fps = self._av_reader.get_avg_fps()
        self._duration = float(len(self._av_reader)) / float(self._fps)
        frame_idxs = list(range(0, len(self._av_reader)))
        try:
            video = self._av_reader.get_batch(frame_idxs)
        except Exception as e:
            raise(f"Failed to decode video with Decord: {video_path}. {e}")
        video = torch.from_numpy(video.asnumpy()).to(
            torch.float32).permute(3, 0, 1, 2)
        if self.video_transform:
            video = self.video_transform(video)
        return video

    def filter(self, filter: dict):
        # 根据提供的字典替换值, key 决定要替换的列, value 是 字符串字典, 决定被替换的值
        if "replace" in filter:
            self.dataframe.replace(
                None if filter["replace"] else filter["replace"], inplace=True)
            # 根据提供的字典删除指定值所在行, key 决定要替换的列, value 是被替换的值列表
        if "dropna" in filter:
            for k, v in filter["dropna"].items():
                self.dataframe = self.dataframe[~self.dataframe[k].isin(v)]
            self.dataframe.dropna(axis=0, how='any', thresh=None,
                                  subset=None, inplace=True)
        if "contains" in filter:
            # 根据提供的字典删除包含指定值的所在行, key 决定要替换的列, value 是被替换的值列表
            for k, v in filter["contains"].items():
                self.dataframe = self.dataframe[self.dataframe[k].str.contains(
                    v, case=True)]
        if "query" in filter:
            self.dataframe.query(filter["query"], inplace=True)
        if "sort_values" in filter:
            self.dataframe.sort_values(
                by=filter["sort_values"], inplace=True, ascending=False)
        self.dataframe.reset_index(drop=True, inplace=True)

    def make_DataFrame(self):
        """
        读取 IEMOCAP 语音数据集, 获取文件名、 情感标签、 文本、 维度值列表。
        """
        # 文件情感标签与序号的对应关系
        tag_to_emotion = {"ang": "angry",  # angry
                          "hap": "happy",  # happy
                          "neu": "neutral",  # neutral
                          "sad": "sad",  # sad
                          "fea": "fearful",  # fearful
                          "sur": "surprised",  # surprise
                          "dis": "disgusted",  # disgusted
                          "exc": "excited",  # excited
                          "fru": "frustrated",  # frustrated
                          "oth": "other",  # other
                          "xxx": "xxx"  # NA
                          }
        # 正则表达式，用来匹配含有 "[任意内容]\n" 格式的数据行，{.：匹配除换行符 \n 之外的任何单字符；+：表示前面要匹配的出现不止一次；\n：表示换行符}
        # 例如：bigin ：[6.2901 - 8.2357]	Ses01F_impro01_F000	neu	[2.5000, 2.5000, 2.5000] ：end
        info_line = re.compile(r'\[.+\]\n', re.IGNORECASE)  # 匹配被 [] 包裹的数据行

        self._pathList = []
        _labelList = []
        _dimvalueList = []
        self._periodList = []
        _textList = []
        _genderList = []
        for sess in range(1, 6):  # 遍历Session{1-5}文件夹
            # 指定和获取文件地址
            # 语音情感标签所在文件夹
            emo_evaluation_dir = os.path.join(
                self.root, f'Session{sess}/dialog/EmoEvaluation/')
            evaluation_files = [l for l in os.listdir(
                emo_evaluation_dir) if (l[:3] == 'Ses')]  # 数据库情感结果文件列表

            # 获取文件名、情感标签、维度值、开始结束时间
            for file in evaluation_files:  # 遍历当前 Session 的 EmoEvaluation 文件
                with open(emo_evaluation_dir + file, encoding='utf-8') as f:
                    content = f.read()
                # 匹配正则化模式对应的所有内容，得到数据列表
                # 匹配被[]包裹的数据行,数据示例: [开始时间 - 结束时间] 文件名称 情感 [V, A, D]
                info_lines = re.findall(info_line, content)
                for line in info_lines[1:]:  # 忽略第一行，无用，不是想要的数据
                    # strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。
                    # .split('\t') 方法用于以制表符为基，分割此字符串
                    start_end_time, wav_file_name, emotion, val_act_dom = line.strip().split('\t')
                    # 文件名，用来获取对应的语音和视频文件

                    # 获取当前语音段的开始时间和结束时间
                    start_time, end_time = start_end_time[1:-1].split('-')
                    val, act, dom = val_act_dom[1:-1].split(',')

                    self._pathList.append(wav_file_name)
                    self._periodList.append(
                        (float(start_time), float(end_time)))
                    # 获取情感维度值 dominance (dom), activation (act) and valence (val)
                    _dimvalueList.append(
                        (float(val),
                            float(act),
                            float(dom))
                    )
                    # 标签列表
                    _labelList.append(tag_to_emotion[emotion])
                    # 文本翻译列表
                    _textList.append("")
                    # 说话人列表
                    _genderList.append(
                        self._get_speaker(wav_file_name))

            transcriptions_dir = os.path.join(
                self.root, f'Session{sess}/dialog/transcriptions/')
            transcription_files = [l for l in os.listdir(
                transcriptions_dir) if (l[:3] == 'Ses')]  # 语音文本翻译所在文件列表
            # 获取音频对应翻译
            for file in transcription_files:
                with open(transcriptions_dir + file, encoding='utf-8') as f:
                    for line in f.readlines():
                        try:
                            filename, _temp = line.strip().split(' [', 1)
                            text = _temp.strip().split(']: ', 1)[1]
                            # text = re.sub("\[.+\]", "", text) # 是否替换Text中的声音词，如哈哈大笑
                            index = self._pathList.index(filename)
                            _textList[index] = text
                        except:  # 出错就跳过
                            continue
        self.dataframe = pd.DataFrame({
            "path": self._pathList,
            "label": _labelList,
            "dimvalue": _dimvalueList,
            "duration": [end-begin for begin, end in self._periodList],
            "text": _textList,
            "gender": _genderList,

        })

    def _load_iemocap_item(self, n: int) -> Dict[str, Optional[Union[int, torch.Tensor, torch.Tensor, int, str, str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor], str]]]:

        return {
            "id": n,
            "video": self._load_video(self.dataframe["path"][n]),
            "audio": self._load_audio(self.dataframe["path"][n]),
            "sample_rate": 16000,
            "text": self.dataframe["text"][n],
            "label": self.dataframe["label"][n],
            "val/act/dom": self.dataframe["dimvalue"][n],
            "gender": self.dataframe["gender"][n],
        }

    def __getitem__(self, n: int) -> Dict[str, Optional[Union[int, torch.Tensor, torch.Tensor, int, str, str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor], str]]]:
        """加载 IEMOCAP 数据库的第 n 个样本数据
        Args:
            n (int): 要加载的样本的索引
        Returns:
            Dict[str, Optional[Union[int, torch.Tensor, torch.Tensor, int, str, str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor], str]]]
        """
        return self._load_iemocap_item(n)

    def __len__(self) -> int:
        """为 IEMOCAP 数据库类型重写 len 默认行为.
        Returns:
            int: IEMOCAP 数据库数据数量
        """
        return len(self.dataframe)


if __name__ == '__main__':
    a = IEMOCAP("/home/visitors2/SCW/deeplearning/data/IEMOCAP", True,
                "/home/visitors2/SCW/deeplearningcopy/utils/haarcascade_frontalface_alt2.xml", 0)
    feature = a.__getitem__(1000)
    pass
