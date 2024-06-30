# 官方库
import gc
from custom_datasets.subset import *
import logging
from utils.filefolderTool import create_folder
from utils.video_featureExtract_extractFace import capture_face_video, capture_video
from tqdm import tqdm
import os
import sys
import io
import re
from typing import Dict, Tuple, Union, Optional
from pathlib import Path
import multiprocessing as mp
# 第三方库
import librosa
import audioread.ffdec
import numpy as np
import decord
import pandas as pd
import torch

class IEMOCAP(MediaDataset):
    """Create a Dataset for IEMOCAP dataset.
        Args:
            root (Union[str, Path]): 官方数据库解压后的路径或存储的数据特征文件.
            filter(dict)
            video (bool, optional): 是否使用视频片段数据, 如果使用, 在视频数据不存在的情况下会进行一次视频数据处理(非常耗时). Defaults to False.
            cascPATH (Union[str, Path], optional): 视频数据处理所用到的 opencv 配置文件地址(需要与 video 同时提供). Defaults to None.
            threads (int, optional): 数据处理进程数(视频提取, 文件加载). Defaults to 0.

        Returns:
            tuple: ``(waveform, sample_rate, transcript, video, val, act, dom, speaker)``
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
        super(IEMOCAP, self).__init__(
            root=root, info=info, filter=filter,
            transform=transform, enhance=enhance, mode=mode,
            savePath=savePath,
        )
        self.dataset_name = "IEMOCAP"
        self.threads = threads
        self.cascPATH = cascPATH

        if self.mode in ["v", "av",  "vt", "avt"] and cascPATH == "":
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
        _sample_rate = []

        logging.info("构建初始数据")
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
                    _sample_rate.append(16000)
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
        if self.mode in ["v", "av",  "vt", "avt"]:
            self.check_video_file()
        self.datadict = {
            "path": self._pathList,
            "dimvalue": _dimvalueList,
            "duration": [end-begin for begin, end in self._periodList],
            "text": _textList,
            "gender": _genderList,
            "label": _labelList,
            "sample_rate": _sample_rate
        }

        if self.mode in ["a", "av",  "at", "avt"]:
            logging.info("构建语音数据")
            audiolist = [None]*len(self._pathList)
            # for n in tqdm(range(len(self.datadict["path"]))):
            #     audiolist[n] = self._load_audio(self.datadict["path"][n])
            if self.threads > 0:
                audiolist = self.sequence_multithread_processing(
                    len(self._pathList), self.thread_load_audio)
            else:
                for i in tqdm(range(len(self.datadict["path"])), desc="音频数据处理中: "):
                    audiolist[i] = self._load_audio(self.datadict["path"][i])

            self.datadict["audio"] = [torch.from_numpy(i) for i in audiolist]
            self.datadict["audio_sampleNum"] = [a.shape[1]
                                                for a in self.datadict["audio"]]
        if self.mode in ["v", "av",  "vt", "avt"]:
            logging.info("构建视频数据")
            videolist = [None]*len(self._pathList)
            for n in tqdm(range(len(self.datadict["path"])), desc="视频数据处理中: "):
                videolist[n] = self._load_video(self.datadict["path"][n])
            self.datadict["video"] = [torch.from_numpy(i) for i in videolist]
            self.datadict["video_frameNum"] = [v.shape[0]
                                               for v in self.datadict["video"]]

    def build_filter(self):
        logging.info("数据筛选...")

        keys = self.filter.keys()
        length = len(self.datadict["path"])
        replaces = self.filter["replace"] if "replace" in keys else {}
        dropna = self.filter["dropna"] if "dropna" in keys else {}
        contains = self.filter["contains"] if "contains" in keys else {}
        query = self.filter["query"] if "query" in keys else {}
        # sort_values = self.filter["sort_values"] if "sort_values" in keys else {
        # }
        dropinds = []
        for ind in range(length):
            if (self.mode in ["v", "av",  "vt", "avt"]) and ("Ses05F_script02_2_M" in self.datadict["path"][ind]):
                dropinds.append(ind)

            # 根据提供的字典删除指定值所在行, key 决定要替换的列, value 是被替换的值列表
            for k, v in dropna.items():
                for t in v:
                    if self.datadict[k][ind] == t:
                        dropinds.append(ind)

            for k, v in contains.items():
                for t in v:
                    if t not in self.datadict[k][ind]:
                        dropinds.append(ind)

            for k, v in replaces.items():
                for s, t in v.items():
                    if self.datadict[k][ind] == s:
                        self.datadict[k][ind] = t

        for k, v in query.items():
            df = pd.DataFrame({k: self.datadict[k]})
            inds = df.query(v).index.tolist()
            inds = [i for i in df.index.tolist() if i not in inds]
            dropinds = dropinds + inds

        for k, v in self.datadict.items():
            self.datadict[k] = [self.datadict[k][i]
                                for i in range(length) if i not in dropinds]


    def check_video_file(self):
        logging.info("检测视频文件夹是否存在...")
        fileExists = list(set([os.path.exists(os.path.join(
            self.root, f"Session{sess}/sentences/video/")) for sess in range(1, 6)]))
        if len(fileExists) != 1 or fileExists[0]:
            logging.info("检测视频文件是否损坏...")
            self.check_videoFile()
        elif not fileExists[0]:
            self._creat_video_process(
                self._pathList)

    def check_videoFile(self) -> bool:
        # 验证处理后或已有的视频文件数量完整性...
        no_exists_filelist = []
        corrupted_filelist = []

        for filename in self._pathList:
            filename_ = filename.split("_")
            sess = int(filename_[0][3:-1])
            if self.videomode == "crop":
                path = os.path.join(
                    self.root, f"Session{sess}/sentences/video/"+"_".join(filename_[:-1])+"/"+filename+"_crop"+".avi")
            else:
                path = os.path.join(
                    self.root, f"Session{sess}/sentences/video/"+"_".join(filename_[:-1])+"/"+filename+".avi")
            if not os.path.exists(path):
                no_exists_filelist.append(filename)
            else:
                with open(path, "rb") as fh:
                    video_file = io.BytesIO(fh.read())
                try:
                    _av_reader = decord.VideoReader(
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
            logging.info("丢失文件:")
            [logging.info("\t"+i) for i in no_exists_filelist]
            logging.info("损坏文件:")
            [logging.info("\t"+i) for i in corrupted_filelist]
            logging.info("尝试重新处理文件...")
            self._creat_video_process(
                no_exists_filelist+corrupted_filelist)
            self.check_video_file()
        else:
            logging.info("文件完整")
        return False

    def _creat_video_process(self, filelist):
        logging.info("自动提取脸部表情视频片段...")
        [create_folder(os.path.join(
            self.root, f"Session{sess}/sentences/video/")) for sess in range(1, 6)]
        # 开始处理视频文件
        if self.threads > 0:
            length = len(filelist)
            manage = mp.Manager()
            pbar_queue = manage.Queue()
            p_pbar = mp.Process(target=self.pbar_listener,
                                args=(pbar_queue, length))
            p_pbar.start()
            pool = []  # 进程池
            step = int(length / self.threads) + 1  # 每份的长度
            for i in range(0, length, step):
                t = mp.Process(target=self._creat_video, args=(
                    filelist[i: i + step], pbar_queue))
                pool.append(t)
            for t in pool:
                t.start()
            for t in pool:
                t.join()
            pbar_queue.put(-1)  # 停止监听
            p_pbar.join()
        else:
            self._creat_video(filelist)
        logging.info("视频文件处理成功")

    def _creat_video(self, filelist, pbar_queue=None):
        # 给定文件名列表, 处理视频文件函数
        for ind, filename in enumerate(self._pathList):  # Ses01F_impro01_F000
            if filename in filelist:
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
                if self.videomode == "crop":
                    video_clip_path = os.path.join(
                        videoPath, filename+"_crop"+".avi")
                else:
                    video_clip_path = os.path.join(videoPath, filename+".avi")

                origin_video_path = os.path.join(
                    self.root, f'Session{sess}/dialog/avi/DivX/'+"_".join(filename_[:-1])+".avi")
                if self.videomode == "crop":
                    capture_video(origin_video_path,
                                  video_clip_path, *self._periodList[ind], speaker)
                else:
                    capture_face_video(self.cascPATH, origin_video_path,
                                       video_clip_path, *self._periodList[ind], speaker)
                if pbar_queue:
                    pbar_queue.put(1)

    def _get_speaker(self, filename) -> str:
        # 给定文件名获取此文件对应说话人性别
        if "_F" in filename:
            return("F")
        elif "_M" in filename:
            return("M")
        else:
            return("")

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

    def thread_load_video(self, data_queue, index, pbar_queue=None):
        for i in index:
            data_queue.put({i: self._load_video(self.datadict["path"][i])})
            if pbar_queue:
                pbar_queue.put(1)

    def thread_load_audio(self, data_queue, index, pbar_queue=None):
        for i in index:
            data_queue.put({i: self._load_audio(self.datadict["path"][i])})
            if pbar_queue:
                pbar_queue.put(1)

    def _load_audio(self, filename) -> torch.Tensor:
        # 指定文件名加载语音文件数据
        filename_ = filename.split("_")
        sess = int(filename_[0][3:-1])
        wav_path = os.path.join(
            self.root, f"Session{sess}/sentences/wav/"+"_".join(filename_[:-1])+"/"+filename+".wav")

        if audioread.ffdec.available():
            wav_path = audioread.ffdec.FFmpegAudioFile(wav_path)

        y, sr = librosa.load(wav_path, sr=None)
        return y[None, :]  # 1xseq

    def _load_video(self, filename) -> torch.Tensor:
        # 指定文件名加载视频文件数据
        filename_ = filename.split("_")
        sess = int(filename_[0][3:-1])
        
        # 视频文件路径
        if self.videomode == "crop":
            video_path = os.path.join(
                self.root, f"Session{sess}/sentences/video/"+"_".join(filename_[:-1])+"/"+filename+"_crop"+".avi")
        else:
            video_path = os.path.join(
                self.root, f"Session{sess}/sentences/video/"+"_".join(filename_[:-1])+"/"+filename+".avi")
        # 3xNxHxW
        return self.process_video(video_path, num_threads=self.threads)

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
            _av_reader = decord.VideoReader(
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
        self._fps = _av_reader.get_avg_fps()
        self._duration = float(len(_av_reader)) / float(self._fps)
        frame_idxs = list(range(0, len(_av_reader)))
        try:
            video = _av_reader.get_batch(frame_idxs)
        except Exception as e:
            raise(f"Failed to decode video with Decord: {video_path}. {e}")
        out = video.asnumpy()[:, 10:160, 110:260, :].copy()  # NxHxWx3
        # out = video.asnumpy()[:, 20:150, 90:280, :].copy()  # NxHxWx3
        del video
        gc.collect()
        return out

    def __getitem__(self, n: int):
        """加载 IEMOCAP 数据库的第 n 个样本数据
        Args:
            n (int): 要加载的样本的索引
        Returns:
            Dict[str, Optional[Union[int, torch.Tensor, torch.Tensor, int, str, str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor], str]]]
        """
        out = {"id": n,
               "label": self.datadict["label"][n],
               "val/act/dom": self.datadict["dimvalue"][n],
               "gender": self.datadict["gender"][n],
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
        """为 IEMOCAP 数据库类型重写 len 默认行为.
        Returns:
            int: IEMOCAP 数据库数据数量
        """
        return len(self.datadict["path"])


if __name__ == '__main__':
    a = IEMOCAP("/home/visitors2/SCW/deeplearning/data/IEMOCAP", True,
                "/home/visitors2/SCW/deeplearningcopy/utils/haarcascade_frontalface_alt2.xml", 0)
    feature = a.__getitem__(1000)

    pass
