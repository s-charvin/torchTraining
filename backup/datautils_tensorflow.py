import os
import sys

import scipy
if sys.platform.startswith('linux'):
    os.chdir("/home/user0/SCW/deeplearning/")
elif sys.platform.startswith('win'):
    pass
import re
import math
import yaml  # yaml文件解析模块

import soundfile as sf
import librosa

import pandas as pd
import numpy as np
from random import random
import matplotlib.pyplot as plt
import sklearn.preprocessing

from transformers import Wav2Vec2FeatureExtractor
import tensorflow_datasets.public_api as tfds
print(os.getcwd())


class AudioDataset():
    def __init__(self, config):
        """类的初始化
        """
        super(AudioDataset, self).__init__()
        self.config = config

        self.data = self._getdata()  # 获取csv文件中的数据库信息
        self.dataset_name = self.config["data"]["data_dir"].split("/")[2]
        self.featuresource = self.config["data"]["featuresource"]
        self._get_info()

        if self.featuresource == "wav_pretrained":
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
                self.config["wev2vec_pretrained"]["path"])
        elif self.featuresource == "local":
            # 从文件中获取
            self.features = np.load(
                self.config["data"]["features_path"], allow_pickle=True).item()

    def _get_info(self):
        self.indexx = None
        self.wav_maxlength = None
        self.wav_minlength = None

        lengths = [len(self.getspeech(self.audio[i])[0])
                   for i in range(len(self.audio))]
        self.wav_maxlength = max(lengths)
        self.wav_minlength = min(lengths)

        sampling_rates = [self.getspeech(self.audio[i])[1]
                          for i in range(len(self.audio))]

        if sampling_rates == self.sampling_rates:
            pass
        else:
            raise Exception("语音采样率不统一")
        le = sklearn.preprocessing.LabelEncoder()
        self.indexx = le.fit(y=self.labels)  # 定义文字标签与数值的对应关系
        # self.classes = self.indexx.classes_
        self.config["data"]["wav_maxlength"] = self.wav_maxlength
        self.config["data"]["wav_minlength"] = self.wav_minlength
        self.config["data"]["indexx"] = self.indexx
        # self.config["data"]["classes"] = self.classes

    def _getdata(self):
        # 读取csv文件,指定xxx,oth和NA(默认)为空，获取数据库信息
        data_dir = self.config["data"]["data_dir"]
        try:
            Data = pd.read_csv(
                data_dir, na_values=self.config["data"]["na_values"], usecols=None)
        except:
            print("读取文件失败, 请确认输入参数或语音数据路径是否正确")

        Data.replace(None if self.config["data"]["mergecols"] ==
                     "None" else self.config["data"]["mergecols"], inplace=True)
        # 指定检查列,按行删除其中包含空字段的行,
        # 要求空字段小于等于1个的留下,
        # False指定不覆盖原数据. how="all":当整行全为空才删除
        Data.dropna(axis=0, how='any', thresh=None,
                    subset=None, inplace=True)

        if self.config["data"]["scrip_impro"]:
            Data = Data[Data['path'].str.contains(
                self.config["data"]["scrip_impro"])]

        if self.config["data"]["conditions"]:
            Data.query(self.config["data"]["conditions"], inplace=True)

        if self.config["data"]["sorting"]:
            Data.sort_values(by="duration", inplace=True, ascending=False)
        try:
            self.audio = Data["path"].tolist()
            self.labels = Data["emotion"].tolist()
            self.duration = Data["duration"].tolist()
            self.sampling_rates = Data["sr"].tolist()
        except Exception as e:
            raise KeyError(
                "语音必需的数据(\"path\", \"emotion\", \"duration\", \"sr\")加载失败")
        try:
            self.text = Data["text"]
        except:
            self.text = None
        try:
            self.valence = Data["val"]
        except:
            self.valence = None
        try:
            self.activation = Data["act"]
        except:
            self.activation = None
        try:
            self.dominance = Data["dom"]
        except:
            self.dominance = None
        try:
            self.gender = Data["gender"]
        except:
            self.gender = None

        return Data

    def _extract_audio_feat(self, file_path):
        """
        :param file_path: 语音文件路径
        :param file_path: 数据库名称,方便获取对应的特征
        :return: extract audio feature.
        """
        feature = []
        if self.featuresource == "local":
            # 从文件中获取
            audio_feature = self.features[file_path]
            return audio_feature
        elif self.featuresource == "wav":
            speech_array, sampling_rate = self.getspeech(file_path)
            speech_array = self.speech_padcut(speech_array)
            return speech_array

        elif self.featuresource == "mfcc":
            speech_array, sr = self.getspeech(file_path)
            speech_array = self.speech_padcut(speech_array)
            mfccs = librosa.feature.mfcc(
                y=speech_array,
                sr=sr,
                n_mfcc=40,
                dct_type=2,
                norm='ortho',
                # log_mels=True,
                n_fft=1024,
                hop_length=256,
                win_length=1024,
                n_mels=128,
                window=scipy.signal.windows.hamming,
                center=True,
                pad_mode="reflect",
                power=2,
                fmin=0.,
                fmax=None,
                htk=True,
            ).T

            return mfccs

        elif self.featuresource == "spectrogram":
            speech_array, sr = self.getspeech(file_path)
            speech_array = self.speech_padcut(speech_array)
            S = np.abs(librosa.stft(
                y=speech_array,
                n_fft=1024,
                hop_length=256,
                win_length=1024,
                window=scipy.signal.windows.hamming,
                center=True,
                pad_mode="reflect",
            ))
            spectrogram = librosa.power_to_db(S ** 2)
            spectrogram = spectrogram[0:400, :]
            return spectrogram.T

        elif self.featuresource == "melspectrogram":

            speech_array, sr = self.getspeech(file_path)
            speech_array = self.speech_padcut(speech_array)
            melspectrogram = librosa.feature.melspectrogram(
                y=speech_array,
                sr=sr,
                n_fft=1024,
                hop_length=256,
                win_length=1024,
                n_mels=128,
                window=scipy.signal.windows.hamming,
                center=True,
                pad_mode="reflect",
                power=2,
                fmin=0.,
                fmax=None,
                norm=None,
                htk=True,
            )

            return melspectrogram.T
        elif self.featuresource == "wav_pretrained":
            speech_array, sr = self.getspeech(file_path)
            speech_array = self.speech_padcut(speech_array)
            input_values = self.processor(
                speech_array, return_tensors="np", sampling_rate=sr).input_values  # Batch size 1
            return input_values

    def getspeech(self, file_path):
        speech_array, sr = librosa.load(file_path, sr=None)
        return speech_array, sr

    def speech_padcut(self, speech_array):
        if self.config["data"]["wavpadcut"] == "wavmid":
            self.wav_midlength = int(
                self.wav_minlength + (self.wav_maxlength-self.wav_minlength)/2.0)
            length = speech_array.shape[0]
            if length <= self.wav_midlength:
                speech_array = np.pad(
                    speech_array, (0, self.wav_midlength-length), 'constant', constant_values=0)
            elif length > self.wav_midlength:
                left_cut = int((length-self.wav_midlength)/2.0)
                right_cut = length-self.wav_midlength-left_cut
                speech_array = speech_array[left_cut:-right_cut]
            return speech_array
        elif self.config["data"]["wavpadcut"] == "wavmax":
            return np.pad(speech_array, (0, self.wav_maxlength-len(speech_array)), 'constant', constant_values=0)
        elif self.config["data"]["wavpadcut"] == "wavmin":
            return speech_array[:self.wav_minlength]
        elif isinstance(self.config["data"]["wavpadcut"], int):
            length = speech_array.shape[0]
            if length <= self.config["data"]["wavpadcut"]:
                return speech_array
            elif length > self.wav_midlength:
                return speech_array[:self.config["data"]["wavpadcut"]]
        else:
            return speech_array

    def get_textlabels(self, index):
        """对于分类型数据库, 输入数字型标签, 返回文本标签。"""
        text_labels = self.indexx.inverse_transform(index)
        return list(text_labels)

    def __len__(self):
        """返回整个数据集的长度"""
        return len(self.audio)

    def __getitem__(self, index):
        """定义怎么读数据，返回指定位置的数据和标签
        return tensor([帧数, 特征数]), int
        """
        feature = self._extract_audio_feat(
            self.audio[index])
        label = self.labels[index]
        a = self.indexx.transform([label])
        return feature, int(a)


class Subset():
    r"""
    Subset of a dataset at specified indices.

    Args:
        dataset (Dataset): The whole Dataset
        indices (list): Indices in the whole set selected for subset
    """

    def __init__(self, dataset, indices) -> None:
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]]
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


def random_split(dataset, lengths):
    r"""
    随机地将数据集分割成给定长度的不重叠的新数据集。
    Args:
        dataset (Dataset): Dataset to be split
        lengths (sequence[int]): lengths of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):
        raise ValueError("分割后的数据总数量与输入数据集的数据数量不等！")
    np.random.shuffle(np.arange(sum(lengths)))
    indices = np.random.permutation(sum(lengths)).tolist()
    temp = 0
    offsets = []
    for i in lengths:
        temp = temp+i
        offsets.append(temp)

    return [Subset(dataset, indices[offset - length: offset]) for offset, length in zip(offsets, lengths)]


class create_csv:
    def __init__(self, name) -> None:
        """创建音频列表文件, CSV格式

        Args:
            name (str): 数据库名称(数据库文件所在文件夹名称)
        """
        self.name = name

        if self.name == "EMODB":
            self.write_EMODB_csv()
        elif self.name == "IEMOCAP":
            self.write_IEMOCAP_csv()
        elif self.name == "RAVDESS":
            self.write_RAVDESS_csv()
        elif self.name == "AESDD":
            self.write_AESDD_csv()
        elif self.name == "SAVEE":
            self.write_SAVEE_csv()
        elif self.name == "TESS":
            self.write_TESS_csv()
        elif self.name == "CASIA":
            self.write_CASIA_csv()
        elif self.name == "eNTERFACE05":
            self.write_eNTERFACE05_csv()

    def write_IEMOCAP_csv(self):
        """
        从data\IEMOCAP\Session{num}}\sentences\wav目录中读取IEMOCAP语音数据集，并将文件名、情感标签、文本、维度值写入CSV文件,默认格式为：数据集名称_*.csv。

        """
        # indexx = {  # 情感与情感序号的对应关系
        #     "angry": 0,
        #     "happy": 1,
        #     "neutral": 2,
        #     "sad": 3,
        #     "fearful": 4,
        #     "surprise": 5,
        #     "disgusted": 6,
        #     "excited": 7,
        #     "frustrated": 8,
        #     "other": 9,
        #     "NA": 10}
        # 文件情感标签与序号的对应关系
        indexx = {"ang": "angry",  # angry
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
        info_line = re.compile(r'\[.+\]\n', re.IGNORECASE)  # 匹配被[]包裹的数据行
        target = {"path": [], "emotion": [], "text": [], "val": [], "act": [
        ], "dom": [], "gender": [], "start_time": [], "end_time": [], "wav_file_name": [], "sr": [], "duration": []}  # 存放语音文件地址及其对应情感标签、文本等标签数据

        for sess in range(1, 6):  # 遍历Session{1-6}文件夹
            # 指定和获取文件地址
            # 语音情感标签所在文件夹
            emo_evaluation_dir = f'./data/IEMOCAP/Session{sess}/dialog/EmoEvaluation/'
            # 语音文本所在文件夹
            transcriptions_dir = f'./data/IEMOCAP/Session{sess}/dialog/transcriptions/'
            wav_dir = f'./data/IEMOCAP/Session{sess}/sentences/wav/'  # 语音所在文件夹
            evaluation_files = [l for l in os.listdir(
                emo_evaluation_dir) if ('Ses' in l and '._Ses' not in l)]  # 语音情感标签所在文件列表
            transcription_files = [l for l in os.listdir(
                transcriptions_dir) if ('Ses' in l and '._Ses' not in l)]  # 语音文本所在文件列表

            # 获取文件名、情感标签、维度值、开始结束时间、性别
            for file in evaluation_files:  # 遍历当前Session的EmoEvaluation文件
                with open(emo_evaluation_dir + file, encoding='utf-8') as f:
                    content = f.read()
                # 匹配正则化模式对应的所有内容，得到数据列表
                # 匹配被[]包裹的数据行,数据示例: [开始时间 - 结束时间] 文件名称 情感 [V, A, D]
                info_lines = re.findall(info_line, content)
                for line in info_lines[1:]:  # 忽略第一行，无用，不是想要的数据
                    # strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。
                    # .split('\t') 方法用于以制表符为基，分割此字符串
                    start_end_time, wav_file_name, emotion, val_act_dom = line.strip().split('\t')
                    # 获取当前语音段的开始时间和结束时间
                    start_time, end_time = start_end_time[1:-1].split('-')
                    start_time, end_time = float(start_time), float(end_time)
                    # 获取情感维度值
                    val, act, dom = val_act_dom[1:-1].split(',')
                    val, act, dom = float(val), float(act), float(dom)

                    # 将数据存储在字典的对应列表中
                    target["emotion"].append(indexx[emotion])
                    target["path"].append(
                        wav_dir+wav_file_name[:-5]+"/"+wav_file_name+".wav")  # 获取当前行对应语音段文件的地址
                    target["text"].append("")  # 文本占位,方便后面存储对应文本数据
                    target["val"].append(val)
                    target["act"].append(act)
                    target["dom"].append(dom)

                    target["start_time"].append(start_time)
                    target["end_time"].append(end_time)
                    target["wav_file_name"].append(wav_file_name)
                    if "_F" in wav_file_name:
                        target["gender"].append("F")
                    elif "_M" in wav_file_name:
                        target["gender"].append("M")
                    else:
                        target["gender"].append("")

                    params = sf.info(
                        wav_dir+wav_file_name[:-5]+"/"+wav_file_name+".wav")
                    target["sr"].append(params.samplerate)  # 采样频率
                    target["duration"].append(params.duration)
            # 获取音频对应翻译
            for file in transcription_files:
                with open(transcriptions_dir + file, encoding='utf-8') as f:
                    for line in f.readlines():
                        try:
                            filename, _temp = line.strip().split(' [', 1)
                            text = _temp.strip().split(']: ', 1)[1]
                            # text = re.sub("\[.+\]", "", text) # 是否替换Text中的声音词，如哈哈大笑
                            index = target["wav_file_name"].index(filename)
                            target["text"][index] = text
                        except:  # 出错就跳过，一般是遇到....（待定）
                            continue

        # 保存为CSV文件,数据包含音频文件地址，情感标签，翻译，维度值，性别，时长
        datanum = len(target['path'])
        pd.DataFrame({
            "path": target['path'],
            "emotion": target['emotion'],
            "text": target["text"],
            "val": target["val"],
            "act": target["act"],
            "dom": target["dom"],
            "gender": target["gender"],
            "sr": target["sr"],
            "duration": target["duration"]
        }).to_csv(f"./data/{self.name}/{self.name}_all.csv", index=False)
        print(f"[{self.name}] Total files to write:", datanum)

    def write_EMODB_csv(self):
        """
        从data\EMODB\wav目录中读取emodb语音数据集, 并将文件名、情感标签、文本写入CSV文件,默认格式为: 数据集名称_all.csv。
        Documentation: 原数据库文件名含义
            Positions 1-2: number of speaker
                03 - male, 31 years old
                08 - female, 34 years
                09 - female, 21 years
                10 - male, 32 years
                11 - male, 26 years
                12 - male, 30 years
                13 - female, 32 years
                14 - female, 35 years
                15 - male, 25 years
                16 - female, 31 years)
            Positions 3-5: 文本编码: textcode
            Position 6: 情感标签编码: categories
            Position 7: if there are more than two versions these are numbered a, b, c
        """

        indexx = {  # 文件标签与情感的对应关系
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
        target = {"path": [], "emotion": [], "gender": [],
                  "text": [], "sr": [], "duration": []}  # 存放语音文件地址及其对应情感标签、文本
        wavpath = f"./data/{self.name}/wav/"  # 数据库所在地址
        # 获取指定目录下所有.wav文件列表
        files = [l for l in os.listdir(wavpath) if ('.wav' in l)]
        for file in files:  # 遍历所有文件
            try:
                emotion = indexx[os.path.basename(file)[5]]  # 从文件名中获取情感标签
                text = textcode[os.path.basename(file)[2:5]]  # 从文件名中获取情感标签
            except KeyError:
                continue
            target['path'].append(wavpath+file)
            target['emotion'].append(emotion)
            target['text'].append(text)
            # 获取说话人性别
            if os.path.basename(file)[0:2] in ["03", "10", "12", "15", "11"]:
                target["gender"].append("M")
            elif os.path.basename(file)[0:2] in ["08", "09", "13", "14", "16"]:
                target["gender"].append("F")
            else:
                target["gender"].append("")
            params = sf.info(wavpath+file)
            target["sr"].append(params.samplerate)  # 采样频率
            target["duration"].append(params.duration)
        # 保存为CSV文件,数据包含音频文件地址，情感标签，翻译，性别
        datanum = len(target['path'])
        pd.DataFrame({"path": target['path'],
                      "emotion": target['emotion'],
                      "text": target['text'],
                      "gender": target["gender"],
                      "sr": target["sr"],
                      "duration": target["duration"]
                      }).to_csv(
            f"./data/{self.name}/{self.name}_all.csv", index=False)
        print(f"[{self.name}] Total files to write:", datanum)

    def write_RAVDESS_csv(self):
        """
        从data\AESDD\目录中读取RAVDESS语音数据集,并将文件名、情感标签、语音类型、情绪强度、文本、性别写入CSV文件,默认格式为: 数据集名称_all.csv。
        Documentation: 原数据库文件名含义
            Positions 0: 模态类型(01 = 音频+视频,02 = 仅视频,03 = 仅音频）。
            Positions 1: 语音类型(01 =语音,02 =歌曲）
            Position 2: 情绪(01 =neutral, 02 =calm, 03 =happy, 04 =sad, 05 =angry, 06 =fearful, 07 =disgust, 08 =surprised)
            Position 3: 情绪强度(01 =正常,02 =强烈）
            Position 4: 语音文本(01 ="Kids are talking by the door",02 ="Dogs are sitting by the door")
            Position 5: Repetition(01 = 1st repetition, 02 = 2nd repetition)
            Position 6: 演员(01至24。奇数演员是男性,偶数演员是女性）。
        """
        indexx = {
            '01': "neutral",  # neutral
            '02': "calm",  # calm
            '03': "happy",  # happy
            '04': "sad",  # sad
            '05': "angry",  # angry
            '06': "fearful",  # fearful
            '07': "disgusted",  # disgust
            '08': "surprised"  # surprised
        }
        path = f"./data/{self.name}/"  # 数据库所在地址
        target = {"path": [], "emotion": [], "Emotion_intensity": [], "Vocaltype": [], "gender": [],
                  "text": [], "sr": [], "duration": []}  # 存放语音文件地址及其对应情感标签、文本
        Actor_i = [l for l in os.listdir(path) if ("Actor_" in l)]
        for Actor in Actor_i:
            for file in os.listdir(path+Actor):
                filename = os.path.basename(file).split(".")[0].split("-")
                target["path"].append(os.path.join(path, Actor, file))
                target["emotion"].append(indexx[filename[2]])
                target["Emotion_intensity"].append(int(filename[3]))
                target["Vocaltype"].append(int(filename[1]))
                if filename[4] == "01":
                    target["text"].append("Kids are talking by the door")
                else:
                    target["text"].append("Dogs are sitting by the door")
                if int(filename[6]) % 2 == 0:
                    target["gender"].append("M")
                else:
                    target["gender"].append("F")
                params = sf.info(os.path.join(path, Actor, file))
                target["sr"].append(params.samplerate)  # 采样频率
                target["duration"].append(params.duration)
        # 保存为CSV文件,数据包含音频文件地址，情感标签，翻译，性别
        datanum = len(target['path'])
        pd.DataFrame({"path": target['path'],
                      "emotion": target['emotion'],
                      "text": target['text'],
                      "gender": target["gender"],
                      "Emotion_intensity": target["Emotion_intensity"],
                      "Vocaltype": target["Vocaltype"],
                      "sr": target["sr"],
                      "duration": target["duration"]
                      }).to_csv(
            f"./data/{self.name}/{self.name}_all.csv", index=False)
        print(f"[{self.name}] Total files to write:", datanum)

    def write_AESDD_csv(self):
        """
        从data\AESDD\wav目录中读取AESDD语音数据集,并将文件名、情感标签、语音类型、情绪强度、文本、性别写入CSV文件,默认格式为: 数据集名称_all.csv。
        Documentation: 原数据库文件名含义
            Positions 0: 情绪(a: anger, d: disgust, f: fear, h: happiness, s: sadness)
            Positions 1-2: 说话人(1-6)
            Position 4: 语音文本(1-20)
            Position -1: 如果是b, 则说明是重复说话
        """
        indexx = {
            'a': "angry",  # anger
            'd': "disgusted",  # disgust
            'f': "fearful",  # fear
            'h': "happy",  # happiness
            's': "sad",  # sadness
        }
        emotions = ["anger", "disgust", "fear", "happiness", "sadness"]
        path = f"./data/{self.name}/"  # 数据库所在地址
        # 存放语音文件地址及其对应情感标签
        target = {"path": [], "emotion": [], "sr": [], "duration": []}

        for emotion in emotions:
            for file in os.listdir(path+emotion):
                filename = os.path.basename(file)
                target["path"].append(os.path.join(path, emotion, file))
                target["emotion"].append(indexx[filename[0]])

                params = sf.info(os.path.join(path, emotion, file))
                target["sr"].append(params.samplerate)  # 采样频率
                target["duration"].append(params.duration)
        # 保存为CSV文件,数据包含音频文件地址，情感标签，翻译，性别
        datanum = len(target['path'])
        pd.DataFrame({"path": target['path'],
                      "emotion": target['emotion'],
                      "sr": target["sr"],
                      "duration": target["duration"]
                      }).to_csv(
            f"./data/{self.name}/{self.name}_all.csv", index=False)
        print(f"[{self.name}] Total files to write:", datanum)

    def write_SAVEE_csv(self):
        """
        从data\SAVEE\wav目录中读取SAVEE语音数据集,并将文件名、情感标签、语音类型、情绪强度、文本、性别写入CSV文件,默认格式为: 数据集名称_all.csv。
        Documentation: 原数据库文件名含义
            Positions 0: 情绪(a: anger, d: disgust, f: fear, h: happiness, n: neutral, sa: sadness, su: surprise)
            Positions 1-2: 语音文本(1-15)
        """
        indexx = {
            'a': "angry",  # anger
            'd': "disgusted",  # disgust
            'f': "fearful",  # fear
            'h': "happy",  # happiness
            'n': "neutral",  # neutral
            'sa': "sad",  # sadness
            'su': "surprised"  # surprise

        }
        actors = ['DC', 'JE', 'JK', 'KL']
        path = f"./data/{self.name}/"  # 数据库所在地址
        # 存放语音文件地址及其对应情感标签、文本
        target = {"path": [], "emotion": [], "sr": [], "duration": []}

        for actor in actors:
            for file in os.listdir(path+actor):
                filename = os.path.basename(file)
                target["path"].append(os.path.join(path, actor, file))
                target["emotion"].append(indexx[filename[:-6]])
                params = sf.info(os.path.join(path, actor, file))
                target["sr"].append(params.samplerate)  # 采样频率
                target["duration"].append(params.duration)
        # 保存为CSV文件,数据包含音频文件地址，情感标签
        datanum = len(target['path'])
        pd.DataFrame({"path": target['path'],
                      "emotion": target['emotion'],
                      "sr": target["sr"],
                      "duration": target["duration"]
                      }).to_csv(
            f"./data/{self.name}/{self.name}_all.csv", index=False)
        print(f"[{self.name}] Total files to write:", datanum)

    def write_TESS_csv(self):
        """
        从data\TESS\wav目录中读取TESS语音数据集,并将文件名、情感标签、语音类型、情绪强度、文本、性别写入CSV文件,默认格式为: 数据集名称_all.csv。
        Documentation: 原数据库文件名含义
            Positions 0: 情绪(anger, disgust, fear, happiness, neutral, pleasant surprise, sadness)
        """
        indexx = {
            'angry': "angry",  # anger
            'disgust': "disgusted",  # disgust
            'fear': "fearful",  # fear
            'happy': "happy",  # happiness
            'neutral': "neutral",  # neutral
            'ps': "surprised-pleased",  # pleasant surprise
            'sad': "sad"  # sadness

        }
        path = f"./data/{self.name}/"  # 数据库所在地址
        # 存放语音文件地址及其对应情感标签、文本
        target = {"path": [], "emotion": [], "sr": [], "duration": []}
        files = [l for l in os.listdir(path) if (".wav" in l)]
        for file in files:
            target["path"].append(os.path.join(path, file))
            target["emotion"].append(
                indexx[os.path.basename(file).split("_")[2][:-4]])

            params = sf.info(os.path.join(path, file))
            target["sr"].append(params.samplerate)  # 采样频率
            target["duration"].append(params.duration)
        # 保存为CSV文件,数据包含音频文件地址，情感标签
        datanum = len(target['path'])
        pd.DataFrame({"path": target['path'],
                      "emotion": target['emotion'],
                      "sr": target["sr"],
                      "duration": target["duration"]
                      }).to_csv(
            f"./data/{self.name}/{self.name}_all.csv", index=False)
        print(f"[{self.name}] Total files to write:", datanum)

    def write_CASIA_csv(self):
        pass

    def write_eNTERFACE05_csv(self):
        """
        从data\eNTERFACE05\wav目录中读取eNTERFACE05语音数据集,并将文件名、情感标签、文本写入CSV文件,默认格式为: 数据集名称_all.csv。
        Documentation: 原数据库文件名含义
            Positions 0: 情绪(angry, disgusted, fearful, happy, sad,surprise)
        """
        path = f"./data/{self.name}/wav"  # 数据库所在地址
        # 存放语音文件地址及其对应情感标签、文本
        target = {"path": [], "emotion": [], "sr": [], "duration": []}
        files = [l for l in os.listdir(path) if (".wav" in l)]
        for file in files:
            target["path"].append(os.path.join(path, file))
            target["emotion"].append(os.path.basename(
                file).split(".")[0].split("_")[1])

            params = sf.info(os.path.join(path, file))
            target["sr"].append(params.samplerate)  # 采样频率
            target["duration"].append(params.duration)
        # 保存为CSV文件,数据包含音频文件地址，情感标签
        datanum = len(target['path'])
        pd.DataFrame({"path": target['path'],
                      "emotion": target['emotion'],
                      "sr": target["sr"],
                      "duration": target["duration"]
                      }).to_csv(
            f"./data/{self.name}/{self.name}_all.csv", index=False)
        print(f"[{self.name}] Total files to write:", datanum)


def search_files(path, all_files):
    # 首先遍历当前目录所有文件及文件夹
    file_list = os.listdir(path)
    # 准备循环判断每个元素是否是文件夹还是文件，是文件的话，把名称传入list，是文件夹的话，递归
    for file in file_list:
        # 利用os.path.join()方法取得路径全名，并存入cur_path变量，否则每次只能遍历一层目录
        cur_path = os.path.join(path, file)
        # 判断是否是文件夹
        if os.path.isdir(cur_path):
            search_files(cur_path, all_files)
        else:
            if ".avi" in os.path.basename(file):
                all_files.append(cur_path)
    return all_files


def extract_audio():
    input_dir = "./data/eNTERFACE05/"
    sr = 16000
    all_files = []
    search_files(input_dir, all_files)
    for file in all_files:
        ouput_path = "./data/eNTERFACE05/wav/" + \
            os.path.basename(file).split(".")[0]+".wav"
        command = f'ffmpeg -i "{file}" -vn -acodec pcm_s16le -f s16le -ac 1 -ar {sr} -f wav "{ouput_path}"'
        os.system(command)


if __name__ == '__main__':
    # # 建立CSV文件，随机建立，不到万不得已, 误用！！
    # extract_audio()
    # create_csv("IEMOCAP")
    # create_csv("EMODB")
    # create_csv("RAVDESS")
    # create_csv("AESDD")
    # create_csv("SAVEE")
    # create_csv("TESS")
    # create_csv("CASIA")
    # create_csv("eNTERFACE05")

    # AudioDataset 测试(输入路径csv,数据库名称建立数据类,然后通过索引获取特征数据)
    config = yaml.safe_load(
        open('./config.yaml', 'r', encoding='utf-8').read())
    # config["data"]["featuresource"] = "melspectrogram"
    a = AudioDataset(config)
    feature, label = a.__getitem__(1000)
    labels = a.get_textlabels([1, 2, 3])

    # mnist = TorchVisionDatasets("CIFAR10")
    # mnist.get_Dataset()

    # # plt.style.use('seaborn-deep')
    # fig, ax = plt.subplots()
    # duration = sorted(a.duration)
    # # duration = duration[int(len(duration)*0.2):-int(len(duration)*0.1)]
    # print(f"min:{duration[0]}, max:{duration[-1]}")
    # ax.plot(duration)
    # # ax.set_xlim(left=1000)
    # # ax.set_ylim(bottom = 2.5)

    # plt.savefig("./test.png")
    pass
