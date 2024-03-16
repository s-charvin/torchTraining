
import numpy as np
import torchaudio
from transformers import AutoConfig, Wav2Vec2Processor, Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC
import csv
import os
import sys
os.environ['CUDA_VISIBLE_DVICES'] = ''
if sys.platform.startswith('linux'):
    os.chdir("/home/user0/SCW/deeplearning/")
elif sys.platform.startswith('win'):
    pass


def read_csv(path):
    """读取csv文件,构成列表

    Args:
        path (str): csv文件路径
    """
    data = []
    emotion = []
    with open(path) as csvfile:
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
        header = next(csv_reader)        # 读取第一行每一列的标题
        for row in csv_reader:            # 将csv文件中的数据保存到data中
            data.append(row[0])  # 选择某一列加入到data数组中
            emotion.append(row[1])
    return data, emotion  # [path1,path2,....]


def gernerate_feature(path_list, output_path):
    """使用wav2vec生成特征

    Args:
        path_list (list): 含有目标文件路径的列表
        indexx (dict): 标签与index对应的字典

    gernerate: {文件名:[特征]}
    """
    content = {}
    model_name_or_path = "lighteternal/wav2vec2-large-xlsr-53-greek"
    processor = Wav2Vec2Processor.from_pretrained(model_name_or_path)
    model = Wav2Vec2ForCTC.from_pretrained(model_name_or_path)
    target_sampling_rate = processor.feature_extractor.sampling_rate
    print(f"The target sampling rate: {target_sampling_rate}")

    for i in range(len(path_list)):
        speech_array, sampling_rate = torchaudio.load(path_list[i])
        resampler = torchaudio.transforms.Resample(
            sampling_rate, target_sampling_rate)  # 将信号从一个频率重新采样到另一个频率。
        speech = resampler(speech_array).squeeze().numpy()

        result = processor(
            speech, sampling_rate=target_sampling_rate, return_tensors="pt")
        output = model(result['input_values'])
        logits = model(result['input_values']).logits
        feature = result["input_values"]
        feature = np.array(feature)
        content.update({path_list[i]: feature})
    np.save(output_path, content)


csv_path = [
    "./data/EMODB/EMODB_all.csv",
    "./data/IEMOCAP/IEMOCAP_all.csv"
]
output_path = [os.path.splitext(i)[0] + "_wa2vec.npy" for i in csv_path]
# 读取路径

for i in range(len(csv_path)):
    path_list, emotion_list = read_csv(csv_path[i])  # 读取
    num_labels = len(set(emotion_list))
    gernerate_feature(path_list, output_path[i])  # 生成
