import os
import sys
import threading
if sys.platform.startswith('linux'):
    os.chdir("/home/user0/SCW/deeplearning/")
elif sys.platform.startswith('win'):
    pass
import csv
import numpy as np
import torchaudio
import torch.nn.functional as F


def read_csv(path):
    """读取csv文件,构成列表

    Args:
        path (str): csv文件路径
    """
    data = []
    with open(path) as csvfile:
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
        header = next(csv_reader)        # 读取第一行每一列的标题
        pass
        for row in csv_reader:            # 将csv文件中的数据保存到data中
            data.append(row[0])  # 选择某一列加入到data数组中
    return data

# 生成特征


def paded_file(file_path, speech_cache_path, max_len):
    metadata = torchaudio.info(file_path)
    speech, sample_rate = torchaudio.load(file_path)
    paded_speech = F.pad(
        speech, (max_len-metadata.num_frames, 0), 'constant', 0)
    torchaudio.save(speech_cache_path, paded_speech, sample_rate=metadata.sample_rate,
                    encoding=metadata.encoding, bits_per_sample=metadata.bits_per_sample)


def get_info(data):
    max_len = 0
    sample_rate = []
    for path in data:
        temp = rf"{path}"
        metadata = torchaudio.info(temp)
        if max_len < metadata.num_frames:
            max_len = metadata.num_frames
        sample_rate.append(metadata.sample_rate)
    return max_len, list(set(sample_rate))


def gernerate_feature(data, output_path, speech_cache_path, max_len):
    """使用opensmile生成特征

    Args:
        data (list): 含有目标文件路径的列表

    gernerate: arff格式的文本文件(数据包含[文件名,特征..,..,..])
    """
    for file_path in data:
        paded_file(file_path, speech_cache_path, max_len)
        cmd = f"SMILExtract "\
            + '-C \"C:\Program Files\opensmile\config\is09-13\IS09_emotion.conf\" '\
            + f"-I \"{speech_cache_path}\" "\
            + f"-instname \"{file_path}\" "\
            + f"-lldarffoutput \"{output_path}\" "\
            + "-appendarfflld 1 "\
            + "-relation openSMILE_features "\
            + "-timestamparfflld 0 "\
            + "-classtype numeric "
        os.system(cmd)

# 格式转换


def arrfToarray(file_path, headline=False):
    with open(file_path, encoding="utf-8") as f:
        content = f.readlines()
        name, ext = os.path.splitext(f.name)  # 获取文件地址
        data_flag = False
        header = ""
        new_content = {}
        for line in content:
            if not data_flag:
                if headline:  # 标题代码未测试,运行正常未知
                    if("@attribute" in line):
                        attributes = line.split()
                        attri_case = "@attribute"
                    elif("@ATTRIBUTE" in line):
                        attributes = line.split()
                        attri_case = "@ATTRIBUTE"
                        column_name = attributes[attributes.index(
                            attri_case) + 1]
                        header = header + column_name + ","  # 获取文件数据的属性,即列标题
                    elif "@DATA" in line or "@data" in line:
                        data_flag = True
                        header = header[:-1] + '\n'  # 去掉最后一个逗号,再加个换行
                        new_content.update({"path": header})  # 将数据属性作为第一个字典内容
                elif "@DATA" in line or "@data" in line:
                    data_flag = True
            elif line == '\n':
                continue
            else:
                line_content = line.rstrip(',?\n')
                line_content = line_content.split(",")
                filepath = line_content[0]
                data = np.array(line_content[1:], dtype='float32')
                feature = new_content.get(filepath)
                if feature is not None:
                    feature = np.concatenate((feature, np.array([data])))
                else:
                    feature = np.array([data])
                new_content.update({filepath: feature})
        np.save(name+'_paded.npy', new_content)


path = [
    # "./data/EMODB/EMODB_all.csv",
    # "./data/IEMOCAP/IEMOCAP_all.csv",
    "./data/eNTERFACE05/eNTERFACE05_all.csv",
    # "./data/AESDD/AESDD_all.csv",
    # "./data/RAVDESS/RAVDESS_all.csv",
    # "./data/SAVEE/SAVEE_all.csv",
    # "./data/TESS/TESS_all.csv"
]
output_path = [os.path.splitext(i)[0] + "_IS09.arff" for i in path]
# 读取路径


def thread_i(i, speech_cache_path):
    if os.path.exists(output_path[i]):
        os.remove(output_path[i])
    if not os.path.exists(os.path.split(speech_cache_path)[0]):
        os.makedirs(os.path.split(speech_cache_path)[0])
    path_list = read_csv(path[i])  # 读取
    max_len, sample_rate = get_info(path_list)
    gernerate_feature(
        path_list, output_path[i], speech_cache_path, max_len)  # 生成
    arrfToarray(output_path[i])  # 转换
    print(f"线程{i}结束")


# 创建线程
thread_0 = threading.Thread(target=thread_i, args=(
    0, "./logs/cache/speech_output0.wav"))
# thread_1 = threading.Thread(target=thread_i, args=(
#     1, "./logs/cache/speech_output1.wav"))
# thread_2 = threading.Thread(target=thread_i, args=(
#     2, "./logs/cache/speech_output2.wav"))
# thread_3 = threading.Thread(target=thread_i, args=(
#     3, "./logs/cache/speech_output3.wav"))
# thread_4 = threading.Thread(target=thread_i, args=(
#     4, "./Log/cache/speech_output4.wav"))
# 启动线程
thread_0.start()
# thread_1.start()
# thread_2.start()
# thread_3.start()
# thread_4.start()
print("结束")
