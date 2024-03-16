import os
import sys
if sys.platform.startswith('linux'):
    os.chdir("/home/user0/SCW/deeplearning/")
elif sys.platform.startswith('win'):
    pass
import csv
import numpy as np


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
            data.append([row[0]])  # 选择某一列加入到data数组中
    return data

# 生成特征


def gernerate_feature(data, output_path):
    """使用opensmile生成特征

    Args:
        data (list): 含有目标文件路径的列表

    gernerate: arff格式的文本文件(数据包含[文件名,特征..,..,..])
    """
    for file_path in data:

        cmd = f"SMILExtract "\
            + '-C \"C:\Program Files\opensmile\config\is09-13\IS09_emotion.conf\" '\
            + f"-I \"{file_path[0]}\" "\
            + f"-instname \"{file_path[0]}\" "\
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
        np.save(name+'.npy', new_content)

# def getfeature(i):
#     path_list = read_csv(path[i])  # 读取
#     gernerate_feature(path_list, output_path[i])  # 生成
#     arrfToarray(output_path[i])  # 转换

#     path_list, output_path = args
#     lenth = int(len(path_list)/4)
#     thread.start_new_thread(program, (path_list[0*lenth:lenth], output_path))
#     thread.start_new_thread(program, (path_list[1*lenth:2*lenth], output_path))
#     thread.start_new_thread(program, (path_list[2*lenth:3*lenth], output_path))
#     thread.start_new_thread(program, (path_list[3*lenth:], output_path))


path = [
    # "./data/EMODB/EMODB_all.csv",
    # "./data/IEMOCAP/IEMOCAP_all.csv",
    # "./data/eNTERFACE05/eNTERFACE05_all.csv",
    "./data/AESDD/AESDD_all.csv",
    # "./data/RAVDESS/RAVDESS_all.csv",
    # "./data/SAVEE/SAVEE_all.csv",
    # "./data/TESS/TESS_all.csv"
]
output_path = [os.path.splitext(i)[0] + "_IS09.arff" for i in path]
# 读取路径

for i in range(len(path)):
    path_list = read_csv(path[i])  # 读取
    gernerate_feature(path_list, output_path[i])  # 生成
    arrfToarray(output_path[i])  # 转换
