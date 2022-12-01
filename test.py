# 官方库
import argparse
import os
import sys
if sys.platform.startswith('linux'):
    os.chdir("/home/visitors2/SCW/torchTraining")
elif sys.platform.startswith('win'):
    pass
import yaml  # yaml文件解析模块
# 第三方库
import numpy as np  # 矩阵运算模块
import torch
from typing import Union
from pathlib import Path
import librosa
import audioread.ffdec

# 自定库
from custom_solver import *
from custom_transforms import *


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
print("程序运行起始文件夹: "+os.getcwd())
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="语音文件列表或文件夹",
                        default="/sdb/visitors2/SCW/data/IEMOCAP/Session3/sentences/wav/Ses03F_impro04/")
    parser.add_argument("--devices", type=str, default="1")
    args = parser.parse_args()
    configpath = '/home/visitors2/SCW/torchTraining/config/'
    config = {}
    [config.update(yaml.safe_load(open(os.path.join(configpath, i),
                   'r', encoding='utf-8').read())) for i in os.listdir(configpath)]
    # 设置CPU/GPU
    print('################ 程序运行环境 ################')
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices

    print(f"# torch_version: {torch.__version__}")  # 软件 torch 版本
    if args.devices and torch.cuda.is_available():  # 如果使用 GPU 加速
        print(
            f"# Cuda: {torch.cuda.is_available()}, Use_Gpu: {args.devices}")
        device = torch.device(f'cuda:{0}')  # 定义主GPU
        if torch.cuda.device_count() >= 1:
            for i in range(torch.cuda.device_count()):
                # 打印所有可用 GPU
                print(f"# GPU: {i}, {torch.cuda.get_device_name(i)}")
    else:
        device = torch.device('cpu')
        print(f"# Cuda: {torch.cuda.is_available()}, Use_Cpu")
    solver = globals()[config["sorlver"]["name"]](
        None, None, config, device)
    if isinstance(args.input, list):
        results = solver.calculate([Path(p).resolve() for p in args.input])
    elif isinstance(args.input, str):
        path = Path(args.input)
        if path.is_dir():
            filelist = list(path.rglob("*.wav"))
        elif path.is_file():
            filelist = [path]
        filelist = [p.resolve() for p in filelist if "._Ses" not in p.stem]

        transforms = config["data"]["transforms"]
        processors = {}
        audio_transform = None
        for key, processor in transforms.items():
            if processor:
                pl = []
                for name, param in processor.items():

                    if param:
                        pl.append(globals()[name](**param))
                    else:
                        pl.append(globals()[name]())
                processors[key] = Compose(pl)
        if processors:
            audio_transform = processors["audio"]
        inputs = []
        for filepath in filelist:
            filepath = str(filepath)
            aro = audioread.ffdec.FFmpegAudioFile(str(filepath))
            y, sr = librosa.load(aro, sr=None)
            inputs.append(audio_transform(torch.from_numpy(y[None, :])))
        results = solver.calculate(inputs)
        print(results)
