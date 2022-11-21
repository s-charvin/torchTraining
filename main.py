# 官方库
import argparse
import os
import sys
if sys.platform.startswith('linux'):
    os.chdir("/home/visitors2/SCW/torchTraining")
elif sys.platform.startswith('win'):
    pass
import yaml  # yaml文件解析模块
import random  # 随机数生成模块
# 第三方库
import numpy as np  # 矩阵运算模块
import torch
from torch.utils.data import DataLoader
# 自定库
from custom_datasets import *
from custom_solver import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
print("程序运行起始文件夹: "+os.getcwd())
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()
    configpath = '/home/visitors2/SCW/torchTraining/config'
    config = {}
    [config.update(yaml.safe_load(open(os.path.join(configpath, i),
                   'r', encoding='utf-8').read())) for i in os.listdir(configpath)]
    if config['logs']['verbose']["config"]:
        print(config)
    SEED = config['train']['seed']
    # 设置CPU/GPU
    print('################ 程序运行环境 ################')
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config['train']['USE_GPU']

    print(f"# torch_version: {torch.__version__}")  # 软件 torch 版本
    if config["train"]['USE_GPU'] and torch.cuda.is_available():  # 如果使用 GPU 加速
        print(
            f"# Cuda: {torch.cuda.is_available()}, Use_Gpu: {config['train']['USE_GPU']}")
        device = torch.device(f'cuda:{0}')  # 定义主GPU
        if torch.cuda.device_count() >= 1:
            for i in range(torch.cuda.device_count()):
                # 打印所有可用 GPU
                print(f"# GPU: {i}, {torch.cuda.get_device_name(i)}")
        torch.cuda.manual_seed_all(SEED)
    else:
        device = torch.device('cpu')
        print(f"# Cuda: {torch.cuda.is_available()}, Use_Cpu")

    # # benchmark 设置False，是为了保证不使用选择最优卷积算法的机制。
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True  # deterministic设置 True 保证使用固定的卷积算法
    # 二者配合起来，才能保证卷积操作的一致性
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.cuda.manual_seed(SEED)

    print('################ 处理和获取数据 ################')
    # dataset
    dataset_name = config["data"]["name"]  # 数据库名称
    dataset_root = config["data"]["root"]
    data = globals()[dataset_name](
        root=dataset_root,
        filter=config["data"]["filters"], transform=config["data"]["transforms"], enhance=config["data"]["enhances"], **config["data"]["para"])

    num_workers = config["train"]['num_workers']  # 数据处理进程
    batch_size = config["train"]['batch_size']  # 数据块大小

    print(f"# 训练使用数据集: {dataset_name}")
    print(f"# 所用数据数量: {len(data)}")

    if config['train']['Kfold']:
        step = 0
        accuracylog = []  # 记录每次训练的最后几次准确率
        for train_data, test_data in random_split_K(dataset=data, K=config['train']['Kfold'], shuffle=True):
            step = step+1
            print(f'################ 开始训练第 {step} 次模型 ################')
            print(f"# 训练集数据数量: {len(train_data)}")
            print(f"# 测试集数据数量: {len(test_data)}")
            train_loader = DataLoader(
                train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
            test_loader = DataLoader(
                test_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
            # 定义新模型，设置参数，确认是否加载历史检查点
            solver = globals()[config["sorlver"]["name"]](
                train_loader, test_loader, config, device)
            accuracylog = []
            try:
                accuracylog.append(solver.train())
                # sendmail(f"训练正常结束", config)
            except Exception as e:
                print(f"# 训练因错误结束, 错误信息：{repr(e)}")
                # sendmail(f"训练因错误结束, 错误信息：{repr(e)}", config)
                raise e
            config["train"]["last_epochs"] = 0
        print(f"# Accuracy Log: {accuracylog}")
        print(
            f"# Mean Accuracy: {np.mean(np.mean(accuracylog, axis=1), axis=0)}")
    else:
        print('################ 开始训练模型 ################')
        train_size = int(len(data)*config['train']["train_rate"])  # 训练集数量
        test_size = len(data) - train_size  # 测试集数量
        train_data, test_data = torch.utils.data.random_split(
            data, [train_size, test_size])
        print(f"# 训练集大小: {len(train_data)}")
        print(f"# 测试集大小: {len(test_data)}")
        train_loader = DataLoader(
            train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
        test_loader = DataLoader(
            test_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
        # 定义新模型，设置参数，确认是否加载历史检查点
        solver = globals()[config["sorlver"]["name"]](
            train_loader, test_loader, config, device)
        accuracylog = []
        try:
            accuracylog.append(solver.train())
            # sendmail(f"训练正常结束", config)
        except Exception as e:
            print(f"# 训练因错误结束, 错误信息：{repr(e)}")
            # sendmail(f"训练因错误结束, 错误信息：{repr(e)}", config)
            raise e
        print(f"# Accuracy Log: {accuracylog}")
