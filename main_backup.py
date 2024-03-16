# 官方库
import argparse
import os
import sys

if sys.platform.startswith("linux"):
    os.chdir("/home/user0/SCW/torchTraining")
elif sys.platform.startswith("win"):
    pass

os.environ['TMPDIR'] = '/sdb/user0/tmp'
os.environ['TEMP'] = '/sdb/user0/tmp'
os.environ['TMP'] = '/sdb/user0/tmp'
# 确保目录存在
if not os.path.exists('/sdb/user0/tmp'):
    os.makedirs('/sdb/user0/tmp')

import yaml  # yaml文件解析模块
import random  # 随机数生成模块
import time

# 第三方库
import numpy as np  # 矩阵运算模块
import torch
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
import torch.utils.data.distributed
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn

# 自定库
from custom_datasets.subset import Subset, random_split_index_K, collate_fn
from custom_datasets import *
from custom_solver import *
from utils.trainutils import collate_fn_pad, RedisClient
from utils.logger import MailSender
import json
# torch.load(cached_file, map_location=map_location)
torch.multiprocessing.set_sharing_strategy("file_system")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    metavar="DIR",
    default="/home/user0/SCW/torchTraining/projects/MERSISF_test",
    help="path to congigs",
)
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(threadName)s %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


def train_worker(local_rank, config, train_data_index, valid_data_index, data):
    config["self_auto"]["local_rank"] = local_rank
    rank = 0
    if local_rank is not None:
        rank = (
            config["train"]["node_rank"] * config["self_auto"]["gpu_nums"] + local_rank
        )
        # backend: 多机器间交换数据协议, init_method: 设置交换数据主节点, world_size:标识使用进程数（主机数*每台主机的GPU数/实际就是机器的个数？）
        # rank: 标识主机和从机，主节点为 0, 剩余的为 1-(N-1), N 为要使用的机器的数量
        dist.init_process_group(
            backend="nccl",  # 通信方式
            init_method="env://",
            world_size=config["self_auto"]["world_size"],  # 总进程数
            rank=rank,  # 当前进程的 id
        )

    if config["train"]["seed"]:
        SEED = config["train"]["seed"] + rank
        # 保证 DataLoader 的随机一致性
        random.seed(SEED)
        np.random.seed(SEED)
        # 保证 torch 的随机一致性
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        # benchmark 设置False，是为了保证不使用选择最优卷积算法的机制。
        cudnn.benchmark = False
        # deterministic设置 True 保证使用固定的卷积算法
        cudnn.deterministic = True
        # 二者配合起来，才能保证卷积操作的一致性

    # dataset
    dataset_name = config["data"]["name"]  # 数据库名称
    dataset_root = config["data"]["root"]

    # data = globals()[dataset_name](
    #     root=dataset_root,
    #     filter=config["data"]["filters"], transform=config["data"]["transforms"], enhance=config["data"]["enhances"], **config["data"]["para"])

    num_workers = config["train"]["num_workers"]  # 数据处理进程
    batch_size = config["train"]["batch_size"]  # 数据块大小

    train_data = Subset(data, train_data_index)
    valid_data = Subset(data, valid_data_index)

    # if config["train"]["loss_weight"] == True:
    #     labels = [train_data[i]["label"]
    #               for i in range(len(train_data))]
    #     categories = data.datadict["categories"]
    #     total = len(labels)
    #     config["self_auto"]["loss_weights"] = [
    #         total/labels.count(emo) for emo in categories]
    # else:
    #     config["self_auto"]["loss_weights"] = None

    train_sampler = None
    val_sampler = None
    if local_rank is not None:
        # 包装模型, 并将其放到 GPU 上
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_data,
            num_replicas=config["self_auto"]["world_size"],
            rank=rank,
        )
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            valid_data,
            num_replicas=config["self_auto"]["world_size"],
            rank=rank,
        )
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        sampler=train_sampler,
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        sampler=val_sampler,
    )

    # 定义新模型，设置参数，确认是否加载历史检查点
    device = (
        torch.device(f"cuda:{local_rank}")
        if (local_rank is not None)
        else torch.device("cpu")
    )
    solver = globals()[config["sorlver"]["name"]](
        train_loader, valid_loader, config, device
    )
    if local_rank in [0, None]:
        solver.print_network()
    solver.train()


def main(config):
    logging.info("\n################ 程序运行环境 ################")
    logging.info(f"torch_version: {torch.__version__}")  # 软件 torch 版本

    if (
        config["train"]["available_gpus"] != "" and torch.cuda.is_available()
    ):  # 如果使用 GPU 加速
        logging.info(
            f"Cuda: {torch.cuda.is_available()}, Use_Gpu: {config['train']['available_gpus']}"
        )
        if torch.cuda.device_count() >= 1:
            for i in range(torch.cuda.device_count()):
                # 打印所有可用 GPU
                logging.info(f"GPU: {i}, {torch.cuda.get_device_name(i)}")

        # 基于节点数和每个节点使用的 gpu 数目计算出单进程单 GPU 条件下需要的 world_size (要运行的进程总数).
        config["self_auto"]["gpu_nums"] = len(
            config["train"]["available_gpus"].split(",")
        )
        config["self_auto"]["world_size"] = (
            config["self_auto"]["gpu_nums"] * config["train"]["nodes"]
        )
    else:
        logging.info(f"CUDA: {torch.cuda.is_available()}, Use_CPU")
        config["self_auto"]["gpu_nums"] = 0
        config["self_auto"]["world_size"] = 1

    if config["train"]["seed"]:
        SEED = config["train"]["seed"]
        # 保证 DataLoader 的随机一致性
        random.seed(SEED)
        np.random.seed(SEED)
        # 保证 torch 的随机一致性
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        # benchmark 设置 False，是为了保证不使用选择最优卷积算法的机制。
        cudnn.benchmark = False
        # deterministic设置 True 保证使用固定的卷积算法
        cudnn.deterministic = True
        # 二者配合起来，才能保证卷积操作的一致性

    # with HiddenPrints():
    dataset_name = config["data"]["name"]  # 数据库名称
    dataset_root = config["data"]["root"]
    data = globals()[dataset_name](
        root=dataset_root,
        filter=config["data"]["filters"],
        transform=config["data"]["transforms"],
        enhance=config["data"]["enhances"],
        **config["data"]["para"],
    )
    data_len = len(data)
    # 删除临时的 data. 这里使用临时 data, 虽然浪费了数据加载时间, 但是是为了实现多折训练所必需的问题解决方法(无法将 data 作为参数传入子进程)
    config["data"]["root"] = data.root
    logging.info(f"训练使用数据集: {dataset_name}")
    logging.info(f"所用数据数量: {data_len}")
    # del data

    last_epochs_ = config["train"]["last_epochs"]  # 记录初次运行的 epoch 数目
    k = 0  # 记录训练折数

    if config["train"]["Kfold"] > 1:
        for train_data_index, valid_data_index in random_split_index_K(
            n_samples=data_len, K=config["train"]["Kfold"], shuffle=True
        ):
            k = k + 1
            config["train"]["last_epochs"] = last_epochs_
            logging.info(f"\n################ 开始训练第 {k} 次模型 ################")
            if config["self_auto"]["gpu_nums"] > 0:
                context = mp.spawn(
                    train_worker,
                    nprocs=config["self_auto"]["gpu_nums"],
                    args=(config, train_data_index, valid_data_index, data),
                    join=False,
                )
                # Loop on join until it returns True or raises an exception.
                try:
                    while not context.join():
                        pass
                except Exception as e:
                    raise e
            elif config["self_auto"]["gpu_nums"] == 0:
                train_worker(None, config, train_data_index, valid_data_index, data)
    else:
        if hasattr(data, "indices"):
            train_data_index = data.indices["train"]
            valid_data_index = data.indices["valid"]
        else:
            indices = np.random.permutation(data_len)
            train_data_index, valid_data_index = (
                indices[: int(config["train"]["train_rate"] * data_len)],
                indices[int(config["train"]["train_rate"] * data_len) :],
            )
        logging.info(f"\n################ 开始训练模型 ################")
        if config["self_auto"]["gpu_nums"] > 0:
            context = mp.spawn(
                train_worker,
                nprocs=config["self_auto"]["gpu_nums"],
                args=(config, train_data_index, valid_data_index, data),
                join=False,
            )
            # Loop on join until it returns True or raises an exception.
            try:
                while not context.join():
                    pass
            except Exception as e:
                raise e
        else:
            train_worker(None, config, train_data_index, valid_data_index, data)


if __name__ == "__main__":
    logging.info("程序运行起始文件夹: " + os.getcwd())
    
    args = parser.parse_args()
    config = {"args": vars(args)}
    [
        config.update(
            yaml.safe_load(
                open(
                    os.path.join(config["args"]["config"], i), "r", encoding="utf-8"
                ).read()
            )
        )
        for i in os.listdir(config["args"]["config"])
    ]
    config["self_auto"] = {}
    
    # 设置 CPU/GPU

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config["train"]["available_gpus"]
    os.environ["MASTER_ADDR"] = config["train"]["master_addr"]
    os.environ["MASTER_PORT"] = config["train"]["master_port"]

    redis_client = None
    if (
        config["train"]["gpu_queuer"]["wait_gpus"]
        and config["train"]["available_gpus"] != ""
    ):
        redis_client = RedisClient(
            **config["train"]["gpu_queuer"]["para"]
        )  # 初始化并连接到 Redis 服务端
        redis_client.check_data()
        task_id = redis_client.join_wait_queue(
            config["train"]["name"], task_config=config
        )  # 注册当前任务进程到等待任务列表
        while redis_client.is_need_wait():
            redis_client.check_data()
            config_ = redis_client.pop_wait_queue()
            if not config_:
                time.sleep(60)  # 休息一下, 再重新尝试
                continue
            if not redis_client.is_can_run():
                time.sleep(60)  # 休息一下, 再重新尝试
                redis_client.pop_run_queue(success=False)
                continue
            try:
                # 定义训练主程序
                config = config_
                if config["logs"]["verbose"]["config"]:
                    logging.info(yaml.dump(config, indent=4))
                main(config)
                redis_client.pop_run_queue(success=True)
                break
            except Exception as e:
                if "CUDA out of memory" in e.args[0]:
                    redis_client.pop_run_queue(success=False)
                    time.sleep(60)
                elif "No space left on device" in e.args[0]:
                    redis_client.pop_run_queue(success=False)
                    time.sleep(60)
                else:
                    # # redis_client.pop_run_queue(success=False)
                    # redis_client.delete_running_task()
                    # logging.info(e.args[0])
                    raise e
        redis_client.delete_waitting_task()
    else:
        if config["logs"]["verbose"]["config"]:
            logging.info(yaml.dump(config, indent=4))
        main(config)