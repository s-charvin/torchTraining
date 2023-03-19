"""
main.py

命令行运行参数:
    -h, --help      : 展示帮助信息和退出。
    --name -n       : 读取config[model][name]并展示对应模型名称。
    --checkpoint -c : 恢复训练的"checkpoint"所在目录。
    # --load_emo      : 加载预训练模型"checkpoint"所在目录。
    --evaluate -e   : 标记以在测试模式下运行模型。
    --alter -a      : Flag to change the config file of a loaded checkpoint to
                      ./config.yaml (this is useful if models are trained in
                      stages as propsed in the project report)（将已加载检查点的配置文件更改为 ./config.yaml 的标志（如果模型按照项目报告中的建议分阶段训练，这很有用））

"""

# 系统文件和目录处理模块
import os
import sys
if sys.platform.startswith('linux'):
    os.chdir("/home/visitors2/SCW/deeplearning/")
elif sys.platform.startswith('win'):
    pass
# 第三方库
import argparse  # 命令行解析模块
import yaml  # yaml文件解析模块
import random  # 随机数生成模块
import numpy as np  # 矩阵运算模块
import tensorflow as tf
from tensorflow.python.client import device_lib
# 自定库
import solver_tensorflow as solver
from datautils_tensorflow import AudioDataset, random_split

# 设置当前可用的GPU,并使这些CPU重新按照[0，1]排序
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
print(os.getcwd())

if __name__ == '__main__':

    # 命令行参数控制(帮助信息、模型名称、参数文件)
    parser = argparse.ArgumentParser(description='**模型主方法', add_help=False)
    parser.add_argument("-h", "--help", action="help",
                        help="查看帮助信息和退出。")
    parser.add_argument("-n", "--name", type=str, default=None,
                        help="更改config[model][name]并展示对应模型名称。")
    parser.add_argument("--config", type=str, help="设置参数存储文件\"config.yaml\"所在目录. 默认为 \"config.yaml\"。",
                        default='./config.yaml')
    parser.add_argument("-a", "--alter", action='store_true',
                        help="是否修改了config文件内容的标志, 默认为False")

    args = parser.parse_args()
    config = yaml.safe_load(open(args.config, 'r', encoding='utf-8').read())
    # checkpoint = config['train']['checkpoint']
    if args.name is not None:
        config['model']['name'] = args.name  # 记录命令行输入的模型名称
    # 设置随机数种子, 当设置了固定的随机数种子,之后依次生成的随机数序列也会被固定下来。
    SEED = config['train']['seed']
    # # benchmark 设置False，是为了保证不使用选择最优卷积算法的机制。
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True  # deterministic设置True保证使用固定的卷积算法
    # 二者配合起来，才能保证卷积操作的一致性
    np.random.seed(SEED)
    random.seed(SEED)
    tf.random.set_seed(SEED)

    # 设置CPU/GPU
    print('################ 程序运行环境 ################')

    print(f"tensorflow_version: {tf.__version__}")  # 软件 torch 版本

    # 如果使用GPU加速
    if config["train"]['num_gpus'] and tf.config.list_physical_devices('GPU'):
        print(
            f"cuda: {tf.config.list_physical_devices('GPU')}, GPU_num: {len(device_lib.list_local_devices())}")
        devices = device_lib.list_local_devices()
        device = devices[0]
        for i in range(1, len(devices)):
            # 打印所有可用GPU
            print(f"GPU: {i}, {devices[i].name}")
    else:
        device = device_lib.list_local_devices()[0]
        print(f"cuda: {tf.config.list_physical_devices('GPU')}, Use:{device}")

    print('################ 处理和获取数据 ################')
    dataset_name = config["data"]["dataset_name"]  # 数据库名称(文件夹名称)
    dataset_type = config['data']["dataset_type"]  # 数据类型
    batchpadsplit = config['data']["batchpadsplit"]  # 数据batch处理方法

    num_workers = config["train"]['num_workers']  # 数据处理进程
    batch_size = config["train"]['batch_size']  # 数据块大小

    if dataset_type == "Audio":
        data = AudioDataset(config)

        train_size = int(len(data)*config['train']["train_size"])  # 训练集数量
        test_size = len(data) - train_size  # 测试集数量
        train_dataset, test_dataset = random_split(
            data, [train_size, test_size])

        print(f"使用{dataset_type}数据集: {dataset_name}")
        print(f"训练集大小: {len(train_dataset)}")
        print(f"测试集大小: {len(test_dataset)}")

    print('################ 开始训练模型 ################')
    # 定义新模型，设置参数，确认是否加载历史检查点
    s = solver.Solver_Classification_CoTeaching(
        train_dataset, test_dataset, config, device)
    if args.alter:
        print(f"将加载的配置更改为新配置 {args.config}.")
        s.config = config
        s.set_configuration()
    # 以情感语音出现的次数，设置分类权重
    s.set_classification_weights(config, train_dataset.dataset.labels)
    # 开始训练主函数
    s.train()
