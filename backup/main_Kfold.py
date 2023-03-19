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
from logger import sendmail
import torch
from torch.utils.data import DataLoader
# 自定库
import SCW.deeplearningcopy.custom_solver.solver as solver
from SCW.deeplearningcopy.custom_datasets.IEMOCAP import IEMOCAP, random_split_K
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# 设置当前可用的GPU,并使这些CPU重新按照[0，1]排序
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
print(os.getcwd())


def collate_fn_pad(batch):

    # 一个batch大小的列表，内部含有数据元组，如：(waveform, label)
    # Gather in lists
    tensors = [item[0] for item in batch]  # [frame, feature]
    targets = [item[1] for item in batch]

    # Group the list of tensors into a batched tensor
    tensors = torch.nn.utils.rnn.pad_sequence(
        tensors, batch_first=True, padding_value=0.)
    targets = torch.LongTensor(targets)

    return tensors, targets


def collate_fn_split(batch):
    # batch: [(spec, label),...] ∈R^batchsize
    fix_len = 250
    hop_len = 125

    # 一个batch大小的列表，内部含有数据元组，如：()
    # Gather in lists
    tensors = []
    targets = []
    nfs = []
    for spec, label in batch:
        # item, labelL: [frame, feature] , int
        # frame 分段为固定大小, 125代表采样率为16000, 帧移为256条件下, 2秒时间长度的数据
        # 计算总分段数
        if spec.shape[0] < fix_len - hop_len:
            nf = 1
        else:
            nf = (spec.shape[0] - fix_len + hop_len) // hop_len + 1
        nfs.append(nf)
        # 计算每段的起始位置
        indf = [i*hop_len for i in range(nf)]
        # 正式开始分段数据
        for i in range(nf-1):
            tensors.append(
                spec[indf[i]:indf[i] + fix_len, :])
            targets.append(label)
        tensors.append(spec[indf[-1]:, :])
        targets.append(label)
    # 对不等长的数据进行填充
    tensors = torch.nn.utils.rnn.pad_sequence(
        tensors, batch_first=True, padding_value=0.)
    targets = torch.LongTensor(targets)
    nfs = torch.LongTensor(nfs)
    return tensors, targets, nfs


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
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.cuda.manual_seed(SEED)

    # 设置CPU/GPU
    print('################ 程序运行环境 ################')

    print(f"torch_version: {torch.__version__}")  # 软件 torch 版本

    if config["train"]['num_gpus'] and torch.cuda.is_available():  # 如果使用GPU加速
        print(
            f"cuda: {torch.cuda.is_available()}, GPU_num: {torch.cuda.device_count()}")
        device = torch.device(f'cuda:{0}')  # 定义主GPU
        if torch.cuda.device_count() >= 1:
            for i in range(torch.cuda.device_count()):
                # 打印所有可用GPU
                print(f"GPU: {i}, {torch.cuda.get_device_name(i)}")
        torch.cuda.manual_seed_all(SEED)
    else:
        device = torch.device('cpu')
        print(f"cuda: {torch.cuda.is_available()}, Use:{device}")

    print('################ 处理和获取数据 ################')
    dataset_name = config["data"]["dataset_name"]  # 数据库名称(文件夹名称)
    dataset_type = config['data']["dataset_type"]  # 数据类型
    batchpadsplit = config['data']["batchpadsplit"]  # 数据batch处理方法

    num_workers = config["train"]['num_workers']  # 数据处理进程
    batch_size = config["train"]['batch_size']  # 数据块大小

    if config['data']['batchpadsplit'] == "pad":
        collate_testfn = collate_fn_pad
        collate_trainfn = collate_fn_pad
    elif config['data']['batchpadsplit'] == "trainsplit":
        collate_testfn = collate_fn_pad
        collate_trainfn = collate_fn_split
    elif config['data']['batchpadsplit'] == "allsplit":
        collate_testfn = collate_fn_split
        collate_trainfn = collate_fn_split
    elif config['data']['batchpadsplit'] == "testsplit":
        collate_testfn = collate_fn_split
        collate_trainfn = collate_fn_pad
    else:
        collate_testfn = None
        collate_trainfn = None

    if dataset_type == "Audio":
        data = IEMOCAP(config)
        train_size = int(len(data)*config['train']["train_size"])  # 训练集数量
        print(f"使用{dataset_type}数据集: {dataset_name}")
        print(f"数据集大小: {len(data)}")

    # elif dataset_type == "Text":
    #     num_steps = config['model']['num_steps']  # 数据块中子序列长度
    #     train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
    #     # 标记：两者暂时相同
    #     test_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
    #     print(f"使用{dataset_type}数据集: {dat aset_name}")
    #     print(f"训练集大小: {len(vocab)}")
    #     print(f"测试集大小: {len(vocab)}")

    print('################ 开始训练模型 ################')
    # 定义新模型，设置参数，确认是否加载历史检查点
    accuracylog = []
    for train_data, test_data in random_split_K(dataset=data, K=5, shuffle=True):
        train_iter = DataLoader(
            train_data, batch_size=batch_size, shuffle=not config['data']["sorting"], num_workers=num_workers, collate_fn=collate_trainfn)
        test_iter = DataLoader(
            test_data, batch_size=batch_size, shuffle=not config['data']["sorting"], num_workers=num_workers, collate_fn=collate_testfn)
        s = solver.Solver_Classification(
            train_iter, test_iter, config, device)
        if args.alter:
            print(f"将加载的配置更改为新配置 {args.config}.")
            s.config = config
            s.set_configuration()
        # 以情感语音出现的次数，设置分类权重
        s.set_classification_weights(config, train_data.dataset.labels)
        # 开始训练主函数
        accuracylog.append(s.train())
        config["train"]["resume_iters"] = 0
    accuracylog = np.mean(np.mean(accuracylog, axis=1), axis=0)
    print(f"{accuracylog}")
    # try:
    #     s.train()
    #     sendmail(f"训练正常结束", config)
    # except Exception as e:
    #     print(f"训练因错误结束, 错误信息：{repr(e)}")
    #     # sendmail(f"训练因错误结束, 错误信息：{repr(e)}", config)
    #     raise e
