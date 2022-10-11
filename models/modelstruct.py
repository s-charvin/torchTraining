import os
import torch
import torch.nn as nn
# pytorch 的 checkpoint 是一种用时间换显存的技术，一般训练模式下，pytorch 每次运算后会保留一些中间变量用于求导，而使用 checkpoint 的函数，则不会保留中间变量，中间变量会在求导时再计算一次，因此减少了显存占用
import torch.utils.checkpoint

from .cnn_example import CNNNetExemple
from .rnn_example import RNNModel
from .wav2vec2_cls import Wav2Vec2_CLS, Wav2Vec2_pretrain
from .cnn_gru import Cnn_Gru
from .resnet import ResNet
from .densenet import DenseNet
from .wav2vec2 import wav2vec2_model
from .emotion_baseline_attention_lstm import EmoAttention_Lstm
from models.resnext import ResNeXt
from models.lstm_res_test import LstmResTest
from models.AFC_ALSTM_ACNN import CLFC
from models.SAE import STLSER
from models.Light_Sernet import LightSerNet, LightSerNet_Change
from models.LCANet import LightSerNet_Multi, LightSerNet_NP
from models.replknet import create_RepLKNetXL
from models.SlowFast import slowfast101
# from models.Testmodel import *
# 预训练模型


class SequenceClassifier(object):
    """序列分类器"""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.model_name = config['model']['name']
        self.build_model()

    def build_model(self):
        print("构建模型....")

        # 指定nn.Module类模型
        # CNNNetExemple ,Wav2Vec2_CLS,Wav2Vec2_pretrain,
        # DenseNet,ResNet
        # EmoAttention_Lstm
        # torch.distributed.init_process_group(backend='nccl')
        # self.net = nn.DataParallel(CLIP(), dim=0)
        self.net = ResNet(self.config)
        # self.net = nn.DataParallel(slowfast101())

        # self.net = nn.DataParallel(DenseNet(self.config))
        con_opt = self.config['optimizer']
        self.optimizer = torch.optim.AdamW(params=self.net.parameters(), lr=con_opt['lr'], betas=[
            con_opt['beta1'], con_opt['beta2']])
        if self.config["logs"]['verbose']:
            # 计算多个网络结构参数
            total = 0
            net_count = self.print_network(self.net, 'net')
            total += net_count
            print("总变量参数: {}".format(total))

    def set_train_mode(self):
        # 设置模型为训练模式，可添加多个模型
        self.net.train()  # 训练时使用BN层和Dropout层

        # 冻结部分前置模块参数的示例
        # for param in self.net.module.wav2vec2.wav2vec2.base_model.parameters():
        #     param.requires_grad = False

    def set_eval_mode(self):
        # 设置模型为评估模式，可添加多个模型
        # 处于评估模式时,批归一化层和dropout层不会在推断时有效果
        self.net.eval()

    def to_device(self, device):
        # 将使用到的单个模型或多个模型放到指定的CPU or GPU上
        self.net.to(device=device)
        # self.optimizer.to(device=device) # 错误，这样会出错

    def print_network(self, model, name):
        """打印网络结构信息"""
        print(f"模型名称: {name}")
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()  # 获得张量的元素个数
        print(f"网络参数数量:{num_params}")
        print("打印网络结构:")
        print(model)
        return num_params

    def reset_grad(self):
        """将梯度清零。"""
        self.optimizer.zero_grad()

    def save(self, save_dir=None, it=0):
        """保存模型"""
        if save_dir is None:
            save_dir = self.config['logs']['model_save_dir']
        path = os.path.join(save_dir, self.model_name)
        if not os.path.exists(path):
            os.makedirs(path)

        self.config['train']['resume_iters'] = it

        state = {'net': self.net.state_dict(),
                 'optimizer': self.optimizer.state_dict(),
                 'config': self.config
                 }
        path = os.path.join(path, f"{it:06}.ckpt")
        torch.save(state, path)
        print(f"模型存放地址: {path}.")

    def load(self, load_dir):
        """
        load_dir: 要加载模型的完整地址
        """
        print(f"模型加载地址: {load_dir}。")

        if self.config['train']['USE_GPU'] and torch.cuda.is_available():
            dictionary = torch.load(load_dir, map_location='cuda:0')
        else:
            dictionary = torch.load(load_dir, map_location='cpu')

        self.config = dictionary['config']
        self.model_name = self.config['model']['name']

        self.net.load_state_dict(dictionary['net'])
        # self.optimizer.load_state_dict(dictionary['optimizer'])
        con_opt = self.config['optimizer']
        self.optimizer = torch.optim.Adam(self.net.parameters(), con_opt['lr'], [
                                          con_opt['beta1'], con_opt['beta2']])

        print("Models 和 optimizers 加载成功")

    def predict(self):
        pass


class SequenceClassifier_CoTeaching(object):
    """序列分类器"""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.model_name = config['model']['name']
        self.build_model()

    def build_model(self):
        print("构建模型....")

        # 定义半并行使用的两个网络
        self.net1 = nn.DataParallel(Testmodel_head())
        self.net2 = nn.DataParallel(ResNet(self.config))
        # self.net2 = nn.DataParallel(create_RepLKNetXL(
        #     num_classes=4, use_checkpoint=False, small_kernel_merged=True))

        con_opt = self.config['optimizer']
        self.optimizer1 = torch.optim.Adam(self.net1.parameters(), con_opt['lr'], [
            con_opt['beta1'], con_opt['beta2']])
        self.optimizer2 = torch.optim.Adam(self.net2.parameters(), con_opt['lr'], [
            con_opt['beta1'], con_opt['beta2']])

        if self.config["logs"]['verbose']:
            # 计算多个网络结构参数
            total = 0
            net1_count = self.print_network(self.net1, 'net1')
            net2_count = self.print_network(self.net2, 'net2')
            total = total + net1_count+net2_count
            print("总变量参数: {}".format(total))
        self.to_device(device='cpu')

    def set_train_mode(self):
        # 设置模型为训练模式，可添加多个模型
        self.net1.train()  # 训练时使用BN层和Dropout层
        self.net2.train()
        # 冻结部分前置模块参数的示例
        # for param in self.net.module.wav2vec2.wav2vec2.base_model.parameters():
        #     param.requires_grad = False

    def set_eval_mode(self):
        # 设置模型为评估模式，可添加多个模型
        # 处于评估模式时,批归一化层和dropout层不会在推断时有效果
        self.net1.eval()
        self.net2.eval()

    def to_device(self, device):
        # 将使用到的单个模型或多个模型放到指定的CPU or GPU上
        self.net1.to(device=device)
        self.net2.to(device=device)

    def print_network(self, model, name):
        """打印网络结构信息"""

        num_params = 0
        for p in model.parameters():
            num_params += p.numel()  # 获得张量的元素个数
        print("打印网络结构:")
        print(model)
        print(f"模型名称: {name}")
        print(f"网络参数数量:{num_params}")
        return num_params

    def reset_grad(self):
        """将梯度清零。"""
        self.optimizer1.zero_grad()
        self.optimizer2.zero_grad()

    def save(self, save_dir=None, it=0):
        """保存模型"""
        if save_dir is None:
            save_dir = self.config['logs']['model_save_dir']
        path = os.path.join(save_dir, self.model_name)
        if not os.path.exists(path):
            os.makedirs(path)

        self.config['train']['resume_iters'] = it

        state = {'net1': self.net1.state_dict(),
                 'net2': self.net2.state_dict(),
                 'optimizer1': self.optimizer1.state_dict(),
                 'optimizer2': self.optimizer2.state_dict(),
                 'config': self.config
                 }
        path = os.path.join(path, f"{it:06}.ckpt")
        torch.save(state, path)
        # print(f"模型存放地址: {path}.")

    def load(self, load_dir):
        """
        load_dir: 要加载模型的完整地址
        """
        print(f"模型加载地址: {load_dir}。")

        if self.config['train']['USE_GPU'] and torch.cuda.is_available():
            dictionary = torch.load(load_dir, map_location='cuda:0')
        else:
            dictionary = torch.load(load_dir, map_location='cpu')

        self.config = dictionary['config']
        self.model_name = self.config['model']['name']

        self.net1.load_state_dict(dictionary['net1'])
        self.net2.load_state_dict(dictionary['net2'])

        con_opt = self.config['optimizer1']
        self.optimizer1 = torch.optim.Adam(self.net.parameters(), con_opt['lr'], [
            con_opt['beta1'], con_opt['beta2']])
        con_opt = self.config['optimizer2']
        self.optimizer2 = torch.optim.Adam(self.net.parameters(), con_opt['lr'], [
            con_opt['beta1'], con_opt['beta2']])
        print("Models 和 optimizers 加载成功")

    def predict(self):
        pass


# class SequenceGenerator(object):
#     """序列结构生成器"""

#     def __init__(self, config, name) -> None:
#         super().__init__()
#         self.config = config
#         self.save_dir = config['logs']['model_save_dir']
#         self.model_name = name
#         self.build_model()

#     def set_train_mode(self):
#         # 设置模型为训练模式，可添加多个模型
#         self.net.train()  # 训练时使用BN层和Dropout层

#     def set_eval_mode(self):
#         # 设置模型为评估模式，可添加多个模型
#         # 批归一化层和dropout层不会在推断时有效果
#         self.net.eval()

#     def to_device(self, device):
#         # 将模型放到指定的CPU or GPU上
#         self.net.to(device=device)
#         # self.optimizer.to(device=device) # 错误，这样会出错

#     def build_model(self):
#         print("构建模型....")
#         # 输入特征维度(or 词元表示维度)

#         self.net = RNNModel(self.config)  # 指定nn模型

#         con_opt = self.config['optimizer']
#         # self.optimizer = torch.optim.Adam(self.net.parameters(), con_opt['lr'], [
#         #                                   con_opt['beta1'], con_opt['beta2']])
#         self.optimizer = torch.optim.SGD(self.net.parameters(), con_opt['lr'])

#         if self.config['verbose']:
#             # 计算多个网络结构参数
#             total = 0
#             net_count = self.print_network(self.net, 'net')
#             total += net_count
#             print("总变量参数: {}".format(total))

#     def print_network(self, model, name):
#         """打印网络结构信息"""
#         print(f"模型名称: {name}")
#         num_params = 0
#         for p in model.parameters():
#             num_params += p.numel()  # 获得张量的元素个数
#         print(f"网络参数数量:{num_params}")
#         print("打印网络结构:")
#         print(model)
#         return num_params

#     def reset_grad(self):
#         """将梯度清零。"""
#         self.optimizer.zero_grad()

#     def save(self, save_dir=None, it=0):
#         """保存模型"""
#         if save_dir is None:
#             save_dir = self.save_dir
#         path = os.path.join(save_dir, self.model_name)
#         if not os.path.exists(path):
#             os.makedirs(path)

#         self.config['model']['resume_iters'] = it

#         state = {'net': self.net.state_dict(),
#                  'optimizer': self.optimizer.state_dict(),
#                  'config': self.config
#                  }
#         path = os.path.join(path, "{:06}.ckpt".format(it))
#         torch.save(state, path)
#         print("模型存放地址: {}.".format(path))

#     def load(self, load_dir):
#         """
#         load_dir: 要加载模型的完整地址
#         """
#         print(f"模型加载地址: {load_dir}。")

#         if torch.cuda.is_available():
#             dictionary = torch.load(load_dir)
#         else:
#             dictionary = torch.load(load_dir, map_location='cpu')

#         self.model_name = self.config['model']['name']
#         self.num_emotions = self.config['model']['num_classes']

#         self.config = dictionary['config']
#         self.net.load_state_dict(dictionary['net'])
#         # self.optimizer.load_state_dict(dictionary['optimizer']) # 似乎这样会加载到CPU中

#         con_opt = self.config['optimizer']
#         self.optimizer = torch.optim.Adam(self.net.parameters(), con_opt['lr'], [
#                                           con_opt['beta1'], con_opt['beta2']])

#         print("Model 和 optimizers 加载成功。")
