from datetime import datetime
import os
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from logger import Logger
from .lr_scheduler import *
from .optimizer import *
from models import *
import torch.distributed as dist


class Audio_Classification(object):

    def __init__(self, train_loader=None, test_loader=None, config: dict = {}, device=None):
        """分类模型训练方法框架
        Args:
            train_loader (_type_): 训练集，可迭代类型数据
            test_loader (_type_): 测试集，可迭代类型数据
            config (_type_): 参数设置
            device (_type_): 运行设备
        """
        self.train_loader = train_loader  # 训练数据迭代器
        self.test_loader = test_loader  # 测试数据迭代器
        self.config = config  # 参数字典
        self.device = device  # 定义主设备
        # 设置要使用的模型
        if self.config['architecture']['net']['para']:
            self.net = globals()[self.config["architecture"]["net"]["name"]](
                **self.config["architecture"]['net']['para'])
        else:
            self.net = globals()[
                self.config["architecture"]["net"]["name"]]()
        # 多 GPU 并行计算
        if self.config["train"]['USE_GPU']:
            self.net = self.net.to(device=self.device)
            self.net = torch.nn.parallel.DistributedDataParallel(
                self.net, device_ids=[self.device])  # device_ids 默认选用本地显示的所有 GPU
        if self.train_loader is not None:
            self.set_configuration()  # 设置参数

        # 加载模型及其参数
        if self.config['train']['checkpoint']:  # 是否加载历史检查点
            self.load(self.config['train']['checkpoint'])

    def set_configuration(self):
        self.n_epochs = self.config['train']['n_epochs']  # 迭代次数
        # 已经训练完成的 epoch, 初次为 -1
        self.last_epochs = self.config['train']["last_epochs"]
        if self.last_epochs < -1:
            raise ValueError("last_epochs 值必须为大于等于 -1 的整数, 当其为 -1 时, 表示重新训练.")
        # 优化器及其参数
        if 'para' in self.config["sorlver"]['optimizer']:
            if self.config["sorlver"]['optimizer']['para']:
                self.optimizer = globals(
                )[self.config["sorlver"]['optimizer']["name"]](params=self.net.parameters(), lr=self.config["sorlver"]['optimizer']['lr'], **self.config["sorlver"]['optimizer']['para'])
            else:
                self.optimizer = globals(
                )[self.config["sorlver"]['optimizer']["name"]](params=self.net.parameters())

        # 使用 tensorboard 记录
        self.use_tensorboard = self.config['logs']['use_tensorboard']
        self.log_every = self.config['logs']['log_every']  # 每几步记录
        self.test_every = self.config['logs']['test_every']

        # 每几步保存模型
        self.model_save_every = self.config['logs']['model_save_every']
        self.model_save_dir = self.config['logs']['model_save_dir']  # 模型保存地址
        self.logname = "-".join([
            str(self.config['train']['name']),  # 当前训练名称
            str(self.config['train']['seed']),
            str(self.config['train']['batch_size']),
            str(self.n_epochs),
            self.config['sorlver']['name'],
            self.config['sorlver']['optimizer']["name"] +
            str(self.config['sorlver']['optimizer']["lr"]),
            self.config['data']['name']
        ])
        if self.use_tensorboard:
            self.logger = Logger(
                self.config['logs']['log_dir'], self.logname)

    def train(self):
        """训练主函数"""
        assert self.train_loader, "未提供训练数据"
        #############################################################
        #                            训练                           #
        #############################################################
        start_iter = self.last_epochs+1  # 训练的新模型则从 1 开始，恢复的模型则从接续的迭代次数开始

        # 更新学习率的方法
        self.lr_method = globals()[self.config["sorlver"]["lr_method"]["name"]](optimizer=self.optimizer,
                                                                                last_epoch=self.last_epochs if self.last_epochs >= 1 else -1, **self.config["sorlver"]["lr_method"]["para"])
        # 预测结果记录aram_groups[0] when resuming
        self.accuracylog = []
        # 训练开始时间记录
        start_time = datetime.now()
        if (self.config['train']['local_rank'] in [0, None]):
            print(f'# 训练开始时间: {start_time}')  # 读取开始运行时间
        step = 0

        # 设置损失函数
        loss_weights = None
        if self.config["sorlver"]["loss_weights"]:
            loss_weights = torch.Tensor(
                self.config["sorlver"]["loss_weights"]).to(device=self.device)
        loss_func = nn.CrossEntropyLoss(
            reduction='mean', weight=loss_weights).to(device=self.device)

        for epoch in range(start_iter, self.n_epochs):  # 迭代循环 [0, epoch)
            for batch_i, data in enumerate(self.train_loader):  # [0, batch)
                step = step + 1
                # 设置训练模式和设备
                self.to_device(device=self.device)
                self.set_train_mode()
                # 获取特征
                if isinstance(data["audio"], torch.nn.utils.rnn.PackedSequence):
                    features, feature_lens = torch.nn.utils.rnn.pad_packed_sequence(
                        data["audio"], batch_first=True)
                elif isinstance(data["audio"], torch.Tensor):
                    features, feature_lens = data["audio"], None
                # 获取标签
                labels = data["labelid"]
                # 将数据迁移到训练设备
                features = features.to(device=self.device)
                if feature_lens is not None:
                    feature_lens = feature_lens.to(device=self.device)
                labels = labels.long().to(device=self.device)

                # 清空梯度数据
                self.reset_grad()
                out, _ = self.net(features.permute(
                    0, 3, 1, 2), feature_lens)
                loss = loss_func(out, labels)

                # 在所有进程运行到这一步之前，先完成此前代码的进程会在这里等待其他进程。
                torch.distributed.barrier()

                loss_ = loss.clone()
                dist.all_reduce(loss_, op=dist.ReduceOp.SUM)
                loss_ /= self.config["train"]["world_size"]
                # print(
                #     f"local_rank: {self.config['train']['local_rank']}, loss: {loss.item()}, avr_loss: {loss_.item()}")
                loss.backward()
                self.optimizer.step()
            #############################################################
            #                            记录                           #
            #############################################################
                # tensorboard 可视化
                if (step % self.log_every == 0) and self.use_tensorboard and (self.config['train']['local_rank'] in [0, None]):
                    log_loss = {}
                    log_loss['train/train_loss'] = loss_.item()
                    for tag, value in log_loss.items():
                        self.logger.scalar_summary(tag, value, step)
                elapsed = datetime.now() - start_time
                if (self.config['train']['local_rank'] in [0, None]):
                    print(
                        f"# <trained>: [Epoch {epoch+1}/{self.n_epochs}] [Batch {batch_i+1}/{len(self.train_loader)}] [lr: {self.optimizer.param_groups[0]['lr']}] [train_loss: {loss_.item()}] [time: {elapsed}]")
                if self.config["sorlver"]["lr_method"]["mode"] == "step":
                    self.lr_method.step(
                        epoch + (batch_i+1) / len(self.train_loader))
            if self.config["sorlver"]["lr_method"]["mode"] == "epoch":
                self.lr_method.step(epoch)
            # 保存模型
            self.last_epochs = epoch  # 记录当前迭代次数
            # 每 epoch 次保存一次模型
            if ((epoch+1) % self.model_save_every) == 0 and (self.config['train']['local_rank'] in [0, None]):
                self.save(save_dir=self.model_save_dir,
                          it=self.last_epochs)
            # 每 epoch 次测试下模型
            if((epoch+1) % self.test_every == 0):
                self.test()

        # 保存最终的模型
        if (self.config['train']['local_rank'] in [0, None]):
            self.save(save_dir=self.model_save_dir, it=self.last_epochs)
        return self.accuracylog[-self.config['logs']['test_accuracy_every']:]

    def test(self):
        assert self.test_loader, "未提供测试数据"
        # 测试模型准确率。
        label_preds = torch.rand(0).to(
            device=self.device, dtype=torch.long)  # 预测标签列表
        total_labels = torch.rand(0).to(
            device=self.device, dtype=torch.long)  # 真实标签列表
        total_loss = torch.rand(0).to(
            device=self.device, dtype=torch.float)  # 损失值列表
        loss_func = nn.CrossEntropyLoss(
            reduction='mean').to(device=self.device)

        for data in self.test_loader:
            self.to_device(device=self.device)
            self.set_eval_mode()

            # 获取特征
            if isinstance(data["audio"], torch.nn.utils.rnn.PackedSequence):
                features, feature_lens = torch.nn.utils.rnn.pad_packed_sequence(
                    data["audio"], batch_first=True)
            elif isinstance(data["audio"], torch.Tensor):
                features, feature_lens = data["audio"], None
            # 获取标签
            true_labels = data["labelid"]
            # 将数据迁移到训练设备
            features = features.to(device=self.device)
            if feature_lens is not None:
                feature_lens = feature_lens.to(device=self.device)
            true_labels = true_labels.long().to(device=self.device)

            with torch.no_grad():
                out, _ = self.net(features.permute(
                    0, 3, 1, 2), feature_lens)  # 计算模型输出分类值
                label_pre = torch.max(out, dim=1)[1]  # 获取概率最大值位置
                label_preds = torch.cat(
                    (label_preds, label_pre), dim=0)  # 保存所有预测结果
                total_labels = torch.cat(
                    (total_labels, true_labels), dim=0)  # 保存所有真实结果
                label_loss = torch.tensor(
                    [loss_func(out, true_labels)], device=self.device)
                total_loss = torch.cat((total_loss, label_loss), dim=0)
        # eval_accuracy = accuracy_score(
        #     total_labels.cpu(), label_preds.cpu())
        eval_accuracy = (total_labels == label_preds).sum() / \
            len(label_preds)  # 计算准确率
        eval_loss = total_loss.mean()
        torch.distributed.barrier()
        eval_loss_ = eval_loss.clone()
        eval_accuracy_ = eval_accuracy.clone()
        dist.all_reduce(eval_loss_, op=dist.ReduceOp.SUM)
        dist.all_reduce(eval_accuracy_, op=dist.ReduceOp.SUM)

        eval_loss_ /= self.config["train"]["world_size"]
        eval_accuracy_ /= self.config["train"]["world_size"]
        eval_loss_ = eval_loss_.item()
        eval_accuracy_ = eval_accuracy_.item()

        if (self.config['train']['local_rank'] in [0, None]):
            print(
                f"# <test>: [eval_loss: {eval_loss_}] [eval_accuracy: {eval_accuracy_}]")
            if self.use_tensorboard:
                self.logger.scalar_summary(
                    "Val/Loss_Eval", eval_loss_, self.last_epochs)
                self.logger.scalar_summary(
                    "Val/Accuracy_Eval", eval_accuracy_, self.last_epochs)
            self.accuracylog.append([eval_accuracy_])

    def calculate(self, inputs: list):
        self.to_device(device=self.device)
        self.set_eval_mode()
        label_pres = []

        for i, inp in enumerate(inputs):
            out, _ = self.net(inputs[i].to(
                device=self.device).unsqueeze(0).permute(0, 3, 1, 2))
            label_pres.append(torch.max(out, dim=1)[1].item())
        return label_pres

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

    def print_network(self, model, name):
        """打印网络结构信息"""
        print(f"# 模型名称: {name}")
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()  # 获得张量的元素个数
        print(f"网络参数数量:{num_params}")
        if self.config["logs"]["verbose"]["model"]:
            print("# 打印网络结构:")
            print(model)
        return num_params

    def reset_grad(self):
        """将梯度清零。"""
        self.optimizer.zero_grad()

    def save(self, save_dir=None, it=0):
        """保存模型"""
        if save_dir is None:
            save_dir = self.config['logs']['model_save_dir']
        path = os.path.join(save_dir, self.logname)
        if not os.path.exists(path):
            os.makedirs(path)

        self.config['train']['last_epochs'] = it

        state = {'net': self.net.state_dict(),
                 'optimizer': self.optimizer.state_dict(),
                 'config': self.config,
                 }
        path = os.path.join(path, f"{it:06}.ckpt")
        torch.save(state, path)
        print(f"# 模型存放地址: {path}.")

    def load(self, load_dir):
        """
        load_dir: 要加载模型的完整地址
        """
        print(f"# 模型加载地址: {load_dir}。")

        if self.config['train']['USE_GPU'] and torch.cuda.is_available():
            dictionary = torch.load(load_dir, map_location='cuda:0')
        else:
            dictionary = torch.load(load_dir, map_location='cpu')

        self.config = dictionary['config']
        self.net.load_state_dict(dictionary['net'])
        self.optimizer.load_state_dict(dictionary['optimizer'])
        # self.optimizer = torch.optim.Adam(**dictionary['optimizer'])
        if self.train_loader:
            self.set_configuration()  # 设置加载的检查点的参数设置
            print("# Models 和 optimizers 加载成功")
        else:
            print("# Models 加载成功, 目前仅可进行计算操作")


if __name__ == '__main__':
    pass
