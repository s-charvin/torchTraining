from datetime import datetime, timedelta
import os
import torch
import torch.nn as nn
from utils.logger import Logger
import torch.distributed as dist
import sklearn

from custom_models import *
from .lr_scheduler import *
from .optimizer import *
from .report import classification_report


class Audio_Classification(object):

    def __init__(self, train_loader=None, test_loader=None, config: dict = {}, device=None, batch_delay=1):
        """分类模型训练方法框架
        Args:
            train_loader (_type_): 训练集，可迭代类型数据
            test_loader (_type_): 测试集，可迭代类型数据
            config (_type_): 参数设置
            device (_type_): 运行设备
        """
        assert batch_delay >= 1, "batch_delay 需要大于等于 1."
        self.train_loader = train_loader  # 训练数据迭代器
        self.test_loader = test_loader  # 测试数据迭代器
        self.config = config  # 参数字典
        self.device = device  # 定义主设备
        # 定义训练延迟, 当 batch 不太大时的一种训练优化方法, 设置延迟几个 epoch 后再优化参数.
        self.batch_delay = batch_delay
        # 设置要使用的模型
        if self.config['architecture']['net']['para']:
            self.net = globals()[self.config["architecture"]["net"]["name"]](
                **self.config["architecture"]['net']['para'])
        else:
            self.net = globals()[
                self.config["architecture"]["net"]["name"]]()

        # 多 GPU 并行计算
        if self.config['self_auto']['gpu_nums'] > 0:
            self.net = self.net.to(device=self.device)
            self.net = torch.nn.parallel.DistributedDataParallel(
                self.net, device_ids=[self.device])  # device_ids 默认选用本地显示的所有 GPU

        # 设置优化器
        if 'para' in self.config["sorlver"]['optimizer'] and self.config["sorlver"]['optimizer']['para']:
            self.optimizer = globals(
            )[self.config["sorlver"]['optimizer']["name"]](params=self.net.parameters(), lr=self.config["sorlver"]['optimizer']['lr'], **self.config["sorlver"]['optimizer']['para'])
        else:
            self.optimizer = globals(
            )[self.config["sorlver"]['optimizer']["name"]](params=self.net.parameters())

        if self.train_loader is not None:
            self.set_configuration()  # 设置参数

        # 加载模型及其参数
        if self.config['train']['checkpoint'] != "":  # 是否加载历史检查点
            self.load(self.config['train']['checkpoint'])

    def set_configuration(self):
        self.n_epochs = self.config['train']['n_epochs']  # 迭代次数
        # 设置已经训练完成的 epoch, 如果是初次为 -1, epoch 从 0 开始
        self.last_epochs = self.config['train']["last_epochs"]
        if self.last_epochs < -1:
            raise ValueError("last_epochs 值必须为大于等于 -1 的整数, 当其为 -1 时, 表示重新训练.")

        self.test_every = self.config['logs']['test_every']
        if (self.config['self_auto']['local_rank'] in [0, None]):
            # 使用 tensorboard 记录
            self.use_tensorboard = self.config['logs']['use_tensorboard']
            self.log_every = self.config['logs']['log_every']  # 每几步记录
            # 每几步保存模型
            self.model_save_every = self.config['logs']['model_save_every']
            # 模型保存地址
            self.model_save_dir = self.config['logs']['model_save_dir']
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

    def label2id(self, label_list):
        label_encoder = sklearn.preprocessing.LabelEncoder()
        label_encoder.fit(y=label_list)
        # id2label = {f"{id}": label for id, label in enumerate()}
        return label_encoder.transform(label_list), label_encoder.classes_

    def train(self):
        """训练主函数"""
        assert self.train_loader, "未提供训练数据"
        #############################################################
        #                            训练                           #
        #############################################################
        start_iter = self.last_epochs+1  # 训练的新模型则从 0 开始，恢复的模型则从接续的迭代次数开始
        step = 0  # 每次迭代的步数, 也就是所有数据总共分的 batch 数量
        # 更新学习率的方法
        if self.config["sorlver"]["lr_method"]["mode"] == "epoch":
            self.lr_method = globals()[self.config["sorlver"]["lr_method"]["name"]](optimizer=self.optimizer,
                                                                                    last_epoch=self.last_epochs if self.last_epochs >= 1 else -1, **self.config["sorlver"]["lr_method"]["para"])
        elif self.config["sorlver"]["lr_method"]["mode"] == "step":
            self.lr_method = globals()[self.config["sorlver"]["lr_method"]["name"]](optimizer=self.optimizer,
                                                                                    last_epoch=self.last_epochs if self.last_epochs >= 1 else -1, steps_per_epoch=len(self.train_loader), **self.config["sorlver"]["lr_method"]["para"])

        # 训练开始时间记录
        start_time = datetime.now()
        if (self.config['self_auto']['local_rank'] in [0, None]):
            print(f'# 训练开始时间: {start_time}')  # 读取开始运行时间

        # 处理标签数据

        self.train_loader.dataset.dataset.datadict["label"], self.categories = self.label2id(
            self.train_loader.dataset.dataset.datadict["label"])

        # 设置损失函数
        loss_weights = None
        if self.config["train"]["loss_weight"]:
            total = len(self.train_loader.dataset.indices)
            labels = [self.train_loader.dataset.dataset[i]["label"]
                      for i in self.train_loader.dataset.indices]
            loss_weights = torch.Tensor([
                total/labels.count(i) for i in range(len(self.categories))]).to(device=self.device)
        else:
            self.config["self_auto"]["loss_weights"] = None

        loss_func = nn.CrossEntropyLoss(
            reduction='mean', weight=loss_weights).to(device=self.device)

        # 保存最好的结果
        self.maxACC = self.WA_ = self.UA_ = self.macro_f1_ = self.w_f1_ = 0
        est_endtime = "..."
        for epoch in range(start_iter, self.n_epochs):  # epoch 迭代循环 [0, epoch)
            # [0, num_batch)
            for batch_i, data in enumerate(self.train_loader):
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
                labels = data["label"]
                # 将数据迁移到训练设备
                features = features.to(device=self.device)
                if feature_lens is not None:
                    feature_lens = feature_lens.to(device=self.device)
                labels = labels.long().to(device=self.device)

                # 清空梯度数据
                self.reset_grad()
                # [In: B,C,F,S]
                out, _ = self.net(features, feature_lens)
                loss = loss_func(out, labels)
                loss_ = loss.clone()
                loss.backward()
                # 在所有进程运行到这一步之前，先完成此前代码的进程会在这里等待其他进程。
                if self.config['self_auto']['gpu_nums'] > 0:
                    torch.distributed.barrier()
                    dist.all_reduce(loss_, op=dist.ReduceOp.SUM)
                    # 计算所有 GPU 的平均损失
                    loss_ /= self.config['self_auto']['world_size']

            #############################################################
            #                            记录                           #
            #############################################################
                # tensorboard 可视化及打印训练信息
                if (self.config['self_auto']['local_rank'] in [0, None]):
                    if (step % self.log_every == 0) and self.use_tensorboard:
                        log_loss = {}
                        log_loss['train/train_loss'] = loss_.item()
                        for tag, value in log_loss.items():
                            self.logger.scalar_summary(tag, value, step)
                    runningtime = datetime.now() - start_time
                    print(
                        f"# <trained>: [Epoch {epoch}/{self.n_epochs-1}] [Batch {batch_i}/{len(self.train_loader)-1}] [lr: {self.optimizer.param_groups[0]['lr']}] [train_loss: {loss_.item():.4f}] [runtime: {runningtime}] [est_endtime: {est_endtime}]")

                # 反向更新
                if ((batch_i+1) % self.batch_delay) == 0:
                    self.optimizer.step()  # 根据反向传播的梯度，更新网络参数

                # 更新学习率
                if self.config["sorlver"]["lr_method"]["mode"] == "step":
                    self.lr_method.step()  # 当更新函数需要的是总 step 数量时, 输入小数
                step = step + 1
            if self.config["sorlver"]["lr_method"]["mode"] == "epoch":
                self.lr_method.step()  # 当更新函数需要的是总 epoch 数量时, 输入整数
            est_endtime = timedelta(
                seconds=self.n_epochs * ((runningtime.days*86400+runningtime.seconds)/(epoch+1)))
            self.last_epochs = epoch  # 记录当前已经训练完的迭代次数

            # 每 epoch 次测试下模型
            if((epoch + 1) % self.test_every == 0):
                self.test()

        # 保存最终的模型
        if (self.config['self_auto']['local_rank'] in [0, None]):
            self.save(save_dir=self.model_save_dir, it=self.last_epochs)
        return True

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

        maxACC = 0
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
            true_labels = data["label"]

            # 将数据迁移到训练设备
            features = features.to(device=self.device)

            if feature_lens is not None:
                feature_lens = feature_lens.to(device=self.device)
            true_labels = true_labels.long().to(device=self.device)

            with torch.no_grad():
                out, _ = self.net(features, feature_lens)  # 计算模型输出分类值
                label_pre = torch.max(out, dim=1)[1]  # 获取概率最大值位置
                label_preds = torch.cat(
                    (label_preds, label_pre), dim=0)  # 保存所有预测结果
                total_labels = torch.cat(
                    (total_labels, true_labels), dim=0)  # 保存所有真实结果
                label_loss = torch.tensor(
                    [loss_func(out, true_labels)], device=self.device)
                total_loss = torch.cat((total_loss, label_loss), dim=0)

        # 计算准确率

        if self.config['self_auto']['gpu_nums'] > 0:
            world_size = self.config['self_auto']['world_size']
            total_loss_l = [torch.zeros_like(total_loss)
                            for i in range(world_size)]
            label_preds_l = [torch.zeros_like(
                label_preds) for i in range(world_size)]
            total_labels_l = [torch.zeros_like(
                total_labels) for i in range(world_size)]

            torch.distributed.barrier()
            dist.all_gather(total_loss_l, total_loss, )
            dist.all_gather(label_preds_l, label_preds, )
            dist.all_gather(total_labels_l, total_labels, )
            total_labels = torch.cat(total_labels_l)
            label_preds = torch.cat(label_preds_l)
            total_loss = torch.cat(total_loss_l)

        if (self.config['self_auto']['local_rank'] in [0, None]):
            WA, UA, ACC, macro_f1, w_f1, report, matrix = classification_report(
                y_true=total_labels.cpu(), y_pred=label_preds.cpu(), target_names=self.categories)

            print(
                f"# <tested>: [WA: {WA}] [UA: {UA}] [ACC: {ACC}] [macro_f1: {macro_f1}] [w_f1: {w_f1}]")
            if self.use_tensorboard:
                self.logger.scalar_summary(
                    "Val/Loss_Eval", total_loss.mean().clone(), self.last_epochs)
                self.logger.scalar_summary(
                    "Val/Accuracy_Eval", ACC, self.last_epochs)

            # 记录最好的一次模型数据
            if self.maxACC < ACC:
                self.maxACC, self.WA_, self.UA_ = ACC, WA, UA
                self.macro_f1_, self.w_f1_ = macro_f1, w_f1
                best_re, best_ma = report, matrix

                # 每次遇到更好的就保存一次模型
                if self.config['logs']['model_save_every']:
                    self.save(save_dir=self.model_save_dir,
                              it=self.last_epochs)
            print(
                f"# <best>: [WA: {self.WA_}] [UA: {self.UA_}] [ACC: {self.maxACC}] [macro_f1: {self.macro_f1_}] [w_f1: {self.w_f1_}]")

    def calculate(self, inputs: list):
        self.to_device(device=self.device)
        self.set_eval_mode()
        label_pres = []
        for i, inp in enumerate(inputs):
            with torch.no_grad():
                out, _ = self.net(inputs[i].to(
                    device=self.device).unsqueeze(0))
                label_id = torch.max(out, dim=1)[1].item()
                label = self.categories[label_id]
                label_pres.append((label_id, label))
        return label_pres

    def set_train_mode(self):
        # 设置模型为训练模式，可添加多个模型
        self.net.train()  # 训练时使用BN层和Dropout层

    def set_eval_mode(self):
        # 设置模型为评估模式，可添加多个模型
        # 处于评估模式时,批归一化层和 dropout 层不会在推断时有效果
        self.net.eval()

    def to_device(self, device):
        # 将使用到的单个模型或多个模型放到指定的CPU or GPU上
        self.net.to(device=device)

    def print_network(self):
        """打印网络结构信息"""
        modelname = self.config["architecture"]["net"]["name"]
        print(f"# 模型名称: {modelname}")
        num_params = 0
        for p in self.net.parameters():
            num_params += p.numel()  # 获得张量的元素个数
        if self.config["logs"]["verbose"]["model"]:
            print("# 打印网络结构:")
            print(self.net)
        print(f"网络参数数量:{num_params}")

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
                 "train_indices": self.train_loader.dataset.indices,  # 训练数据索引
                 "test_indices": self.test_loader.dataset.indices,  # 测试数据索引
                 }
        path = os.path.join(path, f"{it:06}.ckpt")
        torch.save(state, path)
        print(f"# 模型存放地址: {path}.")

    def load(self, load_dir):
        """
        load_dir: 要加载模型的完整地址
        """
        print(f"# 模型加载地址: {load_dir}。")

        # 保存每个训练进程专属的信息
        self_auto_config = self.config["self_auto"]
        self.last_epochs = self.config['train']["last_epochs"]

        # 加载模型, 参数配置, 和数据索引信息
        dictionary = torch.load(load_dir, map_location=self.device)
        self.config = dictionary['config']
        self.config["self_auto"] = self_auto_config

        self.net.load_state_dict(dictionary['net'])
        self.optimizer.load_state_dict(dictionary['optimizer'])

        # 设置已经训练完成的 epoch, 如果是初次为 -1, epoch 从 0 开始
        if self.last_epochs > -1:
            self.config['train']["last_epochs"] = self.last_epochs
        else:
            self.last_epochs = self.config['train']["last_epochs"]

        # 保证当前的训练数据与之前的训练数据一致(这里是为了防止迁移训练环境后导致训练数据的随机分布发生变化)
        if self.train_loader:
            self.train_loader.dataset.indices = dictionary["train_indices"]
            self.test_loader.dataset.indices = dictionary["test_indices"]
            self.set_configuration()  # 设置加载的检查点的参数设置
            print("# Models 和 optimizers 加载成功")
        else:
            print("# Models 加载成功, 目前仅可进行计算操作")
        del dictionary


class Audio_Classification_CoTeaching(object):

    def __init__(self, train_loader=None, test_loader=None, config: dict = {}, device=None, batch_delay=1):
        """分类模型训练方法框架
        Args:
            train_loader (_type_): 训练集，可迭代类型数据
            test_loader (_type_): 测试集，可迭代类型数据
            config (_type_): 参数设置
            device (_type_): 运行设备
        """
        assert batch_delay >= 1, "batch_delay 需要大于等于 1."
        self.train_loader = train_loader  # 训练数据迭代器
        self.test_loader = test_loader  # 测试数据迭代器
        self.config = config  # 参数字典
        self.device = device  # 定义主设备
        # 定义训练延迟, 当 batch 不太大时的一种训练优化方法, 设置延迟几个 epoch 后再优化参数.
        self.batch_delay = batch_delay
        # 设置要使用的模型
        if self.config['architecture']['net']['para']:
            self.net = globals()[self.config["architecture"]["net"]["name"]](
                **self.config["architecture"]['net']['para'])
        else:
            self.net = globals()[
                self.config["architecture"]["net"]["name"]]()

        # 多 GPU 并行计算
        if self.config['self_auto']['gpu_nums'] > 0:
            self.net = self.net.to(device=self.device)
            self.net = torch.nn.parallel.DistributedDataParallel(
                self.net, device_ids=[self.device])  # device_ids 默认选用本地显示的所有 GPU

        # 设置优化器
        if 'para' in self.config["sorlver"]['optimizer'] and self.config["sorlver"]['optimizer']['para']:
            self.optimizer1 = globals(
            )[self.config["sorlver"]['optimizer']["name"]](params=self.net.module.coopnet0.parameters(), lr=self.config["sorlver"]['optimizer']['lr'], **self.config["sorlver"]['optimizer']['para'])
            self.optimizer2 = globals(
            )[self.config["sorlver"]['optimizer']["name"]](params=self.net.module.coopnet1.parameters(), lr=self.config["sorlver"]['optimizer']['lr'], **self.config["sorlver"]['optimizer']['para'])

        else:
            self.optimizer1 = globals(
            )[self.config["sorlver"]['optimizer']["name"]](params=self.net.module.coopnet0.parameters())
            self.optimizer2 = globals(
            )[self.config["sorlver"]['optimizer']["name"]](params=self.net.module.coopnet1.parameters())

        if self.train_loader is not None:
            self.set_configuration()  # 设置参数

        # 加载模型及其参数
        if self.config['train']['checkpoint'] != "":  # 是否加载历史检查点
            self.load(self.config['train']['checkpoint'])

    def set_configuration(self):
        self.n_epochs = self.config['train']['n_epochs']  # 迭代次数
        # 设置已经训练完成的 epoch, 如果是初次为 -1, epoch 从 0 开始
        self.last_epochs = self.config['train']["last_epochs"]
        if self.last_epochs < -1:
            raise ValueError("last_epochs 值必须为大于等于 -1 的整数, 当其为 -1 时, 表示重新训练.")

        self.test_every = self.config['logs']['test_every']
        if (self.config['self_auto']['local_rank'] in [0, None]):
            # 使用 tensorboard 记录
            self.use_tensorboard = self.config['logs']['use_tensorboard']
            self.log_every = self.config['logs']['log_every']  # 每几步记录
            # 每几步保存模型
            self.model_save_every = self.config['logs']['model_save_every']
            # 模型保存地址
            self.model_save_dir = self.config['logs']['model_save_dir']
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

    def label2id(self, label_list):
        label_encoder = sklearn.preprocessing.LabelEncoder()
        label_encoder.fit(y=label_list)
        # id2label = {f"{id}": label for id, label in enumerate()}
        return label_encoder.transform(label_list), label_encoder.classes_

    def train(self):
        """训练主函数"""
        assert self.train_loader, "未提供训练数据"
        #############################################################
        #                            训练                           #
        #############################################################
        start_iter = self.last_epochs+1  # 训练的新模型则从 0 开始，恢复的模型则从接续的迭代次数开始
        step = 0  # 每次迭代的步数, 也就是所有数据总共分的 batch 数量
        # 更新学习率的方法
        if self.config["sorlver"]["lr_method"]["mode"] == "epoch":
            self.lr_method1 = globals()[self.config["sorlver"]["lr_method"]["name"]](optimizer=self.optimizer1,
                                                                                     last_epoch=self.last_epochs if self.last_epochs >= 1 else -1, **self.config["sorlver"]["lr_method"]["para"])
            self.lr_method2 = globals()[self.config["sorlver"]["lr_method"]["name"]](optimizer=self.optimizer2,
                                                                                     last_epoch=self.last_epochs if self.last_epochs >= 1 else -1, **self.config["sorlver"]["lr_method"]["para"])
        elif self.config["sorlver"]["lr_method"]["mode"] == "step":
            self.lr_method1 = globals()[self.config["sorlver"]["lr_method"]["name"]](optimizer=self.optimizer1,
                                                                                     last_epoch=self.last_epochs if self.last_epochs >= 1 else -1, steps_per_epoch=len(self.train_loader), **self.config["sorlver"]["lr_method"]["para"])
            self.lr_method2 = globals()[self.config["sorlver"]["lr_method"]["name"]](optimizer=self.optimizer2,
                                                                                     last_epoch=self.last_epochs if self.last_epochs >= 1 else -1, steps_per_epoch=len(self.train_loader), **self.config["sorlver"]["lr_method"]["para"])

        # 训练开始时间记录
        start_time = datetime.now()
        if (self.config['self_auto']['local_rank'] in [0, None]):
            print(f'# 训练开始时间: {start_time}')  # 读取开始运行时间

        # 处理标签数据

        self.train_loader.dataset.dataset.datadict["label"], self.categories = self.label2id(
            self.train_loader.dataset.dataset.datadict["label"])

        # 设置损失函数
        self.loss_weights = None
        if self.config["train"]["loss_weight"]:
            total = len(self.train_loader.dataset.indices)
            labels = [self.train_loader.dataset.dataset[i]["label"]
                      for i in self.train_loader.dataset.indices]
            self.loss_weights = torch.Tensor([
                total/labels.count(i) for i in range(len(self.categories))]).to(device=self.device)
        else:
            self.config["self_auto"]["loss_weights"] = None

        loss_func = nn.CrossEntropyLoss(
            reduction='none', weight=self.loss_weights).to(device=self.device)

        forget_schedule = np.ones(self.n_epochs)*0.2
        forget_schedule[:int(self.n_epochs/5)] = np.linspace(0,
                                                             0.2**1, int(self.n_epochs/5))
        forget_schedule[-int(self.n_epochs/5):] = np.linspace(0.2**1,
                                                              0, int(self.n_epochs/5))

        # 保存最好的结果
        self.maxACC = self.WA_ = self.UA_ = self.macro_f1_ = self.w_f1_ = 0
        est_endtime = "..."
        for epoch in range(start_iter, self.n_epochs):  # epoch 迭代循环 [0, epoch)
            # [0, num_batch)
            for batch_i, data in enumerate(self.train_loader):
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
                labels = data["label"]
                # 将数据迁移到训练设备
                features = features.to(device=self.device)
                if feature_lens is not None:
                    feature_lens = feature_lens.to(device=self.device)
                true_labels = labels.long().to(device=self.device)

                # 清空梯度数据
                self.reset_grad()
                # [In: B,C,F,S]

                out1, _ = self.net.module.coopnet0(features)

                out2, _ = self.net.module.coopnet1(features.detach())

                loss_1, loss_2 = self.loss_coteaching(
                    out1, out2, true_labels, forget_schedule[epoch-1], loss_func=loss_func)

                loss_1.backward()
                loss_2.backward()

                loss_1_ = loss_1.clone().detach()
                loss_2_ = loss_2.clone().detach()

                # 在所有进程运行到这一步之前，先完成此前代码的进程会在这里等待其他进程
                if self.config['self_auto']['gpu_nums'] > 0:
                    torch.distributed.barrier()
                    dist.all_reduce(loss_1_, op=dist.ReduceOp.SUM)
                    dist.all_reduce(loss_2_, op=dist.ReduceOp.SUM)
                    # 计算所有 GPU 的平均损失
                    loss_1_ /= self.config['self_auto']['world_size']
                    loss_2_ /= self.config['self_auto']['world_size']

            #############################################################
            #                            记录                           #
            #############################################################
                # tensorboard 可视化及打印训练信息
                if (self.config['self_auto']['local_rank'] in [0, None]):
                    if (step % self.log_every == 0) and self.use_tensorboard:
                        log_loss = {}
                        log_loss['train/train_loss1'] = loss_1_.item()
                        log_loss['train/train_loss2'] = loss_2_.item()
                        for tag, value in log_loss.items():
                            self.logger.scalar_summary(tag, value, step)
                    runningtime = datetime.now() - start_time

                    print(
                        f"# <trained>: [Epoch {epoch}/{self.n_epochs-1}] [Batch {batch_i}/{len(self.train_loader)-1}] [lr: {self.optimizer1.param_groups[0]['lr']}] [train_loss: {loss_1_+loss_2_.item():.4f}] [runtime: {runningtime}] [est_endtime: {est_endtime}]")

                # 反向更新
                if ((batch_i+1) % self.batch_delay) == 0:
                    self.optimizer1.step()  # 根据反向传播的梯度，更新网络参数
                    self.optimizer2.step()

                # 更新学习率
                if self.config["sorlver"]["lr_method"]["mode"] == "step":
                    self.lr_method1.step()  # 当更新函数需要的是总 step 数量时, 输入小数
                    self.lr_method2.step()  # 当更新函数需要的是总 step 数量时, 输入小数
                step = step + 1
            if self.config["sorlver"]["lr_method"]["mode"] == "epoch":
                self.lr_method1.step()  # 当更新函数需要的是总 epoch 数量时, 输入整数
                self.lr_method2.step()  # 当更新函数需要的是总 epoch 数量时, 输入整数
            est_endtime = timedelta(
                seconds=self.n_epochs * ((runningtime.days*86400+runningtime.seconds)/(epoch+1)))
            self.last_epochs = epoch  # 记录当前已经训练完的迭代次数

            # 每 epoch 次测试下模型
            if((epoch + 1) % self.test_every == 0):
                self.test()

        # 保存最终的模型
        if (self.config['self_auto']['local_rank'] in [0, None]):
            self.save(save_dir=self.model_save_dir, it=self.last_epochs)
        return True

    def test(self):
        assert self.test_loader, "未提供测试数据"
        # 测试模型准确率。
        label_preds1 = torch.rand(0).to(
            device=self.device, dtype=torch.long)  # 预测标签列表
        label_preds2 = torch.rand(0).to(
            device=self.device, dtype=torch.long)  # 预测标签列表
        total_labels1 = torch.rand(0).to(
            device=self.device, dtype=torch.long)  # 真实标签列表
        total_labels2 = torch.rand(0).to(
            device=self.device, dtype=torch.long)  # 真实标签列表
        total_loss1 = torch.rand(0).to(
            device=self.device, dtype=torch.float)  # 损失值列表
        total_loss2 = torch.rand(0).to(
            device=self.device, dtype=torch.float)  # 损失值列表
        loss_func = nn.CrossEntropyLoss(
            reduction='mean').to(device=self.device)

        maxACC = 0
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
            true_labels = data["label"]
            # 将数据迁移到训练设备
            features = features.to(device=self.device)
            if feature_lens is not None:
                feature_lens = feature_lens.to(device=self.device)
            true_labels = true_labels.long().to(device=self.device)

            with torch.no_grad():

                out1, _ = self.net.module.coopnet0(features)
                out2, _ = self.net.module.coopnet1(features.detach())

                label_pre1 = torch.max(out1, dim=1)[1]  # 获取概率最大值位置
                label_preds1 = torch.cat(
                    (label_preds1, label_pre1), dim=0)  # 保存所有预测结果
                total_labels1 = torch.cat(
                    (total_labels1, true_labels), dim=0)  # 保存所有真实结果
                label_loss1 = torch.tensor(
                    [loss_func(out1, true_labels)], device=self.device)
                total_loss1 = torch.cat((total_loss1, label_loss1), dim=0)

                label_pre2 = torch.max(out2, dim=1)[1]  # 获取概率最大值位置
                label_preds2 = torch.cat(
                    (label_preds2, label_pre2), dim=0)  # 保存所有预测结果
                total_labels2 = torch.cat(
                    (total_labels2, true_labels), dim=0)  # 保存所有真实结果
                label_loss2 = torch.tensor(
                    [loss_func(out2, true_labels)], device=self.device)
                total_loss2 = torch.cat((total_loss2, label_loss2), dim=0)

        # 计算准确率

        if self.config['self_auto']['gpu_nums'] > 0:
            world_size = self.config['self_auto']['world_size']
            total_loss_l1 = [torch.zeros_like(total_loss1)
                             for i in range(world_size)]
            label_preds_l1 = [torch.zeros_like(
                label_preds1) for i in range(world_size)]
            total_labels_l1 = [torch.zeros_like(
                total_labels1) for i in range(world_size)]

            total_loss_l2 = [torch.zeros_like(total_loss2)
                             for i in range(world_size)]
            label_preds_l2 = [torch.zeros_like(
                label_preds2) for i in range(world_size)]
            total_labels_l2 = [torch.zeros_like(
                total_labels2) for i in range(world_size)]

            torch.distributed.barrier()

            dist.all_gather(total_loss_l1, total_loss1, )
            dist.all_gather(label_preds_l1, label_preds1, )
            dist.all_gather(total_labels_l1, total_labels1, )
            total_labels1 = torch.cat(total_labels_l1)
            label_preds1 = torch.cat(label_preds_l1)
            total_loss1 = torch.cat(total_loss_l1)

            dist.all_gather(total_loss_l2, total_loss2, )
            dist.all_gather(label_preds_l2, label_preds2, )
            dist.all_gather(total_labels_l2, total_labels2, )
            total_labels2 = torch.cat(total_labels_l2)
            label_preds2 = torch.cat(label_preds_l2)
            total_loss2 = torch.cat(total_loss_l2)

        if (self.config['self_auto']['local_rank'] in [0, None]):
            WA1, UA1, ACC1, macro_f11, w_f11, report1, matrix1 = classification_report(
                y_true=total_labels1.cpu(), y_pred=label_preds1.cpu(), target_names=self.categories)
            WA2, UA2, ACC2, macro_f12, w_f12, report2, matrix2 = classification_report(
                y_true=total_labels2.cpu(), y_pred=label_preds2.cpu(), target_names=self.categories)
            print(
                f"# <tested>: [WA1: {WA1}] [UA1: {UA1}] [ACC1: {ACC1}] [macro_f11: {macro_f11}] [w_f11: {w_f11}]")
            print(
                f"# <tested>: [WA2: {WA2}] [UA2: {UA2}] [ACC2: {ACC2}] [macro_f12: {macro_f12}] [w_f12: {w_f12}]")

            if self.use_tensorboard:
                self.logger.scalar_summary(
                    "Val/Loss_Eval1", total_loss1.mean().clone(), self.last_epochs)
                self.logger.scalar_summary(
                    "Val/Accuracy_Eval1", ACC1, self.last_epochs)
                self.logger.scalar_summary(
                    "Val/Loss_Eval2", total_loss2.mean().clone(), self.last_epochs)
                self.logger.scalar_summary(
                    "Val/Accuracy_Eval2", ACC2, self.last_epochs)

            # 记录最好的一次模型数据
            if self.maxACC < (ACC1+ACC2)/2:
                self.maxACC, self.WA_, self.UA_ = (
                    ACC1+ACC2)/2, (WA1+WA2)/2, (UA1+UA2)/2
                self.macro_f1_, self.w_f1_ = (
                    macro_f11, macro_f12), (w_f11, w_f12)
                best_re, best_ma = (report1, report2), (matrix1, matrix2)

                # 每次遇到更好的就保存一次模型
                if self.config['logs']['model_save_every']:
                    self.save(save_dir=self.model_save_dir,
                              it=self.last_epochs)
                print(best_re)
                print(best_ma)
            print(
                f"# <best>: [WA: {self.WA_}] [UA: {self.UA_}] [ACC: {self.maxACC}] [macro_f1: {self.macro_f1_}] [w_f1: {self.w_f1_}]")

    def loss_coteaching(self, logits1, logits2, labels, forget_rate, loss_func=nn.CrossEntropyLoss(reduction='none')):
        # 计算此batch内数据的所有损失
        # 设置损失函数和分类权重，权重数据在s.train()前定义了
        loss_func = loss_func
        device_ = logits1.device
        loss_1 = loss_func(logits1, labels)
        ind_1_sorted = np.argsort(loss_1.data.cpu())  # 位置
        loss_1_sorted = loss_1[ind_1_sorted]  # 位置对应数据

        loss_2 = loss_func(logits2, labels)
        ind_2_sorted = np.argsort(loss_2.data.cpu())
        loss_2_sorted = loss_2[ind_2_sorted]

        remember_rate = 1 - forget_rate
        # 计算需要记住的数据数量
        num_remember = int(remember_rate * len(loss_1_sorted))
        ind_1_update = ind_1_sorted[:num_remember]
        ind_2_update = ind_2_sorted[:num_remember]
        # 计算筛选后的数据Loss, 此处通过index进行了数据位置交换
        loss_func_mean = nn.CrossEntropyLoss(
            weight=self.loss_weights, reduction='mean')

        loss_1_update = loss_func_mean(
            logits1[ind_2_update], labels[ind_2_update])
        loss_2_update = loss_func_mean(
            logits2[ind_1_update], labels[ind_1_update])
        # loss_1_update = torch.sum(loss_1_update)/num_remember
        # loss_1_update = torch.sum(loss_2_update)/num_remember
        return loss_1_update.to(device=device_), loss_2_update.to(device=device_)

        # return torch.mean(loss_1).to(device=device_), torch.mean(loss_2).to(device=device_)

    def calculate(self, inputs: list):
        self.to_device(device=self.device)
        self.set_eval_mode()
        label_pres = []
        for i, inp in enumerate(inputs):
            with torch.no_grad():
                out, _ = self.net(inputs[i].to(
                    device=self.device).unsqueeze(0))
                label_id = torch.max(out, dim=1)[1].item()
                label = self.categories[label_id]
                label_pres.append((label_id, label))
        return label_pres

    def set_train_mode(self):
        # 设置模型为训练模式，可添加多个模型
        self.net.train()  # 训练时使用BN层和Dropout层

    def set_eval_mode(self):
        # 设置模型为评估模式，可添加多个模型
        # 处于评估模式时,批归一化层和 dropout 层不会在推断时有效果
        self.net.eval()

    def to_device(self, device):
        # 将使用到的单个模型或多个模型放到指定的CPU or GPU上
        self.net.to(device=device)

    def print_network(self):
        """打印网络结构信息"""
        modelname = self.config["architecture"]["net"]["name"]
        print(f"# 模型名称: {modelname}")
        num_params = 0
        for p in self.net.parameters():
            num_params += p.numel()  # 获得张量的元素个数
        if self.config["logs"]["verbose"]["model"]:
            print("# 打印网络结构:")
            print(self.net)
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
        path = os.path.join(save_dir, self.logname)
        if not os.path.exists(path):
            os.makedirs(path)

        self.config['train']['last_epochs'] = it

        state = {'net': self.net.state_dict(),
                 'optimizer1': self.optimizer1.state_dict(),
                 'optimizer2': self.optimizer2.state_dict(),
                 'config': self.config,
                 "train_indices": self.train_loader.dataset.indices,  # 训练数据索引
                 "test_indices": self.test_loader.dataset.indices,  # 测试数据索引
                 }
        path = os.path.join(path, f"{it:06}.ckpt")
        torch.save(state, path)
        print(f"# 模型存放地址: {path}.")

    def load(self, load_dir):
        """
        load_dir: 要加载模型的完整地址
        """
        print(f"# 模型加载地址: {load_dir}。")

        # 保存每个训练进程专属的信息
        self_auto_config = self.config["self_auto"]
        self.last_epochs = self.config['train']["last_epochs"]

        # 加载模型, 参数配置, 和数据索引信息
        dictionary = torch.load(load_dir, map_location=self.device)
        self.config = dictionary['config']
        self.config["self_auto"] = self_auto_config

        self.net.load_state_dict(dictionary['net'])
        self.optimizer1.load_state_dict(dictionary['optimizer1'])
        self.optimizer2.load_state_dict(dictionary['optimizer2'])

        # 设置已经训练完成的 epoch, 如果是初次为 -1, epoch 从 0 开始
        if self.last_epochs > -1:
            self.config['train']["last_epochs"] = self.last_epochs
        else:
            self.last_epochs = self.config['train']["last_epochs"]

        # 保证当前的训练数据与之前的训练数据一致(这里是为了防止迁移训练环境后导致训练数据的随机分布发生变化)
        if self.train_loader:
            self.train_loader.dataset.indices = dictionary["train_indices"]
            self.test_loader.dataset.indices = dictionary["test_indices"]
            self.set_configuration()  # 设置加载的检查点的参数设置
            print("# Models 和 optimizers 加载成功")
        else:
            print("# Models 加载成功, 目前仅可进行计算操作")
        del dictionary


if __name__ == '__main__':
    pass
