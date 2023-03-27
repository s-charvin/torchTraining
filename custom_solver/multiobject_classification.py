from datetime import datetime
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.logger import Logger
from .lr_scheduler import *
from .optimizer import *
from custom_models import *
from .min_norm_solvers import MinNormSolver, gradient_normalizers
import torch.distributed as dist
from .report import classification_report
import sklearn


class Audio_Classification2MultiObject_MGDA_UB(object):

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
        if self.config['self_auto']['gpu_nums']:
            self.net = self.net.to(device=self.device)
            self.net.encoder = self.net.encoder.to(device=self.device)
            self.net.encoder = torch.nn.parallel.DistributedDataParallel(
                self.net.encoder, device_ids=[self.device])  # device_ids 默认选用本地显示的所有 GPU
            self.net.classifier = self.net.classifier.to(device=self.device)
            self.net.classifier = torch.nn.parallel.DistributedDataParallel(
                self.net.classifier, device_ids=[self.device])  # device_ids 默认选用本地显示的所有 GPU

        # 设置优化器

        model_params = []
        model_params += self.net.encoder.parameters()
        model_params += self.net.classifier.parameters()
        if 'para' in self.config["sorlver"]['optimizer']:
            if self.config["sorlver"]['optimizer']['para']:
                self.optimizer = globals(
                )[self.config["sorlver"]['optimizer']["name"]](params=model_params, lr=self.config["sorlver"]['optimizer']['lr'], **self.config["sorlver"]['optimizer']['para'])
        else:
            self.optimizer = globals(
            )[self.config["sorlver"]['optimizer']["name"]](params=model_params)

        if self.train_loader:
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
        start_iter = self.last_epochs+1  # 训练的新模型则从 1 开始，恢复的模型则从接续的迭代次数开始
        step = 0  # 每次迭代的步数, 也就是所有数据总共分的 batch 数量
        # 更新学习率的方法
        if self.config["sorlver"]["lr_method"]["mode"] == "epoch":
            self.lr_method = globals()[self.config["sorlver"]["lr_method"]["name"]](optimizer=self.optimizer,
                                                                                    last_epoch=self.last_epochs if self.last_epochs >= 1 else -1, **self.config["sorlver"]["lr_method"]["para"])
        elif self.config["sorlver"]["lr_method"]["mode"] == "step":
            self.lr_method = globals()[self.config["sorlver"]["lr_method"]["name"]](optimizer=self.optimizer,
                                                                                    last_epoch=self.last_epochs if self.last_epochs >= 1 else -1, num_batch=len(self.train_loader), **self.config["sorlver"]["lr_method"]["para"])

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
                labels = data["label"]
                # 将数据迁移到训练设备
                features = features.to(device=self.device)
                if feature_lens is not None:
                    feature_lens = feature_lens.to(device=self.device)
                labels = F.one_hot(labels.long(), num_classes=self.net.num_classes).to(
                    device=self.device).float()
                # [batch, class]
                # 根据算法计算损失值的权重 scale
                loss_data = {}
                grads = {}
                scale = {}
                # 清空梯度数据
                self.reset_grad()
                # First compute representations (z)
                # 首先计算共有表征 (z)
                with torch.no_grad():
                    # 使用使用共享参数 (Encoder) 提取共有特征
                    rep, _ = self.net.encoder(features, feature_lens)
                    # As an approximate solution we only need gradients for input
                    # 作为近似解, 我们只需要每个任务的特有梯度
                rep_variable = rep.clone().requires_grad_()

                # 计算每个独立的二分类任务的损失函数关于其特有参数的梯度
                self.reset_grad()
                out_t, _ = self.net.classifier(rep_variable)
                # list([batch,1], ..),None
                for t in range(len(out_t)):  # 遍历所有分类任务
                    loss_func = nn.BCEWithLogitsLoss(
                        reduction='mean', pos_weight=loss_weights[t] if (loss_weights is not None) else None).to(device=self.device)  #
                    # 获取第 t 个分类任务的交叉熵损失
                    loss = loss_func(out_t[t], labels[:, t])
                    loss_data[str(t)] = loss.item()
                    loss.backward()
                    grads[str(t)] = []  # 保留每个独立分类任务反向传播后, rep_variable 的梯度
                    grads[str(t)].append(
                        rep_variable.grad.data.clone().requires_grad_(False))
                    rep_variable.grad.data.zero_()  # 清空当前任务反向传播的梯度

                gn = gradient_normalizers(
                    grads, loss_data, 'loss+')
                # 将每个任务的梯度均除以自身的梯度权重
                for t in range(len(out_t)):
                    for gr_i in range(len(grads[str(t)])):
                        grads[str(t)][gr_i] = grads[str(t)][gr_i] / gn[str(t)]

                # 使用 Frank-Wolfe 算法计算梯度权重
                sol, min_norm = MinNormSolver.find_min_norm_element(
                    [grads[str(t)] for t in range(len(out_t))])
                sol_ = sol.clone().to(device=self.device)  #

                # 在所有进程运行到这一步之前，先完成此前代码的进程会在这里等待其他进程。
                if self.config['self_auto']['gpu_nums']:
                    torch.distributed.barrier()  # 获取所有任务进程计算的平均权重
                    dist.all_reduce(sol_, op=dist.ReduceOp.SUM)
                    sol_ /= self.config['self_auto']['world_size']

                for t in range(len(out_t)):
                    scale[str(t)] = float(sol_[t])  # 存储每个任务的损失权重

                # 清空之前使用算法计算损失权重时利用到的梯度(如果有)
                self.reset_grad()
                # 使用使用共享参数 (Encoder) 提取共有特征
                rep, _ = self.net.encoder(features, feature_lens)
                # 使用特有参数 (Decoder) 获取每个任务的目标输出
                out_t, _ = self.net.classifier(rep)
                for t in range(len(out_t)):
                    # 计算每个任务输出的损失值
                    loss_func = nn.BCEWithLogitsLoss(
                        reduction='mean', pos_weight=loss_weights[t] if (loss_weights is not None) else None).to(device=self.device)
                    # 获取第 t 个分类任务的交叉熵损失
                    loss_t = loss_func(out_t[t], labels[:, t])

                    if t > 0:  # 对其余的任务,
                        loss = loss + scale[str(t)]*loss_t
                    else:  # 对第一个任务
                        loss = scale[str(t)]*loss_t

                loss_ = loss.clone()
                # 在所有进程运行到这一步之前，先完成此前代码的进程会在这里等待其他进程。
                if self.config['self_auto']['gpu_nums']:
                    torch.distributed.barrier()
                    dist.all_reduce(loss_, op=dist.ReduceOp.SUM)
                    # 计算所有 GPU 的平均损失
                    loss_ /= self.config['self_auto']['world_size']

                loss.backward()

            #############################################################
            #                            记录                           #
            #############################################################
                # tensorboard 可视化
                if (self.config['self_auto']['local_rank'] in [0, None]):
                    if (step % self.log_every == 0) and self.use_tensorboard:
                        log_loss = {}
                        log_loss['train/train_loss'] = loss.item()
                        for tag, value in log_loss.items():
                            self.logger.scalar_summary(tag, value, step)
                    runningtime = datetime.now() - start_time
                    est_endtime = (self.n_epochs *
                                   runningtime.seconds/(epoch+1))
                    print(
                        f"# <trained>: [Epoch {epoch}/{self.n_epochs-1}] [Batch {batch_i}/{len(self.train_loader)-1}] [lr: {self.optimizer.param_groups[0]['lr']}] [train_loss: {loss_.item():.4f}] [runtime: {runningtime}] [est_endtime: {est_endtime/360:.2f}h]")
                # 反向更新
                if ((batch_i+1) % self.batch_delay) == 0:
                    self.optimizer.step()  # 根据反向传播的梯度，更新网络参数
                # 更新学习率
                if self.config["sorlver"]["lr_method"]["mode"] == "step":
                    self.lr_method.step()  # 当更新函数需要的是总 step 数量时, 输入小数
                step = step + 1
            # 当更新函数需要的是总 epoch 数量
            if self.config["sorlver"]["lr_method"]["mode"] == "epoch":
                self.lr_method.step()

            self.last_epochs = epoch  # 记录当前已经训练完的迭代次数

            # 每 epoch 次测试下模型
            if((epoch+1) % self.test_every == 0):
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
        loss_func = nn.CrossEntropyLoss(reduction='mean')

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
                rep, _ = self.net.encoder(features, feature_lens)
                out_t, _ = self.net.classifier(rep)
                out = torch.stack(out_t).permute(1, 0)
                label_pre = torch.max(out, dim=1)[1]  # 获取概率最大值位置
                label_preds = torch.cat(
                    (label_preds, label_pre), dim=0)  # 保存所有预测结果
                total_labels = torch.cat(
                    (total_labels, true_labels), dim=0)  # 保存所有真实结果
                label_loss = torch.tensor(
                    [loss_func(out, true_labels)], device=self.device)
                total_loss = torch.cat((total_loss, label_loss), dim=0)

        # 计算准确率

        if self.config['self_auto']['gpu_nums']:
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
            if maxACC < ACC:
                maxACC, WA_, UA_ = ACC, WA, UA
                macro_f1_, w_f1_ = macro_f1, w_f1
                best_re, best_ma = report, matrix
                print(report)
                print(matrix)
                print('The best result ----------')
                print(
                    f'WA: {WA_:.4f}%, UA: {UA_:.4f}%, macro f1: {macro_f1_:.4f}%, weighted f1: {w_f1_:.4f}%')
                print('--------------------------')

                # 每次遇到更好的就保存一次模型
                if self.config['logs']['model_save_every']:
                    self.save(save_dir=self.model_save_dir,
                              it=self.last_epochs)

    def calculate(self, inputs: list):
        self.to_device(device=self.device)
        self.set_eval_mode()
        label_pres = []

        for i, inp in enumerate(inputs):
            with torch.no_grad():
                rep, _ = self.net.encoder(inputs)
                out_t, _ = self.net.classifier(rep)
                out = torch.stack(out_t).permute(1, 0)
                label_id = torch.max(out, dim=1)[1].item()
                label = self.categories[label_id]
                label_pres.append((label_id, label))
        return label_pres

    def set_train_mode(self):
        # 设置模型为训练模式，可添加多个模型
        self.net.encoder.train()  # 训练时使用 BN 层和 Dropout 层
        self.net.classifier.train()
        # 冻结部分前置模块参数的示例
        # for param in self.net.module.wav2vec2.wav2vec2.base_model.parameters():
        #     param.requires_grad = False

    def set_eval_mode(self):
        # 设置模型为评估模式，可添加多个模型
        # 处于评估模式时,批归一化层和dropout层不会在推断时有效果
        self.net.encoder.eval()
        self.net.classifier.eval()

    def to_device(self, device):
        # 将使用到的单个模型或多个模型放到指定的CPU or GPU上
        self.net.encoder.to(device=device)
        self.net.classifier.to(device=device)

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

        state = {'encoder': self.net.encoder.state_dict(),
                 "classifier": self.net.classifier.state_dict(),
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

        self.net.encoder.load_state_dict(dictionary['encoder'])
        self.net.classifier.load_state_dict(dictionary['classifier'])
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


if __name__ == '__main__':
    pass
