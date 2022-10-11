from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from models import SequenceClassifier
from logger import Logger


class Audio_Classification(object):

    def __init__(self, train_loader, test_loader, config, device):
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
        self.model = SequenceClassifier(self.config)
        self.set_configuration()  # 设置参数
        # 加载模型及其参数
        if self.config['train']['checkpoint']:  # 是否加载历史检查点
            self.load_checkpoint()

    def load_checkpoint(self):
        self.model.load(self.config['train']['checkpoint'])
        self.config = self.model.config
        self.set_configuration()  # 设置加载的检查点的参数设置

    def set_configuration(self):
        self.n_epochs = self.config['train']['n_epochs']  # 迭代次数
        # 更新 lr 的延迟
        self.num_epochs_decay = self.config['train']['num_epochs_decay']
        # 恢复迭代的起始步，默认为0
        self.resume_epochs = self.config['train']['resume_epochs']
        self.current_epochs = self.resume_epochs  # 当前迭代次数
        # 优化器其余参数
        self.optimizer_lr = self.config["sorlver"]['optimizer']['lr']  # 学习率
        self.optimizer_para = None
        if 'para' in self.config["sorlver"]['optimizer']:
            self.optimizer_para = self.config["sorlver"]['optimizer']['para']
        # 使用 tensorboard 记录
        self.use_tensorboard = self.config['logs']['use_tensorboard']
        self.log_every = self.config['logs']['log_every']  # 每几步记录
        self.test_every = self.config['logs']['test_every']
        # 每几步保存模型
        self.model_save_every = self.config['logs']['model_save_every']
        self.model_save_dir = self.config['logs']['model_save_dir']  # 模型保存地址
        if self.use_tensorboard:
            logname = "_".join(
                self.config['train']['name'],  # 当前训练名称
                self.config['train']['seed'],
                self.config['train']['batch_size'],
                self.n_epochs,
                self.config['model']['architectures'],
                self.config['sorlver']['name'],
                self.config['sorlver']['optimizer'] +
                self.config['sorlver']['optimizer']["lr"],
                self.config['data']['name'],
            )
            self.logger = Logger(
                self.config['logs']['log_dir'], logname)

    def train(self):
        """训练主函数"""

        #############################################################
        #                            训练                           #
        #############################################################
        print('################ 开始循环训练 ################')
        start_iter = self.resume_epochs + 1  # 训练的新模型则从 1 开始，恢复的模型则从接续的迭代次数开始
        self.update_lr(start_iter)  # 根据开始的 epoch 次数初始化学习率
        self.accuracylog = []
        start_time = datetime.now()
        print(f'# 训练开始时间: {start_time}')  # 读取开始运行时间
        for epoch in range(start_iter, self.n_epochs+1):  # 迭代循环
            for step, data in enumerate(self.train_loader):
                print(
                    f"# [Epoch {epoch}/{self.n_epochs}] [Batch {step}/{len(self.train_loader)}] [lr: {self.model.optimizer.param_groups[0]['lr']}]")
                self.model.to_device(device=self.device)
                self.model.set_train_mode()
                self.current_epochs = step  # 记录当前迭代次数
                if isinstance(data["audio"][0], torch.nn.utils.rnn.PackedSequence):
                    features, feature_lens = torch.nn.utils.rnn.pad_packed_sequence(
                        data["audio"], batch_first=True)
                elif isinstance(data["audio"][0], torch.Tensor):
                    features, feature_lens = data["audio"], None
                labels = data["label"]

                features = features.to(device=self.device)
                labels = labels.long().to(device=self.device)

                loss_func = nn.CrossEntropyLoss(reduction='mean')
                self.model.reset_grad()  # 清空梯度

                y = self.model.net(features, feature_lens)
                loss = loss_func(out, true_labels)
                loss.backward()
                self.model.optimizer.step()
            #############################################################
            #                            记录                           #
            #############################################################
                print(
                    f"# [train_loss: {loss.item()}]")
                if step % self.log_every == 0:  # 每几次迭代记录信息
                    log_loss = {}
                    log_loss['train/train_loss'] = loss.item()
                    for name, val in log_loss.items():
                        print("# {:20} = {:.4f}".format(name, val))

                    if self.use_tensorboard:
                        for tag, value in log_loss.items():
                            self.logger.scalar_summary(tag, value, step)
                else:
                    # print("No log output this iteration.")
                    pass

                # save checkpoint
                if step % self.model_save_every == 0:  # 每几次保存一次模型
                    self.model.save(save_dir=self.model_save_dir,
                                    it=self.current_epochs)
                else:
                    # print("No model saved this iteration.")
                    pass

                if step % self.test_every == 0:  # 每几次测试下模型
                    self.test()
                # update learning rates
                self.update_lr(step)
                elapsed = datetime.now() - start_time
                print(f'# 第 {step:04} 次迭代结束, 耗时 {elapsed}')

        self.model.save(save_dir=self.model_save_dir, it=self.current_epochs)
        return self.accuracylog[-20:]

    def update_lr(self, i):
        """衰减学习率方法"""
        if self.n_epochs - self.num_epochs_decay < i:  # 如果总迭代次数-当前迭代次数<更新延迟次数，开始每次迭代都更新学习率
            decay_delta = self.optimizer_lr / self.num_epochs_decay  # 学习率除以延迟次数，缩小学习率
            decay_start = self.n_epochs - self.num_epochs_decay
            decay_iter = i - decay_start  # 当前迭代次数-总迭代次数+更新延迟次数
            lr = self.optimizer_lr - decay_iter * decay_delta
            print(f"# 更新学习率lr:{lr} ")
            for param_group in self.model.optimizer.param_groups:
                param_group['lr'] = lr  # 写入optimizer学习率参数

    def test(self):

        print("# 测试模型准确率。")
        self.model.set_eval_mode()
        emo_preds = torch.rand(0).to(
            device=self.device, dtype=torch.long)  # 预测标签列表
        total_labels = torch.rand(0).to(
            device=self.device, dtype=torch.long)  # 真实标签列表
        total_loss = torch.rand(0).to(
            device=self.device, dtype=torch.float)  # 损失值列表
        loss_func = nn.CrossEntropyLoss(
            weight=self.loss_weights, reduction='mean')

        data_iter = iter(self.test_loader)
        while True:
            try:
                packed_x, genders, labels, lenths = next(data_iter)
            except:
                break

            x, x_lens = torch.nn.utils.rnn.pad_packed_sequence(
                packed_x, batch_first=True)

            x = x.to(device=self.device)
            # labels = self.getNplabels(labels) # 将标签转换为Np标签
            labels = labels.long().to(device=self.device)

            with torch.no_grad():
                y = self.model.net(x)  # 计算模型输出分类值
                out = y
                true_labels = labels
                if lenths is not None and self.resultmerge in ["test", "allmerge"]:
                    out = torch.zeros(
                        (len(lenths), y.shape[1]), device=self.device)
                    true_labels = torch.zeros(
                        len(lenths), device=self.device, dtype=torch.long)
                    begin = 0
                    for i, num in enumerate(lenths):
                        out[i] = y[begin:begin+num].mean(dim=0)
                        true_labels[i] = labels[begin:begin+num][0]
                        begin = begin+num

                emo_pre = torch.max(out, dim=1)[1]  # 获取概率最大值位置
                emo_preds = torch.cat((emo_preds, emo_pre), dim=0)  # 保存所有预测结果
                total_labels = torch.cat(
                    (total_labels, true_labels), dim=0)  # 保存所有真实结果
                emo_loss = torch.tensor(
                    [loss_func(out, true_labels)], device=self.device)
                total_loss = torch.cat((total_loss, emo_loss), dim=0)
        eval_accuracy = accuracy_score(
            total_labels.cpu(), emo_preds.cpu())  # 计算准确率
        eval_loss = total_loss.mean().item()
        l = ["Accuracy_Eval",  "Loss_Eval"]
        print('# {:20} = {:.3f}'.format(l[0], eval_accuracy))
        print('# {:20} = {:.3f}'.format(l[1], eval_loss))

        if self.use_tensorboard:
            self.logger.scalar_summary(
                "Val/Loss_Eval", eval_loss, self.current_epochs)
            self.logger.scalar_summary(
                "Val/Accuracy_Eval", eval_accuracy, self.current_epochs)
        self.accuracylog.append([eval_accuracy])


if __name__ == '__main__':
    pass
