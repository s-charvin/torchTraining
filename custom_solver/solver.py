
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.autograd import Variable

from custom_models import SequenceClassifier, SequenceClassifier_CoTeaching
from SCW.torchTraining.utils.logger import Logger
from transformers import BertTokenizer,  Wav2Vec2Processor

from sklearn.metrics import accuracy_score


class Solver_Classification(object):

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

        self.set_configuration()  # 设置参数

        # 设置要使用的模型
        self.model = SequenceClassifier(self.config)

        self.load_dir = self.config['train']['checkpoint']
        if self.load_dir != "":  # 是否加载历史检查点
            self.load_checkpoint()

    def load_checkpoint(self):
        self.model.load(self.load_dir)
        self.config = self.model.config  # 重点参数有:
        self.set_configuration()  # 设置加载的检查点的参数设置

    def set_configuration(self):
        self.model_name = self.config['model']['name']  # 模型名称
        self.num_iters = self.config['train']['num_iters']  # 迭代次数
        # 更新lr的延迟
        self.num_iters_decay = self.config['train']['num_iters_decay']
        # 恢复迭代的起始步，默认为0
        self.resume_iters = self.config['train']['resume_iters']
        self.current_iter = self.resume_iters  # 当前迭代次数

        self.lr = self.config['optimizer']['lr']  # 学习率
        # self.beta = self.config['optimizer']['beta'] # 学习率

        # 是否使用tensorboard
        self.use_tensorboard = self.config['logs']['use_tensorboard']
        self.log_every = self.config['logs']['log_every']  # 每几步记录
        if 'test_every' in self.config['logs']:  # 每几步测试
            self.test_every = self.config['logs']['test_every']
        # 每几步保存模型
        self.model_save_every = self.config['logs']['model_save_every']
        self.model_save_dir = self.config['logs']['model_save_dir']  # 模型保存地址
        if self.use_tensorboard:
            self.logger = Logger(
                self.config['logs']['log_dir'], self.model_name)

    def set_classification_weights(self, config, labels):
        print("设置分类权重。")
        if config['train']['loss_weight']:
            labels = list(labels)
            categories = config["data"]["indexx"].classes_
            total = len(labels)
            counts = [total/labels.count(emo)
                      for emo in categories]  # 出现次数少的权重大
            self.loss_weights = torch.Tensor(counts).to(self.device)
            print(f"权重: {self.loss_weights.tolist()}")
            print(f"标签: {categories}")
        else:
            self.loss_weights = None

    def train(self):
        """训练主函数"""

        #############################################################
        #                            训练                           #
        #############################################################
        print('################ 开始循环训练 ################')
        start_iter = self.resume_iters + 1  # 训练的新模型则从1开始，恢复的模型则从接续的迭代次数开始

        self.update_lr(start_iter)  # 根据开始迭代次数更新学习率

        data_iter = iter(self.train_loader)  # 构造数据块迭代器
        self.accuracylog = []
        start_time = datetime.now()
        print(f'训练开始时间: {start_time}')  # 读取开始运行时间

        for epoch in range(start_iter, self.num_iters+1):  # 迭代循环
            print(
                f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Iteration {epoch:02}/{self.num_iters:02} ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(
                f"Iteration {epoch:02} lr = {self.model.optimizer.param_groups[0]['lr']:.8f}")

            self.model.to_device(device=self.device)
            self.model.set_train_mode()
            self.current_iter = epoch  # 记录当前迭代次数
            # 计算结果是否需要融合(split之后才需要此操作)
            self.resultmerge = self.config['data']["resultmerge"]

            try:
                packed_x, genders, labels, lenths = next(
                    data_iter)  # 加载没有被split的数据, 即仅有特征和标签
            except StopIteration as ex:
                data_iter = iter(self.train_loader)
                packed_x, genders, labels, lenths = next(
                    data_iter)  # 加载没有被split的数据, 即仅有特征和标签
            x, x_lens = torch.nn.utils.rnn.pad_packed_sequence(
                packed_x, batch_first=True)

            # labels = self.getNplabels(labels) # 将标签转换为Np标签
            x = x.to(device=self.device)
            labels = labels.long().to(device=self.device)

            # 设置损失函数和分类权重，权重数据在s.train()前定义了
            loss_func = nn.CrossEntropyLoss(
                weight=self.loss_weights, reduction='mean')
            self.model.reset_grad()  # 清空梯度

            y = self.model.net(x)
            out, true_labels = self.mergeResult(lenths, labels, y)
            loss = loss_func(out, true_labels)
            loss.backward()
            self.model.optimizer.step()
        #############################################################
        #                            记录                           #
        #############################################################
            if epoch % self.log_every == 0:  # 每几次迭代记录信息
                log_loss = {}
                log_loss['train/train_loss'] = loss.item()
                for name, val in log_loss.items():
                    print("{:20} = {:.4f}".format(name, val))

                if self.use_tensorboard:
                    for tag, value in log_loss.items():
                        self.logger.scalar_summary(tag, value, epoch)
            else:
                # print("No log output this iteration.")
                pass

            # save checkpoint
            if epoch % self.model_save_every == 0:  # 每几次保存一次模型
                self.model.save(save_dir=self.model_save_dir,
                                it=self.current_iter)
            else:
                # print("No model saved this iteration.")
                pass

            if epoch % self.test_every == 0:  # 每几次测试下模型
                self.test()
            # update learning rates
            self.update_lr(epoch)

            elapsed = datetime.now() - start_time
            print(f'第 {epoch:04} 次迭代结束, 耗时 {elapsed}')

        self.model.save(save_dir=self.model_save_dir, it=self.current_iter)
        return self.accuracylog[-20:]

    def update_lr(self, i):
        """衰减学习率方法"""
        if self.num_iters - self.num_iters_decay < i:  # 如果总迭代次数-当前迭代次数<更新延迟次数，开始每次迭代都更新学习率
            decay_delta = self.lr / self.num_iters_decay  # 学习率除以延迟次数，缩小学习率
            decay_start = self.num_iters - self.num_iters_decay
            decay_iter = i - decay_start  # 当前迭代次数-总迭代次数+更新延迟次数
            lr = self.lr - decay_iter * decay_delta
            print(f"更新学习率lr:{lr} ")
            for param_group in self.model.optimizer.param_groups:
                param_group['lr'] = lr  # 写入optimizer学习率参数

    def test(self):

        print("测试模型准确率。")
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
        print('{:20} = {:.3f}'.format(l[0], eval_accuracy))
        print('{:20} = {:.3f}'.format(l[1], eval_loss))

        if self.use_tensorboard:
            self.logger.scalar_summary(
                "Val/Loss_Eval", eval_loss, self.current_iter)
            self.logger.scalar_summary(
                "Val/Accuracy_Eval", eval_accuracy, self.current_iter)
        self.accuracylog.append([eval_accuracy])

    def mergeResult(self, lenths, labels, y):
        out = y
        true_labels = labels
        # 是否融合结果
        if lenths is not None and self.resultmerge in ["train", "allmerge"]:
            out = y[0:lenths[0]].mean(dim=0).unsqueeze(0)
            true_labels = labels[0:lenths[0]][0].unsqueeze(0)
            begin = lenths[0]
            for index, num in enumerate(lenths):
                if index != 0:
                    out = torch.cat(
                        (out, y[begin:begin+num].mean(dim=0).unsqueeze(0)), dim=0)
                    true_labels = torch.cat(
                        (true_labels, labels[begin:begin+num][0].unsqueeze(0)), dim=0)
                    begin = begin+num
        return out, true_labels

    def getNplabels(self, labels):
        Nplabels = labels
        positivetag = ["excited", "happy", "surprised", "surprised pleased"]
        negativetag = ["angry", "bored", "disgusted",
                       "fearful", "frustrated", "sad"]
        neutraltag = ["calm", "neutral"]

        indexx = self.config["data"]["indexx"]
        # classes = {v: k for k,
        #            v in self.config["data"]["indexx"].classes_.items()}
        for index, label in enumerate(labels):
            if indexx.inverse_transform([label.item()])[0] in positivetag:
                Nplabels[index] = self.config["data"]["Npclasses"]["positive"]
            elif indexx.inverse_transform([label.item()])[0] in negativetag:
                Nplabels[index] = self.config["data"]["Npclasses"]["negative"]
            elif indexx.inverse_transform([label.item()])[0] in neutraltag:
                Nplabels[index] = self.config["data"]["Npclasses"]["neutral"]
        return Nplabels


class Solver_Classification_CoTeaching(object):

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

        self.set_configuration()  # 设置参数

        # 设置要使用的模型
        self.model = SequenceClassifier_CoTeaching(self.config)

        self.load_dir = self.config['train']['checkpoint']
        if self.load_dir != "":  # 是否加载历史检查点
            self.load_checkpoint()

    def load_checkpoint(self):
        self.model.load(self.load_dir)
        self.config = self.model.config  # 重点参数有:
        self.set_configuration()  # 设置加载的检查点的参数设置

    def set_configuration(self):
        self.model_name = self.config['model']['name']  # 模型名称
        self.num_iters = self.config['train']['num_iters']  # 迭代次数
        # 更新lr的延迟
        self.num_iters_decay = self.config['train']['num_iters_decay']
        # 恢复迭代的起始步，默认为0
        self.resume_iters = self.config['train']['resume_iters']
        self.current_iter = self.resume_iters  # 当前迭代次数

        self.lr = self.config['optimizer']['lr']  # 学习率
        # self.beta = self.config['optimizer']['beta'] # 学习率

        # 是否使用tensorboard
        self.use_tensorboard = self.config['logs']['use_tensorboard']
        self.log_every = self.config['logs']['log_every']  # 每几步记录
        if 'test_every' in self.config['logs']:  # 每几步测试
            self.test_every = self.config['logs']['test_every']
        # 每几步保存模型
        self.model_save_every = self.config['logs']['model_save_every']
        self.model_save_dir = self.config['logs']['model_save_dir']  # 模型保存地址
        if self.use_tensorboard:
            self.logger = Logger(
                self.config['logs']['log_dir'], self.model_name)

    def set_classification_weights(self, config, labels):
        print("设置分类权重。")
        if config['train']['loss_weight']:
            labels = list(labels)
            categories = config["data"]["indexx"].classes_
            total = len(labels)
            counts = [total/labels.count(emo)
                      for emo in categories]  # 出现次数少的权重大
            self.loss_weights = torch.Tensor(counts).to(self.device)
            print(f"权重: {self.loss_weights.tolist()}")
            print(f"标签: {categories}")
        else:
            self.loss_weights = None

    def train(self):
        """训练主函数"""

        #############################################################
        #                            训练                           #
        #############################################################
        print('################ 开始循环训练 ################')
        start_iter = self.resume_iters + 1  # 训练的新模型则从1开始，恢复的模型则从接续的迭代次数开始

        self.update_lr(start_iter)  # 根据开始迭代次数更新学习率
        rate_schedule = np.ones(self.num_iters)*0.2
        rate_schedule[:int(self.num_iters/10)] = np.linspace(0,
                                                             0.2**1, int(self.num_iters/10))
        rate_schedule[-int(self.num_iters/5):] = np.linspace(0.2**1,
                                                             0, int(self.num_iters/5))
        data_iter = iter(self.train_loader)  # 构造数据块迭代器
        self.accuracylog = []
        start_time = datetime.now()
        print(f'训练开始时间: {start_time}')  # 读取开始运行时间

        for epoch in range(start_iter, self.num_iters+1):  # 迭代循环
            print(
                f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Iteration {epoch:02}/{self.num_iters:02} ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(
                f"Iteration {epoch:02} lr1 = {self.model.optimizer1.param_groups[0]['lr']:.8f}, lr2 = {self.model.optimizer2.param_groups[0]['lr']:.8f}")

            self.model.to_device(device=self.device)
            self.model.set_train_mode()
            self.current_iter = epoch  # 记录当前迭代次数
            # 计算结果是否需要融合(split之后才需要此操作)
            self.resultmerge = self.config['data']["resultmerge"]

            try:
                packed_x, genders, labels, lenths = next(
                    data_iter)  # 加载没有被split的数据, 即仅有特征和标签
            except StopIteration as ex:
                data_iter = iter(self.train_loader)
                packed_x, genders, labels, lenths = next(
                    data_iter)  # 加载没有被split的数据, 即仅有特征和标签
            x, x_lens = torch.nn.utils.rnn.pad_packed_sequence(
                packed_x, batch_first=True)

            # labels = self.getNplabels(labels) # 将标签转换为Np标签
            x = x.to(device=self.device)
            labels = labels.long().to(device=self.device)

            if self.config["data"]["Nplabel"]:
                labels = self.getNplabels(labels)  # 将标签转换为Np标签

            lenths = lenths.to(device=self.device)
            x = x.to(device=self.device)
            labels = labels.long().to(device=self.device)

            self.model.reset_grad()  # 清空梯度

            y1 = self.model.net1(x)
            # 是否融合结果
            out1, true_labels1 = self.mergeResult(lenths, labels, y1)

            y2 = self.model.net2(x.detach())
            # 是否融合结果
            out2, true_labels2 = self.mergeResult(lenths, labels, y2)

            loss_1, loss_2 = self.loss_coteaching(
                out1, out2, true_labels1, rate_schedule[epoch-1])

            loss_1.backward()
            loss_2.backward()
            self.model.optimizer1.step()
            self.model.optimizer2.step()
        #############################################################
        #                            记录                           #
        #############################################################
            if epoch % self.log_every == 0:  # 每几次迭代记录信息
                log_loss = {}
                log_loss['train/train_loss1'] = loss_1.item()
                log_loss['train/train_loss2'] = loss_2.item()
                for name, val in log_loss.items():
                    print("{:20} = {:.4f}".format(name, val))

                if self.use_tensorboard:
                    for tag, value in log_loss.items():
                        self.logger.scalar_summary(tag, value, epoch)
            else:
                # print("No log output this iteration.")
                pass

            # save checkpoint
            if epoch % self.model_save_every == 0:  # 每几次保存一次模型
                self.model.save(save_dir=self.model_save_dir,
                                it=self.current_iter)
            else:
                # print("No model saved this iteration.")
                pass

            if epoch % self.test_every == 0:  # 每几次测试下模型
                self.test()
            # update learning rates
            self.update_lr(epoch)

            elapsed = datetime.now() - start_time
            print(f'第 {epoch:04} 次迭代结束, 耗时 {elapsed}')

        self.model.save(save_dir=self.model_save_dir, it=self.current_iter)
        return self.accuracylog[-20:]

    def update_lr(self, i):
        """衰减学习率方法"""
        if self.num_iters - self.num_iters_decay < i:  # 如果总迭代次数-当前迭代次数<更新延迟次数，开始每次迭代都更新学习率
            decay_delta = self.lr / self.num_iters_decay  # 学习率除以延迟次数，缩小学习率
            decay_start = self.num_iters - self.num_iters_decay
            decay_iter = i - decay_start  # 当前迭代次数-总迭代次数+更新延迟次数
            lr = self.lr - decay_iter * decay_delta
            print(f"更新学习率lr:{lr} ")
            for param_group in self.model.optimizer1.param_groups:
                param_group['lr'] = lr  # 写入optimizer学习率参数
            for param_group in self.model.optimizer2.param_groups:
                param_group['lr'] = lr  # 写入optimizer学习率参数

    def test(self):

        print("测试模型准确率。")
        self.model.set_eval_mode()
        emo_preds1 = torch.rand(0).to(
            device=self.device, dtype=torch.long)  # 预测标签列表1
        emo_preds2 = torch.rand(0).to(
            device=self.device, dtype=torch.long)  # 预测标签列表2

        total_labels1 = torch.rand(0).to(
            device=self.device, dtype=torch.long)  # 真实标签列表1

        total_labels2 = torch.rand(0).to(
            device=self.device, dtype=torch.long)  # 真实标签列表2

        total_loss1 = torch.rand(0).to(
            device=self.device, dtype=torch.float)  # 损失值列表1
        total_loss2 = torch.rand(0).to(
            device=self.device, dtype=torch.float)  # 损失值列表2

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

            x = x.to(device=self.device)
            if self.config["data"]["Nplabel"]:
                labels = self.getNplabels(labels)  # 将标签转换为Np标签
            labels = labels.long().to(device=self.device)

            with torch.no_grad():
                y1 = self.model.net1(x)  # 计算模型输出分类值
                y2 = self.model.net2(x)
                out1, true_labels1 = self.mergeResult(lenths, labels, y1)
                out2, true_labels2 = self.mergeResult(lenths, labels, y2)

                emo_pre1 = torch.max(out1, dim=1)[1]  # 获取概率最大值位
                emo_preds1 = torch.cat(
                    (emo_preds1, emo_pre1), dim=0)  # 保存所有预测结果
                total_labels1 = torch.cat(
                    (total_labels1, true_labels1), dim=0)  # 保存所有真实结果
                emo_loss1 = torch.tensor(
                    [loss_func(out1, true_labels1)], device=self.device)
                total_loss1 = torch.cat((total_loss1, emo_loss1), dim=0)

                emo_pre2 = torch.max(out2, dim=1)[1]  # 获取概率最大值位
                emo_preds2 = torch.cat(
                    (emo_preds2, emo_pre2), dim=0)  # 保存所有预测结果
                total_labels2 = torch.cat(
                    (total_labels2, true_labels2), dim=0)  # 保存所有真实结果
                emo_loss2 = torch.tensor(
                    [loss_func(out2, true_labels2)], device=self.device)
                total_loss2 = torch.cat((total_loss2, emo_loss2), dim=0)
        eval_accuracy1 = accuracy_score(
            total_labels1.cpu(), emo_preds1.cpu())  # 计算准确率
        eval_loss1 = total_loss1.mean().item()

        eval_accuracy2 = accuracy_score(
            total_labels2.cpu(), emo_preds2.cpu())  # 计算准确率
        eval_loss2 = total_loss2.mean().item()

        l = ["Accuracy_Eval",  "Loss_Eval"]
        print('{:20} = {:.3f}'.format(l[0], eval_accuracy1))
        print('{:20} = {:.3f}'.format(l[1], eval_loss1))

        print('{:20} = {:.3f}'.format(l[0], eval_accuracy2))
        print('{:20} = {:.3f}'.format(l[1], eval_loss2))

        if self.use_tensorboard:
            self.logger.scalar_summary(
                "Val/Loss1_Eval", eval_loss1, self.current_iter)
            self.logger.scalar_summary(
                "Val/Accuracy1_Eval", eval_accuracy1, self.current_iter)
            self.logger.scalar_summary(
                "Val/Loss2_Eval", eval_loss2, self.current_iter)
            self.logger.scalar_summary(
                "Val/Accuracy2_Eval", eval_accuracy2, self.current_iter)
        self.accuracylog.append([eval_accuracy1, eval_accuracy2])

    def mergeResult(self, lenths, labels, y):
        out = y
        true_labels = labels
        # 是否融合结果
        if lenths is not None and self.resultmerge in ["train", "allmerge"]:
            out = y[0:lenths[0]].mean(dim=0).unsqueeze(0)
            true_labels = labels[0:lenths[0]][0].unsqueeze(0)
            begin = lenths[0]
            for index, num in enumerate(lenths):
                if index != 0:
                    out = torch.cat(
                        (out, y[begin:begin+num].mean(dim=0).unsqueeze(0)), dim=0)
                    true_labels = torch.cat(
                        (true_labels, labels[begin:begin+num][0].unsqueeze(0)), dim=0)
                    begin = begin+num
        return out, true_labels

    def getNplabels(self, labels):
        Nplabels = labels
        positivetag = ["excited", "happy", "surprised", "surprised pleased"]
        negativetag = ["angry", "bored", "disgusted",
                       "fearful", "frustrated", "sad"]
        neutraltag = ["calm", "neutral"]

        indexx = self.config["data"]["indexx"]
        # classes = {v: k for k,
        #            v in self.config["data"]["indexx"].classes_.items()}
        for index, label in enumerate(labels):
            if indexx.inverse_transform([label.item()])[0] in positivetag:
                Nplabels[index] = self.config["data"]["Npclasses"]["positive"]
            elif indexx.inverse_transform([label.item()])[0] in negativetag:
                Nplabels[index] = self.config["data"]["Npclasses"]["negative"]
            elif indexx.inverse_transform([label.item()])[0] in neutraltag:
                Nplabels[index] = self.config["data"]["Npclasses"]["neutral"]
        return Nplabels

    def loss_coteaching(self, logits1, logits2, labels, forget_rate):
        # 计算此batch内数据的所有损失
        # 设置损失函数和分类权重，权重数据在s.train()前定义了
        loss_func = nn.CrossEntropyLoss(
            weight=self.loss_weights, reduction='none')
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


class Solver_Classification_RandomLearning(object):

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

        self.set_configuration()  # 设置参数

        # 设置要使用的模型
        self.model = SequenceClassifier(self.config)

        self.load_dir = self.config['train']['checkpoint']
        if self.load_dir != "":  # 是否加载历史检查点
            self.load_checkpoint()

    def load_checkpoint(self):
        self.model.load(self.load_dir)
        self.config = self.model.config  # 重点参数有:
        self.set_configuration()  # 设置加载的检查点的参数设置

    def set_configuration(self):
        self.model_name = self.config['model']['name']  # 模型名称
        self.num_iters = self.config['train']['num_iters']  # 迭代次数
        # 更新lr的延迟
        self.num_iters_decay = self.config['train']['num_iters_decay']
        # 恢复迭代的起始步，默认为0
        self.resume_iters = self.config['train']['resume_iters']
        self.current_iter = self.resume_iters  # 当前迭代次数

        self.lr = self.config['optimizer']['lr']  # 学习率
        # self.beta = self.config['optimizer']['beta'] # 学习率

        # 是否使用tensorboard
        self.use_tensorboard = self.config['logs']['use_tensorboard']
        self.log_every = self.config['logs']['log_every']  # 每几步记录
        if 'test_every' in self.config['logs']:  # 每几步测试
            self.test_every = self.config['logs']['test_every']
        # 每几步保存模型
        self.model_save_every = self.config['logs']['model_save_every']
        self.model_save_dir = self.config['logs']['model_save_dir']  # 模型保存地址
        if self.use_tensorboard:
            self.logger = Logger(
                self.config['logs']['log_dir'], self.model_name)

    def set_classification_weights(self, config, labels):
        print("设置分类权重。")
        if config['train']['loss_weight']:
            labels = list(labels)
            categories = config["data"]["indexx"].classes_
            total = len(labels)
            counts = [total/labels.count(emo)
                      for emo in categories]  # 出现次数少的权重大
            self.loss_weights = torch.Tensor(counts).to(self.device)
            print(f"权重: {self.loss_weights.tolist()}")
            print(f"标签: {categories}")
        else:
            self.loss_weights = None

    def train(self):
        """训练主函数"""

        #############################################################
        #                            训练                           #
        #############################################################
        print('################ 开始循环训练 ################')
        start_iter = self.resume_iters + 1  # 训练的新模型则从1开始，恢复的模型则从接续的迭代次数开始

        self.update_lr(start_iter)  # 根据开始迭代次数更新学习率

        good_rate = np.ones(self.num_iters)*0.8
        good_rate[:int(self.num_iters/5)] = np.linspace(1,
                                                        0.8**1, int(self.num_iters/5))
        good_rate[-int(self.num_iters/5):] = np.linspace(0.8**1,
                                                         1, int(self.num_iters/5))

        bad_rate = np.ones(self.num_iters)*0.4
        bad_rate[:int(self.num_iters/5)] = np.linspace(1,
                                                       0.4**1, int(self.num_iters/5))
        bad_rate[-int(self.num_iters/5):] = np.linspace(0.4**1,
                                                        1, int(self.num_iters/5))

        data_iter = iter(self.train_loader)  # 构造数据块迭代器
        self.accuracylog = []
        start_time = datetime.now()
        print(f'训练开始时间: {start_time}')  # 读取开始运行时间

        for epoch in range(start_iter, self.num_iters+1):  # 迭代循环
            print(
                f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Iteration {epoch:02}/{self.num_iters:02} ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(
                f"Iteration {epoch:02} lr = {self.model.optimizer.param_groups[0]['lr']:.8f}")

            self.model.to_device(device=self.device)
            self.model.set_train_mode()
            self.current_iter = epoch  # 记录当前迭代次数
            # 计算结果是否需要融合(split之后才需要此操作)
            self.resultmerge = self.config['data']["resultmerge"]

            try:
                packed_x, genders, labels, lenths = next(
                    data_iter)  # 加载没有被split的数据, 即仅有特征和标签
            except StopIteration as ex:
                data_iter = iter(self.train_loader)
                packed_x, genders, labels, lenths = next(
                    data_iter)  # 加载没有被split的数据, 即仅有特征和标签
            x, x_lens = torch.nn.utils.rnn.pad_packed_sequence(
                packed_x, batch_first=True)

            x = x.to(device=self.device)
            labels = labels.long().to(device=self.device)

            self.model.reset_grad()  # 清空梯度

            y = self.model.net(x)
            out, true_labels = self.mergeResult(lenths, labels, y)

            loss = self.loss_randomlearning(
                out, true_labels, bad_rate[epoch-1], good_rate[epoch-1])

            loss.backward()
            self.model.optimizer.step()
        #############################################################
        #                            记录                           #
        #############################################################
            if epoch % self.log_every == 0:  # 每几次迭代记录信息
                log_loss = {}
                log_loss['train/train_loss'] = loss.item()
                for name, val in log_loss.items():
                    print("{:20} = {:.4f}".format(name, val))

                if self.use_tensorboard:
                    for tag, value in log_loss.items():
                        self.logger.scalar_summary(tag, value, epoch)
            else:
                # print("No log output this iteration.")
                pass

            # save checkpoint
            if epoch % self.model_save_every == 0:  # 每几次保存一次模型
                self.model.save(save_dir=self.model_save_dir,
                                it=self.current_iter)
            else:
                # print("No model saved this iteration.")
                pass

            if epoch % self.test_every == 0:  # 每几次测试下模型
                self.test()
            # update learning rates
            self.update_lr(epoch)

            elapsed = datetime.now() - start_time
            print(f'第 {epoch:04} 次迭代结束, 耗时 {elapsed}')

        self.model.save(save_dir=self.model_save_dir, it=self.current_iter)
        return self.accuracylog[-20:]

    def update_lr(self, i):
        """衰减学习率方法"""
        if self.num_iters - self.num_iters_decay < i:  # 如果总迭代次数-当前迭代次数<更新延迟次数，开始每次迭代都更新学习率
            decay_delta = self.lr / self.num_iters_decay  # 学习率除以延迟次数，缩小学习率
            decay_start = self.num_iters - self.num_iters_decay
            decay_iter = i - decay_start  # 当前迭代次数-总迭代次数+更新延迟次数
            lr = self.lr - decay_iter * decay_delta
            print(f"更新学习率lr:{lr} ")
            for param_group in self.model.optimizer.param_groups:
                param_group['lr'] = lr  # 写入optimizer学习率参数

    def test(self):

        print("测试模型准确率。")
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
        print('{:20} = {:.3f}'.format(l[0], eval_accuracy))
        print('{:20} = {:.3f}'.format(l[1], eval_loss))

        if self.use_tensorboard:
            self.logger.scalar_summary(
                "Val/Loss_Eval", eval_loss, self.current_iter)
            self.logger.scalar_summary(
                "Val/Accuracy_Eval", eval_accuracy, self.current_iter)
        self.accuracylog.append([eval_accuracy])

    def mergeResult(self, lenths, labels, y):
        out = y
        true_labels = labels
        # 是否融合结果
        if lenths is not None and self.resultmerge in ["train", "allmerge"]:
            out = y[0:lenths[0]].mean(dim=0).unsqueeze(0)
            true_labels = labels[0:lenths[0]][0].unsqueeze(0)
            begin = lenths[0]
            for index, num in enumerate(lenths):
                if index != 0:
                    out = torch.cat(
                        (out, y[begin:begin+num].mean(dim=0).unsqueeze(0)), dim=0)
                    true_labels = torch.cat(
                        (true_labels, labels[begin:begin+num][0].unsqueeze(0)), dim=0)
                    begin = begin+num
        return out, true_labels

    def loss_randomlearning(self, logits, labels, bad_rate, good_rate):
        # 计算此batch内数据的所有损失
        # 设置损失函数和分类权重，权重数据在s.train()前定义了
        loss_func = nn.CrossEntropyLoss(
            weight=self.loss_weights, reduction='none')
        device_ = logits.device
        loss = loss_func(logits, labels)

        ind_sorted = np.argsort(loss.data.cpu())  # 位置

        # 前 80% 的 Loss 低的数据数量
        num_good = int(len(ind_sorted)*0.8)
        # 后 20% 的 Loss 高的数据数量
        num_bad = len(ind_sorted)-int(len(ind_sorted)*0.8)

        # 参与训练的好数据和坏数据
        ind_update = np.concatenate([
            np.random.choice(
                ind_sorted[:num_good], int(num_good*good_rate), replace=False),
            np.random.choice(
                ind_sorted[-num_bad:], int(num_bad*bad_rate), replace=False)
        ], axis=0)

        # 计算筛选后的数据Loss, 此处通过 index 进行了数据位置交换
        loss_func_mean = nn.CrossEntropyLoss(
            weight=self.loss_weights, reduction='mean')

        loss_update = loss_func_mean(
            logits[ind_update], labels[ind_update])

        return loss_update.to(device=device_)


class Solver_Classification_Clip(object):

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

        self.set_configuration()  # 设置参数

        # 设置要使用的模型
        self.model = SequenceClassifier(self.config)
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True)
        self.processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-base-960h")
        self.load_dir = self.config['train']['checkpoint']
        if self.load_dir != "":  # 是否加载历史检查点
            self.load_checkpoint()

    def load_checkpoint(self):
        self.model.load(self.load_dir)
        self.config = self.model.config  # 重点参数有:
        self.set_configuration()  # 设置加载的检查点的参数设置

    def set_configuration(self):
        self.model_name = self.config['model']['name']  # 模型名称
        self.num_iters = self.config['train']['num_iters']  # 迭代次数
        # 更新lr的延迟
        self.num_iters_decay = self.config['train']['num_iters_decay']
        # 恢复迭代的起始步，默认为0
        self.resume_iters = self.config['train']['resume_iters']
        self.current_iter = self.resume_iters  # 当前迭代次数

        self.lr = self.config['optimizer']['lr']  # 学习率
        # self.beta = self.config['optimizer']['beta'] # 学习率

        # 是否使用tensorboard
        self.use_tensorboard = self.config['logs']['use_tensorboard']
        self.log_every = self.config['logs']['log_every']  # 每几步记录
        if 'test_every' in self.config['logs']:  # 每几步测试
            self.test_every = self.config['logs']['test_every']
        # 每几步保存模型
        self.model_save_every = self.config['logs']['model_save_every']
        self.model_save_dir = self.config['logs']['model_save_dir']  # 模型保存地址
        if self.use_tensorboard:
            self.logger = Logger(
                self.config['logs']['log_dir'], self.model_name)

    def set_classification_weights(self, config, labels):
        print("设置分类权重。")
        if config['train']['loss_weight']:
            labels = list(labels)
            categories = config["data"]["indexx"].classes_
            total = len(labels)
            counts = [total/labels.count(emo)
                      for emo in categories]  # 出现次数少的权重大
            self.loss_weights = torch.Tensor(counts).to(self.device)
            print(f"权重: {self.loss_weights.tolist()}")
            print(f"标签: {categories}")
        else:
            self.loss_weights = None

    def processorwav(self, wavs):
        for i, data in enumerate(wavs):
            wavs[i] = self.processor(data)
        return wavs

    def batch_encode_plus(self, texts):
        encoded_texts = self.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=texts,   # 输入文本 batch
            # is_split_into_words=True,
            add_special_tokens=True,  # 添加 '[CLS]' 和 '[SEP]'
            # max_length=128,           # 填充 & 截断长度
            padding='longest',  # 设置填充到本 batch 中的最长长度
            return_attention_mask=True,   # 返回 attn. masks.
            return_tensors='pt',     # 返回 pytorch tensors 格式的数据
        )
        return encoded_texts['input_ids'], encoded_texts["attention_mask"]

    def train(self):
        """训练主函数"""

        #############################################################
        #                            训练                           #
        #############################################################
        print('################ 开始循环训练 ################')
        start_iter = self.resume_iters + 1  # 训练的新模型则从1开始，恢复的模型则从接续的迭代次数开始

        self.update_lr(start_iter)  # 根据开始迭代次数更新学习率

        # coteach 的学习比率
        rate_schedule = np.ones(self.num_iters)*0.1
        rate_schedule[:int(self.num_iters/10)] = np.linspace(0,
                                                             0.1**1, int(self.num_iters/10))
        rate_schedule[-int(self.num_iters/5):] = np.linspace(0.1**1,
                                                             0, int(self.num_iters/5))
        data_iter = iter(self.train_loader)  # 构造数据块迭代器
        self.accuracylog = []
        start_time = datetime.now()
        print(f'训练开始时间: {start_time}')  # 读取开始运行时间

        for epoch in range(start_iter, self.num_iters+1):  # 迭代循环
            print(
                f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Iteration {epoch:02}/{self.num_iters:02} ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(
                f"Iteration {epoch:02} lr = {self.model.optimizer.param_groups[0]['lr']:.8f}")

            self.model.to_device(device=self.device)
            self.model.set_train_mode()
            self.current_iter = epoch  # 记录当前迭代次数

            try:
                packed_x, texts, genders, labels, lenths = next(
                    data_iter)  # 加载没有被split的数据, 即仅有特征和标签
            except StopIteration as ex:
                data_iter = iter(self.train_loader)
                packed_x, texts, genders, labels, lenths = next(
                    data_iter)  # 加载没有被split的数据, 即仅有特征和标签
            input_values, x_lens = torch.nn.utils.rnn.pad_packed_sequence(
                packed_x, batch_first=True)

            # targets = ["This female felt very "+i +
            #            "." for i in self.config["data"]["indexx"].classes_]

            targets = ["This man felt very "+i +
                       "." for i in self.config["data"]["indexx"].classes_]+["This woman felt very "+i +
                                                                             "." for i in self.config["data"]["indexx"].classes_]

            for i, l in enumerate(labels):
                if genders[i] == "M":
                    pass
                elif genders[i] == "F":
                    labels[i] = labels[i]+4

            target_text_ids, target_text_mask = self.batch_encode_plus(
                targets)
            text_ids, text_mask = self.batch_encode_plus(texts)
            input_values = input_values.unsqueeze(
                1).to(device=self.device).detach()  # .detach()
            text_ids = text_ids.to(device=self.device).detach()
            text_mask = text_mask.to(device=self.device).detach()
            target_text_mask = target_text_mask.to(device=self.device).detach()
            target_text_ids = target_text_ids.to(device=self.device).detach()
            labels = labels.to(device=self.device).detach()
            # 设置损失函数和分类权重，权重数据在s.train()前定义了
            loss_func = nn.CrossEntropyLoss(weight=None, reduction='mean')
            self.model.reset_grad()  # 清空梯度

            logits_per_speech, logits_per_text = self.model.net(speechs=input_values,
                                                                text_ids=text_ids, text_mask=text_mask,
                                                                target_text_ids=target_text_ids, target_text_mask=target_text_mask)

            loss_sl = loss_func(logits_per_speech, labels)
            loss_tl = loss_func(logits_per_text, labels)
            # loss_sl, loss_tl = self.loss_coteaching(
            #     logits_per_speech, logits_per_text, labels, forget_rate=rate_schedule[epoch-1])
            loss = (loss_sl + loss_tl)/2
            loss.backward()
            self.model.optimizer.step()
        #############################################################
        #                            记录                           #
        #############################################################
            if epoch % self.log_every == 0:  # 每几次迭代记录信息
                log_loss = {}
                log_loss['train/train_loss'] = loss.item()
                log_loss['train/train_loss_sl'] = loss_sl.item()
                log_loss['train/train_loss_tl'] = loss_tl.item()
                for name, val in log_loss.items():
                    print("{:20} = {:.4f}".format(name, val))

                if self.use_tensorboard:
                    for tag, value in log_loss.items():
                        self.logger.scalar_summary(tag, value, epoch)
            else:
                # print("No log output this iteration.")
                pass

            # save checkpoint
            if epoch % self.model_save_every == 0:  # 每几次保存一次模型
                self.model.save(save_dir=self.model_save_dir,
                                it=self.current_iter)
            else:
                # print("No model saved this iteration.")
                pass

            if epoch % self.test_every == 0:  # 每几次测试下模型
                self.test()
            # update learning rates
            self.update_lr(epoch)

            elapsed = datetime.now() - start_time
            print(f'第 {epoch:04} 次迭代结束, 耗时 {elapsed}')

        self.model.save(save_dir=self.model_save_dir, it=self.current_iter)
        return self.accuracylog[-20:]

    def update_lr(self, i):
        """衰减学习率方法"""
        if self.num_iters - self.num_iters_decay < i:  # 如果总迭代次数-当前迭代次数<更新延迟次数，开始每次迭代都更新学习率
            decay_delta = self.lr / self.num_iters_decay  # 学习率除以延迟次数，缩小学习率
            decay_start = self.num_iters - self.num_iters_decay
            decay_iter = i - decay_start  # 当前迭代次数-总迭代次数+更新延迟次数
            lr = self.lr - decay_iter * decay_delta
            print(f"更新学习率lr:{lr} ")
            for param_group in self.model.optimizer.param_groups:
                param_group['lr'] = lr  # 写入optimizer学习率参数

    def test(self):
        print("测试模型准确率。")
        self.model.set_eval_mode()
        emo_preds_sl = torch.rand(0).to(
            device=self.device, dtype=torch.long)  # 预测标签列表
        emo_preds_tl = torch.rand(0).to(
            device=self.device, dtype=torch.long)  # 预测标签列表
        total_labels = torch.rand(0).to(
            device=self.device, dtype=torch.long)  # 真实标签列表
        total_loss_sl = torch.rand(0).to(
            device=self.device, dtype=torch.float)  # 损失值列表
        total_loss_tl = torch.rand(0).to(
            device=self.device, dtype=torch.float)  # 损失值列表
        loss_func = nn.CrossEntropyLoss(reduction='mean')

        data_iter = iter(self.test_loader)
        while True:
            try:
                packed_x, texts, genders, labels, lenths = next(data_iter)
            except:
                break

            input_values, x_lens = torch.nn.utils.rnn.pad_packed_sequence(
                packed_x, batch_first=True)

            targets = ["The man felt very "+i +
                       "." for i in self.config["data"]["indexx"].classes_]
            target_text_ids, target_text_mask = self.batch_encode_plus(
                targets)
            text_ids, text_mask = self.batch_encode_plus(texts)
            input_values = input_values.unsqueeze(
                1).to(device=self.device)
            labels = labels.to(device=self.device)
            text_ids = text_ids.to(device=self.device)
            text_mask = text_mask.to(device=self.device)
            target_text_mask = target_text_mask.to(device=self.device)
            target_text_ids = target_text_ids.to(device=self.device)

            with torch.no_grad():

                logits_per_speech, logits_per_text = self.model.net(speechs=input_values,
                                                                    text_ids=text_ids, text_mask=text_mask,
                                                                    target_text_ids=target_text_ids, target_text_mask=target_text_mask)  # 计算模型输出分类值
                probs_sl = logits_per_speech.softmax(dim=-1)
                probs_tl = logits_per_text.softmax(dim=-1)
                #########################
                true_labels = labels

                # 获取 spech to target 概率最大值位置
                emo_pre_sl = torch.max(probs_sl, dim=1)[1]
                # 获取 text to target 概率最大值位置
                emo_pre_tl = torch.max(probs_tl, dim=1)[1]

                # 保存 spech to target 预测结果
                emo_preds_sl = torch.cat(
                    (emo_preds_sl, emo_pre_sl), dim=0)
                # 保存 text to target 预测结果
                emo_preds_tl = torch.cat((emo_preds_tl, emo_pre_tl), dim=0)

                # 保存所有真实结果
                total_labels = torch.cat(
                    (total_labels, true_labels), dim=0)

                # 保存 spech to target 测试损失
                emo_loss_sl = torch.tensor(
                    [loss_func(logits_per_speech, true_labels)], device=self.device)
                total_loss_sl = torch.cat(
                    (total_loss_sl, emo_loss_sl), dim=0)
                # 保存 text to target 测试损失
                emo_loss_tl = torch.tensor(
                    [loss_func(logits_per_text, true_labels)], device=self.device)
                total_loss_tl = torch.cat(
                    (total_loss_tl, emo_loss_tl), dim=0)

        eval_accuracy_sl = accuracy_score(
            total_labels.cpu(), emo_preds_sl.cpu())  # 计算准确率
        eval_accuracy_tl = accuracy_score(
            total_labels.cpu(), emo_preds_tl.cpu())  # 计算准确率
        eval_loss_sl = total_loss_sl.mean().item()
        eval_loss_tl = total_loss_tl.mean().item()
        l = ["Sl_Accuracy_Eval",  "Tl_Accuracy_Eval",
             "Sl_Loss_Eval", "Tl_Loss_Eval", ]
        print('{:20} = {:.3f}'.format(l[0], eval_accuracy_sl))
        print('{:20} = {:.3f}'.format(l[1], eval_accuracy_tl))
        print('{:20} = {:.3f}'.format(l[2], eval_loss_sl))
        print('{:20} = {:.3f}'.format(l[3], eval_loss_tl))

        if self.use_tensorboard:
            self.logger.scalar_summary(
                "Val/Sl_Accuracy_Eval", eval_accuracy_sl, self.current_iter)
            self.logger.scalar_summary(
                "Val/Tl_Accuracy_Eval", eval_accuracy_tl, self.current_iter)

            self.logger.scalar_summary(
                "Val/Sl_Loss_Eval", eval_loss_sl, self.current_iter)
            self.logger.scalar_summary(
                "Val/Tl_Loss_Eval", eval_loss_tl, self.current_iter)
            self.logger.scalar_summary(
                "Val/Total_Loss_Eval", (eval_loss_sl+eval_loss_tl)/2, self.current_iter)
        self.accuracylog.append(
            [eval_accuracy_sl, eval_accuracy_tl, (eval_loss_sl+eval_loss_tl)/2])

    def loss_coteaching(self, logits1, logits2, labels, forget_rate):
        # 计算此 batch 内数据的所有损失
        # 设置损失函数和分类权重，权重数据在s.train()前定义了
        loss_func = nn.CrossEntropyLoss(weight=None, reduction='none')

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
        # 计算筛选后的数据Loss, 此处通过 index 进行了数据位置交换
        loss_func_mean = nn.CrossEntropyLoss(weight=None, reduction='mean')

        loss_1_update = loss_func_mean(
            logits1[ind_2_update], labels[ind_2_update])
        loss_2_update = loss_func_mean(
            logits2[ind_1_update], labels[ind_1_update])
        # loss_1_update = torch.sum(loss_1_update)/num_remember
        # loss_1_update = torch.sum(loss_2_update)/num_remember
        return loss_1_update.to(device=device_), loss_2_update.to(device=device_)


class Solver_Classification_Clip_DSM(object):

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

        self.set_configuration()  # 设置参数

        # 设置要使用的模型
        self.model = SequenceClassifier(self.config)
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True)
        self.load_dir = self.config['train']['checkpoint']
        if self.load_dir != "":  # 是否加载历史检查点
            self.load_checkpoint()

    def load_checkpoint(self):
        self.model.load(self.load_dir)
        self.config = self.model.config  # 重点参数有:
        self.set_configuration()  # 设置加载的检查点的参数设置

    def set_configuration(self):
        self.model_name = self.config['model']['name']  # 模型名称
        self.num_iters = self.config['train']['num_iters']  # 迭代次数
        # 更新lr的延迟
        self.num_iters_decay = self.config['train']['num_iters_decay']
        # 恢复迭代的起始步，默认为0
        self.resume_iters = self.config['train']['resume_iters']
        self.current_iter = self.resume_iters  # 当前迭代次数

        self.lr = self.config['optimizer']['lr']  # 学习率
        # self.beta = self.config['optimizer']['beta'] # 学习率

        # 是否使用tensorboard
        self.use_tensorboard = self.config['logs']['use_tensorboard']
        self.log_every = self.config['logs']['log_every']  # 每几步记录
        if 'test_every' in self.config['logs']:  # 每几步测试
            self.test_every = self.config['logs']['test_every']
        # 每几步保存模型
        self.model_save_every = self.config['logs']['model_save_every']
        self.model_save_dir = self.config['logs']['model_save_dir']  # 模型保存地址
        if self.use_tensorboard:
            self.logger = Logger(
                self.config['logs']['log_dir'], self.model_name)

    def set_classification_weights(self, config, labels):
        print("设置分类权重。")
        if config['train']['loss_weight']:
            labels = list(labels)
            categories = config["data"]["indexx"].classes_
            total = len(labels)
            counts = [total/labels.count(emo)
                      for emo in categories]  # 出现次数少的权重大
            self.loss_weights = torch.Tensor(counts).to(self.device)
            print(f"权重: {self.loss_weights.tolist()}")
            print(f"标签: {categories}")
        else:
            self.loss_weights = None

    def batch_encode_plus(self, texts):
        encoded_texts = self.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=texts,   # 输入文本 batch
            # is_split_into_words=True,
            add_special_tokens=True,  # 添加 '[CLS]' 和 '[SEP]'
            # max_length=128,           # 填充 & 截断长度
            padding='longest',  # 设置填充到本 batch 中的最长长度
            return_attention_mask=True,   # 返回 attn. masks.
            return_tensors='pt',     # 返回 pytorch tensors 格式的数据
        )
        return encoded_texts['input_ids'], encoded_texts["attention_mask"]

    def train(self):
        """ 训练主函数 """

        #############################################################
        #                            训练                           #
        #############################################################
        print('################ 开始循环训练 ################')
        start_iter = self.resume_iters + 1  # 训练的新模型则从1开始，恢复的模型则从接续的迭代次数开始

        self.update_lr(start_iter)  # 根据开始迭代次数更新学习率

        data_iter = iter(self.train_loader)  # 构造数据块迭代器
        self.accuracylog = []
        start_time = datetime.now()
        print(f'训练开始时间: {start_time}')  # 读取开始运行时间

        for epoch in range(start_iter, self.num_iters+1):  # 迭代循环
            print(
                f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Iteration {epoch:02}/{self.num_iters:02} ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(
                f"Iteration {epoch:02} lr = {self.model.optimizer.param_groups[0]['lr']:.8f}")

            self.model.to_device(device=self.device)
            self.model.set_train_mode()
            self.current_iter = epoch  # 记录当前迭代次数

            try:
                input_values, texts, genders, labels = next(
                    data_iter)  # 加载没有被split的数据, 即仅有特征和标签
            except StopIteration as ex:
                data_iter = iter(self.train_loader)
                input_values, texts, genders, labels = next(
                    data_iter)  # 加载没有被split的数据, 即仅有特征和标签

            # targets = ["This female felt very "+i +
            #            "." for i in self.config["data"]["indexx"].classes_]

            targets = ["This man felt very "+i +
                       "." for i in self.config["data"]["indexx"].classes_]+["This woman felt very "+i +
                                                                             "." for i in self.config["data"]["indexx"].classes_]

            for i, l in enumerate(labels):
                if genders[i] == "M":
                    pass
                elif genders[i] == "F":
                    labels[i] = labels[i]+4

            target_text_ids, target_text_mask = self.batch_encode_plus(
                targets)
            text_ids, text_mask = self.batch_encode_plus(texts)
            input_values = input_values.unsqueeze(
                1).to(device=self.device).detach()  # .detach()
            text_ids = text_ids.to(device=self.device).detach()
            text_mask = text_mask.to(device=self.device).detach()
            target_text_mask = target_text_mask.to(device=self.device).detach()
            target_text_ids = target_text_ids.to(device=self.device).detach()
            labels = labels.to(device=self.device).detach()
            # 设置损失函数和分类权重，权重数据在s.train()前定义了
            loss_func = nn.CrossEntropyLoss(weight=None, reduction='mean')
            self.model.reset_grad()  # 清空梯度

            logits_per_speech, logits_per_text = self.model.net(speechs=input_values,
                                                                text_ids=text_ids, text_mask=text_mask,
                                                                target_text_ids=target_text_ids, target_text_mask=target_text_mask)

            loss_sl = loss_func(logits_per_speech, labels)
            loss_tl = loss_func(logits_per_text, labels)
            # loss_sl, loss_tl = self.loss_coteaching(
            #     logits_per_speech, logits_per_text, labels, forget_rate=rate_schedule[epoch-1])
            loss = (loss_sl + loss_tl)/2
            loss.backward()
            self.model.optimizer.step()
        #############################################################
        #                            记录                           #
        #############################################################
            if epoch % self.log_every == 0:  # 每几次迭代记录信息
                log_loss = {}
                log_loss['train/train_loss'] = loss.item()
                log_loss['train/train_loss_sl'] = loss_sl.item()
                log_loss['train/train_loss_tl'] = loss_tl.item()
                for name, val in log_loss.items():
                    print("{:20} = {:.4f}".format(name, val))

                if self.use_tensorboard:
                    for tag, value in log_loss.items():
                        self.logger.scalar_summary(tag, value, epoch)
            else:
                # print("No log output this iteration.")
                pass

            # save checkpoint
            if epoch % self.model_save_every == 0:  # 每几次保存一次模型
                self.model.save(save_dir=self.model_save_dir,
                                it=self.current_iter)
            else:
                # print("No model saved this iteration.")
                pass

            if epoch % self.test_every == 0:  # 每几次测试下模型
                self.test()
            # update learning rates
            self.update_lr(epoch)

            elapsed = datetime.now() - start_time
            print(f'第 {epoch:04} 次迭代结束, 耗时 {elapsed}')

        self.model.save(save_dir=self.model_save_dir, it=self.current_iter)
        return self.accuracylog[-20:]

    def update_lr(self, i):
        """衰减学习率方法"""
        if self.num_iters - self.num_iters_decay < i:  # 如果总迭代次数-当前迭代次数<更新延迟次数，开始每次迭代都更新学习率
            decay_delta = self.lr / self.num_iters_decay  # 学习率除以延迟次数，缩小学习率
            decay_start = self.num_iters - self.num_iters_decay
            decay_iter = i - decay_start  # 当前迭代次数-总迭代次数+更新延迟次数
            lr = self.lr - decay_iter * decay_delta
            print(f"更新学习率lr:{lr} ")
            for param_group in self.model.optimizer.param_groups:
                param_group['lr'] = lr  # 写入optimizer学习率参数

    def test(self):
        print("测试模型准确率。")
        self.model.set_eval_mode()
        emo_preds_sl = torch.rand(0).to(
            device=self.device, dtype=torch.long)  # 预测标签列表
        emo_preds_tl = torch.rand(0).to(
            device=self.device, dtype=torch.long)  # 预测标签列表
        total_labels = torch.rand(0).to(
            device=self.device, dtype=torch.long)  # 真实标签列表
        total_loss_sl = torch.rand(0).to(
            device=self.device, dtype=torch.float)  # 损失值列表
        total_loss_tl = torch.rand(0).to(
            device=self.device, dtype=torch.float)  # 损失值列表
        loss_func = nn.CrossEntropyLoss(reduction='mean')

        for i, (input_values, texts, genders, labels) in enumerate(self.test_loader):

            targets = ["The man felt very "+i +
                       "." for i in self.config["data"]["indexx"].classes_]
            target_text_ids, target_text_mask = self.batch_encode_plus(
                targets)
            text_ids, text_mask = self.batch_encode_plus(texts)
            input_values = input_values.unsqueeze(
                1).to(device=self.device)
            labels = labels.to(device=self.device)
            text_ids = text_ids.to(device=self.device)
            text_mask = text_mask.to(device=self.device)
            target_text_mask = target_text_mask.to(device=self.device)
            target_text_ids = target_text_ids.to(device=self.device)

            with torch.no_grad():

                logits_per_speech, logits_per_text = self.model.net(speechs=input_values,
                                                                    text_ids=text_ids, text_mask=text_mask,
                                                                    target_text_ids=target_text_ids, target_text_mask=target_text_mask)  # 计算模型输出分类值
                probs_sl = logits_per_speech.softmax(dim=-1)
                probs_tl = logits_per_text.softmax(dim=-1)
                #########################
                true_labels = labels

                # 获取 spech to target 概率最大值位置
                emo_pre_sl = torch.max(probs_sl, dim=1)[1]
                # 获取 text to target 概率最大值位置
                emo_pre_tl = torch.max(probs_tl, dim=1)[1]

                # 保存 spech to target 预测结果
                emo_preds_sl = torch.cat(
                    (emo_preds_sl, emo_pre_sl), dim=0)
                # 保存 text to target 预测结果
                emo_preds_tl = torch.cat((emo_preds_tl, emo_pre_tl), dim=0)

                # 保存所有真实结果
                total_labels = torch.cat(
                    (total_labels, true_labels), dim=0)

                # 保存 spech to target 测试损失
                emo_loss_sl = torch.tensor(
                    [loss_func(logits_per_speech, true_labels)], device=self.device)
                total_loss_sl = torch.cat(
                    (total_loss_sl, emo_loss_sl), dim=0)
                # 保存 text to target 测试损失
                emo_loss_tl = torch.tensor(
                    [loss_func(logits_per_text, true_labels)], device=self.device)
                total_loss_tl = torch.cat(
                    (total_loss_tl, emo_loss_tl), dim=0)

        eval_accuracy_sl = accuracy_score(
            total_labels.cpu(), emo_preds_sl.cpu())  # 计算准确率
        eval_accuracy_tl = accuracy_score(
            total_labels.cpu(), emo_preds_tl.cpu())  # 计算准确率
        eval_loss_sl = total_loss_sl.mean().item()
        eval_loss_tl = total_loss_tl.mean().item()
        l = ["Sl_Accuracy_Eval",  "Tl_Accuracy_Eval",
             "Sl_Loss_Eval", "Tl_Loss_Eval", ]
        print('{:20} = {:.3f}'.format(l[0], eval_accuracy_sl))
        print('{:20} = {:.3f}'.format(l[1], eval_accuracy_tl))
        print('{:20} = {:.3f}'.format(l[2], eval_loss_sl))
        print('{:20} = {:.3f}'.format(l[3], eval_loss_tl))

        if self.use_tensorboard:
            self.logger.scalar_summary(
                "Val/Sl_Accuracy_Eval", eval_accuracy_sl, self.current_iter)
            self.logger.scalar_summary(
                "Val/Tl_Accuracy_Eval", eval_accuracy_tl, self.current_iter)

            self.logger.scalar_summary(
                "Val/Sl_Loss_Eval", eval_loss_sl, self.current_iter)
            self.logger.scalar_summary(
                "Val/Tl_Loss_Eval", eval_loss_tl, self.current_iter)
            self.logger.scalar_summary(
                "Val/Total_Loss_Eval", (eval_loss_sl+eval_loss_tl)/2, self.current_iter)
        self.accuracylog.append(
            [eval_accuracy_sl, eval_accuracy_tl, (eval_loss_sl+eval_loss_tl)/2])


if __name__ == '__main__':
    pass
