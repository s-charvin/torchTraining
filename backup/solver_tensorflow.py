import numpy as np
from datetime import datetime

from tqdm import tqdm

from models import SequenceClassifier, SequenceClassifier_CoTeaching
from logger import Logger

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
                data = next(data_iter)  # 加载没有被split的数据, 即仅有特征和标签
            except StopIteration as ex:
                data_iter = iter(self.train_loader)
                data = next(data_iter)  # 加载没有被split的数据, 即仅有特征和标签

            try:
                (x, labels) = data
                lenths = None
            except ValueError as ex:
                # 加载被split的数据, 即还含有记录的长度
                (x, labels, lenths) = data
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

        # for i, (x, labels, lenths) in enumerate(self.test_loader):
        for i, data in enumerate(self.test_loader):
            try:
                (x, labels) = data  # 正常数据
                lenths = None
            except:
                (x, labels, lenths) = data  # split后的情况
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
        """适用于CoTeaching的分类模型训练方法框架, 两个网络互相学习

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

        start_time = datetime.now()
        print(f'训练开始时间: {start_time}')  # 读取开始运行时间

        rate_schedule = np.ones(self.num_iters)*0.2
        rate_schedule[:10] = np.linspace(0, 0.2**1, 10)

        for epoch in range(start_iter, self.num_iters+1):  # 迭代循环

            self.model.to_device(device=self.device)
            self.model.set_train_mode()
            self.current_iter = epoch  # 记录当前迭代次数
            # 计算结果是否需要融合(split之后才需要此操作)
            self.resultmerge = self.config['data']["resultmerge"]
            pbar = tqdm(enumerate(self.train_loader),
                        desc=f"epoch:{epoch:02}, lr1 = {self.model.optimizer1.param_groups[0]['lr']:.8f}, lr2 = {self.model.optimizer2.param_groups[0]['lr']:.8f}/n")
            for i, (data) in pbar:  # 在当前迭代周期遍历所有数据
                try:
                    (x, labels) = data
                    lenths = None
                except ValueError as ex:
                    # 加载被split的数据, 即还含有记录的长度
                    (x, labels, lenths) = data

                lenths = lenths.to(device=self.device)
                x = x.to(device=self.device)
                labels = labels.long().to(device=self.device)
                # Nplabels = self.getNplabels(labels)

                # 设置损失函数和分类权重，权重数据在s.train()前定义了
                loss_func = nn.CrossEntropyLoss(
                    weight=self.loss_weights, reduction='none')

                self.model.reset_grad()  # 清空梯度

                y1, hide = self.model.net1(x)
                # 是否融合结果
                out1, true_labels1 = self.mergeResult(lenths, labels, y1)

                y2 = self.model.net2(x.detach(), hide.detach())
                # 是否融合结果
                out2, true_labels2 = self.mergeResult(lenths, labels, y2)

                loss_1, loss_2 = self.loss_coteaching(
                    y1, y2, labels, rate_schedule, loss_func)

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
                if epoch >= 100:
                    if epoch % self.test_every == 0:  # 每几次测试下模型
                        self.test()
                # update learning rates
                self.update_lr(epoch)

                elapsed = datetime.now() - start_time
                print(f'第 {epoch:04} 次迭代结束, 耗时 {elapsed}')

        self.model.save(save_dir=self.model_save_dir, it=self.current_iter)

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

        # if i >= self.num_iters/2:
        #     i = i - self.num_iters/2
        #     if self.num_iters/2 - self.num_iters_decay/2 < i:  # 如果总迭代次数-当前迭代次数<更新延迟次数，开始每次迭代都更新学习率
        #         decay_delta = self.lr / self.num_iters_decay/2  # 学习率除以延迟次数，缩小学习率
        #         decay_start = self.num_iters/2 - self.num_iters_decay/2
        #         decay_iter = i - decay_start  # 当前迭代次数-总迭代次数+更新延迟次数
        #         lr = self.lr - decay_iter * decay_delta
        #         print(f"更新学习率lr:{lr} ")
        #         for param_group in self.model.optimizer2.param_groups:
        #             param_group['lr'] = lr  # 写入optimizer学习率参数

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

        # for i, (x, labels, lenths) in enumerate(self.test_loader):
        for i, data in enumerate(self.test_loader):
            try:
                (x, labels) = data  # 正常数据
                lenths = None
            except:
                (x, labels, lenths) = data  # split后的情况

            x = x.to(device=self.device)
            labels = self.getNplabels(labels)
            labels = labels.long().to(device=self.device)

            with torch.no_grad():
                y1, hide = self.model.net1(x)  # 计算模型输出分类值
                y = self.model.net2(x, hide)
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

    def loss_coteaching(self, logits1, logits2, labels, forget_rate, loss_func):
        # 计算此batch内数据的所有损失
        loss_1 = loss_func(logits1, labels)
        ind_1_sorted = np.argsort(loss_1.data).cuda()  # 位置
        loss_1_sorted = loss_1[ind_1_sorted]  # 位置对应数据

        loss_2 = loss_func(logits2, labels)
        ind_2_sorted = np.argsort(loss_2.data).cuda()
        loss_2_sorted = loss_2[ind_2_sorted]

        remember_rate = 1 - forget_rate
        # 计算需要记住的数据数量
        num_remember = int(remember_rate * len(loss_1_sorted))

        ind_1_update = ind_1_sorted[:num_remember]
        ind_2_update = ind_2_sorted[:num_remember]
        # 计算筛选后的数据Loss, 此处通过index进行了数据位置交换
        loss_1_update = F.cross_entropy(
            logits1[ind_2_update], labels[ind_2_update])
        loss_2_update = F.cross_entropy(
            logits2[ind_1_update], labels[ind_1_update])

        return torch.sum(loss_1_update)/num_remember, torch.sum(loss_2_update)/num_remember
        # class Solver_Generate(object):

        #     def __init__(self, train_loader, test_loader, config, device, load_dir=None):
        #         """RNN序列生成模型训练方法框架

        #         Args:
        #             train_loader (_type_): 训练集，可迭代类型数据
        #             test_loader (_type_): 测试集，可迭代类型数据
        #             config (_type_): 参数设置
        #             device (_type_): 运行设备
        #             load_dir (_type_, optional): 欲加载检查点地址. Defaults to None.
        #         """
        #         self.train_loader = train_loader  # 训练数据迭代器
        #         self.test_loader = test_loader  # 测试数据迭代器
        #         self.vocab = train_loader.vocab  # 词元字典
        #         self.config = config  # 参数字典
        #         self.model_name = self.config['model']['name']  # 模型名称
        #         self.model = SequenceGenerator(
        #             self.config, self.model_name)  # 设置要使用的模型，分类模型、生成模型

        #         self.device = device  # 定义主设备
        #         self.set_configuration()  # 设置参数
        #         self.model = self.model

        #         if load_dir is not None:  # 是否加载历史检查点
        #             self.load_checkpoint(load_dir)

        #     def load_checkpoint(self, load_dir):
        #         self.model.load(load_dir)
        #         self.config = self.model.config  # 重点参数有:
        #         self.set_configuration()  # 设置加载的检查点的参数设置

        #     def set_configuration(self):
        #         self.batch_size = self.config['model']['batch_size']
        #         self.num_iters = self.config['model']['num_iters']  # 迭代次数
        #         # 更新lr的延迟
        #         self.num_iters_decay = self.config['model']['num_iters_decay']
        #         # 恢复迭代的起始步，默认为0
        #         self.resume_iters = self.config['model']['resume_iters']
        #         self.current_iter = self.resume_iters  # 当前迭代次数

        #         self.lr = self.config['optimizer']['lr']  # 学习率
        #         # 输入数据块是否是随机取样
        #         self.use_random_iter = self.config['model']['use_random_iter']
        #         # 是否使用tensorboard
        #         self.use_tensorboard = self.config['logs']['use_tensorboard']
        #         self.log_every = self.config['logs']['log_every']  # 每几步记录
        #         if 'test_every' in self.config['logs']:  # 每几步测试
        #             self.test_every = self.config['logs']['test_every']
        #         self.model_save_dir = self.config['logs']['model_save_dir']  # 模型保存地址
        #         # 每几步保存
        #         self.model_save_every = self.config['logs']['model_save_every']
        #         if self.use_tensorboard:
        #             self.logger = Logger(
        #                 self.config['logs']['log_dir'], self.model_name)

        #     def train(self):
        #         """训练主函数"""

        #         #############################################################
        #         #                            训练                           #
        #         #############################################################
        #         print('################ 开始循环训练 ################')
        #         start_iter = self.resume_iters + 1  # 训练的新模型则从1开始，恢复的模型则从接续的迭代次数开始

        #         # self.update_lr(start_iter)  # 根据开始迭代次数更新学习率

        #         data_iter = iter(self.train_loader)  # 构造数据块迭代器

        #         start_time = datetime.now()
        #         print(f'训练开始时间: {start_time}')  # 读取开始运行时间
        #         state = None  # 第一次输入的标志

        #         for i in range(start_iter, self.num_iters+1):  # 迭代循环
        #             print(
        #                 f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Iteration {i:02}/{self.num_iters:02} ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        #             print(
        #                 f"Iteration {i:02} lr = {self.model.optimizer.param_groups[0]['lr']:.6f}")

        #             self.model.to_device(device=self.device)
        #             self.model.set_train_mode()
        #             self.current_iter = i  # 记录当前迭代次数

        #             print('获取数据块。')
        #             try:
        #                 X, Y = next(data_iter)
        #             except:
        #                 data_iter = iter(self.train_loader)
        #                 X, Y = next(data_iter)
        #             y = Y.T.reshape(-1)
        #             X = X.to(device=self.device)
        #             y = y.long().to(device=self.device)

        #             # 初始化隐藏层向量
        #             if state is None or self.use_random_iter:
        #                 # 初始化 `state`当第一次迭代或使用随机取样时
        #                 # (随机取样,说明序列顺序随机,每一个数据块和数据块之间并无上下文背景关系)
        #                 state = self.model.net.begin_state(
        #                     batch_size=X.shape[0], device=self.device)
        #             else:
        #                 if isinstance(self.model.net, nn.Module) and not isinstance(state, tuple):
        #                     # `state` is a tensor for `nn.GRU`
        #                     state.detach_()
        #                 else:
        #                     # `state` is a tuple of tensors for `nn.LSTM` and
        #                     # for our custom scratch implementation
        #                     for s in state:
        #                         s.detach_()

        #             # 设置损失函数,和最后计算:求平均
        #             loss_func = nn.CrossEntropyLoss(reduction='mean')

        #             # 模型计算
        #             self.model.reset_grad()  # 模型计算前清空梯度, 防止梯度爆炸
        #             y_hat, state = self.model.net(X, state)
        #             loss = loss_func(y_hat, y.long())
        #             loss.backward()
        #             self.grad_clipping(self.model.net, 1)  # 梯度裁剪
        #             self.model.optimizer.step()

        #             # 更新学习率
        #             # self.update_lr(i)
        #         #############################################################
        #         #                            记录                           #
        #         #############################################################
        #             if i % self.log_every == 0:  # 每几次迭代记录信息
        #                 log_loss = {}
        #                 log_loss['total_loss'] = loss.mean().item()
        #                 for name, val in log_loss.items():
        #                     print("{:20} = {:.4f}".format(name, val))

        #                 if self.use_tensorboard:
        #                     for tag, value in log_loss.items():
        #                         self.logger.scalar_summary(tag, value, i)
        #             else:
        #                 print("No log output this iteration.")

        #             # save checkpoint
        #             if i % self.model_save_every == 0:  # 每几次保存一次模型
        #                 self.model.save(save_dir=self.model_save_dir,
        #                                 it=self.current_iter)
        #             else:
        #                 print("No model saved this iteration.")

        #             elapsed = datetime.now() - start_time
        #             print('第 {:04} 次迭代结束, 耗时 {}, '.format(i, elapsed))
        #         #############################################################
        #         #                            预测                           #
        #         #############################################################
        #             self.num_preds = self.config['model']['num_outputs']
        #             if i % self.test_every == 0:  # 每几次测试下模型
        #                 preds = self.predict('time traveller', self.num_preds,
        #                                      self.model.net, self.vocab, self.device)
        #                 print(preds)

        #         self.model.save(save_dir=self.model_save_dir, it=self.current_iter)

        #     def update_lr(self, i):
        #         """衰减学习率方法"""
        #         if self.num_iters - self.num_iters_decay < i:  # 如果总迭代次数-当前迭代次数<更新延迟次数，开始每次迭代都更新学习率
        #             decay_delta = self.lr / self.num_iters_decay  # 学习率除以延迟次数，缩小学习率
        #             decay_start = self.num_iters - self.num_iters_decay
        #             decay_iter = i - decay_start  # 当前迭代次数-总迭代次数+更新延迟次数
        #             lr = self.lr - decay_iter * decay_delta
        #             print(f"更新学习率lr:{lr} ")
        #             for param_group in self.model.optimizer.param_groups:
        #                 param_group['lr'] = lr  # 写入optimizer学习率参数

        #     def predict(self, prefix, num_preds, net, vocab, device):
        #         """在前言`prefix`后预测生成新序列"""
        #         print("正在预测...")
        #         self.model.set_eval_mode()

        #         state = net.begin_state(batch_size=1, device=device)
        #         outputs = [vocab[prefix[0]]]

        #         def get_input():
        #             return torch.tensor([outputs[-1]], device=device).reshape((1, 1))

        #         for y in prefix[1:]:  # Warm-up period
        #             _, state = net(get_input(), state)
        #             outputs.append(vocab[y])
        #         for _ in range(num_preds):  # 预测 `num_preds` 步
        #             y, state = net(get_input(), state)
        #             outputs.append(int(y.argmax(dim=1).reshape(1)))
        #         return ''.join([vocab.idx_to_token[i] for i in outputs])

        #     def grad_clipping(self, net, theta):
        #         """Clip the gradient."""
        #         if isinstance(net, nn.Module):
        #             params = [p for p in net.parameters() if p.requires_grad]
        #         else:
        #             params = net.params
        #         norm = torch.sqrt(sum(torch.sum((p.grad**2)) for p in params))
        #         if norm > theta:
        #             for param in params:
        #                 param.grad[:] *= theta / norm


if __name__ == '__main__':
    pass
