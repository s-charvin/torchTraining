from datetime import datetime, timedelta
import os
import torch
import torch.nn as nn
from utils.logger import Logger
import torch.distributed as dist
import sklearn
import logging
from custom_models import *
from .lr_scheduler import *
from .optimizer import *
from .report import classification_report

class Audio_Video_Progressive_Joint_Classification_Customloss(object):
    def __init__(
        self,
        train_loader=None,
        test_loader=None,
        config: dict = {},
        device=None,
        batch_delay=1,
    ):
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
        if self.config["architecture"]["net"]["para"]:
            self.net = globals()[self.config["architecture"]["net"]["name"]](
                **self.config["architecture"]["net"]["para"]
            )
        else:
            self.net = globals()[self.config["architecture"]["net"]["name"]]()

        # 多 GPU 并行计算
        if self.config["self_auto"]["gpu_nums"]:
            self.net = self.net.to(device=self.device)
            self.net = torch.nn.parallel.DistributedDataParallel(
                self.net, device_ids=[self.device]
                # , find_unused_parameters=True
            )  # device_ids 默认选用本地显示的所有 GPU

        # 设置优化器
        if (
            "para" in self.config["sorlver"]["optimizer"]
            and self.config["sorlver"]["optimizer"]["para"]
        ):
            self.optimizer = globals()[self.config["sorlver"]["optimizer"]["name"]](
                params=self.net.parameters(),
                lr=self.config["sorlver"]["optimizer"]["lr"],
                **self.config["sorlver"]["optimizer"]["para"],
            )
        else:
            self.optimizer = globals()[self.config["sorlver"]["optimizer"]["name"]](
                params=self.net.parameters()
            )

        if self.train_loader is not None:
            self.set_configuration()  # 设置参数
        # 加载模型及其参数
        if self.config["train"]["checkpoint"]:  # 是否加载历史检查点
            self.load(self.config["train"]["checkpoint"])

    def set_configuration(self):
        self.n_epochs = self.config["train"]["n_epochs"]  # 迭代次数
        # 设置已经训练完成的 epoch, 如果是初次为 -1, epoch 从 0 开始
        self.last_epochs = self.config["train"]["last_epochs"]
        if self.last_epochs < -1:
            raise ValueError("last_epochs 值必须为大于等于 -1 的整数, 当其为 -1 时, 表示重新训练.")

        self.test_every = self.config["logs"]["test_every"]
        if self.config["self_auto"]["local_rank"] in [0, None]:
            # 使用 tensorboard 记录
            self.use_tensorboard = self.config["logs"]["use_tensorboard"]
            self.log_every = self.config["logs"]["log_every"]  # 每几步记录
            # 每几步保存模型
            self.model_save_every = self.config["logs"]["model_save_every"]
            # 模型保存地址
            self.model_save_dir = self.config["logs"]["model_save_dir"]
            self.logname = "-".join(
                [
                    str(self.config["train"]["name"]),  # 当前训练名称
                    str(self.config["train"]["seed"]),
                    str(self.config["train"]["batch_size"]),
                    str(self.n_epochs),
                    self.config["sorlver"]["name"],
                    self.config["sorlver"]["optimizer"]["name"]
                    + str(self.config["sorlver"]["optimizer"]["lr"]),
                    self.config["data"]["name"],
                ]
            )
            if self.use_tensorboard:
                self.logger = Logger(self.config["logs"]["log_dir"], self.logname)

    def label2id(self, label_list):
        label_encoder = sklearn.preprocessing.LabelEncoder()
        label_encoder.fit(y=label_list)
        # id2label = {f"{id}": label for id, label in enumerate()}
        return label_encoder.transform(label_list), label_encoder.classes_

    def adjust_targets_for_tasks(self, targets):
        # 二分类任务：情绪好坏
        # 假设中性情绪标签ID为2（例如从categories中获取，如果categories['neutral'] == 2）
        categories = list(self.categories)
        angry_id = categories.index('angry')
        excited_id = categories.index('excited')
        frustrated_id = categories.index('frustrated')
        happy_id = categories.index('happy')
        neutral_id = categories.index('neutral')
        sad_id = categories.index('sad')
        
        targets2 = targets.clone()  # 好坏情绪
        targets2[targets == angry_id] = 0
        targets2[targets == frustrated_id] = 0
        targets2[targets == sad_id] = 0
        targets2[targets == excited_id] = 1
        targets2[targets == happy_id] = 1
        targets2[targets == neutral_id] = 1
        
        # 五分类任务：愤怒，快乐(含兴奋)，悲伤，中性，其他
        
        targets5 = targets.clone()
        targets5[targets == angry_id] = 0
        targets5[targets == happy_id] = 1
        targets5[targets == neutral_id] = 2
        targets5[targets == sad_id] = 3
        targets5[targets == excited_id] = 1  # 将兴奋合并到快乐
        targets5[targets == frustrated_id] = 4
        
        return targets2, targets5, targets

    def calculate_loss_weights(self, targets, num_classes, device):
        if self.config["train"]["loss_weight"]:
            label_counts = torch.zeros(num_classes)
            for target in targets:
                label_counts[target] += 1
            total_samples = sum(label_counts)
            loss_weights = total_samples / (num_classes * label_counts)
            return torch.tensor(loss_weights, dtype=torch.float).to(device)
        else:
            return None
        
    def calculate_task_weights(self, epoch, total_epochs):
        # 二分类任务权重: 初始权重为 3，逐渐下降至1直到1/4周期，然后保持1
        if epoch < total_epochs / 4:
            weight_2class = 3 - 2 * (epoch / (total_epochs / 4))
        else:
            weight_2class = 1

        # 四分类任务权重: 从0.5开始，增加至6直到1/4周期，然后逐步下降至3直到1/2周期，保持3
        if epoch < total_epochs / 4:
            weight_4class = 0.5 + 5.5 * (epoch / (total_epochs / 4))
        elif epoch < total_epochs / 2:
            weight_4class = 6 - 3 * ((epoch - total_epochs / 4) / (total_epochs / 4))
        else:
            weight_4class = 3

        # 六分类任务权重: 前1/4周期为0，从1/4到1/2周期权重从0.5增加到6，然后逐步下降至3直到3/4周期，保持3
        if epoch < total_epochs / 4:
            weight_6class = 0
        elif epoch < total_epochs / 2:
            weight_6class = 0.5 + 5.5 * ((epoch - total_epochs / 4) / (total_epochs / 4))
        elif epoch < 3 * total_epochs / 4:
            weight_6class = 6 - 3 * ((epoch - total_epochs / 2) / (total_epochs / 4))
        else:
            weight_6class = 3

        return weight_2class, weight_4class, weight_6class
    
    def train(self):
        """训练主函数"""
        assert self.train_loader, "未提供训练数据"
        #############################################################
        #                            训练                           #
        #############################################################
        start_iter = self.last_epochs + 1  # 训练的新模型则从 0 开始, 恢复的模型则从接续的迭代次数开始
        step = 0  # 每次迭代的步数, 也就是所有数据总共分的 batch 数量
        # 更新学习率的方法
        if self.config["sorlver"]["lr_method"]["mode"] == "epoch":
            self.lr_method = globals()[self.config["sorlver"]["lr_method"]["name"]](
                optimizer=self.optimizer,
                last_epoch=self.last_epochs if self.last_epochs >= 1 else -1,
                **self.config["sorlver"]["lr_method"]["para"],
            )
        elif self.config["sorlver"]["lr_method"]["mode"] == "step":
            self.lr_method = globals()[self.config["sorlver"]["lr_method"]["name"]](
                optimizer=self.optimizer,
                last_epoch=self.last_epochs if self.last_epochs >= 1 else -1,
                num_batch=len(self.train_loader),
                **self.config["sorlver"]["lr_method"]["para"],
            )

        # 训练开始时间记录
        start_time = datetime.now()
        if self.config["self_auto"]["local_rank"] in [0, None]:
            logging.info(f"# 训练开始时间: {start_time}")  # 读取开始运行时间


        # 计算每个类别的数量
        unique_labels, counts = np.unique(self.train_loader.dataset.dataset.datadict["label"], return_counts=True)
        label_counts = dict(zip(unique_labels, counts))
        logging.info(f"# 标签计数: {label_counts}")
        # 处理标签数据
        (
            self.train_loader.dataset.dataset.datadict["label"],
            self.categories,
        ) = self.label2id(self.train_loader.dataset.dataset.datadict["label"])
        
        

        # 保存最好的结果
        self.maxACC = self.WA_ = self.UA_ = self.macro_f1_ = self.w_f1_ = 0
        self.best_re = self.best_ma = None
        est_endtime = "..."
        if self.config["train"]["init_test"]:
            logging.info(f"# 计算模型的初始性能:")
            self.test()
            
        for epoch in range(start_iter, self.n_epochs):  # epoch 迭代循环 [0, epoch)
            task_weights = self.calculate_task_weights(epoch, self.n_epochs)
            # [0, num_batch)
            for batch_i, data in enumerate(self.train_loader):
                # 设置训练模式和设备
                self.to_device(device=self.device)
                self.set_train_mode()
                # 获取特征
                if isinstance(data["audio"], torch.nn.utils.rnn.PackedSequence):
                    (
                        audio_features,
                        audio_feature_lens,
                    ) = torch.nn.utils.rnn.pad_packed_sequence(
                        data["audio"], batch_first=True
                    )
                elif isinstance(data["audio"], torch.Tensor):
                    audio_features, audio_feature_lens = data["audio"], None

                if isinstance(data["video"], torch.nn.utils.rnn.PackedSequence):
                    (
                        video_features,
                        video_feature_lens,
                    ) = torch.nn.utils.rnn.pad_packed_sequence(
                        data["video"], batch_first=True
                    )
                elif isinstance(data["video"], torch.Tensor):
                    video_features, video_feature_lens = data["video"], None
                # 获取标签
                labels = data["label"]
                targets2, targets5, targets6 = self.adjust_targets_for_tasks(labels)
                targets2, targets5, targets6 = targets2.to(self.device), targets5.to(self.device), targets6.to(self.device)
                
                # 将数据迁移到训练设备
                video_features = video_features.to(
                    device=self.device, dtype=torch.float32
                )
                audio_features = audio_features.to(
                    device=self.device, dtype=torch.float32
                )

                if video_feature_lens is not None:
                    video_feature_lens = video_feature_lens.to(device=self.device)
                if audio_feature_lens is not None:
                    audio_feature_lens = audio_feature_lens.to(device=self.device)

                labels = labels.long().to(device=self.device)
                outs, dis_loss, sim_loss = self.net(
                    audio_features,
                    video_features,
                    audio_feature_lens,
                    video_feature_lens,
                )
                
                criterions = [
                    nn.CrossEntropyLoss(reduction="mean", weight=torch.Tensor([0.9095538312318138, 1.110420367081113])).to(device=self.device),
                    nn.CrossEntropyLoss(reduction="mean", weight=torch.Tensor([1.3348754448398576,0.9016826923076923,1.3126859142607175,0.8753792298716453,0.807969843834141])).to(device=self.device),
                    nn.CrossEntropyLoss(reduction="mean", weight=torch.Tensor([1.1123962040332147,1.1696289367009667,0.6733082031951175,2.1014005602240897,0.7294826915597044,1.0939049285505977])).to(device=self.device)
                ]
                
                loss = sum(
                    criterion(output, target) * weight 
                    for output, criterion, target, weight in zip(outs, criterions, [targets2, targets5, targets6], task_weights))
                
                loss_ = loss.clone()
                dis_loss_ = dis_loss.clone()
                sim_loss_ = sim_loss.clone()
                all_loss = (loss + dis_loss + sim_loss) / 3
                all_loss = all_loss / self.batch_delay
                all_loss.backward()
                # 在所有进程运行到这一步之前，先完成此前代码的进程会在这里等待其他进程。
                if self.config["self_auto"]["gpu_nums"]:
                    torch.distributed.barrier()
                    dist.all_reduce(loss_, op=dist.ReduceOp.SUM)
                    # 计算所有 GPU 的平均损失
                    loss_ /= self.config["self_auto"]["world_size"]
                    dis_loss_ /= self.config["self_auto"]["world_size"]
                    sim_loss_ /= self.config["self_auto"]["world_size"]

                #############################################################
                #                            记录                           #
                #############################################################
                # tensorboard 可视化
                if self.config["self_auto"]["local_rank"] in [0, None]:
                    if (step % self.log_every == 0) and self.use_tensorboard:
                        log_loss = {}
                        log_loss["train/train_loss"] = loss_.item()
                        for tag, value in log_loss.items():
                            self.logger.scalar_summary(tag, value, step)
                    runningtime = datetime.now() - start_time

                    logging.info(
                        f"# <trained>: [Epoch {epoch+1}/{self.n_epochs}] [Batch {batch_i+1}/{len(self.train_loader)}] [lr: {self.optimizer.param_groups[0]['lr']}] [train_loss: {loss_.item():.4f} | {dis_loss_.item():.4f} | {sim_loss_.item():.4f} ] [runtime: {runningtime}] [est_endtime: {est_endtime}]"
                    )

                # 反向更新
                if ((batch_i + 1) % self.batch_delay) == 0 or (batch_i + 1 == len(self.train_loader)):
                    self.optimizer.step()  # 根据反向传播的梯度，更新网络参数
                    self.reset_grad()

                # 更新学习率
                if self.config["sorlver"]["lr_method"]["mode"] == "step":
                    self.lr_method.step()  # 当更新函数需要的是总 step 数量时, 输入小数
                step = step + 1
            if self.config["sorlver"]["lr_method"]["mode"] == "epoch":
                self.lr_method.step()  # 当更新函数需要的是总 epoch 数量时, 输入整数

            est_endtime = timedelta(
                seconds=self.n_epochs
                * ((runningtime.days * 86400 + runningtime.seconds) / (epoch + 1))
            )
            self.last_epochs = epoch  # 记录当前已经训练完的迭代次数

            # 每 epoch 次测试下模型
            if (epoch + 1) % self.test_every == 0:
                self.test()

        # 保存最终的模型
        if self.config["self_auto"]["local_rank"] in [0, None]:
            self.save(save_dir=self.model_save_dir, it=self.last_epochs)
            logging.info(self.best_re)
            logging.info(self.best_ma)
        return True

    def test(self):
        """测试模型准确率和其他性能指标。"""
        assert self.test_loader, "未提供测试数据"
        
        num_samples = 0
        
        # 损失函数
        loss_func = nn.CrossEntropyLoss(reduction="sum").to(self.device)
        total_loss = torch.tensor(0.0).to(device=self.device)
        
        # 初始化存储测试结果的列表
        all_predictions = {2: torch.tensor([], dtype=torch.long).to(self.device),
                           5: torch.tensor([], dtype=torch.long).to(self.device),
                           6: torch.tensor([], dtype=torch.long).to(self.device)}
        all_true_labels = {2: torch.tensor([], dtype=torch.long).to(self.device),
                           5: torch.tensor([], dtype=torch.long).to(self.device),
                           6: torch.tensor([], dtype=torch.long).to(self.device)}

        for data in self.test_loader:
            self.to_device(device=self.device)
            self.set_eval_mode()

            # 获取特征
            if isinstance(data["audio"], torch.nn.utils.rnn.PackedSequence):
                (
                    audio_features,
                    audio_feature_lens,
                ) = torch.nn.utils.rnn.pad_packed_sequence(
                    data["audio"], batch_first=True
                )
            elif isinstance(data["audio"], torch.Tensor):
                audio_features, audio_feature_lens = data["audio"], None

            if isinstance(data["video"], torch.nn.utils.rnn.PackedSequence):
                (
                    video_features,
                    video_feature_lens,
                ) = torch.nn.utils.rnn.pad_packed_sequence(
                    data["video"], batch_first=True
                )
            elif isinstance(data["video"], torch.Tensor):
                video_features, video_feature_lens = data["video"], None

            # 获取标签
            true_labels = data["label"]
            # 将数据迁移到训练设备
            video_features = video_features.to(device=self.device, dtype=torch.float32)
            audio_features = audio_features.to(device=self.device, dtype=torch.float32)

            if video_feature_lens is not None:
                video_feature_lens = video_feature_lens.to(device=self.device)
            if audio_feature_lens is not None:
                audio_feature_lens = audio_feature_lens.to(device=self.device)
                
            labels = true_labels.long().to(self.device)
        
            with torch.no_grad():
                outputs, _,_ = self.net(
                    audio_features,
                    video_features,
                    audio_feature_lens,
                    video_feature_lens,
                )  # 计算模型输出分类值
                
                targets2, targets5, targets6 = self.adjust_targets_for_tasks(labels)
                targets2, targets5, targets6 = targets2.to(self.device), targets5.to(self.device), targets6.to(self.device)

                # 计算损失
                losses = [loss_func(output, target) for output, target in zip(outputs, [targets2, targets5, targets6])]
                total_loss += sum(losses).item() / len(losses)
                
                # 解析预测结果
                predictions = [torch.max(F.softmax(output, dim=1), dim=1)[1] for output in outputs]
                all_predictions[2] = torch.cat((all_predictions[2], predictions[0]), dim=0)
                all_predictions[5] = torch.cat((all_predictions[5], predictions[1]), dim=0)
                all_predictions[6] = torch.cat((all_predictions[6], predictions[2]), dim=0)
                all_true_labels[2] = torch.cat((all_true_labels[2], targets2), dim=0)
                all_true_labels[5] = torch.cat((all_true_labels[5], targets5), dim=0)
                all_true_labels[6] = torch.cat((all_true_labels[6], targets6), dim=0)

                num_samples += labels.size(0)
        # 计算准确率

        if self.config["self_auto"]["gpu_nums"]:
            world_size = self.config["self_auto"]["world_size"]
            # 跨GPU同步
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            total_loss /= world_size * num_samples
            
            torch.distributed.barrier()
            for task in all_predictions.keys():
                # 创建收集列表
                gathered_predictions = [torch.empty_like(all_predictions[task]) for _ in range(world_size)]
                gathered_labels = [torch.empty_like(all_true_labels[task]) for _ in range(world_size)]
                
                # 收集每个进程的预测和标签
                dist.all_gather(gathered_predictions, all_predictions[task])
                dist.all_gather(gathered_labels, all_true_labels[task])
                
                # 更新存储
                all_predictions[task] = torch.cat(gathered_predictions, dim=0)
                all_true_labels[task] = torch.cat(gathered_labels, dim=0)
            
        if self.config["self_auto"]["local_rank"] in [0, None]:
            avg_accuracy = 0
            num_tasks = 0
            target_names = {
                2: ['Negative', 'Positive'],
                5: ['Angry', 'Happy/Excited', 'Neutral', 'Sad', 'Other'],
                6: ['Angry', 'Excited', 'Frustrated', 'Happy', 'Neutral', 'Sad']
            }
            
            reports = {}
            for task, predictions in all_predictions.items():
                # WA, UA, ACC, macro_f1, w_f1, report, matrix
                WA, UA, ACC, macro_f1, w_f1, report, matrix = classification_report(all_true_labels[task].cpu(), predictions.cpu(), target_names=target_names[task])
                reports[task] = {
                    "WA": WA,
                    "UA": UA,
                    "ACC": ACC,
                    "macro_f1": macro_f1,
                    "w_f1": w_f1,
                    "report": report,
                    "matrix": matrix,
                }
                
                logging.info(f"# <tested, {task} classification>: [WA: {WA}] [UA: {UA}] [ACC: {ACC}] [macro_f1: {macro_f1}] [w_f1: {w_f1}]")
                avg_accuracy += ACC
                num_tasks += 1
            avg_accuracy /= num_tasks
            if self.use_tensorboard:
                self.logger.scalar_summary("Val/Loss_Eval", total_loss, self.last_epochs)
                self.logger.scalar_summary("Val/Accuracy_Eval", avg_accuracy, self.last_epochs)
            
            logging.info(f"# <tested>: [ACC: {avg_accuracy}]")
            
            # 记录最好的一次模型数据
            if self.maxACC < avg_accuracy:
                self.maxACC = avg_accuracy
                self.reports = reports
                # Save model
                self.save(save_dir=self.model_save_dir, it=self.last_epochs)
                logging.info(f"# <best>: [ACC: {avg_accuracy}]")
                self.format_and_print_reports()
                # 每次遇到更好的就保存一次模型
                try:
                    if self.config['logs']['model_save_every']:
                        self.save(save_dir=self.model_save_dir,
                                it=self.last_epochs)
                except Exception as e:
                    if "No space left on device" in e.args[0]:
                        pass
                    else:
                        raise e
                    
    def format_and_print_reports(self):
        """格式化并打印self.reports字典中存储的所有性能指标和分类报告。"""
        if not hasattr(self, 'reports') or not isinstance(self.reports, dict):
            logging.error("Reports attribute is missing or not a dictionary.")
            return

        for task, metrics in self.reports.items():
            logging.info(f"===== Task {task} Classification Metrics =====")
            logging.info(f"Weighted Accuracy (WA): {metrics['WA']}")
            logging.info(f"Unweighted Accuracy (UA): {metrics['UA']}")
            logging.info(f"Accuracy (ACC): {metrics['ACC']}")
            logging.info(f"Macro F1-Score: {metrics['macro_f1']}")
            logging.info(f"Weighted F1-Score: {metrics['w_f1']}")

            logging.info(f"\nClassification Report:\n{metrics['report']}")
            logging.info(f"\nConfusion Matrix:\n{metrics['matrix']}")
            logging.info("\n")  # Add a newline for better readability between tasks

    def calculate(self, inputs: list):
        self.to_device(device=self.device)
        self.set_eval_mode()
        label_pres = []
        for i, inp in enumerate(inputs):
            with torch.no_grad():
                out, _ = self.net(
                    inputs[i][0].to(device=self.device),
                    inputs[i][1].to(device=self.device),
                )
            label_id = torch.max(out, dim=1)[1].item()
            label = self.categories[label_id]
            label_pres.append((label_id, label))
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

    def print_network(self):
        """打印网络结构信息"""
        modelname = self.config["architecture"]["net"]["name"]
        logging.info(f"# 模型名称: {modelname}")
        num_params = 0
        for p in self.net.parameters():
            num_params += p.numel()  # 获得张量的元素个数
        if self.config["logs"]["verbose"]["model"]:
            logging.info("# 打印网络结构:")
            logging.info(self.net)
        logging.info(f"网络参数数量:{num_params}")
        return num_params

    def reset_grad(self):
        """将梯度清零。"""
        self.optimizer.zero_grad()

    def save(self, save_dir=None, it=0):
        """保存模型"""
        if save_dir is None:
            save_dir = self.config["logs"]["model_save_dir"]
        path = os.path.join(save_dir, self.logname)
        if not os.path.exists(path):
            os.makedirs(path)

        self.config["train"]["last_epochs"] = it

        state = {
            "net": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config,
            "train_indices": self.train_loader.dataset.indices,  # 训练数据索引
            "test_indices": self.test_loader.dataset.indices,  # 测试数据索引
        }
        path = os.path.join(path, f"{it:06}.ckpt")
        torch.save(state, path)
        logging.info(f"# 模型存放地址: {path}.")

    def load(self, load_dir):
        """
        load_dir: 要加载模型的完整地址
        """
        logging.info(f"# 模型加载地址: {load_dir}。")

        # 保存每个训练进程专属的信息
        self_auto_config = self.config["self_auto"]
        self.last_epochs = self.config["train"]["last_epochs"]

        # 加载模型, 参数配置, 和数据索引信息
        dictionary = torch.load(load_dir, map_location=self.device)
        self.config = dictionary["config"]

        self.config["self_auto"] = self_auto_config
        self.net.load_state_dict(dictionary["net"])
        self.optimizer.load_state_dict(dictionary["optimizer"])

        # 设置已经训练完成的 epoch, 如果是初次为 -1, epoch 从 0 开始
        if self.last_epochs > -1:
            self.config["train"]["last_epochs"] = self.last_epochs
        else:
            self.last_epochs = self.config["train"]["last_epochs"]

        # 保证当前的训练数据与之前的训练数据一致(这里是为了防止迁移训练环境后导致训练数据的随机分布发生变化)
        if self.train_loader:
            self.train_loader.dataset.indices = dictionary["train_indices"]
            self.test_loader.dataset.indices = dictionary["test_indices"]
            self.set_configuration()  # 设置加载的检查点的参数设置
            logging.info("# Models 和 optimizers 加载成功")
        else:
            logging.info("# Models 加载成功, 目前仅可进行计算操作")
        del dictionary


if __name__ == "__main__":
    pass
