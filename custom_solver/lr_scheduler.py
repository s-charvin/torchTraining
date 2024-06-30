import math
import warnings
import torch
from torch.optim.lr_scheduler import *
from torch.optim.lr_scheduler import _LRScheduler, MultiplicativeLR
import matplotlib.pyplot as plt

__all__ = ["LambdaLR", "MultiplicativeLR", "StepLR",
           "MultiStepLR", "ExponentialLR", "CosineAnnealingLR",
           "ReduceLROnPlateau", "CyclicLR",
           "CosineAnnealingWarmRestarts", "OneCycleLR", "CosineLearningRateWithWarmRestarts", "LinearStepDecay", "CosineLearningRateWithWarmRestartsProgress"]

"""
Args:
    optimizer (_type_): 
    steps_per_epoch (_type_): 每 epoch 的训练步数, 也是 batch 的总数量
    epochs (_type_): 要迭代的总 epoch 数目
    num_decay_step (_type_): 
    last_epoch (int, optional): 初始 epoch, 即上次训练结束时的 epoch 数目, 从 0 开始计算. Defaults to -1.
    verbose (bool, optional): . Defaults to False.
"""


class LinearStepDecay(_LRScheduler):
    def __init__(self, optimizer, steps_per_epoch, epochs, num_decay_step, last_epoch=-1, verbose=False):

        self.steps_per_epoch = steps_per_epoch
        self.num_iter_step = epochs * steps_per_epoch
        self.num_decay_step = num_decay_step
        self.last_step = last_epoch if last_epoch < 0 else (
            last_epoch+1) * self.num_iter_step

        super(LinearStepDecay, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        decay_deltas = [
            base_lr / self.num_decay_step for base_lr in self.base_lrs]

        decay_iter = self.last_step - \
            (self.num_iter_step - self.num_decay_step)  # 当前迭代次数-总迭代次数+更新延迟次数

        if (self.last_step == 0) or (self.num_iter_step - self.num_decay_step < self.last_step):
            lrlist = []
            for base_lr, decay_delta in zip(self.base_lrs, decay_deltas):
                lrlist.append(base_lr - decay_iter * decay_delta)
            return lrlist
        return [group['lr'] for group in self.optimizer.param_groups]

    def step(self):
        class _enable_get_lr_call:

            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False
                return self

        with _enable_get_lr_call(self):
            self.last_step += 1
            if (self.last_step % self.steps_per_epoch) == 1:
                self.last_epoch += 1

            for i, data in enumerate(zip(self.optimizer.param_groups, self.get_lr())):
                param_group, lr = data
                param_group['lr'] = lr
                self.print_lr(self.verbose, i, lr, None)

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]


class LinearDecay(_LRScheduler):
    r"""
    """

    def __init__(self, optimizer, num_iter_epoch, num_decay_epoch, last_epoch=-1, verbose=False):

        self.num_iter_epoch = num_iter_epoch
        self.num_decay_epoch = num_decay_epoch
        super(LinearDecay, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if (self.last_epoch == 0) or (self.num_decay_epoch < self.last_epoch):
            resume_epoch = self.num_iter_epoch - self.num_decay_epoch
            return [base_lr * ((resume_epoch - (self.last_epoch - self.num_decay_epoch)) / resume_epoch) for base_lr in self.base_lrs]
        return [group['lr'] for group in self.optimizer.param_groups]


class CosineLearningRateWithWarmRestarts(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, epochs, cosine_end_lr=1e-6, warmup_start_lr=1e-8, last_epoch=-1, verbose=False):

        self.warmup_epochs = warmup_epochs
        self.epochs = epochs
        self.cosine_end_lr = cosine_end_lr
        self.warmup_start_lr = warmup_start_lr
        self.offset = self.warmup_epochs if self.warmup_epochs > 0.0 else 0.0
        super(CosineLearningRateWithWarmRestarts, self).__init__(
            optimizer, last_epoch, verbose)
        for base_lr in self.base_lrs:
            assert (cosine_end_lr < base_lr)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if (self.last_epoch < self.warmup_epochs):
            return [self.last_epoch * ((self.get_lr_cosine(self.warmup_epochs, base_lr) - self.warmup_start_lr) / self.warmup_epochs) + self.warmup_start_lr for base_lr in self.base_lrs]
        return [self.get_lr_cosine(self.last_epoch, base_lr) for base_lr in self.base_lrs]

    def get_lr_cosine(self, cur_epoch, lr):
        return self.cosine_end_lr + (lr - self.cosine_end_lr) * (math.cos(math.pi * (cur_epoch - self.offset) / (self.epochs - self.offset)) + 1.0) * 0.5

class CosineLearningRateWithWarmRestartsProgress(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, epochs, restarts, restart_weights, cosine_end_lr=1e-6, warmup_start_lr=1e-8, last_epoch=-1, verbose=False):
        self.warmup_epochs = warmup_epochs
        self.epochs = epochs
        self.cosine_end_lr = cosine_end_lr
        self.warmup_start_lr = warmup_start_lr
        assert max(restarts) < self.epochs
        self.restarts = [0] + restarts  # 将起始点加为第一个重启点，以平滑预热到第一次退火的过渡
        self.restart_weights = [1] + restart_weights
        
        super(CosineLearningRateWithWarmRestartsProgress, self).__init__(optimizer, last_epoch, verbose)
        for base_lr in self.base_lrs:
            assert cosine_end_lr < base_lr

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, please use `get_last_lr()`.", UserWarning)

        if self.last_epoch < self.warmup_epochs:
            lr = []
            for base_lr in self.base_lrs:
                end_warmup_lr = self.get_lr_cosine(self.warmup_epochs, base_lr, 0)  # 获取预热结束后的学习率
                lr.append(self.warmup_start_lr + self.last_epoch * ((end_warmup_lr - self.warmup_start_lr) / self.warmup_epochs))
            return lr

        lr = []
        restart_ind = 0
        for base_lr in self.base_lrs:
            current_lr = base_lr * self.restart_weights[restart_ind]  # 默认使用首个重启权重
            for ind, restart in enumerate(self.restarts):
                if self.last_epoch > restart:
                    restart_ind = ind
                    current_lr = base_lr * self.restart_weights[restart_ind]
                else:
                    break
            lr.append(self.get_lr_cosine(self.last_epoch, current_lr, restart_ind))
        return lr

    def get_lr_cosine(self, cur_epoch, lr, restart_ind):
        restart_epoch = self.restarts[restart_ind]
        next_restart_epoch = self.epochs if (restart_ind + 1 == len(self.restarts)) else self.restarts[restart_ind + 1]
        # 计算当前周期数减去上一个重启周期数后，相对于下一个重启周期的比例
        progress = (cur_epoch - restart_epoch) / (next_restart_epoch - restart_epoch)
        # 使用余弦退火公式来计算当前的学习率
        return self.cosine_end_lr + (lr - self.cosine_end_lr) * 0.5 * (1 + math.cos(math.pi * progress))

    
if __name__ == '__main__':
    epochs = 200
    warmup_epochs = 40
    base_lr = 0.00025
    cosine_end_lr = 0.000001
    warmup_start_lr = 0.00000001
    
    restarts = [100, 200, 300]
    restart_weights = [1, 1, 1]
    
    data_num = 1152
    batch_size = 32
    num_batch = data_num//batch_size + 1

    model = torch.nn.Sequential(
        torch.nn.Linear(1000, 100),
        torch.nn.ReLU(),
        torch.nn.Linear(100, 10),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)

    # scheduler = LinearDecay(optimizer, epoch, decay_step)

    # def lambda2(epoch): return 0.95 ** epoch
    # scheduler = LambdaLR(optimizer, lr_lambda=[lambda2])

    # def lmbda(epoch): return 0.95
    # scheduler = MultiplicativeLR(optimizer, lr_lambda=lmbda)

    # scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    # scheduler = MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)

    # scheduler = ExponentialLR(optimizer, gamma=0.1)
    
    scheduler = CosineLearningRateWithWarmRestarts(
        optimizer, 
        warmup_epochs=warmup_epochs, 
        epochs=epochs,
        cosine_end_lr=cosine_end_lr, 
        warmup_start_lr=warmup_start_lr, 
        )
    
    # scheduler = CosineLearningRateWithWarmRestartsProgress(
    #     optimizer, 
    #     warmup_epochs, 
    #     epochs, 
    #     restarts,
    #     restart_weights,
    #     cosine_end_lr=cosine_end_lr, 
    #     warmup_start_lr=warmup_start_lr, 
    #     )
    
    # scheduler = LinearStepDecay(
    #     optimizer, steps_per_epoch=num_batch, epochs=epochs, num_decay_step=0.8*num_batch*epochs)

    last_lrs = []
    last_epochs = []
    for e in range(0, epochs):
        for batch_i in range(0, num_batch):
            pass
        scheduler.step()
        last_lrs.append(scheduler.get_last_lr()[0])
        last_epochs.append(scheduler.last_epoch)
    # print(last_lrs)
    # print(last_epochs)
    plt.plot(last_epochs, last_lrs)
    plt.savefig("/home/visitors2/SCW/torchTraining/custom_solver/lr0.png")
