import math
import warnings
import torch
from torch.optim.lr_scheduler import *
from torch.optim.lr_scheduler import _LRScheduler
import matplotlib.pyplot as plt


class CustomScheduler(_LRScheduler):
    r"""
    """

    def __init__(self, optimizer, epoch, batchlen, decay, last_epoch=-1, verbose=False):

        self.batchlen = batchlen
        self.num_iter = epoch * batchlen
        self.decay = decay
        super(CustomScheduler, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        decay_deltas = [base_lr / self.decay for base_lr in self.base_lrs]

        decay_iter = self.last_epoch - \
            (self.num_iter - self.decay)  # 当前迭代次数-总迭代次数+更新延迟次数

        if (self.last_epoch == 0) or (self.num_iter - self.decay < self.last_epoch):
            lrlist = []
            for base_lr, decay_delta in zip(self.base_lrs, decay_deltas):
                lrlist.append(base_lr - decay_iter * decay_delta)
            return lrlist
        return [group['lr'] for group in self.optimizer.param_groups]

    def step(self, epoch=None):
        """
        """

        if epoch is None and self.last_epoch < 0:
            epoch = 0

        if epoch is None:
            epoch = self.last_epoch + 1
        else:
            if epoch < 0:
                raise ValueError(
                    "Expected non-negative epoch, but got {}".format(epoch))

        self.last_epoch = int(math.floor(epoch) * self.batchlen +
                              (epoch-math.floor(epoch)) * self.batchlen)

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
            for i, data in enumerate(zip(self.optimizer.param_groups, self.get_lr())):
                param_group, lr = data
                param_group['lr'] = lr
                self.print_lr(self.verbose, i, lr, epoch)

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]


if __name__ == '__main__':
    epoch = 885
    batchsize = 113
    decay = 99905
    model = torch.nn.Sequential(
        torch.nn.Linear(1000, 100),
        torch.nn.ReLU(),
        torch.nn.Linear(100, 10),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = CustomScheduler(
        optimizer, epoch=epoch, batchlen=batchsize, decay=decay)
    last_lr = []
    last_epoch = []
    for e in range(0, epoch):
        for batch_i in range(0, batchsize):
            optimizer.step()
            last_lr.append(scheduler.get_last_lr()[0])
            last_epoch.append(scheduler.last_epoch)
            scheduler.step(e + (batch_i+1) / batchsize)
    print(last_lr[-1])
    plt.plot(last_epoch, last_lr)
    plt.savefig("/home/visitors2/SCW/torchTraining/custom_solver/squares.png")
