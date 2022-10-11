import torch.nn as nn
import torch


class DenseBlock(nn.Module):
    def __init__(self, num_convs, input_channels, growth_channels):
        """
        x :           BN(64)->RELU->Conv(64,64)
        [x,dense(x)]: BN(128)->RELU->Conv(128,64)
        ......
        """
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):  # Dense层含有的卷积层个数
            layer.append(
                nn.Sequential(
                    # 每一层输入数量都在累加，卷积块的通道数控制了输出通道数相对于输入通道数的增长，近似于成长率growth_rate
                    # 每一层卷积块比上一层大 growth_channels
                    nn.BatchNorm2d(growth_channels * i + input_channels),
                    nn.ReLU(),
                    nn.Conv2d(growth_channels * i + input_channels, growth_channels, kernel_size=3, padding=1))
            )
        self.net = nn.Sequential(*layer)

    def forward(self, X):  # 修改每层传入方法为稠密连接
        # 遍历每一层dense层，连接得到[x, dense(x), dense([x,dense(x)]), dense([x,dense(x),dense([x,dense(x)])])].
        for blk in self.net:
            Y = blk(X)
            # 连接通道维度上每个块的输入和输出，注意此时输出维度会增加
            X = torch.cat((X, Y), dim=1)
        return X
