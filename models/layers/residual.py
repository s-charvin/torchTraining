import torch.nn as nn
from torch.nn import functional as F


class Residual(nn.Module):
    # 残差块(卷积-BN-激活-卷积-BN-残差连接（Y=Y+X）-激活)
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        """
        para use_1x1conv: 是否利用1*1卷积修改输入通道与输出对应
        """
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, num_channels,
                      kernel_size=3, padding=1, stride=strides),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels)
        )
        if use_1x1conv:
            self.conv2 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv2 = None

    def forward(self, X):  # 修改每层传入方法为残差累加

        Y = self.conv1(X)
        if self.conv2:
            X = self.conv2(X)
        Y += X
        return F.relu(Y)
