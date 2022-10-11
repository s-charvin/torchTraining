import torch.nn as nn
import torch


class Inception(nn.Module):
    """ 并行使用多个不同核大小的卷积层
        不同大小的滤波器可以有效地识别不同范围的图像细节。
    """

    def __init__(self, in_channels, c1, c2, c3, c4):
        super(Inception, self).__init__()
        # 线路1，单1 x 1卷积层
        self.conv1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # 线路2，1 x 1卷积层后接3 x 3卷积层
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, c2[0], kernel_size=1), nn.ReLU(),
            nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1), nn.ReLU()
        )
        # 线路3，1 x 1卷积层后接5 x 5卷积层
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, c3[0], kernel_size=1), nn.ReLU(),
            nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2), nn.ReLU()
        )
        # 线路4，3 x 3最大汇聚层后接1 x 1卷积层
        self.conv4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, c4, kernel_size=1), nn.ReLU(),
        )

    def forward(self, X):
        return torch.cat(
            (
                self.conv1(X),
                self.conv2(X),
                self.conv3(X),
                self.conv4(X),
            ), dim=1)
