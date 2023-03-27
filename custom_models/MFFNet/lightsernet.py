from .components import *
import torch
from torch import Tensor, nn
import numpy as np


class LightSerNet_Encoder(nn.Module):
    """
    paper: Light-SERNet: A lightweight fully convolutional neural network for speech emotion recognition
    """

    def __init__(self, in_channels=1) -> None:
        super().__init__()
        self.path1 = nn.Sequential(
            Conv2dSame(in_channels=in_channels, out_channels=32, kernel_size=(
                11, 1), stride=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2)
        )
        self.path2 = nn.Sequential(
            Conv2dSame(in_channels=1, out_channels=32, kernel_size=(
                1, 9), stride=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2)
        )
        self.path3 = nn.Sequential(
            Conv2dSame(in_channels=1, out_channels=32, kernel_size=(
                3, 3), stride=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2)
        )
        self.conv_extractor = get_Conv2d_extractor(
            in_channels=32*3,
            shapes=[  # [channel:int, kernel:list|int, stride:list|int, norm:str|None, pool:str|None, pool_kernel:list|int]
                [64, [3, 3], 1, "bn", "ap", [2, 2]],
                [96, [3, 3], 1, "bn", "ap", [2, 2]],
                [128, [3, 3], 1, "bn", "ap", [2, 1]],
                [160, [3, 3], 1, "bn", "ap", [2, 1]],
                [320, [1, 1], 1, "bn", "gap", 1]
            ],
            bias=False
        )
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: Tensor, feature_lens=None) -> Tensor:
        # [B , C, S, F]
        out1 = self.path1(x)
        out2 = self.path2(x)
        out3 = self.path3(x)

        x = torch.cat([out1, out2, out3], dim=1).contiguous()

        x = self.conv_extractor(x)
        x = self.dropout(x.squeeze(-1).squeeze(-1))
        return x


if __name__ == '__main__':
    test = np.random.random((4, 1, 126, 40)).astype(np.float32)
    test = torch.Tensor(test)
    macnn = LightSerNet_Encoder(in_channels=1)
    print(macnn(test).shape)
