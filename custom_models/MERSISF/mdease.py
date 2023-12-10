from . import components
import torch
from torch import Tensor, nn
import torch.nn.functional as F


class MDEASE(nn.Module):
    """
    model: 多维度语音情感感知编码器
    """

    def __init__(self, in_channels=1, out_features=4) -> None:
        super().__init__()
        self.path1 = nn.Sequential(
            components.Conv2dSame(
                in_channels=in_channels, out_channels=32, kernel_size=(11, 1)
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
        )
        self.path2 = nn.Sequential(
            components.Conv2dSame(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=(1, 9),
                stride=(1, 1),
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
        )
        self.path3 = nn.Sequential(
            components.Conv2dSame(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=(3, 3),
                stride=(1, 1),
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
        )
        self.conv_extractor = components._get_ResMultiConv2d_extractor(
            in_channels=32 * 3,
            shapes=[  # [channel:int, kernel:list|int, stride:list|int, norm:str|None, pool:str|None, pool_kernel:list|int]
                [32, 1, "bn", "ap", [2, 2]],
                [64, 1, "bn", "ap", [2, 2]],
                [256, 1, "bn", "ap", [2, 1]],
                [512, 1, "bn", "ap", [2, 1]],
            ],
            bias=False,
        )
        self.gap = components.Conv2dLayerBlock(
            in_channels=512,
            out_channels=320,
            kernel_size=[1, 1],
            stride=1,
            bias=False,
            layer_norm=nn.BatchNorm2d(320),
            layer_poolling=nn.AdaptiveAvgPool2d(1),
            activate_func=nn.functional.relu,
        )
        self.dropout = nn.Dropout(0.3)
        self.last_linear = nn.Linear(in_features=320, out_features=out_features)

    def forward(self, x: Tensor, feature_lens=None) -> Tensor:
        out1 = self.path1(x)
        out2 = self.path2(x)
        out3 = self.path3(x)

        x = torch.cat([out1, out2, out3], dim=1).contiguous()
        x = self.conv_extractor(x)
        x = self.gap(x)
        x = self.dropout(x.squeeze(2).squeeze(2))
        x = self.last_linear(x)
        return x
