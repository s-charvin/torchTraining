from . import components
import torch
from torch import Tensor, nn
import torch.nn.functional as F


class LightSerNet(nn.Module):
    """
    paper: Light-SERNet: A lightweight fully convolutional neural network for speech emotion recognition
    """

    def __init__(self, in_channels=1, out_features=4) -> None:
        super().__init__()
        self.path1 = nn.Sequential(
            components.Conv2dSame(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=(11, 1),
                stride=(1, 1),
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
        )
        self.path2 = nn.Sequential(
            components.Conv2dSame(
                in_channels=1, out_channels=32, kernel_size=(1, 9), stride=(1, 1)
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
        )
        self.path3 = nn.Sequential(
            components.Conv2dSame(
                in_channels=1, out_channels=32, kernel_size=(3, 3), stride=(1, 1)
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
        )

        self.conv_extractor = components.get_Conv2d_extractor(
            in_channels=32 * 3,
            shapes=[  # [channel:int, kernel:list|int, stride:list|int, norm:str|None, pool:str|None, pool_kernel:list|int]
                [64, [3, 3], 1, "bn", "ap", [2, 2]],
                [96, [3, 3], 1, "bn", "ap", [2, 2]],
                [128, [3, 3], 1, "bn", "ap", [2, 1]],
                [160, [3, 3], 1, "bn", "ap", [2, 1]],
                [320, [1, 1], 1, "bn", "gap", 1],
            ],
            bias=False,
        )
        self.dropout = nn.Dropout(0.3)
        self.last_linear = nn.Linear(in_features=320, out_features=out_features)

    def forward(self, x: Tensor, feature_lens=None) -> Tensor:
        out1 = self.path1(x)
        out2 = self.path2(x)
        out3 = self.path3(x)

        x = torch.cat([out1, out2, out3], dim=1).contiguous()

        x = self.conv_extractor(x)
        x = self.dropout(x.squeeze(2).squeeze(2))
        x = self.last_linear(x)

        return x, None


class LightResMultiSerNet(nn.Module):
    """
    paper: Light-SERNet: A lightweight fully convolutional neural network for speech emotion recognition
    """

    def __init__(self) -> None:
        super().__init__()
        self.path1 = nn.Sequential(
            components.Conv2dSame(
                in_channels=1, out_channels=32, kernel_size=(11, 1), stride=(1, 1)
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
        )
        self.path2 = nn.Sequential(
            components.Conv2dSame(
                in_channels=1, out_channels=32, kernel_size=(1, 9), stride=(1, 1)
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
        )
        self.path3 = nn.Sequential(
            components.Conv2dSame(
                in_channels=1, out_channels=32, kernel_size=(3, 3), stride=(1, 1)
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
        self.last_linear = nn.Linear(in_features=320, out_features=4)

    def forward(self, x: Tensor, feature_lens=None) -> Tensor:
        x = x.permute(0, 3, 1, 2)  # [B, C, Seq, F]
        out1 = self.path1(x)
        out2 = self.path2(x)
        out3 = self.path3(x)

        x = torch.cat([out1, out2, out3], dim=1).contiguous()

        x = self.conv_extractor(x)
        x = self.gap(x)
        x = self.dropout(x.squeeze(2).squeeze(2))
        x = self.last_linear(x)
        return x, None


class LightTestSerNet(nn.Module):
    """
    paper: Light-SERNet: A lightweight fully convolutional neural network for speech emotion recognition
    """

    def __init__(self) -> None:
        super().__init__()
        self.path11 = nn.Sequential(
            components.Conv2dSame(
                in_channels=1, out_channels=16 * 1, kernel_size=(3, 1), stride=(1, 1)
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
        )
        self.path21 = nn.Sequential(
            components.Conv2dSame(
                in_channels=1, out_channels=16 * 1, kernel_size=(1, 3), stride=(1, 1)
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
        )
        self.path31 = nn.Sequential(
            components.Conv2dSame(
                in_channels=1, out_channels=16 * 1, kernel_size=(3, 3), stride=(1, 1)
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
        )

        self.path12 = nn.Sequential(
            components.Conv2dSame(
                in_channels=16 * 3,
                out_channels=16 * 3,
                kernel_size=(7, 1),
                stride=(1, 1),
            ),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
        )
        self.path22 = nn.Sequential(
            components.Conv2dSame(
                in_channels=16 * 3,
                out_channels=16 * 3,
                kernel_size=(1, 7),
                stride=(1, 1),
            ),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
        )
        self.path32 = nn.Sequential(
            components.Conv2dSame(
                in_channels=16 * 3,
                out_channels=16 * 3,
                kernel_size=(7, 7),
                stride=(1, 1),
            ),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
        )

        self.path13 = nn.Sequential(
            components.Conv2dSame(
                in_channels=16 * 9,
                out_channels=16 * 3,
                kernel_size=(11, 1),
                stride=(1, 1),
            ),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
        )
        self.path23 = nn.Sequential(
            components.Conv2dSame(
                in_channels=16 * 9,
                out_channels=16 * 3,
                kernel_size=(1, 9),
                stride=(1, 1),
            ),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
        )
        self.path33 = nn.Sequential(
            components.Conv2dSame(
                in_channels=16 * 9,
                out_channels=16 * 3,
                kernel_size=(11, 9),
                stride=(1, 1),
            ),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
        )

        self.conv_extractor = components._get_ModulatedDeformConv2d_extractor(
            in_channels=48 * 3,
            shapes=[  # [channel:int, kernel:list|int, stride:list|int, norm:str|None, pool:str|None, pool_kernel:list|int]
                [64, [3, 3], 1, "bn", "ap", [2, 2]],
                [96, [3, 3], 1, "bn", "ap", [2, 2]],
                [128, [3, 3], 1, "bn", "ap", [2, 1]],
                [160, [3, 3], 1, "bn", "ap", [2, 1]],
                [320, [1, 1], 1, "bn", "gap", 1],
            ],
            bias=False,
        )
        self.dropout = nn.Dropout(0.3)
        self.last_linear = nn.Linear(in_features=320, out_features=4)

    def forward(self, x: Tensor, feature_lens=None) -> Tensor:
        x = x.permute(0, 3, 1, 2)  # [B, C, Seq, F]
        out1 = self.path11(x)
        out2 = self.path21(x)
        out3 = self.path31(x)
        x = torch.cat([out1, out2, out3], dim=1).contiguous()

        out1 = self.path12(x)
        out2 = self.path22(x)
        out3 = self.path32(x)
        x = torch.cat([out1, out2, out3], dim=1).contiguous()

        out1 = self.path13(x)
        out2 = self.path23(x)
        out3 = self.path33(x)
        x = torch.cat([out1, out2, out3], dim=1).contiguous()

        x = self.conv_extractor(x)
        x = self.dropout(x.squeeze(2).squeeze(2))
        x = self.last_linear(x)

        return x, None


class LightResMultiSerNet_Encoder(nn.Module):
    """
    paper: Light-SERNet: A lightweight fully convolutional neural network for speech emotion recognition
    """

    def __init__(self) -> None:
        super().__init__()
        self.path1 = nn.Sequential(
            components.Conv2dSame(in_channels=1, out_channels=32, kernel_size=(11, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
        )
        self.path2 = nn.Sequential(
            components.Conv2dSame(
                in_channels=1, out_channels=32, kernel_size=(1, 9), stride=(1, 1)
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
        )
        self.path3 = nn.Sequential(
            components.Conv2dSame(
                in_channels=1, out_channels=32, kernel_size=(3, 3), stride=(1, 1)
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

    def forward(self, x: Tensor, feature_lens=None) -> Tensor:
        out1 = self.path1(x)
        out2 = self.path2(x)
        out3 = self.path3(x)

        x = torch.cat([out1, out2, out3], dim=1).contiguous()
        x = self.conv_extractor(x)
        x = self.gap(x)
        x = self.dropout(x.squeeze(2).squeeze(2))
        return x
