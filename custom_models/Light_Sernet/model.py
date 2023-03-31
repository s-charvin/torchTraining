from . import components
import torch
from torch import Tensor, nn
import torch.nn.functional as F


class LightSerNet(nn.Module):
    """
    paper: Light-SERNet: A lightweight fully convolutional neural network for speech emotion recognition
    """

    def __init__(self) -> None:
        super().__init__()
        self.path1 = nn.Sequential(
            components.Conv2dSame(in_channels=1, out_channels=32, kernel_size=(
                11, 1), stride=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2)
        )
        self.path2 = nn.Sequential(
            components.Conv2dSame(in_channels=1, out_channels=32, kernel_size=(
                1, 9), stride=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2)
        )
        self.path3 = nn.Sequential(
            components.Conv2dSame(in_channels=1, out_channels=32, kernel_size=(
                3, 3), stride=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2)
        )
        self.conv_extractor = components._get_Conv2d_extractor(
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
        self.last_linear = nn.Linear(in_features=320, out_features=4)

    def forward(self, x: Tensor, feature_lens=None) -> Tensor:

        out1 = self.path1(x)
        out2 = self.path2(x)
        out3 = self.path3(x)

        x = torch.cat([out1, out2, out3], dim=1).contiguous()

        x = self.conv_extractor(x)
        x = self.dropout(x.squeeze(2).squeeze(2))
        x = self.last_linear(x)

        return x


class LightMultiSerNet(nn.Module):
    """
    paper: Light-SERNet: A lightweight fully convolutional neural network for speech emotion recognition
    """

    def __init__(self) -> None:
        super().__init__()
        self.path1 = nn.Sequential(
            components.Conv2dSame(in_channels=1, out_channels=32, kernel_size=(
                11, 1), stride=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2)
        )
        self.path2 = nn.Sequential(
            components.Conv2dSame(in_channels=1, out_channels=32, kernel_size=(
                1, 9), stride=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2)
        )
        self.path3 = nn.Sequential(
            components.Conv2dSame(in_channels=1, out_channels=32, kernel_size=(
                3, 3), stride=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2)
        )
        self.conv_extractor = components._get_MultiConv2d_extractor(
            in_channels=32*3,
            out_channels=[64, 96,  128, 160]
        )
        self.gap = components.Conv2dLayerBlock(in_channels=160, out_channels=320, kernel_size=[
                                               1, 1], stride=1, bias=False, layer_norm=nn.BatchNorm2d(320), layer_poolling=nn.AdaptiveAvgPool2d(1), activate_func=nn.functional.relu)

        self.dropout = nn.Dropout(0.3)
        self.last_linear = nn.Linear(in_features=320, out_features=4)

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


class LightResMultiSerNet(nn.Module):
    """
    paper: Light-SERNet: A lightweight fully convolutional neural network for speech emotion recognition
    """

    def __init__(self) -> None:
        super().__init__()
        self.path1 = nn.Sequential(
            components.Conv2dSame(in_channels=1, out_channels=32, kernel_size=(
                11, 1), stride=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2)
        )
        self.path2 = nn.Sequential(
            components.Conv2dSame(in_channels=1, out_channels=32, kernel_size=(
                1, 9), stride=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2)
        )
        self.path3 = nn.Sequential(
            components.Conv2dSame(in_channels=1, out_channels=32, kernel_size=(
                3, 3), stride=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2)
        )
        self.conv_extractor = components._get_ResMultiConv2d_extractor(
            in_channels=32*3,
            out_channels=[64, 96,  128, 160]
        )
        self.gap = components.Conv2dLayerBlock(in_channels=160, out_channels=320, kernel_size=[
                                               1, 1], stride=1, bias=False, layer_norm=nn.BatchNorm2d(320), layer_poolling=nn.AdaptiveAvgPool2d(1), activate_func=nn.functional.relu)
        self.dropout = nn.Dropout(0.3)
        self.last_linear = nn.Linear(in_features=320, out_features=4)

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


class LightSerNet_Change(nn.Module):
    """
    """

    def __init__(self, config,) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=40, hidden_size=144,
                            num_layers=1, bidirectional=False, bias=False, batch_first=True)
        self.path1 = nn.Sequential(
            components.Conv2dSame(in_channels=1, out_channels=16, kernel_size=(
                11, 1), stride=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2)
        )
        self.path2 = nn.Sequential(
            components.Conv2dSame(in_channels=1, out_channels=16, kernel_size=(
                1, 9), stride=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2)
        )
        self.path3 = nn.Sequential(
            components.Conv2dSame(in_channels=1, out_channels=16, kernel_size=(
                3, 3), stride=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2)
        )
        self.conv_extractor = components._get_Conv2d_extractor(
            in_channels=16*3,
            shapes=[  # [channel:int, kernel:list|int, stride:list|int, norm:str|None, pool:str|None, pool_kernel:list|int]
                [64, [3, 3], 1, "bn", "ap", [2, 2]],
                [80, [3, 3], 1, "bn", "ap", [2, 2]],
                [96, [3, 3], 1, "bn", "ap", [2, 1]],
                [112, [3, 3], 1, "bn", "ap", [2, 1]],
                [128, [3, 3], 1, "bn", "ap", [2, 1]],
                [144, [1, 1], 1, "bn", "gap", 1]
            ],
            bias=False
        )
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(in_features=288, out_features=4)

    def forward(self, x: Tensor) -> Tensor:
        # [batch, seq, feature]
        x, (h, c) = self.lstm(x)
        # [batch, seq, hidden_size]
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        out1 = self.path1(x)
        out2 = self.path2(x)
        out3 = self.path3(x)

        x = torch.cat([out1, out2, out3], dim=1)

        x = self.conv_extractor(x).squeeze(2).squeeze(2)
        x = torch.cat([x, h.squeeze(0)], dim=1)
        x = self.dropout(x)
        x = self.classifier(x)

        return F.softmax(x, dim=1)
