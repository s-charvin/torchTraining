import torch
from torch import Tensor, nn
from typing import Tuple, Optional
import torch.nn.functional as F
from .components_all import _get_Conv2d_extractor, Conv2d_same


class LightSerNet_Multi(nn.Module):
    """
    paper: Light-SERNet: A lightweight fully convolutional neural network for speech emotion recognition
    """

    def __init__(self, config,) -> None:
        super().__init__()
        in_channels = config["LCANet"]["in_channels"]
        out_channels = config["LCANet"]["MultiSer"]["out_channels"]
        self.path1 = nn.Sequential(
            Conv2d_same(in_channels=in_channels, out_channels=out_channels, kernel_size=(
                11, 1), stride=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2)
        )
        self.path2 = nn.Sequential(
            Conv2d_same(in_channels=in_channels, out_channels=out_channels, kernel_size=(
                1, 9), stride=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2)
        )
        self.path3 = nn.Sequential(
            Conv2d_same(in_channels=in_channels, out_channels=out_channels, kernel_size=(
                3, 3), stride=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2)
        )
        convshape = config["LCANet"]["MultiSer"]["convshape"]
        self.conv_extractor = _get_Conv2d_extractor(
            in_channels=32*3,
            # [channel:int, kernel:list|int, stride:list|int, norm:str|None, pool:str|None, pool_kernel:list|int]
            shapes=convshape,
            bias=False,
            padding="same"
        )
        dropout = config["LCANet"]["MultiSer"]["dropout"]
        self.dropout = nn.Dropout(dropout)
        out_features = config["model"]["num_outputs"]
        self.classifier = nn.Linear(
            in_features=convshape[-1][0], out_features=out_features)

    def forward(self, x: Tensor, hide) -> Tensor:
        if len(x.shape) == 3:
            x = x.unsqueeze(1)

        out1 = self.path1(x)
        out2 = self.path2(x)
        out3 = self.path3(x)
        # [batch, channel, seq, fea]
        out = torch.cat([out1, out2, out3], dim=1)
        # [batch, channel*3, seq, fea]
        out = torch.cat([out, hide], dim=-1)
        # [batch, channel*3, seq, fea*2]
        out = self.conv_extractor(out).squeeze(2).squeeze(2)

        out = self.dropout(out)
        out = self.classifier(out)

        return F.softmax(out, dim=1), out


class LightSerNet_NP(nn.Module):
    """
    paper: Light-SERNet: A lightweight fully convolutional neural network for speech emotion recognition
    """

    def __init__(self, config,) -> None:
        super().__init__()
        in_channels = config["LCANet"]["in_channels"]
        out_channels = config["LCANet"]["NPSer"]["out_channels"]
        self.path1 = nn.Sequential(
            Conv2d_same(in_channels=in_channels, out_channels=out_channels, kernel_size=(
                11, 1), stride=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2)
        )
        self.path2 = nn.Sequential(
            Conv2d_same(in_channels=in_channels, out_channels=out_channels, kernel_size=(
                1, 9), stride=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2)
        )
        self.path3 = nn.Sequential(
            Conv2d_same(in_channels=in_channels, out_channels=out_channels, kernel_size=(
                3, 3), stride=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2)
        )
        convshape = config["LCANet"]["NPSer"]["convshape"]
        self.conv_extractor = _get_Conv2d_extractor(
            in_channels=32*3,
            shapes=convshape,
            bias=False
        )
        dropout = config["LCANet"]["NPSer"]["dropout"]
        self.dropout = nn.Dropout(dropout)
        out_features = config["LCANet"]["NPSer"]["num_outputs"]
        self.classifier = nn.Linear(
            in_features=convshape[-1][0], out_features=out_features)

    def forward(self, x: Tensor) -> Tensor:
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        out1 = self.path1(x)
        out2 = self.path2(x)
        out3 = self.path3(x)

        x = torch.cat([out1, out2, out3], dim=1)

        x = self.conv_extractor(x).squeeze(2).squeeze(2)
        hide = self.dropout(x)
        x = self.classifier(hide)

        return F.softmax(x, dim=1), hide
