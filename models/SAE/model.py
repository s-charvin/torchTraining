from . import components
from typing import Optional, Tuple, List
import torch
from torch import Tensor, nn
from torch.nn import Module
import torch.nn.functional as F


class STLSER(Module):
    """
    paper: A Novel end-to-end Speech Emotion Recognition Network with Stacked Transformer Layers
    """

    def __init__(self, config):
        super().__init__()

        norm_mode = config["STLSER"]["conv2d_extractor"]["norm_mode"]
        pool_mode = config["STLSER"]["conv2d_extractor"]["pool_mode"]
        in_channels = config["STLSER"]["conv2d_extractor"]["in_channels"]
        shapes = config["STLSER"]["conv2d_extractor"]["shapes"]
        conv_bias = config["STLSER"]["conv2d_extractor"]["bias"]
        feature_dim = config["STLSER"]["lstm"]["feature_dim"]
        hidden_size = config["STLSER"]["lstm"]["hidden_size"]
        num_layers = config["STLSER"]["lstm"]["num_layers"]
        lstm_bias = config["STLSER"]["lstm"]["bias"]
        lstm_dropout = config["STLSER"]["lstm"]["dropout"]
        bidirectional = config["STLSER"]["lstm"]["bidirectional"]
        if bidirectional:
            bi_num = 2
        else:
            bi_num = 1

        embed_dim = hidden_size * bi_num
        num_heads = config["STLSER"]["stl"]["num_heads"]
        layers = config["STLSER"]["stl"]["layers"]

        fcdim = config["STLSER"]["fc"]
        num_outputs = config["model"]["num_outputs"]

        self.extractor = components._get_Conv2d_extractor(norm_mode=norm_mode,
                                                          pool_mode=pool_mode, in_channels=in_channels, shapes=shapes, bias=conv_bias)
        self.bilstm = nn.LSTM(input_size=feature_dim, hidden_size=hidden_size, num_layers=num_layers,
                              bias=lstm_bias, batch_first=True, dropout=lstm_dropout, bidirectional=bidirectional)

        self.STL = components.SAEncoder(
            embed_dim=embed_dim, num_heads=num_heads, layers=layers)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(in_features=embed_dim, out_features=fcdim)
        self.fc2 = nn.Linear(in_features=fcdim, out_features=num_outputs)

    def forward(self, x: Tensor):
        if x.ndim == 3:
            x = x.unsqueeze(1)
        # [batch, 1 , frames, feature]
        x = self.extractor(x)
        # [batch, channel , out_frames, out_feature]
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        self.bilstm.flatten_parameters()
        # [batch, out_frames, channel*out_feature]
        x, _ = self.bilstm(x)
        # [batch, out_frames, hidden*bidirection]
        x = x.permute(1, 0, 2)
        # [out_frames, batch, hidden*bidirection]
        x = self.STL(x)
        # [out_frames, batch, hidden]
        x = x.permute(1, 2, 0)
        # [batch, out_frames, hidden*bidirection]
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        # [batch, out_frames, hidden]
        x = self.fc2(x)
        # [batch, out_frames, num_outputs]
        return x
