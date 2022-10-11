from . import components
import torch
from torch import Tensor, nn

import torch.nn.functional as F


class AAA_CLFC(nn.Module):
    """
    """

    def __init__(self, config,) -> None:
        super().__init__()

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        norm_mode = config["AAA_CLFC"]["conv2d_extractor"]["norm_mode"]
        pool_mode = config["AAA_CLFC"]["conv2d_extractor"]["pool_mode"]
        in_channels = config["AAA_CLFC"]["conv2d_extractor"]["in_channels"]
        shapes = config["AAA_CLFC"]["conv2d_extractor"]["shapes"]
        conv_bias = config["AAA_CLFC"]["conv2d_extractor"]["bias"]

        feature_dim = config["AAA_CLFC"]["lstm"]["feature_dim"]
        hidden_size = config["AAA_CLFC"]["lstm"]["hidden_size"]
        num_layers = config["AAA_CLFC"]["lstm"]["num_layers"]
        lstm_bias = config["AAA_CLFC"]["lstm"]["bias"]
        lstm_dropout = config["AAA_CLFC"]["lstm"]["dropout"]
        bidirectional = config["AAA_CLFC"]["lstm"]["bidirectional"]
        if bidirectional:
            bi_num = 2
        else:
            bi_num = 1
        embed_dim = hidden_size * bi_num
        num_heads = config["AAA_CLFC"]["attention"]["num_heads"]
        att_dropout = config["AAA_CLFC"]["attention"]["dropout"]

        fcdim = config["AAA_CLFC"]["fc"]
        num_outputs = config["model"]["num_outputs"]
        self.extractor = components._get_Conv2d_extractor(norm_mode=norm_mode,
                                                          pool_mode=pool_mode, in_channels=in_channels, shapes=shapes, bias=conv_bias)
        self.bilstm = nn.LSTM(input_size=feature_dim, hidden_size=hidden_size, num_layers=num_layers,
                              bias=lstm_bias, batch_first=True, dropout=lstm_dropout, bidirectional=bidirectional)
        self.headattention = components.SelfAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=att_dropout)
        self.fc1 = nn.Linear(in_features=embed_dim, out_features=fcdim)
        self.fc2 = nn.Linear(in_features=fcdim, out_features=num_outputs)

    def forward(self, x: Tensor) -> Tensor:
        x = self.extractor(x)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        self.bilstm.flatten_parameters()
        x, _ = self.bilstm(x)
        x = self.headattention(x)
        x = x.flatten()
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class CLFC(nn.Module):
    """ 
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        norm_mode = config["AAA_CLFC"]["conv2d_extractor"]["norm_mode"]
        pool_mode = config["AAA_CLFC"]["conv2d_extractor"]["pool_mode"]
        in_channels = config["AAA_CLFC"]["conv2d_extractor"]["in_channels"]
        shapes = config["AAA_CLFC"]["conv2d_extractor"]["shapes"]
        conv_bias = config["AAA_CLFC"]["conv2d_extractor"]["bias"]
        # feature_dim = shapes[-1][0]
        feature_dim = config["AAA_CLFC"]["lstm"]["feature_dim"]
        hidden_size = config["AAA_CLFC"]["lstm"]["hidden_size"]
        num_layers = config["AAA_CLFC"]["lstm"]["num_layers"]
        lstm_bias = config["AAA_CLFC"]["lstm"]["bias"]
        lstm_dropout = config["AAA_CLFC"]["lstm"]["dropout"]
        bidirectional = config["AAA_CLFC"]["lstm"]["bidirectional"]
        if bidirectional:
            bi_num = 2
        else:
            bi_num = 1
        embed_dim = hidden_size * bi_num
        num_heads = config["AAA_CLFC"]["attention"]["num_heads"]
        att_dropout = config["AAA_CLFC"]["attention"]["dropout"]

        fcdim = config["AAA_CLFC"]["fc"]
        num_outputs = config["model"]["num_outputs"]

        self.extractor = components._get_Conv2d_extractor(norm_mode=norm_mode,
                                                          pool_mode=pool_mode, in_channels=in_channels, shapes=shapes, bias=conv_bias)
        self.bilstm = nn.LSTM(input_size=feature_dim, hidden_size=hidden_size, num_layers=num_layers,
                              bias=lstm_bias, batch_first=True, dropout=lstm_dropout, bidirectional=bidirectional)
        self.headattention = components.SelfAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=att_dropout)
        self.fc1 = nn.Linear(in_features=embed_dim, out_features=fcdim)
        self.fc2 = nn.Linear(in_features=fcdim, out_features=num_outputs)

    def forward(self, x: Tensor) -> Tensor:
        # [batch, in_frames, feature]
        if x.ndim == 3:
            x = x.unsqueeze(1)
        x = self.extractor(x)  # [batch, out_frames, out_feature]
        x = torch.flatten(x.permute(0, 2, 1, 3), 2)
        self.bilstm.flatten_parameters()
        x, _ = self.bilstm(x)
        x = self.headattention(x)
        x = self.fc1(x)
        x = F.adaptive_avg_pool1d(x.permute(0, 2, 1), 1)
        x = self.fc2(x.squeeze())
        # x = x.sum(dim=1)
        return x
