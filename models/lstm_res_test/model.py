from . import components
import torch
from torch import Tensor, nn


class LstmResTest(nn.Module):
    """ 
    """

    def __init__(self, config,) -> None:
        super().__init__()

        self.input_size = config["lstm_res"]["lstm"]["feature_dim"]
        self.bidirectional = config["lstm_res"]["lstm"]["bidirectional"]

        self.bias = config["lstm_res"]["lstm"]["bias"]
        num_directions = 2 if self.bidirectional else 1
        self.lstm1_hiddens_num = config["lstm_res"]["lstm"]["lstm1_hiddens_num"]
        self.lstm1_layers_num = config["lstm_res"]["lstm"]["lstm1_layers_num"]
        self.lstm2_hiddens_num = config["lstm_res"]["lstm"]["lstm2_hiddens_num"]
        self.lstm2_layers_num = config["lstm_res"]["lstm"]["lstm2_layers_num"]

        # [seq_len, batch, hidden_size]-> [seq_len, batch, hidden_size * num_directions]
        self.lstm1 = components.DiyLSTM(input_size=self.input_size, hidden_size=self.lstm1_hiddens_num,
                                        num_layers=self.lstm1_layers_num, bias=self.bias, bidirectional=self.bidirectional)

        self.lstm2 = components.DiyLSTM(input_size=self.lstm1_hiddens_num * num_directions, hidden_size=self.lstm2_hiddens_num,
                                        num_layers=self.lstm2_layers_num, bias=self.bias, bidirectional=self.bidirectional)
        # [batch, channel=1, seq_len, hidden_size]- > [[batch, channel, _, _]]
        self.resnet = components.ResNet(config)

    def forward(self, x: Tensor) -> Tensor:
        if len(x.shape) == 4:
            x = x.squeeze(1)
        x = x.permute(1, 0, 2)
        # self.lstm1.flatten_parameters()
        x, (_, _) = self.lstm1(x)
        # self.lstm2.flatten_parameters()
        x, (_, _) = self.lstm2(x)
        x = self.resnet(x.permute(1, 0, 2))
        # # 输出结构
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)
        return x
