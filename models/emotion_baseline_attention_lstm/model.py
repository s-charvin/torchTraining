from . import components
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from typing import Tuple, Optional
import torch
from torch import Tensor, nn
from torch.nn import Parameter


class EmoAttention_Lstm(nn.Module):
    """
    Args:

    paper:Y. Xie, R. Liang, Z. Liang, C. Huang, C. Zou and B. Schuller,
          "Speech Emotion Classification Using Attention-Based LSTM,"
          in IEEE/ACM Transactions on Audio, Speech, and Language Processing,
          vol. 27, no. 11, pp. 1675-1685, Nov. 2019, doi: 10.1109/TASLP.2019.2925934.
    """

    def __init__(self, config,) -> None:
        super().__init__()
        self.input_size = config["EmoAttention_Lstm"]["feature_dim"]
        self.bias = config["EmoAttention_Lstm"]["bias"]
        self.bidirectional = config["EmoAttention_Lstm"]["bidirectional"]
        num_directions = 2 if self.bidirectional else 1

        self.lstm1_hiddens_num = config["EmoAttention_Lstm"]["lstm1_hiddens_num"]
        self.lstm1_layers_num = config["EmoAttention_Lstm"]["lstm1_layers_num"]
        self.lstm2_hiddens_num = config["EmoAttention_Lstm"]["lstm2_hiddens_num"]
        self.lstm2_layers_num = config["EmoAttention_Lstm"]["lstm2_layers_num"]
        self.linear_hiddens_num = config["EmoAttention_Lstm"]["linear_hiddens_num"]

        aux_num = config["EmoAttention_Lstm"]["aux_num"]
        if aux_num == "None":
            self.num_outputs = config["model"]["num_outputs"]
        else:
            self.num_outputs = aux_num

        self.lstm1 = components.LSTM(input_size=self.input_size, hidden_size=self.lstm1_hiddens_num,
                                     num_layers=self.lstm1_layers_num, bias=self.bias, bidirectional=self.bidirectional)

        self.lstm2 = components.LSTM(input_size=self.lstm1_hiddens_num * num_directions, hidden_size=self.lstm2_hiddens_num,
                                     num_layers=self.lstm2_layers_num, bias=self.bias, bidirectional=self.bidirectional)

        self.linear = nn.Linear(self.lstm2_hiddens_num * num_directions *
                                2, self.linear_hiddens_num, bias=self.bias)
        self.out = nn.Linear(self.linear_hiddens_num,
                             self.num_outputs, bias=self.bias)

        self.W_T = Parameter(torch.Tensor(
            self.lstm2_hiddens_num * num_directions, self.lstm2_hiddens_num * num_directions))
        self.W_F = Parameter(torch.Tensor(
            self.lstm2_hiddens_num * num_directions, self.lstm2_hiddens_num * num_directions))
        self.V_F = Parameter(torch.Tensor(
            self.lstm2_hiddens_num * num_directions, self.lstm2_hiddens_num * num_directions))

    def forward(self, input: Tensor, hidden: Optional[Tuple[Tensor, Tensor]] = None):
        # input: [batch_size, seq_len, hidden_size]
        # output: [seq_len, batch, hidden_size * num_directions]
        # hn,cn: [num_layers*num_directions, batch, hidden_size]
        output, (_, _) = self.lstm1(input.permute(1, 0, 2), hidden)
        output, (_, _) = self.lstm2(output, None)
        # [batch, seq_len, hidden_size * num_directions]
        output = output.permute(1, 0, 2)

        # [batch,seq_len, hidden_size * num_directions]
        s_t_ = F.linear(output, self.W_T, None)
        # [batch, 1 , hidden_size * num_directions]
        output_maxtime = output[:, -1, :].unsqueeze(1)
        # [batch,1, seq_len]
        s_t = torch.softmax(torch.matmul(
            output_maxtime, s_t_.permute(0, 2, 1)), dim=2)
        # [batch, 1 , hidden_size * num_directions]
        output_t = torch.matmul(s_t, output)

        # [batch, seq_len, hidden_size * num_directions]
        s_f = torch.softmax(
            F.linear(torch.tanh(F.linear(output, self.W_F, None)), self.V_F, None), dim=2)
        #  [batch, 1, hidden_size * num_directions]
        output_f = torch.sum(s_f * output, dim=1).unsqueeze(1)

        # [batch, 1 , hidden_size * num_directions*2]
        output = torch.cat([output_t, output_f], dim=2)
        # [batch, hidden_size]
        output = self.linear(output.squeeze(1))
        # [batch, class]
        output = self.out(output)
        return output
