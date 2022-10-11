from typing import Type, Any, Callable, Union, List, Optional, Tuple, Iterable
import math
import torch
from torch import Tensor, nn
from torch.nn import Parameter
from torch.autograd import Variable
import torch.nn.functional as F
# import torchaudio
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
from transformers import Wav2Vec2Processor, Wav2Vec2Model

all = ["BiLSTM_Attention", ]

# LSTMCell 结构变体


class LSTMCell(nn.LSTMCell):
    """ 自实现的LSTMCell结构, 能够进行单次输入的计算.

    Args:
        input_size (int):
        hidden_size (int):
        bias (bool, optional): . Defaults to True.
    Math::
        \begin{array}{ll} \\
            i_t = \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi}) \\
            f_t = \sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf}) \\
            g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg}) \\
            o_t = \sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho}) \\
            c_t = f_t \odot c_{t-1} + i_t \odot g_t \\
            h_t = o_t \odot \tanh(c_t) \\
        \end{array}
    """

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True) -> None:
        super().__init__(input_size, hidden_size, bias)

    def forward(self, input, hidden=None):
        self.check_forward_input(input)
        if hidden is None:
            hx = input.new_zeros(input.size(
                0), self.hidden_size, dtype=input.dtype, device=input.device)
            cx = input.new_zeros(input.size(
                0), self.hidden_size, dtype=input.dtype, device=input.device)
        else:
            hx, cx = hidden

        self.check_forward_hidden(input, hx, '[0]')
        self.check_forward_hidden(input, cx, '[1]')

        gates = F.linear(input, self.weight_ih, self.bias_ih) \
            + F.linear(hx, self.weight_hh, self.bias_hh)

        i_, f_, g_, o_ = gates.chunk(4, 1)  # 沿1轴分为4块
        i = torch.sigmoid(i_)
        f = torch.sigmoid(f_)
        g = torch.tanh(g_)
        o = torch.sigmoid(o_)
        cy = (f * cx) + (i * g)
        hy = o * torch.tanh(cy)
        return hy, cy


class AttLSTMCell(nn.Module):
    """ 自实现的AttLSTMCell结构, 能够进行单次输入的计算.
        通过将遗忘门替换成了Attention结构,
        用长时记忆 C 通过 Attention 思想,
        计算出自身需要重点关注的部分, 以及其他不需要重点关注的部分.

    Args:
        input_size (int):
        hidden_size (int):
        bias (bool, optional): . Defaults to True.

    Paper: Speech Emotion Classification Using Attention-Based LSTM

    Math::

        \begin{array}{ll}
        c` = f * c + (1-f) * g \\
        g = \tanh(W_{xg} x + b_{xg} + W_{hg} h + b_{hg}) \\

        f = \sigma( W_{ff} \tanh(W_{cf} c + b_{cf}))  \\

        h' = o * \tanh(c') \\
        o = \sigma(W_{xo} x + b_{xo} + W_{ho} h + b_{ho}) \\

        \end{array}

        """

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True) -> None:
        super(AttLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.W_xg = Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hg = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.W_ff = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.W_cf = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.W_xo = Parameter(torch.Tensor(hidden_size, input_size))
        self.W_ho = Parameter(torch.Tensor(hidden_size, hidden_size))

        if bias:
            self.b_xg = Parameter(torch.Tensor(hidden_size))
            self.b_hg = Parameter(torch.Tensor(hidden_size))

            self.b_cf = Parameter(torch.Tensor(hidden_size))
            self.b_xo = Parameter(torch.Tensor(hidden_size))
            self.b_ho = Parameter(torch.Tensor(hidden_size))
        else:
            self.register_parameter('b_xg', None)
            self.register_parameter('b_hg', None)

            self.register_parameter('b_cf', None)
            self.register_parameter('b_xo', None)
            self.register_parameter('b_ho', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def check_forward_input(self, input: Tensor) -> None:
        if input.size(1) != self.input_size:
            raise RuntimeError(
                "input has inconsistent input_size: got {}, expected {}".format(
                    input.size(1), self.input_size))

    def check_forward_hidden(self, input: Tensor, hx: Tensor, hidden_label: str = '') -> None:
        if input.size(0) != hx.size(0):
            raise RuntimeError(
                "Input batch size {} doesn't match hidden{} batch size {}".format(
                    input.size(0), hidden_label, hx.size(0)))

        if hx.size(1) != self.hidden_size:
            raise RuntimeError(
                "hidden{} has inconsistent hidden_size: got {}, expected {}".format(
                    hidden_label, hx.size(1), self.hidden_size))

    def forward(self, input: Tensor, hidden: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, Tensor]:
        self.check_forward_input(input)
        if hidden is None:
            hx = input.new_zeros(input.size(
                0), self.hidden_size, dtype=input.dtype, device=input.device)
            cx = input.new_zeros(input.size(
                0), self.hidden_size, dtype=input.dtype, device=input.device)
        else:
            hx, cx = hidden
        self.check_forward_hidden(input, hx, '[0]')
        self.check_forward_hidden(input, cx, '[1]')

        f = torch.sigmoid(
            F.linear(torch.tanh(F.linear(cx, self.W_cf, self.b_cf)), self.W_ff, None))
        g = torch.tanh(F.linear(input, self.W_xg, self.b_xg) +
                       F.linear(hx, self.W_hg, self.b_hg))
        cy = f * cx + (1-f) * g
        o = torch.sigmoid(F.linear(input, self.W_xo, self.b_xo) +
                          F.linear(hx, self.W_ho, self.b_ho))
        hy = o * torch.tanh(cy)

        return hy, cy


class LayerNormLSTMCell(nn.LSTMCell):
    """ 通过在隐藏层向量和输出向量的计算过程中
        添加LayerNorm层,对数据进行Norm操作,优化LSTM结构

    Args:
        input_size (_type_):
        hidden_size (_type_):
        dropout (float, optional): . Defaults to 0.0.
        bias (bool, optional): . Defaults to True.
        use_layer_norm (bool, optional): . Defaults to True.
    """

    def __init__(self, input_size, hidden_size, dropout=0.0, bias=True):
        super().__init__(input_size, hidden_size, bias)

        # 对输入x进行运算, 得到的Wih, 在本次计算前进行LayerNorm
        self.ln_ih = nn.LayerNorm(4 * hidden_size)
        # 对中间隐藏层h进行运算的Whh, 在本次计算前进行LayerNorm
        self.ln_hh = nn.LayerNorm(4 * hidden_size)
        # 隐藏层h运算
        self.ln_ho = nn.LayerNorm(hidden_size)
        # DropConnect on the recurrent hidden to hidden weight
        self.dropout = dropout

    def forward(self, input, hidden=None):
        self.check_forward_input(input)
        if hidden is None:
            hx = input.new_zeros(input.size(
                0), self.hidden_size, requires_grad=False)
            cx = input.new_zeros(input.size(
                0), self.hidden_size, requires_grad=False)
        else:
            hx, cx = hidden
        self.check_forward_hidden(input, hx, '[0]')
        self.check_forward_hidden(input, cx, '[1]')

        weight_hh = nn.functional.dropout(  # 隐藏层向量的计算权重添加dropout
            self.weight_hh, p=self.dropout, training=self.training)

        gates = self.ln_ih(F.linear(input, self.weight_ih, self.bias_ih)) \
            + self.ln_hh(F.linear(hx, weight_hh, self.bias_hh))

        i_, f_, g_, o_ = gates.chunk(4, 1)
        i = torch.sigmoid(i_)
        f = torch.sigmoid(f_)
        g = torch.tanh(g_)
        o = torch.sigmoid(o_)
        cy = (f * cx) + (i * g)
        hy = o * self.ln_ho(torch.tanh(cy))

        return hy, cy


class TestLSTMCell(nn.Module):
    r""" 通过Parameter自定义的LSTMCell,
         不过这里似乎有问题, 建议使用官方Cell或者LSTMCell


    .. math::

        \begin{array}{ll}
        c' = f * c + u \\

        t = \tanh(W_xt x + b_xt) + h

        f = \sigma( W_tf t+ W_cf c + b) \\

        u = \sigma( W_tu t+ W_cu c + b) t \\

        h' = \sigma( W_th t+ W_ch c' + b) c'\\

        \end{array}
        """

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True) -> None:
        super(TestLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.W_xt = Parameter(torch.Tensor(hidden_size, input_size))
        self.W_tf = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.W_cf = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.W_tu = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.W_cu = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.W_th = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.W_ch = Parameter(torch.Tensor(hidden_size, hidden_size))

        if bias:
            self.b_xt = Parameter(torch.Tensor(hidden_size))
            self.b_tf = Parameter(torch.Tensor(hidden_size))
            self.b_cf = Parameter(torch.Tensor(hidden_size))
            self.b_tu = Parameter(torch.Tensor(hidden_size))
            self.b_cu = Parameter(torch.Tensor(hidden_size))
            self.b_th = Parameter(torch.Tensor(hidden_size))
            self.b_ch = Parameter(torch.Tensor(hidden_size))
        else:
            self.register_parameter('b_xt', None)
            self.register_parameter('b_tf', None)
            self.register_parameter('b_cf', None)
            self.register_parameter('b_tu', None)
            self.register_parameter('b_cu', None)
            self.register_parameter('b_th', None)
            self.register_parameter('b_ch', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def check_forward_input(self, input: Tensor) -> None:
        if input.size(1) != self.input_size:
            raise RuntimeError(
                "input has inconsistent input_size: got {}, expected {}".format(
                    input.size(1), self.input_size))

    def check_forward_hidden(self, input: Tensor, hx: Tensor, hidden_label: str = '') -> None:
        if input.size(0) != hx.size(0):
            raise RuntimeError(
                "Input batch size {} doesn't match hidden{} batch size {}".format(
                    input.size(0), hidden_label, hx.size(0)))

        if hx.size(1) != self.hidden_size:
            raise RuntimeError(
                "hidden{} has inconsistent hidden_size: got {}, expected {}".format(
                    hidden_label, hx.size(1), self.hidden_size))

    def forward(self, input: Tensor, hidden: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, Tensor]:
        self.check_forward_input(input)
        if hidden is None:
            hx = input.new_zeros(input.size(
                0), self.hidden_size, dtype=input.dtype, device=input.device)
            cx = input.new_zeros(input.size(
                0), self.hidden_size, dtype=input.dtype, device=input.device)
        else:
            hx, cx = hidden
        self.check_forward_hidden(input, hx, '[0]')
        self.check_forward_hidden(input, cx, '[1]')

        t = torch.tanh(F.linear(input, self.W_xt, self.b_xt)) + hx
        f = torch.sigmoid(F.linear(t, self.W_tf, self.b_tf) +
                          F.linear(cx, self.W_cf, self.b_cf))
        u = torch.sigmoid(F.linear(t, self.W_tu, self.b_tu) +
                          F.linear(cx, self.W_cu, self.b_cu)) * t
        cy = torch.tanh(f * cx+u)
        hy = torch.tanh(torch.sigmoid(F.linear(t, self.W_th, self.b_th) +
                                      F.linear(cy, self.W_ch, self.b_ch)) * cy)

        return hy, cy

# LSTM 结构变体


class DiyLSTM(nn.Module):
    """ 自定义实现的LSTM结构, 可以通过更换Cell实现
        多种类型的LSTM, 并且支持多layer和双向结构

    Args:
        input_size (int): 
        hidden_size (int): 
        num_layers (int, optional): . Defaults to 1.
        bias (bool, optional): . Defaults to False.
        bidirectional (bool, optional): . Defaults to False.
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int = 1,
                 bias: bool = False,
                 bidirectional: bool = False,) -> None:

        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1

        self.forward_layer = nn.ModuleList(
            [
                AttLSTMCell(input_size=(input_size if layer == 0 else hidden_size * num_directions),
                            hidden_size=hidden_size, bias=bias)
                for layer in range(num_layers)
            ]
        )

        if self.bidirectional:
            self.backward_layer = nn.ModuleList(
                [
                    AttLSTMCell(input_size=(input_size if layer == 0 else hidden_size * num_directions),
                                hidden_size=hidden_size, bias=bias)
                    for layer in range(num_layers)
                ]
            )

    def check_hidden_size(self, hx: Tensor, expected_hidden_size: Tuple[int, int, int],
                          msg: str = 'Expected hidden size {}, got {}') -> None:
        if hx.size() != expected_hidden_size:
            raise RuntimeError(msg.format(
                expected_hidden_size, list(hx.size())))

    def check_forward_args(self, input: Tensor, hx: Tensor, cx: Tensor):

        # check_input

        expected_input_dim = 3
        if input.dim() != expected_input_dim:
            raise RuntimeError(
                'input must have {} dimensions, got {}'.format(
                    expected_input_dim, input.dim()))
        if self.input_size != input.size(-1):
            raise RuntimeError(
                'input.size(-1) must be equal to input_size. Expected {}, got {}'.format(
                    self.input_size, input.size(-1)))
        # check_hidden_size
        mini_batch = input.size(1)
        num_directions = 2 if self.bidirectional else 1
        expected_hidden_size = (
            self.num_layers * num_directions, mini_batch, self.hidden_size)
        self.check_hidden_size(hx, expected_hidden_size,
                               'Expected hx size {}, got {}')
        # check_cell_size
        expected_cell_size = (
            self.num_layers * num_directions, mini_batch, self.hidden_size)
        self.check_hidden_size(cx, expected_cell_size,
                               'Expected cx size {}, got {}')

    def copy_parameters(self, rnn_old):
        for param in rnn_old.named_parameters():
            name_ = param[0].split("_")
            layer = int(name_[2].replace("l", ""))
            sub_name = "_".join(name_[:2])
            if len(name_) > 3:
                self.backward_layer[layer].register_parameter(
                    sub_name, param[1])
            else:
                self.forward_layer[layer].register_parameter(
                    sub_name, param[1])

    def forward(self, input: Tensor, hidden: Optional[Tuple[Tensor, Tensor]] = None):
        # [seq_len, batch_size, hidden_size]
        # [num_layers * num_directions, batch, hidden_size]
        seq_len, batch_size, _ = input.size()

        num_directions = 2 if self.bidirectional else 1
        if hidden is None:

            hx = input.new_zeros(self.num_layers * num_directions,
                                 batch_size, self.hidden_size, requires_grad=False)
            # hx = torch.zeros(self.num_layers * num_directions,
            #                  batch_size, self.hidden_size,
            #                  dtype=input.dtype, device=input.device)
            cx = input.new_zeros(self.num_layers * num_directions,
                                 batch_size, self.hidden_size, requires_grad=False)
        else:
            hx, cx = hidden

        self.check_forward_args(input, hx, cx)

        ht = []  # [seq_len, num_layers*num_directions] , 保存
        for i in range(seq_len):
            ht.append([None] * (self.num_layers * num_directions))
        ct = []  # [seq_len, num_layers*num_directions]
        for i in range(seq_len):
            ct.append([None] * (self.num_layers * num_directions))

        if self.bidirectional:
            hy = []
            cy = []
            xs = input

            # [(f_layer, b_layer), ... ], 数量是层数
            for l, (f_layer, b_layer) in enumerate(zip(self.forward_layer, self.backward_layer)):  # l是第几层
                # 获取初始输入的第 l 层的前向和后向 (h, c) 向量所在位置
                f_l, b_l = 2 * l, 2 * l + 1  # [(0,1) (2,3), (4,5) ... ]
                # 获取初始输入的第 l 层的前向和后向 (h, c) 向量
                f_h, f_c, b_h, b_c = hx[f_l], cx[f_l], hx[b_l], cx[b_l]
                # [(正向输入, 反向输入), ... ], 数量是序列时间点长度
                for t, (f_x, b_x) in enumerate(zip(xs, reversed(xs))):
                    # forward
                    # 输入前向输入的时间 t 点得到的 (h, c)
                    # ht_, ct_: [batch, hidden_size]
                    ht_, ct_ = f_layer(f_x, (f_h, f_c))
                    # t 时间点, 前向向量所在位置更新值
                    # ht: --> [seq_len, num_layers*num_directions, batch, hidden_size]
                    ht[t][f_l] = ht_
                    # ct: --> [seq_len, num_layers*num_directions, batch, hidden_size]
                    ct[t][f_l] = ct_
                    # 在每次计算完后更新计算后的前向向量
                    # f_h, f_c: [batch, hidden_size]
                    f_h, f_c = ht[t][f_l], ct[t][f_l]

                    # backward
                    t = seq_len - 1 - t  # 反向时间
                    # ht_, ct_: [batch, hidden_size]
                    ht_, ct_ = b_layer(b_x, (b_h, b_c))
                    # t 时间点, 后向向量所在位置更新值
                    # ht: --> [seq_len, num_layers*num_directions, batch, hidden_size]
                    ht[t][b_l] = ht_
                    # ct: --> [seq_len, num_layers*num_directions, batch, hidden_size]
                    ct[t][b_l] = ct_
                    # 在每次计算完后更新计算后的后向向量
                    # b_h, b_c: [batch, hidden_size]
                    b_h, b_c = ht[t][b_l], ct[t][b_l]

                # 第l层，所有时间点，所有方向的特征
                # xs: [seq_len, batch, hidden_size * num_directions]
                xs = [
                    torch.cat((h[f_l], h[b_l]), dim=1) for h in ht
                ]

                # 第l层, 所有时间点, 所有方向, 特征向量
                # ht_temp: [seq_len, num_directions, batch, hidden_size]
                ht_temp = torch.stack(
                    [
                        # [num_layers*num_directions, batch, hidden_size]
                        torch.stack([h[f_l], h[b_l]]) for h in ht
                    ]
                )
                # ct_temp: [seq_len, num_directions, batch, hidden_size]
                ct_temp = torch.stack(
                    [
                        torch.stack([c[f_l], c[b_l]]) for c in ct
                    ]
                )

                if len(hy) == 0:  # 初始
                    hy = ht_temp[-1]
                else:
                    # [num_layers*num_directions, batch, hidden_size]
                    hy = torch.cat((hy, ht_temp[-1]), dim=0)
                if len(cy) == 0:
                    cy = ct_temp[-1]
                else:
                    cy = torch.cat((cy, ct_temp[-1]), dim=0)
            # [seq_len, batch, hidden_size * num_directions]
            y = torch.stack(xs)

        else:
            h, c = hx, cx  # [num_layers, batch, hidden_size]

            for t, x in enumerate(input):  # [0-seq_len]
                # [0, num_layers]
                for l, layer in enumerate(self.forward_layer):
                    ht_, ct_ = layer(x, (h[l], c[l]))  # [batch, hidden_size]
                    # [seq_len, num_layers]-->[seq_len, num_layers, batch, hidden_size]
                    ht[t][l] = ht_
                    ct[t][l] = ct_
                    x = ht[t][l]  # [batch, hidden_size]
                ht[t] = torch.stack(ht[t])
                ct[t] = torch.stack(ct[t])
                h, c = ht[t], ct[t]  # [num_layers, batch, hidden_size]
            # y: [seq_len, batch, hidden_size] 最后一层，所有时间点，所有方向的特征
            y = torch.stack(
                [
                    h[-1] for h in ht

                ]
            )
            #
            hy = torch.stack(list(ht[-1]))  # [num_layers, batch, hidden_size]
            cy = torch.stack(list(ct[-1]))
        return y, (hy, cy)


class LayerNormLSTM(nn.Module):
    """ 网络上找的自定义LSTM的参考, 其中有个mask操作还未迁移
        所以先占个位置.

    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 dropout=0.0,
                 weight_dropout=0.0,
                 bias=True,
                 bidirectional=False,
                 use_layer_norm=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # using variational dropout
        self.dropout = dropout
        self.bidirectional = bidirectional

        num_directions = 2 if bidirectional else 1
        self.hidden0 = nn.ModuleList([
            LayerNormLSTMCell(input_size=(input_size if layer == 0 else hidden_size * num_directions),
                              hidden_size=hidden_size, dropout=weight_dropout, bias=bias, use_layer_norm=use_layer_norm)
            for layer in range(num_layers)
        ])

        if self.bidirectional:
            self.hidden1 = nn.ModuleList([
                LayerNormLSTMCell(input_size=(input_size if layer == 0 else hidden_size * num_directions),
                                  hidden_size=hidden_size, dropout=weight_dropout, bias=bias, use_layer_norm=use_layer_norm)
                for layer in range(num_layers)
            ])

    def copy_parameters(self, rnn_old):
        for param in rnn_old.named_parameters():
            name_ = param[0].split("_")
            layer = int(name_[2].replace("l", ""))
            sub_name = "_".join(name_[:2])
            if len(name_) > 3:
                self.hidden1[layer].register_parameter(sub_name, param[1])
            else:
                self.hidden0[layer].register_parameter(sub_name, param[1])

    def forward(self, input, hidden=None, seq_lens=None):
        seq_len, batch_size, _ = input.size()
        num_directions = 2 if self.bidirectional else 1
        if hidden is None:
            hx = input.new_zeros(self.num_layers * num_directions,
                                 batch_size, self.hidden_size, requires_grad=False)
            cx = input.new_zeros(self.num_layers * num_directions,
                                 batch_size, self.hidden_size, requires_grad=False)
        else:
            hx, cx = hidden

        ht = []
        for i in range(seq_len):
            ht.append([None] * (self.num_layers * num_directions))
        ct = []
        for i in range(seq_len):
            ct.append([None] * (self.num_layers * num_directions))

        seq_len_mask = input.new_ones(
            batch_size, seq_len, self.hidden_size, requires_grad=False)
        if seq_lens != None:
            for i, l in enumerate(seq_lens):
                seq_len_mask[i, l:, :] = 0
        seq_len_mask = seq_len_mask.transpose(0, 1)

        if self.bidirectional:
            # if use cuda, change 'torch.LongTensor' to 'torch.cuda.LongTensor'
            indices_ = (torch.LongTensor(seq_lens) - 1).unsqueeze(1).unsqueeze(0).unsqueeze(0).repeat(
                [1, 1, 1, self.hidden_size])
            # if use cuda, change 'torch.LongTensor' to 'torch.cuda.LongTensor'
            indices_reverse = torch.LongTensor([0] * batch_size).unsqueeze(1).unsqueeze(0).unsqueeze(0).repeat(
                [1, 1, 1, self.hidden_size])
            indices = torch.cat((indices_, indices_reverse), dim=1)
            hy = []
            cy = []
            xs = input
            # Variational Dropout
            if not self.training or self.dropout == 0:
                dropout_mask = input.new_ones(
                    self.num_layers, 2, batch_size, self.hidden_size)
            else:
                dropout_mask = input.new(
                    self.num_layers, 2, batch_size, self.hidden_size).bernoulli_(1 - self.dropout)
                dropout_mask = Variable(
                    dropout_mask, requires_grad=False) / (1 - self.dropout)

            for l, (layer0, layer1) in enumerate(zip(self.hidden0, self.hidden1)):
                l0, l1 = 2 * l, 2 * l + 1
                h0, c0, h1, c1 = hx[l0], cx[l0], hx[l1], cx[l1]
                for t, (x0, x1) in enumerate(zip(xs, reversed(xs))):
                    ht_, ct_ = layer0(x0, (h0, c0))
                    ht[t][l0] = ht_ * seq_len_mask[t]
                    ct[t][l0] = ct_ * seq_len_mask[t]
                    h0, c0 = ht[t][l0], ct[t][l0]
                    t = seq_len - 1 - t
                    ht_, ct_ = layer1(x1, (h1, c1))
                    ht[t][l1] = ht_ * seq_len_mask[t]
                    ct[t][l1] = ct_ * seq_len_mask[t]
                    h1, c1 = ht[t][l1], ct[t][l1]

                xs = [torch.cat((h[l0]*dropout_mask[l][0], h[l1]
                                * dropout_mask[l][1]), dim=1) for h in ht]
                ht_temp = torch.stack(
                    [torch.stack([h[l0], h[l1]]) for h in ht])
                ct_temp = torch.stack(
                    [torch.stack([c[l0], c[l1]]) for c in ct])
                if len(hy) == 0:
                    hy = torch.stack(
                        list(ht_temp.gather(dim=0, index=indices).squeeze(0)))
                else:
                    hy = torch.cat(
                        (hy, torch.stack(list(ht_temp.gather(dim=0, index=indices).squeeze(0)))), dim=0)
                if len(cy) == 0:
                    cy = torch.stack(
                        list(ct_temp.gather(dim=0, index=indices).squeeze(0)))
                else:
                    cy = torch.cat(
                        (cy, torch.stack(list(ct_temp.gather(dim=0, index=indices).squeeze(0)))), dim=0)
            y = torch.stack(xs)
        else:
            # if use cuda, change 'torch.LongTensor' to 'torch.cuda.LongTensor'
            indices = (torch.LongTensor(seq_lens) - 1).unsqueeze(1).unsqueeze(0).unsqueeze(0).repeat(
                [1, self.num_layers, 1, self.hidden_size])
            h, c = hx, cx
            # Variational Dropout
            if not self.training or self.dropout == 0:
                dropout_mask = input.new_ones(
                    self.num_layers, batch_size, self.hidden_size)
            else:
                dropout_mask = input.new(
                    self.num_layers, batch_size, self.hidden_size).bernoulli_(1 - self.dropout)
                dropout_mask = Variable(
                    dropout_mask, requires_grad=False) / (1 - self.dropout)

            for t, x in enumerate(input):
                for l, layer in enumerate(self.hidden0):
                    ht_, ct_ = layer(x, (h[l], c[l]))
                    ht[t][l] = ht_ * seq_len_mask[t]
                    ct[t][l] = ct_ * seq_len_mask[t]
                    x = ht[t][l] * dropout_mask[l]
                ht[t] = torch.stack(ht[t])
                ct[t] = torch.stack(ct[t])
                h, c = ht[t], ct[t]
            y = torch.stack([h[-1]*dropout_mask[-1] for h in ht])
            hy = torch.stack(list(torch.stack(ht).gather(
                dim=0, index=indices).squeeze(0)))
            cy = torch.stack(list(torch.stack(ct).gather(
                dim=0, index=indices).squeeze(0)))

        return y, (hy, cy)


class BiLSTM_Attention(nn.Module):
    """ 利用最后一层的最后一个的隐藏层向量, 
        计算输出的Attention权重, 然后对输出进行处理

    Args:
        input_size (int): 
        hidden_size (int): 
        num_layers (int, optional): . Defaults to 1.
        bias (bool, optional): . Defaults to False.
        bidirectional (bool, optional): . Defaults to False.
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int = 1,
                 bias: bool = False,
                 bidirectional: bool = False,):

        super(BiLSTM_Attention, self).__init__()
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, bidirectional=bidirectional, bias=bias)

    # lstm_output : [batch_size, n_step, hidden_size * num_directions(=2)], F matrix

    def attention_net(self, lstm_output, final_state):
        # [num_layers(=1) * num_directions(=1), batch_size, hidden_size]
        hidden = final_state.view(-1, self.hidden_size *
                                  self.num_directions, self.num_layers)
        # hidden : [batch_size, hidden_size * num_directions(=1), 1(=n_layer)]

        # [batch_size, len_seq, num_directions * hidden_size] * [batch_size, hidden_size * num_directions, 1]
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(
            2)
        # attn_weights : [batch_size, len_seq]
        soft_attn_weights = F.softmax(attn_weights, 1)  # 对每个输入的所有序列点的权重, 即重视程度

        # [batch_size, hidden_size * num_directions, len_seq] * [batch_size, len_seq, 1]
        # 对每个seq点的特征乘以了一个权重
        context = torch.bmm(lstm_output.transpose(
            1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)

        # context : [batch_size, hidden_size * num_directions]
        return context, soft_attn_weights.data.numpy()

    def forward(self, X):
        # X : [batch_size, len_seq, input_size]
        X = X.permute(1, 0, 2)
        # X : [len_seq, batch_size, input_size]

        # [num_layers * num_directions, batch_size, hidden_size]
        hidden_state = torch.zeros(
            self.num_layers*self.num_directions, len(X), self.hidden_size)
        cell_state = torch.zeros(
            self.num_layers*self.num_directions, len(X), self.hidden_size)

        # final_hidden_state, final_cell_state : [num_layers * num_directions, batch_size, hidden_size]
        output, (final_hidden_state, final_cell_state) = self.lstm(
            input, (hidden_state, cell_state))

        output = output.permute(1, 0, 2)
        # output : [batch_size, len_seq, num_directions * hidden_size]
        attn_output, attention = self.attention_net(output, final_hidden_state)
        # model : [batch_size, num_classes], attention : [batch_size, n_step]
        return attn_output, attention


class LSTM_Attention(nn.Module):
    """ 添加了一个LSTM结构, 对LSTM最后一层得到的
        结果向量, 通过最后输出的状态向量
        计算每个序列点的权重系数, 即添加Attention结构.

    Args:
        input_size (int): 
        hidden_size (int): 
        num_layers (int, optional): . Defaults to 1.
        bias (bool, optional): . Defaults to True.
        bidirectional (bool, optional): . Defaults to False.
    Returns:
        attn_output (Tensor): 被 Attention 的输出
        attention (Tensor): Attention 过程中的权重系数
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int = 1,
                 bias: bool = True,
                 bidirectional: bool = False,
                 dropout: int = 0):
        super(LSTM_Attention, self).__init__()

        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, bidirectional=bidirectional, bias=bias, dropout=dropout)

    # lstm_output : [batch_size, n_step, hidden_size * num_directions(=2)], F matrix

    def attention_net(self, lstm_output, final_state):
        # [num_layers(=1) * num_directions(=1), batch_size, hidden_size]
        hidden = final_state.view(-1, self.hidden_size *
                                  self.num_directions, self.num_layers)
        # hidden : [batch_size, hidden_size * num_directions(=1), 1(=n_layer)]

        # [batch_size, len_seq, num_directions * hidden_size] * [batch_size, hidden_size * num_directions, 1]
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(
            2)
        # attn_weights : [batch_size, len_seq]
        soft_attn_weights = F.softmax(attn_weights, 1)  # 对每个输入的所有序列点的权重, 即重视程度

        # [batch_size, hidden_size * num_directions, len_seq] * [batch_size, len_seq, 1]
        # 对每个seq点的特征乘以了一个权重
        context = torch.bmm(lstm_output.transpose(
            1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)

        # context : [batch_size, hidden_size * num_directions]
        return context, soft_attn_weights.data.numpy()

    def forward(self, X):
        # X : [batch_size, len_seq, input_size]
        X = X.permute(1, 0, 2)
        # X : [len_seq, batch_size, input_size]

        # [num_layers * num_directions, batch_size, hidden_size]
        hidden_state = torch.zeros(
            self.num_layers*self.num_directions, len(X), self.hidden_size)
        cell_state = torch.zeros(
            self.num_layers*self.num_directions, len(X), self.hidden_size)

        # final_hidden_state, final_cell_state : [num_layers * num_directions, batch_size, hidden_size]
        output, (final_hidden_state, final_cell_state) = self.lstm(
            input, (hidden_state, cell_state))

        output = output.permute(1, 0, 2)
        # output : [batch_size, len_seq, num_directions * hidden_size]
        attn_output, attention = self.attention_net(output, final_hidden_state)
        # model : [batch_size, num_classes], attention : [batch_size, n_step]
        return attn_output, attention

# CNN 结构变体


class Conv2dLayerBlock(nn.Module):
    """ 单个2D卷积块, 可自定义norm ,activate, pool层
        并且支持"same"模式, 这在老板是不支持的
        Args:
            in_channels (int): 
            out_channels (int): 
            kernel_size (_type_): 
            stride (int): 
            bias (bool): 
            layer_norm (Optional[nn.Module]): 
            layer_poolling (Optional[nn.Module]): 
            activate_func (Optional[nn.Module]): . Defaults to Optional[nn.Module].
            padding (int, optional): . Defaults to 0.
        example:
            net = Conv2dLayerBlock(64,128,[3,3],[1,1],True,nn.BatchNorm2d,nn.MaxPool2d,ReLU,"same","zeros")
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride: int,
        bias: bool,
        layer_norm: Optional[nn.Module],
        layer_poolling: Optional[nn.Module],
        activate_func: Optional[nn.Module],
        padding=0,  # 最新版可用"same"模式, 老版本没有, 替代品为自定义的Conv2d_same
        padding_mode="zeros",
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        if padding == "same":
            self.conv = Conv2d_same(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,  # 步长
                bias=bias,  # 是否使用偏置数
                padding_mode=padding_mode
            )
        else:
            self.conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,  # 输出通道数,即使用几个卷积核
                kernel_size=kernel_size,
                stride=stride,  # 步长
                bias=bias,  # 是否使用偏置数
                padding=padding,
                padding_mode=padding_mode
            )
        self.layer_norm = layer_norm
        self.activate_func = activate_func
        self.leyer_poolling = layer_poolling

    def forward(self, x: Tensor,) -> Tensor:
        """
        Args:
            x (Tensor): Shape ``[batch, in_channels, H, W]``.
        Returns:
            Tensor: Shape ``[batch, out_channels, out_frames]``.
        """
        x = self.conv(x)
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        if self.activate_func is not None:
            x = self.activate_func(x)
        if self.leyer_poolling is not None:
            x = self.leyer_poolling(x)
        return x


class Conv2dFeatureExtractor(nn.Module):
    """构建连续的2D卷积网络结构
    Args:
        conv_layers (nn.ModuleList): 模型所需的网络结构列表
    """

    def __init__(self, conv_layers: nn.ModuleList,):
        super().__init__()
        self.conv_blocks = conv_layers

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor):
                shape: ``[batch,in_channels, frame, feature]``.
        Returns:
            Tensor:
                输出结果, shape: ``[batch,out_channels, frame, feature]``
        """
        # 检查参数
        if x.ndim != 4:
            raise ValueError(
                f"期望输入的维度为 4D (batch, channel, H, W), 但得到的输入数据维度为 {list(x.shape)}")
        for block in self.conv_blocks:
            x = block(x)
        return x


def _get_Conv2d_extractor(
        in_channels: int,
        # [channel, kernel, stride, norm, pool_mode, pool_kernel]
        shapes,
        bias: bool = False,
        padding="same",
) -> Conv2dFeatureExtractor:
    """ 返回一个由多层Conv2dBlock构成的特征提取器, 可以通过列表进行自定义结构.

    Args:
        in_channels (int): 
        shapes (Iterable[List[int, List[int] or int, List[int] or int, str, str, List[int] or int]]): 
            [out_channel, kernel, stride, norm_mode, pool_mode, pool_kernel]
            norm: ["gn", "ln", "bn", "None"]
            pool: ["mp", "ap", "gap", "None"]
        bias (bool): 

    Returns:
        Conv2dFeatureExtractor: 一个由Conv2d block组成的特征提取器
    """
    blocks = []
    in_channels = in_channels  # 输入数据的通道

    for i, (out_channels, kernel_size, stride, norm_mode, pool_mode, pool_kernel) in enumerate(shapes):

        # Norm设置
        assert norm_mode in ["gn", "ln", "bn", "None"]
        assert pool_mode in ["mp", "ap", "gap", "None"]
        if norm_mode == "gn":
            normalization = nn.GroupNorm(
                num_groups=out_channels,
                num_channels=out_channels,
                affine=True,
            )
        elif norm_mode == "ln":
            normalization = nn.LayerNorm(
                normalized_shape=out_channels,
                elementwise_affine=True,
            )
        elif norm_mode == "bn":
            normalization = nn.BatchNorm2d(out_channels)
        else:
            normalization = None

        if pool_mode == "mp":
            poolling = nn.MaxPool2d(pool_kernel)
        elif pool_mode == "ap":
            poolling = nn.AvgPool2d(pool_kernel)
        elif pool_mode == "gap":  # GlobalAveragePooling2D
            pool_kernel = 1
            poolling = nn.AdaptiveAvgPool2d(1)
        else:
            poolling = None
        # 卷积块设置
        blocks.append(
            Conv2dLayerBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                bias=bias,
                layer_norm=normalization,
                layer_poolling=poolling,
                activate_func=nn.ReLU(),
                padding=padding
            )
        )

        in_channels = out_channels  # 修改输入通道为 Extractor 的输出通道
    return Conv2dFeatureExtractor(nn.ModuleList(blocks))


class Conv2d_same(torch.nn.Conv2d):
    """ 等同于"same"模式的卷积, 因为引入了合适的展平操作,
        其得到的输出维度和输入维度相等.
    """

    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        # 计算输入维度应该填充的维度大小
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.size()[-2:]  # 获取输入维度

        pad_h = self.calc_same_pad(
            i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0])
        pad_w = self.calc_same_pad(
            i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1])

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2,
                    pad_h // 2, pad_h - pad_h // 2]
            )
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class Conv1dLayerBlock(nn.Module):
    """单个Conv1d块, 可自定义norm ,activate, pool层 
    Input:
        Tensor: Shape ``[batch, in_channels, in_frames]``.
    Returns:
        Tensor: Shape ``[batch, out_channels, out_frames]``.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size,
            stride: int,
            bias: bool,
            layer_norm: Optional[nn.Module],
            layer_poolling: Optional[nn.Module],
            activate_func=nn.functional.relu,
            padding=0,  # 最新版可用"same"模式, 老版本没有, 替代品为自定义的Conv2d_same
            padding_mode="zeros",
    ):

        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

        self.conv = nn.Conv1d(  # 卷积核是一维的，卷积操作是在[batch, in_channels, in_frame]第三维上进行, 覆盖全部channel。
            in_channels=in_channels,
            out_channels=out_channels,  # 输出通道数,即使用几个卷积核
            kernel_size=kernel_size,  # 竖着进行卷积的frame的数量
            stride=stride,  # 步长, 即frame跨越步数
            bias=bias,  # 是否使用偏置数
            padding=padding,
            padding_mode=padding_mode
        )

        self.layer_norm = layer_norm
        self.activate_func = activate_func
        self.leyer_poolling = layer_poolling

    def forward(
            self,
            x: Tensor,) -> Tuple[Tensor]:
        x = self.conv(x)
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        if self.activate_func is not None:
            x = self.activate_func(x)
        if self.leyer_poolling is not None:
            x = self.leyer_poolling(x)
        return x


class Conv1dFeatureExtractor(nn.Module):
    """构建连续的1D卷积网络结构
    Args:
        conv_layers (nn.ModuleList): 模型所需的网络结构列表
        batch_first(bool): .
    """

    def __init__(self, conv_layers: nn.ModuleList,):
        super().__init__()
        self.conv_blocks = conv_layers

    def forward(self, x: Tensor) -> Tensor:
        # 检查参数
        if x.ndim != 3:
            raise ValueError(
                f"期望输入的维度为 3D (batch,seqs, channels), 但得到的输入数据维度为 {list(x.shape)}")
        x = x.permute(0, 2, 1)  # [batch, channels, seqs]
        for block in self.conv_blocks:
            x = block(x)
        return x.permute(0, 2, 1)


def _get_Conv1d_extractor(
    in_channels,
    # [channel:int, kernel:list|int, stride:list|int, norm:str|None, pool:str|None, pool_kernel:list|int]
    shapes,
        bias: bool,) -> Conv1dFeatureExtractor:

    blocks = []
    in_channels = in_channels  # 输入数据的通道

    for i, (out_channels, kernel_size, stride, norm_mode, pool_mode, pool_kernel) in enumerate(shapes):

        # Norm设置
        assert norm_mode in ["gn", "ln", "bn", "None"]
        assert pool_mode in ["mp", "None"]

        if norm_mode == "gn":
            normalization = nn.GroupNorm(
                num_groups=out_channels,
                num_channels=out_channels,
                affine=True,
            )
        elif norm_mode == "ln":
            normalization = nn.LayerNorm(
                normalized_shape=out_channels,
                elementwise_affine=True,
            )
        elif norm_mode == "bn":
            normalization = nn.BatchNorm1d(out_channels)
        else:
            normalization = None

        if pool_mode == "mp":
            poolling = nn.MaxPool1d(pool_kernel)
        else:
            poolling = None
        # 卷积块设置
        blocks.append(
            Conv1dLayerBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                bias=bias,
                layer_norm=normalization,
                layer_poolling=poolling,
                activate_func=nn.functional.relu
            )
        )

        in_channels = out_channels  # 修改输入通道为 Extractor 的输出通道
    return Conv1dFeatureExtractor(nn.ModuleList(blocks))

# Resnet 结构


class ResNet_BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,  # conv2 的 stride
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,  # 固定值
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                "BasicBlock 仅支持 groups=1 或 base_width=64")
        if dilation > 1:
            raise NotImplementedError(
                "BasicBlock 不支持 Dilation > 1")
        # 当stride != 1时, self.conv1 和 self.downsample 都会对输入进行下采样操作, 维度发生变化
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,
                               stride=stride, padding=1, groups=1, bias=False, dilation=1,)

        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, groups=1, bias=False, dilation=1,)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet_Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:

        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # 当stride != 1时, self.conv2 和 self.downsample 都会对输入进行下采样操作, 维度发生变化
        self.conv1 = nn.Conv2d(
            inplanes, width, kernel_size=1, stride=1, bias=False)
        self.bn1 = norm_layer(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, groups=groups, bias=False, dilation=dilation,)
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(
            width, planes * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        res = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            res = self.downsample(x)

        out += res
        out = self.relu(out)

        return out


class ResNeXt_Bottleneck_C(nn.Module):
    """
    RexNeXt bottleneck type C (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)
    """

    def __init__(self, in_channels, out_channels, stride, cardinality, base_width, widen_factor):
        """ Constructor
        Args:
            in_channels: 输入通道维度
            out_channels: 输出通道维度
            stride: 卷积步长. Replaces pooling layer.
            cardinality: 卷积 group 数目.
            base_width: 每 group 平均通道数.
            widen_factor: 加宽因子, factor to reduce the input dimensionality before convolution.
        """
        super(ResNeXt_Bottleneck_C, self).__init__()

        # # 分组通道数, 根据不同输出通道数而变化
        # D = int(base_width * (out_channels / (widen_factor * 64.)))
        # C = cardinality  # 卷积分组数

        width_ratio = out_channels / (widen_factor * 64.)
        D = cardinality * int(base_width * width_ratio)
        self.conv_reduce = nn.Conv2d(
            in_channels, D, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_reduce = nn.BatchNorm2d(D)
        self.conv_conv = nn.Conv2d(
            D, D, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn = nn.BatchNorm2d(D)
        self.conv_expand = nn.Conv2d(
            D, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_expand = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        # 将输入映射到输出维度,方便做残差连接.
        # 显然只在第一层做
        if in_channels != out_channels:
            self.shortcut.add_module('shortcut_conv',
                                     nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0,
                                               bias=False))
            self.shortcut.add_module(
                'shortcut_bn', nn.BatchNorm2d(out_channels))

    def forward(self, x):
        bottleneck = self.conv_reduce.forward(x)
        bottleneck = F.relu(self.bn_reduce.forward(bottleneck), inplace=True)
        bottleneck = self.conv_conv.forward(bottleneck)
        bottleneck = F.relu(self.bn.forward(bottleneck), inplace=True)
        bottleneck = self.conv_expand.forward(bottleneck)
        bottleneck = self.bn_expand.forward(bottleneck)
        residual = self.shortcut.forward(x)
        return F.relu(residual + bottleneck, inplace=True)


def resnet18() -> ResNet:
    """ ResNet-18 
    paper: https://arxiv.org/pdf/1512.03385.pdf
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    # if weights is not None:
    #         model.load_state_dict(weights.get_state_dict(progress=progress))
    return model


def resnet34() -> ResNet:
    """ ResNet-34
    paper: https://arxiv.org/pdf/1512.03385.pdf
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3])
    # if weights is not None:
    #         model.load_state_dict(weights.get_state_dict(progress=progress))
    return model


def resnet50() -> ResNet:
    """ ResNet-50
    paper: https://arxiv.org/pdf/1512.03385.pdf
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    # if weights is not None:
    #         model.load_state_dict(weights.get_state_dict(progress=progress))
    return model


def resnet101() -> ResNet:
    """ ResNet-101
    paper: https://arxiv.org/pdf/1512.03385.pdf
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3])
    # if weights is not None:
    #         model.load_state_dict(weights.get_state_dict(progress=progress))
    return model


def resnet152() -> ResNet:
    """ ResNet-152
    paper: https://arxiv.org/pdf/1512.03385.pdf
    Args:

    """
    model = ResNet(Bottleneck, [3, 8, 36, 3])
    # if weights is not None:
    #         model.load_state_dict(weights.get_state_dict(progress=progress))
    return model


def resnext50_32x4d() -> ResNet:
    """ ResNeXt-50 32x4d
    paper: https://arxiv.org/abs/1611.05431
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], groups=32, width_per_group=4)
    # if weights is not None:
    #         model.load_state_dict(weights.get_state_dict(progress=progress))
    return model


def resnet101_32x8d() -> ResNet:
    """ ResNeXt-101 32x8d
    paper: https://arxiv.org/abs/1611.05431
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], groups=32, width_per_group=8)
    # if weights is not None:
    #         model.load_state_dict(weights.get_state_dict(progress=progress))
    return model


def resnext101_64x4d() -> ResNet:
    """ ResNeXt-101 64x4d
    paper: https://arxiv.org/abs/1611.05431
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], groups=64, width_per_group=4)
    # if weights is not None:
    #         model.load_state_dict(weights.get_state_dict(progress=progress))
    return model


def wide_resnet50_2() -> ResNet:
    """ Wide ResNet-50-2
    paper: https://arxiv.org/abs/1605.07146
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], width_per_group=64 * 2)
    # if weights is not None:
    #         model.load_state_dict(weights.get_state_dict(progress=progress))
    return model


def wide_resnet101_2() -> ResNet:
    """ Wide ResNet-101-2
    paper: https://arxiv.org/abs/1611.05431
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], width_per_group=64 * 2)
    # if weights is not None:
    #         model.load_state_dict(weights.get_state_dict(progress=progress))
    return model

# Attention 结构变体


class SelfAttention(nn.Module):
    """多头注意力机制 (Multihead Self Attention module )

    Args:
        embed_dim (int): Total dimension of the model.
        num_heads (int): The number of heads.
        dropout (float, optional):
            Dropout probabiliry on attn_output_weights. Default: ``0.0``
    """

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dropout: float = 0.0,):
        super().__init__()
        head_dim = embed_dim // num_heads  # 计算每个 Head 对应的 Q, K, V 的输出特征维度
        if head_dim * num_heads != embed_dim:
            raise ValueError(
                f"`输入嵌入向量为度 embed_dim({embed_dim})`不能整除 `num_heads ({num_heads})`")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = head_dim

        self.scaling = self.head_dim ** -0.5

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(
            self,
            x: Tensor,
            attention_mask: Optional[Tensor] = None,) -> Tensor:
        """
        Args:
            x (Tensor): shape: ``[batch_size, sequence_length, embed_dim]``.
            attention_mask (Tensor or None, optional):
                shape: ``[batch_size, 1, sequence_length, sequence_length]``

        Returns:
            Tensor: The resulting tensor. shape: ``[batch, sequence_length, embed_dim]``
        """
        # 检查参数
        if x.ndim != 3 or x.shape[2] != self.embed_dim:
            raise ValueError(
                f"期望输入维度大小为 (batch, sequence, embed_dim=={self.embed_dim}), 但得到的却是 {x.shape}."
            )
        batch_size, length, embed_dim = x.size()
        if attention_mask is not None:
            shape_ = (batch_size, 1, length, length)
            if attention_mask.size() != shape_:
                raise ValueError(
                    f"期望的 attention mask 维度大小为 {shape_}, 但得到的却是 {attention_mask.size()}."
                )

        shape = (batch_size, length, self.num_heads, self.head_dim)
        # [batch_size, num_heads, sequence_length, head_dim], 这里 num_heads*head_dim =embed_dim
        q = self.q_proj(x).view(*shape).transpose(2, 1)  # B, nH, L, Hd
        # [batch_size, num_heads, head_dim, sequence_length]
        k = self.k_proj(x).view(*shape).permute(0, 2, 3, 1)  # B, nH, Hd, L
        # [batch_size, num_heads, sequence_length, head_dim]
        v = self.v_proj(x).view(*shape).transpose(2, 1)  # B, nH, L, Hd

        # 求平均, 进行特征缩放
        weights = self.scaling * (q @ k)  # B, nH, L, L , 注意 @ = torch.matmul
        if attention_mask is not None:
            weights += attention_mask

        weights = torch.nn.functional.softmax(weights, dim=-1)
        weights = torch.nn.functional.dropout(
            weights, p=self.dropout, training=self.training)
        # [batch_size, num_heads, sequence_length, head_dim]
        output = weights @ v  # B, nH, L, Hd
        # [batch_size, sequence_length, embed_dim]
        output = output.transpose(2, 1).reshape(batch_size, length, embed_dim)

        output = self.out_proj(output)
        return output


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, attn_dropout=0.,
                 bias=True, add_bias_kv=False, add_zero_attn=False):
        super().__init__()
        # 在 SAE中，　还未迁移过来


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim, padding_idx=0, left_pad=0, init_size=128):
        super().__init__()
        # 在 SAE中，　还未迁移过来


class SAEncoder(nn.Module):

    def __init__(self, embed_dim, num_heads, layers, attn_dropout=0.0, relu_dropout=0.0, res_dropout=0.0,
                 embed_dropout=0.0, attn_mask=False):
        super().__init__()
        # 在 SAE中，　还未迁移过来


class TransformerEncoderLayer(nn.Module):

    def __init__(self, embed_dim, num_heads=4, attn_dropout=0.1, relu_dropout=0.1, res_dropout=0.1,
                 attn_mask=False):
        super().__init__()
        # 在 SAE中，　还未迁移过来

# wav2Vec
