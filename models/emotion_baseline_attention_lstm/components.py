from typing import Tuple, Optional
import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter


class LSTMCell(nn.LSTMCell):
    """
    .. math::
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
    r"""A long short-term memory (LSTM) cell.

    .. math::

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
    def __init__(self, input_size, hidden_size, dropout=0.0, bias=True, use_layer_norm=True):
        super().__init__(input_size, hidden_size, bias)
        self.use_layer_norm = use_layer_norm
        if self.use_layer_norm:
            self.ln_ih = nn.LayerNorm(4 * hidden_size)
            self.ln_hh = nn.LayerNorm(4 * hidden_size)
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

        weight_hh = nn.functional.dropout(
            self.weight_hh, p=self.dropout, training=self.training)
        if self.use_layer_norm:
            gates = self.ln_ih(F.linear(input, self.weight_ih, self.bias_ih)) \
                + self.ln_hh(F.linear(hx, weight_hh, self.bias_hh))
        else:
            gates = F.linear(input, self.weight_ih, self.bias_ih) \
                + F.linear(hx, weight_hh, self.bias_hh)

        i, f, c, o = gates.chunk(4, 1)
        i_ = torch.sigmoid(i)
        f_ = torch.sigmoid(f)
        c_ = torch.tanh(c)
        o_ = torch.sigmoid(o)
        cy = (f_ * cx) + (i_ * c_)
        if self.use_layer_norm:
            hy = o_ * self.ln_ho(torch.tanh(cy))
        else:
            hy = o_ * torch.tanh(cy)
        return hy, cy


class TestLSTMCell(nn.Module):
    r"""


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


class LayerNormLSTM(nn.Module):
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


class LSTM(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int = 1,
                 bias: bool = True,
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


# class BiLSTM(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers=2, sequences=True, dropout: float = 0., bidirectional: bool = False,):
#         """

#         Args:
#             input_size (_type_): [batch, sequence,feature]
#             hidden_size (_type_): _description_
#             layers (int, optional): _description_. Defaults to 2.
#             sequences (bool, optional): _description_. Defaults to True.
#             num_layers (int, optional): _description_. Defaults to 1.
#             batch_first (bool, optional): _description_. Defaults to False.
#             dropout (float, optional): _description_. Defaults to 0..
#             bidirectional (bool, optional): _description_. Defaults to False.
#         """
#         super(LSTM, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.lstm00 = AttLSTMCell(input_size, hidden_size)
#         self.bidirectional = bidirectional
#         if bidirectional:  # 双向
#             self.lstm10 = AttLSTMCell(input_size, hidden_size)
#             self.num_directions = 2
#         else:
#             self.num_directions = 1

#         if num_layers > 1:  # 多层
#             for i in range(1, num_layers):
#                 lstm = AttLSTMCell(hidden_size, hidden_size)
#                 setattr(self, 'lstm0{}'.format(i), lstm)
#                 if bidirectional:  # 双向
#                     setattr(self, 'lstm1{}'.format(i), lstm)

#     def forward(self, input, initial_states=None):

#         seq_len, batch = input.shape[0], input.shape[1]
#         input = input.view(batch, seq_len, -1)
#         hn = []
#         cn = []
#         if self.bidirectional:
#             input_reverse = torch.flip(input, dims=[1])
#         if initial_states is None:  # 默认隐藏层向量
#             zeros = Variable(torch.zeros(input.size(0), self.hidden_size))
#             initial_states = [(zeros, zeros), ] * self.num_layers
#         states = initial_states
#         outputs = []
#         length = input.size(1)  # 序列长度
#         for t in range(length):  # 遍历序列, 输入对应时间点的整个batch的值
#             x = input[:, t, :]
#             for l in range(self.num_layers):
#                 hc = getattr(self, 'lstm0{}'.format(l))(x, states[l])
#                 states[l] = hc  # 保存当前时间点, 当前层的输出状态向量
#                 x = hc[0]  # 隐藏层向量作为下一层的输入
#                 if t == length-1:
#                     hn.append(hc[0])
#                     cn.append(hc[1])
#             outputs.append(hc)  # 保存当前时间点的最终输出

#         if self.bidirectional:
#             states = initial_states
#             back_outputs = []
#             length = input_reverse.size(1)  # 序列长度

#             for t in range(length):  # 遍历序列, 输入对应时间点的整个batch的值
#                 x = input_reverse[:, t, :]
#                 for l in range(self.num_layers):
#                     hc = getattr(self, 'lstm1{}'.format(l))(x, states[l])
#                     states[l] = hc  # 上一层的h和c作为下一层的状态输入
#                     x = hc[0]  # 隐藏层向量作为下一层的输入
#                     if t == length-1:
#                         hn.append(hc[0])
#                         cn.append(hc[1])
#                 back_outputs.append(hc)  # 保存当前时间点的最终输出

#         hs, cs = zip(*outputs)
#         h = torch.stack(hs).transpose(0, 1)
#         # c = torch.stack(cs).transpose(0, 1)

#         if self.bidirectional:
#             back_hs, back_cs = zip(*back_outputs)
#             back_h = torch.stack(back_hs).transpose(0, 1)
#             # back_c = torch.stack(back_cs).transpose(0, 1)

#         # 最后一层,所有时间点的输出
#         output = h
#         if self.bidirectional:
#             # (num_layers * num_directions, batch, hidden_size)
#             output = torch.cat((h, back_h), -1).view(seq_len, batch, -1)

#         # # 最后一层,最后时间点的输出
#         hn = torch.stack(hn, dim=0)
#         cn = torch.stack(cn, dim=0)

#         # hn, cn = outputs[-1]
#         # if self.bidirectional:
#         #     hn = torch.cat((hn, back_outputs[-1][0]), -1)
#         #     cn = torch.cat((hn, back_outputs[-1][1]), -1)
#         # 输出最后时间点, 所有层的输出

#         return output, (hn, cn)


# class LSTM(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers=1):
#         super(LSTM, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.bidirectional = None
#         self.lay0 = AttLSTMCell(input_size, hidden_size)
#         if num_layers > 1:
#             for i in range(1, num_layers):
#                 lay = AttLSTMCell(hidden_size, hidden_size)
#                 setattr(self, 'lay{}'.format(i), lay)

#     def forward(self, input, initial_states=None):
#         if initial_states is None:
#             zeros = Variable(torch.zeros(input.size(
#                 0), self.hidden_size, device=input.device))
#             initial_states = [(zeros, zeros), ] * self.num_layers
#         else:
#             initial_states = [initial_states]
#         states = list(initial_states)

#         outputs = []
#         hn = []
#         cn = []
#         length = input.size(1)
#         for t in range(length):
#             x = input[:, t, :]
#             for l in range(self.num_layers):
#                 hx, cx = getattr(self, 'lay{}'.format(l))(x, states[l])
#                 if t == length-1:
#                     hn.append(hx)
#                     cn.append(hx)
#                 x = hx
#                 states[l] = (hx, cx)
#             outputs.append(hx)
#         outputs = torch.stack(outputs, dim=0)

#         hn = torch.stack(hn, dim=0)
#         cn = torch.stack(cn, dim=0)
#         return outputs, (hn, cn)
