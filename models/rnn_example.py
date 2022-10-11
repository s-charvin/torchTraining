import torch.nn as nn
import torch


class RNNModel(nn.Module):
    """循环神经网络模型"""

    def __init__(self, config) -> None:
        """
        Args:
            rnn_layer (_type_): 单个RNN神经元
            input_size (_type_): 词表内词元数量
        """
        super(RNNModel, self).__init__()
        self.config = config
        self.input_size = config['model']['feature_dim']
        self.num_outputs = self.config['model']['num_outputs']  # 输出序列长度
        self.num_hiddens = self.config['model']['num_hiddens']  # 隐藏层向量
        # (单个时间点输入特征维度(or onehot维度), 隐藏层向量维度)
        self.rnn = nn.RNN(self.input_size, self.num_hiddens)
        self.num_hiddens = self.rnn.hidden_size  # 隐藏层向量维度

        # 如果RNN是双向的（之后将介绍），num_directions应该是2，否则应该是1
        if not self.rnn.bidirectional:
            self.num_directions = 1  # 单向RNN
            self.linear = nn.Linear(
                self.num_hiddens, self.input_size)  # 定义输出神经网络
        else:
            self.num_directions = 2  # 双向RNN
            self.linear = nn.Linear(
                self.num_hiddens * 2, self.input_size)   # 定义输出神经网络

    def forward(self, X, state):
        X = X.to(torch.float32)
        # 获取cat后的所有输出,以及最后一层隐藏层向量
        # (时间步数, 块大小, 隐藏层向量维度),(1, 块大小, 隐藏层向量维度)
        Y, state = self.rnn(X, state)

        # 全连接层首先将Y的形状改为(时间步数*批量大小,隐藏单元数)
        # 它的输出形状是(时间步数*批量大小,词表大小)。时间步数*批量大小 = 当前输入块输入的所有时间步
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        # 初始化隐藏层向量
        if not isinstance(self.rnn, nn.LSTM):
            # nn.GRU以张量作为隐状态
            return torch.zeros((self.num_directions * self.rnn.num_layers,
                                batch_size, self.num_hiddens),
                               device=device)
        else:
            # nn.LSTM以元组作为隐状态
            return (torch.zeros((
                self.num_directions * self.rnn.num_layers,
                batch_size, self.num_hiddens), device=device),
                torch.zeros((
                    self.num_directions * self.rnn.num_layers,
                    batch_size, self.num_hiddens), device=device))
