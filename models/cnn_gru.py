import torch.nn.functional as F  # nn.Conv2是一个类，而F.conv2d是一个函数
import torch.nn as nn
import torch


class Cnn_Gru(nn.Module):
    """循环神经网络模型"""

    def __init__(self, config) -> None:
        """
        Args:
            rnn_layer (_type_): 单个RNN神经元
            input_size (_type_): 词表内词元数量
        """
        super(Cnn_Gru, self).__init__()
        self.config = config
        self.input_size = config['model']['input_chanel']
        self.num_outputs = self.config['model']['num_outputs']  # 输出维度
        self.num_hiddens = self.config['model']['num_hiddens']  # 隐藏层向量
        # (单个时间点输入特征维度(or onehot维度), 隐藏层向量维度)
        self.bidirectional = self.config['model']['bidirectional']
        # self.rnn = nn.GRU(num_inputs, num_hiddens, batch_first=True)
        # 如果RNN是双向的（之后将介绍），num_directions应该是2，否则应该是1
        if self.bidirectional:  # 双向RNN
            self.num_directions = 2
            self.linear = nn.Linear(
                self.num_hiddens * 2, self.num_outputs)  # 定义输出神经网络
        else:  # 单向RNN
            self.num_directions = 1
            self.linear = nn.Linear(
                self.num_hiddens * self.num_hiddens, self.num_outputs)   # 定义输出神经网络
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.05),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.05),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.05),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.05),
            nn.Flatten(2, -1),  # 多帧特征压缩到一点
        )
        self.rnn = nn.GRU(input_size=76, hidden_size=self.num_hiddens,
                          num_layers=1, batch_first=True, bidirectional=False)

    def forward(self, X):
        X = X.to(torch.float32)
        X = X.unsqueeze(1)
        # tensor([batch数,1,帧数, 特征数])
        # 获取cat后的所有输出,以及最后一层隐藏层向量
        # (时间步数, 块大小, 隐藏层向量维度),(1, 块大小, 隐藏层向量维度)
        output = self.net(X)
        h0 = self.begin_state(batch_size=X.shape[0], device=X.device)
        output, _ = self.rnn(output, h0)
        # 全连接层首先将Y的形状改为(时间步数*批量大小,隐藏单元数)
        # 它的输出形状是(时间步数*批量大小,词表大小)。时间步数*批量大小 = 当前输入块输入的所有时间步

        output = output.reshape((output.shape[0], -1))
        Y = self.linear(output)
        return F.softmax(Y, dim=1)

    def begin_state(self, batch_size, device):
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
