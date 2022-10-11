from .cnn_example import CNNNetExemple
from transformers import Wav2Vec2Model, Wav2Vec2Config
import torch.nn.functional as F  # nn.Conv2是一个类，而F.conv2d是一个函数
import torch.nn as nn
import torch


class Wav2Vec2_CLS(nn.Module):
    def __init__(self, config):
        """使用Wav2Vec2得到语音向量,再通过自定义的分类模型(如最简单的Linear)得到分类向量

        Args:
            hidden_size (_type_): _description_
            output (int, optional): _description_. Defaults to 4.
            dropout (float, optional): _description_. Defaults to 0.05.
            config (_type_, optional): _description_. Defaults to None.
        """
        super(Wav2Vec2_CLS, self).__init__()
        self.config = config
        self.wav2vec2 = Wav2Vec2(self.config)
        self.dropout = nn.Dropout(self.config['wev2vec']['vec_dropout'])
        # 使用分类层对隐藏层输出向量进行分类
        self.cls_head = CNNNetExemple(self.config)

    def forward(self, X):
        # 最后一层隐藏层向量
        hidden_states = self.wav2vec2(X)
        hidden_states = self.dropout(hidden_states)
        Y = self.cls_head(hidden_states)
        return F.softmax(Y, dim=1)


class Wav2Vec2_pretrain(nn.Module):
    def __init__(self, config):
        """使用Wav2Vec2得到语音向量,再通过自定义的分类模型(如最简单的Linear)得到分类向量

        Args:
            hidden_size (_type_): _description_
            output (int, optional): _description_. Defaults to 4.
            dropout (float, optional): _description_. Defaults to 0.05.
            config (_type_, optional): _description_. Defaults to None.
        """
        super(Wav2Vec2_pretrain, self).__init__()
        self.config = config
        # wev2vec输出特征维度
        self.num_outputs = self.config['model']['num_outputs']  # 输出分类数
        self.vec_hiddens = self.config['wev2vec']['vec_hiddens']  # 输出隐藏层维度
        self.wav2vec2 = Wav2Vec2(self.config)
        # 设置输出特征pooling模式,以便作为Linear的输入
        self.pooling_mode = self.config['wev2vec']['pooling_mode']
        self.dropout = nn.Dropout(self.config['wev2vec']['vec_dropout'])
        # 使用分类层对隐藏层输出向量进行分类
        self.cls_head = nn.Sequential(
            nn.Linear(self.vec_hiddens, self.vec_hiddens),
            nn.Linear(self.vec_hiddens, self.num_outputs)
        )

    def merged_strategy(
            self,
            hidden_states,
            mode="mean"
    ):
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")
        return outputs

    def forward(self, X):
        # 最后一层隐藏层向量
        hidden_states = self.wav2vec2(
            X,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            # tuple: (ctc_labels, cls_labels), shape=(batch_size, target_length)
        )
        hidden_states = self.merged_strategy(
            hidden_states, mode=self.pooling_mode)
        Y = self.cls_head(hidden_states)
        return F.softmax(Y, dim=1)


class Wav2Vec2(nn.Module):
    def __init__(self, config):
        """使用预处理的Wav2Vec2,输入音频数据块,得到对应的特征向量

        Args:
            wav2cec2config (_type_, optional): _description_. Defaults to None.
        """
        super().__init__()
        self.config = config
        self.vec_hiddens = self.config['wev2vec']['vec_hiddens']
        self.path = self.config['wev2vec']['path']
        self.wav2cec2config = Wav2Vec2Config.from_pretrained(self.path)
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(
            self.path, config=self.wav2cec2config)

    def forward(
        self,
        input,
        attention_mask=None
    ):

        outputs = self.wav2vec2(
            input,
            attention_mask=attention_mask,
            output_attentions=None,
            output_hidden_states=self.vec_hiddens,
            return_dict=None,
        )

        hidden_states = outputs[0]  # wav2vec最后一层网络层的隐藏层向量
        return hidden_states
