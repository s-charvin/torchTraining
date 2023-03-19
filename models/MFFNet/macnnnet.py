import torch
import torch.nn as nn
import numpy as np
import area_attention
import torch.nn.functional as F
import math


class Conv2dSame(nn.Conv2d):

    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.size()[-2:]

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


class MACNN_Encoder(nn.Module):
    def __init__(self, attention_heads=4, attention_size=32):
        super(MACNN_Encoder, self).__init__()
        # activation='ReLU')
        self.conv1a = Conv2dSame(1, 16, (10, 2))
        self.bn1a = nn.BatchNorm2d(16)
        # activation='ReLU')
        self.conv1b = Conv2dSame(1, 16, (2, 8))
        self.bn1b = nn.BatchNorm2d(16)
        # activation='ReLU')
        self.conv2 = Conv2dSame(32, 32, (3, 3))
        self.bn2 = nn.BatchNorm2d(32)
        # activation='ReLU')
        self.conv3 = Conv2dSame(32, 48, (3, 3))
        self.bn3 = nn.BatchNorm2d(48)
        # activation='ReLU')
        self.conv4 = Conv2dSame(48, 64, (3, 3))
        self.bn4 = nn.BatchNorm2d(64)
        # activation='ReLU')
        self.conv5 = Conv2dSame(64, 80, (3, 3))
        self.bn5 = nn.BatchNorm2d(80)
        self.maxp = nn.MaxPool2d((2, 2))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.attention_query = nn.ModuleList()
        self.attention_key = nn.ModuleList()
        self.attention_value = nn.ModuleList()
        self.attention_heads = attention_heads
        self.attention_size = attention_size
        for i in range(self.attention_heads):
            self.attention_query.append(
                Conv2dSame(80, self.attention_size, 1))
            self.attention_key.append(
                Conv2dSame(80, self.attention_size, 1))
            self.attention_value.append(
                Conv2dSame(80, self.attention_size, 1))

    def forward(self, *input):
        x = input[0]
        xa = self.conv1a(x)
        xa = self.bn1a(xa)
        xa = F.relu(xa)

        xb = self.conv1b(x)
        xb = self.bn1b(xb)
        xb = F.relu(xb)

        x = torch.cat([xa, xb], 1)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.maxp(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.maxp(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)

        attn = None
        for i in range(self.attention_heads):
            # Q = self.attention_query[i](x)
            # Q = tf.transpose(Q, perm=[0, 3, 1, 2])
            # K = self.attention_key[i](x)
            # K = tf.transpose(K, perm=[0, 3, 2, 1])
            # V = self.attention_value[i](x)
            # V = tf.transpose(V, perm=[0, 3, 1, 2])
            # attention = tf.nn.softmax(tf.matmul(Q, K))
            # attention = tf.matmul(attention, V)
            Q = self.attention_query[i](x)
            K = self.attention_key[i](x)
            V = self.attention_value[i](x)
            attention = F.softmax(torch.multiply(Q, K))
            attention = torch.multiply(attention, V)
            if (attn is None):
                attn = attention
            else:
                attn = torch.cat([attn, attention], 2)

        x = F.relu(attn)
        x = self.gap(x)
        x = self.flatten(x)
        return x


if __name__ == '__main__':
    test = np.random.random((4, 1, 126, 40)).astype(np.float32)
    test = torch.Tensor(test)
    macnn = MACNN_Encoder(out_features=320)
    print(macnn(test))
