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


class MACNN(nn.Module):
    def __init__(self, attention_heads=8, attention_hidden=256, out_features=4):
        super(MACNN, self).__init__()
        # activation='ReLU')
        self.attention_heads = attention_heads
        self.attention_hidden = attention_hidden
        self.conv1a = nn.Conv2d(kernel_size=(
            10, 2), in_channels=1, out_channels=8, padding=(4, 0))
        self.conv1b = nn.Conv2d(kernel_size=(
            2, 8), in_channels=1, out_channels=8, padding=(0, 3))
        self.conv2 = nn.Conv2d(kernel_size=(
            3, 3), in_channels=16, out_channels=32, padding=1)
        self.conv3 = nn.Conv2d(kernel_size=(
            3, 3), in_channels=32, out_channels=48, padding=1)
        self.conv4 = nn.Conv2d(kernel_size=(
            3, 3), in_channels=48, out_channels=64, padding=1)
        self.conv5 = nn.Conv2d(kernel_size=(
            3, 3), in_channels=64, out_channels=80, padding=1)
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(8)
        self.bn1b = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(48)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(80)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.last_linear = nn.Linear(in_features=self.attention_hidden,
                                     out_features=out_features)
        self.dropout = nn.Dropout(0.5)
        self.attention_query = nn.ModuleList()
        self.attention_key = nn.ModuleList()
        self.attention_value = nn.ModuleList()

        for i in range(self.attention_heads):
            self.attention_query.append(
                nn.Conv2d(in_channels=80, out_channels=self.attention_hidden, kernel_size=1))
            self.attention_key.append(
                nn.Conv2d(in_channels=80, out_channels=self.attention_hidden, kernel_size=1))
            self.attention_value.append(
                nn.Conv2d(in_channels=80, out_channels=self.attention_hidden, kernel_size=1))

    def forward(self, x, xlen=None):

        xa = self.conv1a(x)
        xa = self.bn1a(xa)

        xa = F.relu(xa)
        xb = self.conv1b(x)
        xb = self.bn1b(xb)

        xb = F.relu(xb)
        x = torch.cat((xa, xb), 1)
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

        # #attention

        attn = None
        for i in range(self.attention_heads):
            Q = self.attention_query[i](x)
            K = self.attention_key[i](x)
            V = self.attention_value[i](x)
            attention = F.softmax(torch.mul(Q, K))
            attention = torch.mul(attention, V)

            # attention_img = attention[0, 0, :, :].squeeze().detach().cpu().numpy()
            # img = Image.fromarray(attention_img, 'L')
            # img.save('img/img_'+str(i)+'.png')

            if (attn is None):
                attn = attention
            else:
                attn = torch.cat((attn, attention), 2)
        x = attn
        x = F.relu(x)
        x = self.gap(x)

        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])

        x = self.last_linear(x)
        return x


if __name__ == '__main__':
    test = np.random.random((4, 1, 40, 40)).astype(np.float32)
    test = torch.Tensor(test)
    macnn = MACNN()
    print(macnn(test))
