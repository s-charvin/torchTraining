#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Zhu Wenjing
# Date: 2022-03-07
# E-mail: zhuwenjing02@duxiaoman.com

import torch
import torch.nn as nn
import torch.nn.functional as F
from .components import *


class AACNN_Encoder(nn.Module):
    '''
    Area Attention, ICASSP 2020
    '''

    def __init__(self, max_area_height=3, max_area_width=3, **kwargs):
        super(AACNN_Encoder, self).__init__()
        self.height = max_area_height
        self.width = max_area_width
        # self.conv1 = nn.Conv2D(32, (3,3), padding='same', data_format='channels_last',)
        self.conv1a = nn.Conv2d(kernel_size=(
            10, 2), in_channels=1, out_channels=16, padding=(4, 0))
        self.conv1b = nn.Conv2d(kernel_size=(
            2, 8), in_channels=1, out_channels=16, padding=(0, 3))
        self.conv2 = nn.Conv2d(kernel_size=(
            3, 3), in_channels=16, out_channels=32, padding=1)
        self.conv3 = nn.Conv2d(kernel_size=(
            3, 3), in_channels=32, out_channels=48, padding=1)
        self.conv4 = nn.Conv2d(kernel_size=(
            3, 3), in_channels=48, out_channels=64, padding=1)
        self.conv5 = nn.Conv2d(kernel_size=(
            3, 3), in_channels=64, out_channels=80, padding=1)
        # self.conv6 = nn.Conv2D(128, (3, 3), padding= )#
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(16)
        self.bn1b = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(48)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(80)
        # self.gap = nn.AdaptiveAvgPool2d(1)
        # self.dropout = nn.Dropout(0.5)

        self.area_attention = AreaAttention(
            key_query_size=80,
            area_key_mode='mean',
            area_value_mode='sum',
            max_area_height=max_area_height,
            max_area_width=max_area_width,
            dropout_rate=0.5,
        )

    def forward(self, *input):
        x = input[0]
        xa = self.conv1a(x)
        xa = self.bn1a(xa)
        xa = F.relu(xa)
        xb = self.conv1b(x)
        xb = self.bn1b(xb)
        xb = F.relu(xb)
        x = torch.cat((xa, xb), 2)

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

        shape = x.shape
        x = x.contiguous().permute(0, 2, 3, 1).view(
            shape[0], shape[3]*shape[2], shape[1])
        x = self.area_attention(x, x, x)
        x = F.relu(x)
        x = x.reshape(*shape)

        x = x.reshape(x.shape[0], -1)
        return x


class AACNN_HeadConcat(nn.Module):
    def __init__(self, height=3, width=3, out_size=4, shape=(26, 63), **kwargs):
        super(AACNN_HeadConcat, self).__init__()
        self.height = height
        self.width = width
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
        i = 80 * ((shape[0] - 1)//4) * ((shape[1] - 1)//4)
        self.fc = nn.Linear(in_features=i, out_features=4)
        self.dropout = nn.Dropout(0.5)
        self.area_attention = AreaAttention(
            key_query_size=80,
            area_key_mode='mean',
            area_value_mode='sum',
            max_area_height=height,
            max_area_width=width,
            dropout_rate=0.5,
            # top_k_areas=0
        )

    def forward(self, *input):
        x = input[0]
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

        # flatten
        shape = x.shape
        x = x.contiguous().permute(0, 2, 3, 1).view(
            shape[0], shape[3]*shape[2], shape[1])

        x = self.area_attention(x, x, x)
        x = F.relu(x)

        x = x.reshape(x.shape[0], -1)

        x = self.fc(x)

        return x


class AAResMultiConv3(nn.Module):
    '''
    Area Attention with Multiscale blocks
    '''

    def __init__(self, head=4, attn_hidden=64, shape=(26, 63), **kwargs):
        super(AAResMultiConv3, self).__init__()
        self.head = head
        self.attn_hidden = attn_hidden
        self.conv1a = nn.Conv2d(kernel_size=(
            3, 1), in_channels=1, out_channels=16, padding=(1, 0))
        self.conv1b = nn.Conv2d(kernel_size=(
            1, 3), in_channels=1, out_channels=16, padding=(0, 1))
        self.conv2 = ResMultiConv(16)
        self.conv3 = ResMultiConv(32)
        self.conv4 = ResMultiConv(64)
        self.conv5 = nn.Conv2d(kernel_size=(
            5, 5), in_channels=128, out_channels=128, padding=2)
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(16)
        self.bn1b = nn.BatchNorm2d(16)
        self.bn5 = nn.BatchNorm2d(128)

        i = 128 * (shape[0]//2) * (shape[1]//4)
        self.fc = nn.Linear(in_features=i, out_features=4)
        # self.dropout = nn.Dropout(0.5)

        self.area_attention = AreaAttention(
            key_query_size=80,
            area_key_mode='mean',
            area_value_mode='sum',
            max_area_height=3,
            max_area_width=3,
            dropout_rate=0.5,
        )

    def forward(self, *input):
        # input[0]: torch.Size([32, 1, 26, 63])
        xa = self.conv1a(input[0])  # (32, 16, 25, 62)
        xa = self.bn1a(xa)  # (32, 16, 25, 62)

        xa = F.relu(xa)
        xb = self.conv1b(input[0])
        xb = self.bn1b(xb)

        xb = F.relu(xb)
        x = torch.cat((xa, xb), 2)  # (32, 16, 50, 62)

        x = self.conv2(x)  # (32, 32, 50, 62)
        x = self.maxp(x)
        x = self.conv3(x)  # (32, 64, 25, 31)
        x = self.maxp(x)
        x = self.conv4(x)  # (32, 128, 12, 15)

        x = self.conv5(x)  # (32, 128, 12, 15)
        x = self.bn5(x)
        x = F.relu(x)

        shape = x.shape
        x = x.contiguous().permute(0, 2, 3, 1).view(
            shape[0], shape[3]*shape[2], shape[1])
        x = self.area_attention(x, x, x)
        x = F.relu(x)
        x = x.reshape(x.shape[0], -1)

        x = self.fc(x)

        return x


if __name__ == '__main__':
    test = np.random.random((4, 1, 126, 40)).astype(np.float32)
    test = torch.Tensor(test)
    macnn = AACNN_Encoder(shape=(126, 40), out_features=320)
    print(macnn(test).shape)
