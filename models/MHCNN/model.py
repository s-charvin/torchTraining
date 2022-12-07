#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Zhu Wenjing
# Date: 2022-03-07
# E-mail: zhuwenjing02@duxiaoman.com

import torch
import torch.nn as nn
import torch.nn.functional as F
from g_mlp_pytorch import gMLP
from g_mlp_pytorch import SpatialGatingUnit
from components import *


class MHCNN(nn.Module):
    '''
    Multi-Head Attention
    '''

    def __init__(self, head=4, attn_hidden=64, **kwargs):
        super(MHCNN, self).__init__()
        self.head = head
        self.attn_hidden = attn_hidden
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
        self.fc = nn.Linear(in_features=self.attn_hidden, out_features=4)
        self.dropout = nn.Dropout(0.5)
        self.attention_query = nn.ModuleList()
        self.attention_key = nn.ModuleList()
        self.attention_value = nn.ModuleList()

        for i in range(self.head):
            self.attention_query.append(
                nn.Conv2d(in_channels=80, out_channels=self.attn_hidden, kernel_size=1))
            self.attention_key.append(
                nn.Conv2d(in_channels=80, out_channels=self.attn_hidden, kernel_size=1))
            self.attention_value.append(
                nn.Conv2d(in_channels=80, out_channels=self.attn_hidden, kernel_size=1))

    def forward(self, *input):
        xa = self.conv1a(input[0])
        xa = self.bn1a(xa)

        xa = F.relu(xa)
        xb = self.conv1b(input[0])
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
        for i in range(self.head):
            Q = self.attention_query[i](x)
            K = self.attention_key[i](x)
            V = self.attention_value[i](x)
            attention = F.softmax(torch.mul(Q, K), dim=1)
            attention = torch.mul(attention, V)

            if (attn is None):
                attn = attention
            else:
                attn = torch.cat((attn, attention), 2)
        x = attn
        x = F.relu(x)
        x = self.gap(x)

        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])

        x = self.fc(x)
        return x


class MHCNN_AreaConcat(nn.Module):
    def __init__(self, head=4, attn_hidden=64, shape=(26, 63), **kwargs):
        super(MHCNN_AreaConcat, self).__init__()
        self.head = head
        self.attn_hidden = attn_hidden
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
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(16)
        self.bn1b = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(48)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(80)
        self.gap = nn.AdaptiveAvgPool2d(1)
        i = self.attn_hidden * ((shape[0]-1)//2) * \
            ((shape[1]-1)//4) * self.head
        self.fc = nn.Linear(in_features=i, out_features=4)
        self.dropout = nn.Dropout(0.5)
        self.attention_query = nn.ModuleList()
        self.attention_key = nn.ModuleList()
        self.attention_value = nn.ModuleList()

        for i in range(self.head):
            self.attention_query.append(
                nn.Conv2d(in_channels=80, out_channels=self.attn_hidden, kernel_size=1))
            self.attention_key.append(
                nn.Conv2d(in_channels=80, out_channels=self.attn_hidden, kernel_size=1))
            self.attention_value.append(
                nn.Conv2d(in_channels=80, out_channels=self.attn_hidden, kernel_size=1))

    def forward(self, *input):
        xa = self.conv1a(input[0])
        xa = self.bn1a(xa)

        xa = F.relu(xa)
        xb = self.conv1b(input[0])
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
        # #attention

        attn = None
        for i in range(self.head):
            Q = self.attention_query[i](x)
            K = self.attention_key[i](x)
            V = self.attention_value[i](x)
            attention = F.softmax(torch.mul(Q, K), dim=1)
            attention = torch.mul(attention, V)

            if (attn is None):
                attn = attention
            else:
                attn = torch.cat((attn, attention), 2)
        x = attn
        x = F.relu(x)
        # x = self.gap(x)

        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])

        x = self.fc(x)
        return x


class MHCNN_AreaConcat_gap(nn.Module):
    def __init__(self, head=4, attn_hidden=64, **kwargs):
        super(MHCNN_AreaConcat_gap, self).__init__()
        self.head = head
        self.attn_hidden = attn_hidden
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
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(16)
        self.bn1b = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(48)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(80)
        self.gap = nn.AdaptiveAvgPool2d(1)
        i = self.attn_hidden
        self.fc = nn.Linear(in_features=i, out_features=4)
        self.dropout = nn.Dropout(0.5)
        self.attention_query = nn.ModuleList()
        self.attention_key = nn.ModuleList()
        self.attention_value = nn.ModuleList()

        for i in range(self.head):
            self.attention_query.append(
                nn.Conv2d(in_channels=80, out_channels=self.attn_hidden, kernel_size=1))
            self.attention_key.append(
                nn.Conv2d(in_channels=80, out_channels=self.attn_hidden, kernel_size=1))
            self.attention_value.append(
                nn.Conv2d(in_channels=80, out_channels=self.attn_hidden, kernel_size=1))

    def forward(self, *input):
        xa = self.conv1a(input[0])
        xa = self.bn1a(xa)

        xa = F.relu(xa)
        xb = self.conv1b(input[0])
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
        # #attention

        attn = None
        for i in range(self.head):
            Q = self.attention_query[i](x)
            K = self.attention_key[i](x)
            V = self.attention_value[i](x)
            attention = F.softmax(torch.mul(Q, K), dim=1)
            attention = torch.mul(attention, V)

            if (attn is None):
                attn = attention
            else:
                attn = torch.cat((attn, attention), 2)
        x = attn
        x = F.relu(x)
        x = self.gap(x)

        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])

        x = self.fc(x)
        return x


class MHCNN_AreaConcat_gap1(nn.Module):
    def __init__(self, head=4, attn_hidden=64, shape=(26, 63), **kwargs):
        super(MHCNN_AreaConcat_gap1, self).__init__()
        self.head = head
        self.attn_hidden = 32
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
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(16)
        self.bn1b = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(48)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(80)
        self.gap = nn.AdaptiveAvgPool2d(1)
        i = self.attn_hidden
        self.fc = nn.Linear(in_features=(shape[0]-1)//2, out_features=4)
        self.dropout = nn.Dropout(0.5)
        self.attention_query = nn.ModuleList()
        self.attention_key = nn.ModuleList()
        self.attention_value = nn.ModuleList()

        for i in range(self.head):
            self.attention_query.append(
                nn.Conv2d(in_channels=80, out_channels=self.attn_hidden, kernel_size=1))
            self.attention_key.append(
                nn.Conv2d(in_channels=80, out_channels=self.attn_hidden, kernel_size=1))
            self.attention_value.append(
                nn.Conv2d(in_channels=80, out_channels=self.attn_hidden, kernel_size=1))

    def forward(self, *input):
        xa = self.conv1a(input[0])
        xa = self.bn1a(xa)

        xa = F.relu(xa)
        xb = self.conv1b(input[0])
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
        # #attention

        attn = None
        for i in range(self.head):
            Q = self.attention_query[i](x)
            K = self.attention_key[i](x)
            V = self.attention_value[i](x)
            attention = F.softmax(torch.mul(Q, K), dim=1)
            attention = torch.mul(attention, V)

            if (attn is None):
                attn = attention
            else:
                attn = torch.cat((attn, attention), 3)
        x = attn
        x = x.contiguous().permute(0, 2, 3, 1)
        x = F.relu(x)
        x = self.gap(x)

        x = x.reshape(x.shape[0], -1)

        x = self.fc(x)
        return x


class MHResMultiConv3(nn.Module):
    '''
    Multi-Head-Attention with Multiscale blocks
    '''

    def __init__(self, head=4, attn_hidden=64, shape=(26, 63), **kwargs):
        super(MHResMultiConv3, self).__init__()
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
        i = self.attn_hidden * self.head * (shape[0]//2) * (shape[1]//4)
        self.fc = nn.Linear(in_features=i, out_features=4)

        self.dropout = nn.Dropout(0.5)
        self.attention_query = nn.ModuleList()
        self.attention_key = nn.ModuleList()
        self.attention_value = nn.ModuleList()

        for i in range(self.head):
            self.attention_query.append(
                nn.Conv2d(in_channels=128, out_channels=self.attn_hidden, kernel_size=1))
            self.attention_key.append(
                nn.Conv2d(in_channels=128, out_channels=self.attn_hidden, kernel_size=1))
            self.attention_value.append(
                nn.Conv2d(in_channels=128, out_channels=self.attn_hidden, kernel_size=1))

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

        attn = None
        for i in range(self.head):
            Q = self.attention_query[i](x)
            K = self.attention_key[i](x)
            V = self.attention_value[i](x)
            attention = F.softmax(torch.mul(Q, K), dim=1)
            attention = torch.mul(attention, V)

            if (attn is None):
                attn = attention
            else:
                attn = torch.cat((attn, attention), 2)
        x = attn
        x = F.relu(x)

        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])

        x = self.fc(x)
        return x
