#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Zhu Wenjing
# Date: 2022-03-07
# E-mail: zhuwenjing02@duxiaoman.com

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from .components import *
from torch import einsum
from einops import rearrange


class MSNet(nn.Module):
    '''
    Multi-Scale Convolutional Network based on Multi-Channel Feature Fusion
    '''

    def __init__(self, in_channels, seq_len, last_hidden_dim):
        super(MSNet, self).__init__()
        self.seq_len = seq_len
        self.last_hidden_dim = last_hidden_dim

        self.path1 = nn.Sequential(
            Conv2dSame(in_channels=in_channels, out_channels=8, kernel_size=(
                11, 1), stride=(1, 1)),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            # nn.AvgPool2d(kernel_size=2)
        )
        self.path2 = nn.Sequential(
            Conv2dSame(in_channels=in_channels, out_channels=8, kernel_size=(
                1, 9), stride=(1, 1)),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            # nn.AvgPool2d(kernel_size=2)
        )
        self.path3 = nn.Sequential(
            Conv2dSame(in_channels=in_channels, out_channels=8, kernel_size=(
                3, 3), stride=(1, 1)),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            # nn.AvgPool2d(kernel_size=2)
        )

        self.avgpooling = nn.AvgPool2d(kernel_size=2)
        self.conv2 = ResMultiScaleConv2d(24)
        self.conv3 = ResMultiScaleConv2d(48)
        self.conv4 = ResMultiScaleConv2d(96)
        self.conv5 = ResMultiScaleConv2d(192)

        self.conv6 = nn.Conv2d(kernel_size=(
            5, 5), in_channels=384, out_channels=self.last_hidden_dim, padding=2)
        self.bn6 = nn.BatchNorm2d(self.last_hidden_dim)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: Tensor, feature_lens=None):
        # size(x): [B, 1, S, 40] (40: MFCC, S: seq_len)

        # 多通道信息提取网络
        out1 = self.path1(x)  # (B, 8, S, 40)
        out2 = self.path2(x)  # (B, 8, S, 40)
        out3 = self.path3(x)  # (B, 8, S, 40)

        x = torch.cat([out1, out2, out3], dim=1)
        # (B, 16*3, S, 40)

        # 多尺度特征降维
        x = self.conv2(x)
        # (B, 48, S, 40)
        x = self.avgpooling(x)
        # (B, 48, S/2, 20)
        x = self.conv3(x)
        x = self.avgpooling(x)
        # (B, 96, S/4, 10)
        x = self.conv4(x)
        x = self.avgpooling(x)
        # (B, 192, S/8, 5)
        x = self.conv5(x)
        # (B, 384, S/8, 5)
        x = self.conv6(x)
        x = self.bn6(x)
        # (B, 320, S/8, 5)
        x = F.relu(x)
        x = self.gap(x)
        # (B, 320, 1)
        x = self.dropout(x.squeeze(-1).squeeze(-1))
        return x  # (B, 320)


class SequenceGatingUnit(nn.Module):
    def __init__(
            self,
            dim,
            init_eps=1e-3,
            act_gate=nn.Tanh(),):

        super().__init__()
        self.norm_gate = nn.LayerNorm(dim//2)
        weight = torch.zeros((dim//2, dim//2))
        self.weight = nn.Parameter(weight)
        nn.init.uniform_(self.weight, -init_eps, init_eps)
        self.bias = nn.Parameter(torch.ones(dim//2))
        self.act_gate = act_gate

    def forward(self, x):
        # size(x): [B,S,dim]
        res, gate = x.chunk(2, dim=-1)
        # [B,S,dim/2]
        gate = self.norm_gate(gate)
        gate = einsum('b n d, d d -> b n d', gate, self.weight)
        gate = gate + rearrange(self.bias, 'd -> () () d')
        return self.act_gate(gate) * res


class gMLPBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        ff_mult=4,
        init_eps=1e-3,
        act_gate=nn.Tanh(),
    ):
        super().__init__()

        dim_ff = dim * ff_mult

        self.proj_in = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim_ff),
            nn.GELU()
        )
        self.sgu = SequenceGatingUnit(
            dim=dim_ff, init_eps=init_eps, act_gate=act_gate)
        self.proj_out = nn.Linear(dim_ff // 2, dim)

    def forward(self, x):
        # size(x): [B,S,F]
        out = self.proj_in(x)  # [B,S,F * ff_mult]
        out = self.sgu(out)  # [B,S,F * ff_mult // 2]
        out = self.proj_out(out)  # [B,S,F]
        return out + x  # 残差连接


class gMLP(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        ff_mult=4,
        init_eps=1e-3,
        act_gate=nn.Tanh(),

    ):
        super().__init__()
        self.layers = nn.Sequential(*nn.ModuleList([
            gMLPBlock(dim=dim, ff_mult=ff_mult,
                      init_eps=init_eps, act_gate=act_gate)
            for i in range(depth)]))

    def forward(self, x):
        return self.layers(x)


class Audio_GloballyGatedMLP_SVC_BoxCox_Net(nn.Module):
    '''
    Globally GatedMLP 通过全局门控单元, 判断哪些信息重要
    BoxCox_Net 将原偏态分布的段向量转换成正太分布
    Segment-level Vectors Correction 对得到的段级向量进行修正, 得到整段语音的表征向量
    '''

    def __init__(self, num_classes=4, seq_len=63*3, last_hidden_dim=320) -> None:
        super().__init__()
        self.af_seq_len = seq_len
        self.num_classes = num_classes
        self.last_hidden_dim = last_hidden_dim
        self.audio_feature_extractor = MSNet(
            in_channels=1, seq_len=seq_len, last_hidden_dim=last_hidden_dim)
        # self.gmlp = gMLP(dim=self.last_hidden_dim, depth=1, act_gate=nn.Tanh())
        self.audio_classifier = nn.Linear(
            in_features=self.last_hidden_dim, out_features=num_classes)

    def forward(self, af: Tensor, af_len=None) -> Tensor:
        """
        Args:
            af (Tensor): size:[B, C,Seq, F]
            af_len (Sequence, optional): size:[B]. Defaults to None.
        Returns:
            Tensor: size:[batch, out]
        """
        # 处理语音数据
        # [B, C, Seq, F]
        if af.shape[2] > self.af_seq_len:
            af = [af[:, :, 63*i:63*i+self.af_seq_len, :]
                  for i in range(int(((af.shape[2]-self.af_seq_len) / 63)+1))]
        else:
            af = [af]
        # af_seg_len * [B, C, af_seq_len, F]

        # 记录分段数量, af_seg_len = ceil or floor(seq/af_seq_len)
        af_seg_len = len(af)
        af = torch.stack(af).permute(1, 0, 2, 3, 4).clone().contiguous()
        # [B, af_seg_len, C, af_seq_len, F]

        # [B * af_seg_len, C, af_seq_len, F]
        af_fea = self.audio_feature_extractor(
            af.view((-1,)+af.size()[2:]))
        # [B * af_seg_len, F]
        af_fea = af_fea.view((-1, af_seg_len) + af_fea.size()[1:])
        # [B, af_seg_len, F]
        if af_len is not None:  # 如果提供了原数据的有效 Seq 长度
            batch_size, max_len, _ = af_fea.shape
            # 计算每一段实际的有效数据段数量

            af_len_ = []
            for l in af_len:
                if l.item() > self.af_seq_len:
                    af_len_.append(int((l.item()-self.af_seq_len)/63+1))
                else:
                    af_len_.append(l.item())
            af_len = torch.Tensor(af_len_).to(device=af_len.device)
            af_mask = torch.arange(max_len, device=af_len.device).expand(
                batch_size, max_len) >= af_len[:, None]
            af_fea[af_mask] = 0.0

        # 全局感知, 过滤非情感向量
        # [B, af_seg_len, F]
        # af_fea = self.gmlp(af_fea)  # [B, af_seg_len, F]
        af_fea = af_fea.mean(dim=1)  # [B, F]

        # 分类
        af_out = self.audio_classifier(af_fea)  # [B, num_classes]
        # 向量聚类
        # af_fea = af_fea.unsqueeze(-1)
        # af_out = F.pairwise_distance(af_fea, self.audio_classifier, p=2)
        return af_out, None
