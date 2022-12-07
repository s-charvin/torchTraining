
from . import components
from typing import Type, Any, Callable, Union, List, Optional
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import tensor2tensor.layers.area_attention as area_attention
from area_attention import AreaAttention


class MACNN(nn.Module):
    def __init__(self, attention_heads=4, attention_size=32, out_size=4):
        super(MACNN, self).__init__()
#         self.conv1a = nn.Conv2d(16, (10, 2), padding_mode='reflect',
#                                 )  # activation='relu')
#         # activation='relu')
#         self.conv1b = nn.Conv2d(, 16, (2, 8), padding_mode='reflect', )
#         # activation='relu')
#         self.conv2 = nn.Conv2d(, 32, (3, 3), padding_mode='reflect',)
#         # activation='relu')
#         self.conv3 = nn.Conv2d(, 48, (3, 3), padding_mode='reflect',)
#         # activation='relu')
#         self.conv4 = nn.Conv2d(, 64, (3, 3), padding_mode='reflect',)
#         # activation='relu')
#         self.conv5 = nn.Conv2d(, 80, (3, 3), padding_mode='reflect',)
#         self.maxp = nn.MaxPool2d((2, 2))
#         self.bn1a = nn.BatchNorm2d(3)
#         self.bn1b = nn.BatchNorm2d(3)
#         self.bn2 = nn.BatchNorm2d(3)
#         self.bn3 = nn.BatchNorm2d(3)
#         self.bn4 = nn.BatchNorm2d(3)
#         self.bn5 = nn.BatchNorm2d(3)
#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.flatten = nn.Flatten()
#         self.fc = nn.Linear(input_size, out_size)

#         self.attention_query = nn.ModuleList()
#         self.attention_key = nn.ModuleList()
#         self.attention_value = nn.ModuleList()
#         self.attention_heads = attention_heads
#         self.attention_size = attention_size

#         for i in range(self.attention_heads):
#             self.attention_query.append(
#                 nn.Conv2d(self.attention_size, 1, padding_mode='reflect'))
#             self.attention_key.append(
#                 nn.Conv2d(self.attention_size, 1, padding_mode='reflect'))
#             self.attention_value.append(
#                 nn.Conv2d(self.attention_size, 1, padding_mode='reflect'))

    def forward(self, *input):
        x = input[0]
#         xa = self.conv1a(x)
#         xa = self.bn1a(xa)
#         xa = torch.functional.relu(xa)
#         xb = self.conv1b(x)
#         xb = self.bn1b(xb)
#         xb = F.relu(xb)
#         x = torch.cat([xa, xb], 1)
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = F.relu(x)
#         x = self.maxp(x)
#         x = self.conv3(x)
#         x = self.bn3(x)
#         x = F.relu(x)
#         x = self.maxp(x)
#         x = self.conv4(x)
#         x = self.bn4(x)
#         x = F.relu(x)
#         x = self.conv5(x)
#         x = self.bn5(x)
#         x = F.relu(x)

#         attn = None
#         for i in range(self.attention_heads):
#             # Q = self.attention_query[i](x)
#             # Q = torch.transpose(Q, perm=[0, 3, 1, 2])
#             # K = self.attention_key[i](x)
#             # K = torch.transpose(K, perm=[0, 3, 2, 1])
#             # V = self.attention_value[i](x)
#             # V = torch.transpose(V, perm=[0, 3, 1, 2])
#             # attention = F.softmax(torch.matmul(Q, K))
#             # attention = torch.matmul(attention, V)
#             Q = self.attention_query[i](x)
#             K = self.attention_key[i](x)
#             V = self.attention_value[i](x)
#             attention = F.softmax(torch.multiply(Q, K))
#             attention = torch.multiply(attention, V)
#             if (attn is None):
#                 attn = attention
#             else:
#                 attn = torch.cat([attn, attention], 2)
#         x = attn.permute(0, 2, 3, 1)
#         x = F.relu(x)
#         x = self.gap(x)
#         x = self.flatten(x)
#         x = self.fc(x)
        return x


class AACNN(nn.Module):
    def __init__(self, config, height=3, width=3, out_size=4):
        super(AACNN, self).__init__()
        self.height = height
        self.width = width
        self.conv1 = nn.Conv2d(1, 32, (3, 3), padding_mode='reflect',)
        # activation='relu')
        self.conv1a = nn.Conv2d(1, 16, (10, 2), padding_mode='reflect')
        # activation='relu')
        self.conv1b = nn.Conv2d(1, 16, (2, 8), padding_mode='reflect', )
        # activation='relu')
        self.conv2 = nn.Conv2d(32, 32, (3, 3), padding_mode='reflect',)
        # activation='relu')
        self.conv3 = nn.Conv2d(32, 48, (3, 3), padding_mode='reflect',)
        # activation='relu')
        self.conv4 = nn.Conv2d(48, 64, (3, 3), padding_mode='reflect',)
        # activation='relu')
        self.conv5 = nn.Conv2d(64, 80, (3, 3), padding_mode='reflect',)
        # activation='relu')
        self.conv6 = nn.Conv2d(80, 128, (3, 3), padding_mode='reflect',)
        self.maxp = nn.MaxPool2d((2, 2))
        self.bn1a = nn.BatchNorm2d(3)
        self.bn1b = nn.BatchNorm2d(3)
        self.bn2 = nn.BatchNorm2d(3)
        self.bn3 = nn.BatchNorm2d(3)
        self.bn4 = nn.BatchNorm2d(3)
        self.bn5 = nn.BatchNorm2d(3)
        self.bn6 = nn.BatchNorm2d(3)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128, out_size)  # softmax
        # self.query = nn.Linear(, 20)
        # self.key = nn.Linear(, 20)
        # self.value = nn.Linear(, 20)
        self.area_attention = AreaAttention(
            key_query_size=20,
            area_key_mode='mean',
            area_value_mode='sum',
            max_area_height=self.height,
            max_area_width=self.width,
            memory_height=1,
            memory_width=1,
            dropout_rate=0.5,
            top_k_areas=0)

    def forward(self, *input):
        x = input[0]
        xa = self.conv1a(x)
        xa = self.bn1a(xa)
        xa = F.relu(xa)
        xb = self.conv1b(x)
        xb = self.bn1b(xb)
        xb = F.relu(xb)
        x = torch.cat([xa, xb], 1)

        # x=input[0]
        # x=self.bn1a(x)
        # x=self.conv1(x)
        # x=F.relu(x)

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

        # x=self.conv6(x)
        # x=self.bn6(x)
        # x=F.relu(x)

        q = x
        k = x
        v = x

        # x = area_attention.dot_product_area_attention(
        #     q, k, v, bias, dropout_rate, None,
        #     save_weights_to=None,
        #     dropout_broadcast_dims=None,
        #     max_area_width=self.width,
        #     max_area_height=self.height,
        #     area_key_mode='mean',
        #     area_value_mode='sum',
        #     training=True)
        x = self.area_attention(q, k, v)
        x = self.flatten(x)
        x = self.fc(x)
        x = F.softmax(x)
        return x
