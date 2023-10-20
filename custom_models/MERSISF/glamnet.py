
import torch
import torch.nn as nn
import torch.nn.functional as F
from g_mlp_pytorch import gMLP
from g_mlp_pytorch import SpatialGatingUnit
from .components import *


class GLAM5(nn.Module):
    '''
    GLobal-Aware Multiscale block with 5x5 convolutional kernels in CNN architecture
    '''

    def __init__(self, shape=(26, 63), **kwargs):
        super(GLAM5, self).__init__()
        self.conv1a = nn.Conv2d(kernel_size=(
            5, 1), in_channels=1, out_channels=16, padding=(2, 0))
        self.conv1b = nn.Conv2d(kernel_size=(
            1, 5), in_channels=1, out_channels=16, padding=(0, 2))
        self.conv2 = ResMultiConv(16)
        self.conv3 = ResMultiConv(32)
        self.conv4 = ResMultiConv(64)
        self.conv5 = nn.Conv2d(kernel_size=(
            5, 5), in_channels=128, out_channels=128, padding=2)
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(16)
        self.bn1b = nn.BatchNorm2d(16)
        self.bn5 = nn.BatchNorm2d(128)
        dim = (shape[0]//2) * (shape[1]//4)
        i = 128 * dim
        self.fc = nn.Linear(in_features=i, out_features=4)
        self.dropout = nn.Dropout(0.5)

        self.gmlp = gMLP(dim=dim, depth=1, seq_len=128, act=nn.Tanh())

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

        # flatten
        shape = x.shape
        x = x.view(*x.shape[:-2], -1)

        x = self.gmlp(x)
        x = F.relu(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x, None


class GLAM(nn.Module):
    '''
    GLobal-Aware Multiscale block with 3x3 convolutional kernels in CNN architecture
    '''

    def __init__(self, shape=(40, 63), out_features=4):
        super(GLAM, self).__init__()
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
        dim = (shape[0]//2) * (shape[1]//4)
        i = 128 * dim
        self.last_linear = nn.Linear(in_features=i, out_features=out_features)
        self.dropout = nn.Dropout(0.5)

        self.gmlp = gMLP(dim=dim, depth=1, seq_len=128, act=nn.Tanh())

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

        # flatten
        shape = x.shape
        x = x.view(*x.shape[:-2], -1)

        x = self.gmlp(x)
        x = F.relu(x)

        x = x.reshape(x.shape[0], -1)  # (32, 128*12*15)
        x = self.last_linear(x)
        return x


class gMLPResConv3(nn.Module):
    '''
    GLAM - Multiscale
    '''

    def __init__(self, shape=(26, 63), **kwargs):
        super(gMLPResConv3, self).__init__()
        self.conv1a = nn.Conv2d(kernel_size=(
            3, 1), in_channels=1, out_channels=16, padding=(1, 0))
        self.conv1b = nn.Conv2d(kernel_size=(
            1, 3), in_channels=1, out_channels=16, padding=(0, 1))
        self.conv2 = ResConv3(16)
        self.conv3 = ResConv3(32)
        self.conv4 = ResConv3(64)
        self.conv5 = nn.Conv2d(kernel_size=(
            5, 5), in_channels=128, out_channels=128, padding=2)
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(16)
        self.bn1b = nn.BatchNorm2d(16)
        self.bn5 = nn.BatchNorm2d(128)
        dim = (shape[0]//2) * (shape[1]//4)
        i = 128 * dim
        self.fc = nn.Linear(in_features=i, out_features=4)
        self.dropout = nn.Dropout(0.5)

        self.gmlp = gMLP(dim=dim, depth=1, seq_len=128, act=nn.Tanh())

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

        # flatten
        shape = x.shape
        x = x.view(*x.shape[:-2], -1)

        x = self.gmlp(x)
        x = F.relu(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x


class gMLPResConv5(nn.Module):
    '''
    GLAM5 - Multiscale
    '''

    def __init__(self, shape=(26, 63), **kwargs):
        super(gMLPResConv5, self).__init__()
        self.conv1a = nn.Conv2d(kernel_size=(
            3, 1), in_channels=1, out_channels=16, padding=(1, 0))
        self.conv1b = nn.Conv2d(kernel_size=(
            1, 3), in_channels=1, out_channels=16, padding=(0, 1))
        self.conv2 = ResConv5(16)
        self.conv3 = ResConv5(32)
        self.conv4 = ResConv5(64)
        self.conv5 = nn.Conv2d(kernel_size=(
            5, 5), in_channels=128, out_channels=128, padding=2)
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(16)
        self.bn1b = nn.BatchNorm2d(16)
        self.bn5 = nn.BatchNorm2d(128)
        dim = (shape[0]//2) * (shape[1]//4)
        i = 128 * dim
        self.fc = nn.Linear(in_features=i, out_features=4)
        self.dropout = nn.Dropout(0.5)

        self.gmlp = gMLP(dim=dim, depth=1, seq_len=128, act=nn.Tanh())

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

        # flatten
        shape = x.shape
        x = x.view(*x.shape[:-2], -1)

        x = self.gmlp(x)
        x = F.relu(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x


class gMLPMultiConv(nn.Module):
    '''
    GLAM - Resnet
    '''

    def __init__(self, shape=(26, 63), **kwargs):
        super(gMLPMultiConv, self).__init__()
        self.conv1a = nn.Conv2d(kernel_size=(
            3, 1), in_channels=1, out_channels=16, padding=(1, 0))
        self.conv1b = nn.Conv2d(kernel_size=(
            1, 3), in_channels=1, out_channels=16, padding=(0, 1))
        self.conv2 = MultiConv(16)
        self.conv3 = MultiConv(32)
        self.conv4 = MultiConv(64)
        self.conv5 = nn.Conv2d(kernel_size=(
            5, 5), in_channels=128, out_channels=128, padding=2)
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(16)
        self.bn1b = nn.BatchNorm2d(16)
        self.bn5 = nn.BatchNorm2d(128)
        dim = (shape[0]//2) * (shape[1]//4)
        i = 128 * dim
        self.fc = nn.Linear(in_features=i, out_features=4)
        self.dropout = nn.Dropout(0.5)

        self.gmlp = gMLP(dim=dim, depth=1, seq_len=128, act=nn.Tanh())

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

        # flatten
        shape = x.shape
        x = x.view(*x.shape[:-2], -1)

        x = self.gmlp(x)
        x = F.relu(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x


class gMLPgResMultiConv3(nn.Module):
    def __init__(self, shape=(26, 63), **kwargs):
        super(gMLPgResMultiConv3, self).__init__()
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
        dim = (shape[0]//4) * (shape[1]//4)
        i = 128 * dim
        self.fc = nn.Linear(in_features=i, out_features=4)
        self.dropout = nn.Dropout(0.5)

        self.gmlp = gMLP(dim=dim, depth=1, seq_len=128, act=nn.Tanh())
        self.sgu = SpatialGatingUnit(
            dim=shape[0] * shape[1] * 2, dim_seq=16, act=nn.Tanh())

    def forward(self, *input):
        # input[0]: torch.Size([32, 1, 26, 63])
        xa = self.conv1a(input[0])  # (32, 16, 26, 63)
        xa = self.bn1a(xa)  # (32, 16, 26, 63)

        xa = F.relu(xa)
        xb = self.conv1b(input[0])
        xb = self.bn1b(xb)

        xb = F.relu(xb)
        x = torch.cat((xa, xb), 2)  # (32, 16, 50, 62)

        shape = x.shape
        x = x.view(*x.shape[:-2], -1)
        x = self.sgu(x)
        x = x.view(shape[0], shape[1], shape[2]//2, shape[3])

        x = self.conv2(x)  # (32, 32, 50, 62)
        x = self.maxp(x)
        x = self.conv3(x)  # (32, 64, 25, 31)
        x = self.maxp(x)
        x = self.conv4(x)  # (32, 128, 12, 15)

        x = self.conv5(x)  # (32, 128, 12, 15)
        x = self.bn5(x)
        x = F.relu(x)

        # flatten
        shape = x.shape
        x = x.view(*x.shape[:-2], -1)

        x = self.gmlp(x)
        x = F.relu(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x


class gMLPResMultiConv35(nn.Module):
    '''
    Temporal and Spatial convolution with multiscales
    '''

    def __init__(self, shape=(26, 63), **kwargs):
        super(gMLPResMultiConv35, self).__init__()
        self.conv1a1 = nn.Conv2d(kernel_size=(
            3, 1), in_channels=1, out_channels=8, padding=(1, 0))
        self.conv1a2 = nn.Conv2d(kernel_size=(
            5, 1), in_channels=1, out_channels=8, padding=(2, 0))
        self.conv1b1 = nn.Conv2d(kernel_size=(
            1, 3), in_channels=1, out_channels=8, padding=(0, 1))
        self.conv1b2 = nn.Conv2d(kernel_size=(
            1, 5), in_channels=1, out_channels=8, padding=(0, 2))
        self.conv2 = ResMultiConv(16)
        self.conv3 = ResMultiConv(32)
        self.conv4 = ResMultiConv(64)
        self.conv5 = nn.Conv2d(kernel_size=(
            5, 5), in_channels=128, out_channels=128, padding=2)
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(16)
        self.bn1b = nn.BatchNorm2d(16)
        self.bn5 = nn.BatchNorm2d(128)
        dim = (shape[0]//2) * (shape[1]//4)
        i = 128 * dim
        self.fc = nn.Linear(in_features=i, out_features=4)
        self.dropout = nn.Dropout(0.5)

        self.gmlp = gMLP(dim=dim, depth=1, seq_len=128, act=nn.Tanh())

    def forward(self, *input):
        # input[0]: torch.Size([32, 1, 26, 63])
        xa1 = self.conv1a1(input[0])  # (32, 8, 26, 63)
        xa2 = self.conv1a2(input[0])  # (32, 8, 26, 63)
        xa = torch.cat((xa1, xa2), 1)
        xa = self.bn1a(xa)  # (32, 16, 26, 63)
        xa = F.relu(xa)

        xb1 = self.conv1b1(input[0])
        xb2 = self.conv1b2(input[0])
        xb = torch.cat((xb1, xb2), 1)
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

        # flatten
        shape = x.shape
        x = x.view(*x.shape[:-2], -1)

        x = self.gmlp(x)
        x = F.relu(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x


class aMLPResMultiConv3(nn.Module):
    def __init__(self, shape=(26, 63), **kwargs):
        super(aMLPResMultiConv3, self).__init__()
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
        dim = (shape[0]//2) * (shape[1]//4)
        i = 128 * dim
        self.fc = nn.Linear(in_features=i, out_features=4)
        self.dropout = nn.Dropout(0.5)

        self.gmlp = gMLP(dim=dim, depth=1, seq_len=128,
                         attn_dim=64, act=nn.Tanh())

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

        # flatten
        shape = x.shape
        x = x.view(*x.shape[:-2], -1)

        x = self.gmlp(x)
        x = F.relu(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x
