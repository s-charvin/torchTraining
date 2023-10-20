import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Conv1dSame(torch.nn.Conv1d):

    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sq = x.size()[-2]

        pad_sq = self.calc_same_pad(
            i=sq, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0])

        if pad_sq > 0:
            x = F.pad(
                x, [pad_sq // 2, pad_sq - pad_sq // 2]
            ).contiguous()
        return F.conv1d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        ).contiguous()


def smooth_labels(labels, factor=0.1):
    # smooth the labels
    labels *= (1 - factor)
    labels += (factor / labels.shape[1])
    return labels


class WeightLayer(nn.Module):
    def __init__(self, in_channels):
        super(WeightLayer, self).__init__()
        self.kernel = nn.Parameter(torch.Tensor(in_channels, 1))
        nn.init.uniform_(self.kernel)

    def forward(self, x):
        tempx = x.transpose(1, 2)
        x = torch.matmul(tempx, self.kernel)
        x = x.squeeze(dim=-1)
        return x


class Temporal_Aware_Block(nn.Module):
    def __init__(self, dilation, nb_filters, kernel_size, dropout_rate=0):
        super(Temporal_Aware_Block, self).__init__()

        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.conv_1_1 = Conv1dSame(
            nb_filters, nb_filters, kernel_size, dilation=dilation)
        self.bn_1_1 = nn.BatchNorm1d(nb_filters)
        self.dropout_1 = nn.Dropout(dropout_rate)

        self.conv_2_1 = Conv1dSame(
            nb_filters, nb_filters, kernel_size, dilation=dilation)
        self.bn_2_1 = nn.BatchNorm1d(nb_filters)
        self.dropout_2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        original_x = x

        conv_1_1 = self.conv_1_1(x)
        conv_1_1 = self.bn_1_1(conv_1_1)
        conv_1_1 = self.activation(conv_1_1)
        output_1_1 = self.dropout_1(conv_1_1)

        conv_2_1 = self.conv_2_1(output_1_1)
        conv_2_1 = self.bn_2_1(conv_2_1)
        conv_2_1 = self.activation(conv_2_1)
        output_2_1 = self.dropout_2(conv_2_1)

        output_2_1 = self.sigmoid(output_2_1)

        F_x = original_x * output_2_1
        return F_x


class TIMNet(nn.Module):
    def __init__(self, num_classes=4, nb_filters=64, kernel_size=2, nb_stacks=1, dilations=8,
                 activation="relu", dropout_rate=0.1, return_sequences=True):
        super(TIMNet, self).__init__()
        self.return_sequences = return_sequences
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.dilations = dilations
        self.nb_stacks = nb_stacks
        self.kernel_size = kernel_size
        self.nb_filters = nb_filters

        self.supports_masking = True
        self.mask_value = 0.

        self.conv_forward = nn.Conv1d(
            40, nb_filters, kernel_size=1, dilation=1)
        self.conv_backward = nn.Conv1d(
            40, nb_filters, kernel_size=1, dilation=1)

        self.temporal_aware_blocks_forward = nn.ModuleList()
        self.temporal_aware_blocks_backward = nn.ModuleList()

        for _ in range(nb_stacks):
            for i in range(dilations):
                self.temporal_aware_blocks_forward.append(
                    Temporal_Aware_Block(2 ** i, nb_filters, kernel_size, dropout_rate))
                self.temporal_aware_blocks_backward.append(
                    Temporal_Aware_Block(2 ** i, nb_filters, kernel_size, dropout_rate))

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        self.decision = WeightLayer(in_channels=nb_stacks * dilations)
        self.predictions = nn.Linear(nb_filters, num_classes)

    def forward(self, x, x_len=None):
        x = x.squeeze().permute(0, 2, 1)
        # [B, F, S]
        forward = x
        backward = torch.flip(x, dims=[1])

        forward_convd = self.conv_forward(forward)
        backward_convd = self.conv_backward(backward)

        final_skip_connection = []

        skip_out_forward = forward_convd
        skip_out_backward = backward_convd

        for i in range(self.nb_stacks):
            for j in range(self.dilations):
                skip_out_forward = self.temporal_aware_blocks_forward[i * self.dilations + j](
                    skip_out_forward)
                skip_out_backward = self.temporal_aware_blocks_backward[i * self.dilations + j](
                    skip_out_backward)

                temp_skip = skip_out_forward + skip_out_backward
                temp_skip = self.global_avg_pool(temp_skip)
                temp_skip = temp_skip.permute(0, 2, 1)
                final_skip_connection.append(temp_skip)

        x = torch.stack(final_skip_connection, dim=1)
        x = x.squeeze()
        x = self.decision(x)
        x = self.predictions(x)
        return x, None
