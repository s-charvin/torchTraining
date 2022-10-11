import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class AreaAttention(nn.Module):
    """
    Area Attention [1]. This module allows to attend to areas of the memory, where each area
    contains a group of items that are either spatially or temporally adjacent.
    [1] Li, Yang, et al. "Area attention." International Conference on Machine Learning. PMLR 2019
    """

    def __init__(
            self, key_query_size: int, area_key_mode: str = 'mean', area_value_mode: str = 'sum',
            max_area_height: int = 1, max_area_width: int = 1, memory_height: int = 1,
            memory_width: int = 1, dropout_rate: float = 0.0, top_k_areas: int = 0
    ):
        """
        Initializes the Area Attention module.
        :param key_query_size: size for keys and queries
        :param area_key_mode: mode for computing area keys, one of {"mean", "max", "sample", "concat", "max_concat", "sum", "sample_concat", "sample_sum"}
        :param area_value_mode: mode for computing area values, one of {"mean", "sum"}
        :param max_area_height: max height allowed for an area
        :param max_area_width: max width allowed for an area
        :param memory_height: memory height for arranging features into a grid
        :param memory_width: memory width for arranging features into a grid
        :param dropout_rate: dropout rate
        :param top_k_areas: top key areas used for attention
        """
        super(AreaAttention, self).__init__()
        assert area_key_mode in ['mean', 'max', 'sample', 'concat',
                                 'max_concat', 'sum', 'sample_concat', 'sample_sum']
        assert area_value_mode in ['mean', 'max', 'sum']
        self.area_key_mode = area_key_mode
        self.area_value_mode = area_value_mode
        self.max_area_height = max_area_height
        self.max_area_width = max_area_width
        self.memory_height = memory_height
        self.memory_width = memory_width
        self.dropout = nn.Dropout(p=dropout_rate)
        self.top_k_areas = top_k_areas
        self.area_temperature = np.power(key_query_size, 0.5)
        if self.area_key_mode in ['concat', 'max_concat', 'sum', 'sample_concat', 'sample_sum']:
            self.area_height_embedding = nn.Embedding(
                max_area_height, key_query_size // 2)
            self.area_width_embedding = nn.Embedding(
                max_area_width, key_query_size // 2)
            if area_key_mode == 'concat':
                area_key_feature_size = 3 * key_query_size
            elif area_key_mode == 'max_concat':
                area_key_feature_size = 2 * key_query_size
            elif area_key_mode == 'sum':
                area_key_feature_size = key_query_size
            elif area_key_mode == 'sample_concat':
                area_key_feature_size = 2 * key_query_size
            else:  # 'sample_sum'
                area_key_feature_size = key_query_size
            self.area_key_embedding = nn.Sequential(
                nn.Linear(area_key_feature_size, key_query_size),
                nn.ReLU(inplace=True),
                nn.Linear(key_query_size, key_query_size)
            )

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Area Attention module.
        :param q: queries Tensor with shape (batch_size, num_queries, key_query_size)
        :param k: keys Tensor with shape (batch_size, num_keys_values, key_query_size)
        :param v: values Tensor with shape (batch_size, num_keys_values, value_size)
        :returns a Tensor with shape (batch_size, num_queries, value_size)
        """
        k_area = self._compute_area_key(k)
        if self.area_value_mode == 'mean':
            v_area, _, _, _, _ = self._compute_area_features(v)
        elif self.area_value_mode == 'max':
            v_area, _, _ = self._basic_pool(v, fn=torch.max)
        elif self.area_value_mode == 'sum':
            _, _, v_area, _, _ = self._compute_area_features(v)
        else:
            raise ValueError(
                f"Unsupported area value mode={self.area_value_mode}")
        logits = torch.matmul(q, k_area.transpose(1, 2))
        # logits = logits / self.area_temperature
        weights = logits.softmax(dim=-1)
        if self.top_k_areas > 0:
            top_k = min(weights.size(-1), self.top_k_areas)
            top_weights, _ = weights.topk(top_k)
            min_values, _ = torch.min(top_weights, dim=-1, keepdim=True)
            weights = torch.where(weights >= min_values,
                                  weights, torch.zeros_like(weights))
            weights /= weights.sum(dim=-1, keepdim=True)
        weights = self.dropout(weights)
        return torch.matmul(weights, v_area)

    def _compute_area_key(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute the key for each area.
        :param features: a Tensor with shape (batch_size, num_features (height * width), feature_size)
        :return: a Tensor with shape (batch_size, num_areas, feature_size)
        """
        area_mean, area_std, _, area_heights, area_widths = self._compute_area_features(
            features)
        if self.area_key_mode == 'mean':
            return area_mean
        elif self.area_key_mode == 'max':
            area_max, _, _ = self._basic_pool(features)
            return area_max
        elif self.area_key_mode == 'sample':
            if self.training:
                area_mean += (area_std * torch.randn(area_std.size()))
            return area_mean

        height_embed = self.area_height_embedding(
            (area_heights[:, :, 0] - 1).long())
        width_embed = self.area_width_embedding(
            (area_widths[:, :, 0] - 1).long())
        size_embed = torch.cat([height_embed, width_embed], dim=-1)
        if self.area_key_mode == 'concat':
            area_key_features = torch.cat(
                [area_mean, area_std, size_embed], dim=-1)
        elif self.area_key_mode == 'max_concat':
            area_max, _, _ = self._basic_pool(features)
            area_key_features = torch.cat([area_max, size_embed], dim=-1)
        elif self.area_key_mode == 'sum':
            area_key_features = size_embed + area_mean + area_std
        elif self.area_key_mode == 'sample_concat':
            if self.training:
                area_mean += (area_std * torch.randn(area_std.size()))
            area_key_features = torch.cat([area_mean, size_embed], dim=-1)
        elif self.area_key_mode == 'sample_sum':
            if self.training:
                area_mean += (area_std * torch.randn(area_std.size()))
            area_key_features = area_mean + size_embed
        else:
            raise ValueError(f"Unsupported area key mode={self.area_key_mode}")
        area_key = self.area_key_embedding(area_key_features)
        return area_key

    def _compute_area_features(
            self, features: torch.Tensor, epsilon: float = 1e-6
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute features for each area.
        :param features: a Tensor with shape (batch_size, num_features (height * width), feature_size)
        :param epsilon: epsilon added to the variance for computing standard deviation
        :return: a tuple with 5 elements:
          - area_mean: a Tensor with shape (batch_size, num_areas, feature_size)
          - area_std: a Tensor with shape (batch_size, num_areas, feature_size)
          - area_sum: a Tensor with shape (batch_size, num_areas, feature_size)
          - area_heights: a Tensor with shape (batch_size, num_areas, 1)
          - area_widths: a Tensor with shape (batch_size, num_areas, 1)
        """
        area_sum, area_heights, area_widths = self._compute_sum_image(features)
        area_squared_sum, _, _ = self._compute_sum_image(features.square())
        area_size = (area_heights * area_widths).float()
        area_mean = area_sum / area_size
        s2_n = area_squared_sum / area_size
        area_variance = s2_n - area_mean.square()
        area_std = (area_variance.abs() + epsilon).sqrt()
        return area_mean, area_std, area_sum, area_heights, area_widths

    def _compute_sum_image(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute area sums for features.
        :param features: a Tensor with shape (batch_size, num_features (height * width), feature_size)
        :return: a tuple with 3 elements:
          - sum_image: a Tensor with shape (batch_size, num_areas, feature_size)
          - area_heights: a Tensor with shape (batch_size, num_areas, 1)
          - area_widths: a Tensor with shape (batch_size, num_areas, 1)
        """
        batch_size, num_features, feature_size = features.size()
        features_2d = features.reshape(
            batch_size, self.memory_height, num_features//self.memory_height, feature_size)
        horizontal_integral = features_2d.cumsum(dim=-2)
        vertical_integral = horizontal_integral.cumsum(dim=-3)
        padded_image = F.pad(vertical_integral, pad=[0, 0, 1, 0, 1, 0])
        heights = []
        widths = []
        dst_images = []
        src_images_diag = []
        src_images_h = []
        src_images_v = []
        size = torch.ones_like(padded_image[:, :, :, 0], dtype=torch.int32)
        for area_height in range(self.max_area_height):
            for area_width in range(self.max_area_width):
                dst_images.append(
                    padded_image[:, area_height + 1:, area_width + 1:, :].reshape(batch_size, -1, feature_size))
                src_images_diag.append(
                    padded_image[:, :-area_height - 1, :-area_width - 1, :].reshape(batch_size, -1, feature_size))
                src_images_h.append(
                    padded_image[:, area_height + 1:, :-area_width - 1, :].reshape(batch_size, -1, feature_size))
                src_images_v.append(
                    padded_image[:, :-area_height - 1, area_width + 1:, :].reshape(batch_size, -1, feature_size))
                heights.append((size[:, area_height + 1:, area_width + 1:]
                               * (area_height + 1)).reshape(batch_size, -1))
                widths.append((size[:, area_height + 1:, area_width + 1:]
                              * (area_width + 1)).reshape(batch_size, -1))
        sum_image = torch.sub(
            torch.cat(dst_images, dim=1) + torch.cat(src_images_diag, dim=1),
            torch.cat(src_images_v, dim=1) + torch.cat(src_images_h, dim=1)
        )
        area_heights = torch.cat(heights, dim=1).unsqueeze(dim=2)
        area_widths = torch.cat(widths, dim=1).unsqueeze(dim=2)
        return sum_image, area_heights, area_widths

    def _basic_pool(self, features: torch.Tensor, fn=torch.max) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Pool for each area based on a given pooling function (fn).
        :param features: a Tensor with shape (batch_size, num_features (height * width), feature_size).
        :param fn: Torch pooling function
        :return: a tuple with 3 elements:
          - pool_results: a Tensor with shape (batch_size, num_areas, feature_size)
          - area_heights: a Tensor with shape (batch_size, num_areas, 1)
          - area_widths: a Tensor with shape (batch_size, num_areas, 1)
        """
        batch_size, num_features, feature_size = features.size()
        features_2d = features.reshape(
            batch_size, self.memory_height, self.memory_width, feature_size)
        heights = []
        widths = []
        pooled_areas = []
        size = torch.ones_like(features_2d[:, :, :, 0], dtype=torch.int32)
        for area_height in range(self.max_area_height):
            for area_width in range(self.max_area_width):
                pooled_areas.append(self._pool_one_shape(
                    features_2d, area_width=area_width + 1, area_height=area_height + 1, fn=fn))
                heights.append((size[:, area_height:, area_width:]
                               * (area_height + 1)).reshape([batch_size, -1]))
                widths.append((size[:, area_height:, area_width:]
                              * (area_width + 1)).reshape([batch_size, -1]))
        pooled_areas = torch.cat(pooled_areas, dim=1)
        area_heights = torch.cat(heights, dim=1).unsqueeze(dim=2)
        area_widths = torch.cat(widths, dim=1).unsqueeze(dim=2)
        return pooled_areas, area_heights, area_widths

    def _pool_one_shape(self, features_2d: torch.Tensor, area_width: int, area_height: int, fn=torch.max) -> torch.Tensor:
        """
        Pool for an area in features_2d.
        :param features_2d: a Tensor with shape (batch_size, height, width, feature_size)
        :param area_width: max width allowed for an area
        :param area_height: max height allowed for an area
        :param fn: PyTorch pooling function
        :return: a Tensor with shape (batch_size, num_areas, feature_size)
        """
        batch_size, height, width, feature_size = features_2d.size()
        flat_areas = []
        for y_shift in range(area_height):
            image_height = max(height - area_height + 1 + y_shift, 0)
            for x_shift in range(area_width):
                image_width = max(width - area_width + 1 + x_shift, 0)
                area = features_2d[:, y_shift:image_height,
                                   x_shift:image_width, :]
                flatten_area = area.reshape(batch_size, -1, feature_size, 1)
                flat_areas.append(flatten_area)
        flat_area = torch.cat(flat_areas, dim=3)
        pooled_area = fn(flat_area, dim=3).values
        return


class MultiConv(nn.Module):
    '''
    Multi-scale block without short-cut connections
    '''

    def __init__(self, channels=16, **kwargs):
        super(MultiConv, self).__init__()
        self.conv3 = nn.Conv2d(kernel_size=(
            3, 3), in_channels=channels, out_channels=channels, padding=1)
        self.conv5 = nn.Conv2d(kernel_size=(
            5, 5), in_channels=channels, out_channels=channels, padding=2)
        self.bn = nn.BatchNorm2d(channels*2)

    def forward(self, x):
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        x = torch.cat((x3, x5), 1)
        x = self.bn(x)
        x = F.relu(x)
        return x


class ResMultiConv(nn.Module):
    '''
    Multi-scale block with short-cut connections
    '''

    def __init__(self, channels=16, **kwargs):
        super(ResMultiConv, self).__init__()
        self.conv3 = nn.Conv2d(kernel_size=(
            3, 3), in_channels=channels, out_channels=channels, padding=1)
        self.conv5 = nn.Conv2d(kernel_size=(
            5, 5), in_channels=channels, out_channels=channels, padding=2)
        self.bn = nn.BatchNorm2d(channels*2)

    def forward(self, x):
        x3 = self.conv3(x) + x
        x5 = self.conv5(x) + x
        x = torch.cat((x3, x5), 1)
        x = self.bn(x)
        x = F.relu(x)
        return x


class ResConv3(nn.Module):
    '''
    Resnet with 3x3 kernels
    '''

    def __init__(self, channels=16, **kwargs):
        super(ResConv3, self).__init__()
        self.conv3 = nn.Conv2d(kernel_size=(
            3, 3), in_channels=channels, out_channels=2*channels, padding=1)
        self.bn = nn.BatchNorm2d(channels*2)

    def forward(self, x):
        x = self.conv3(x) + torch.cat((x, x), 1)
        x = self.bn(x)
        x = F.relu(x)
        return x


class ResConv5(nn.Module):
    '''
    Resnet with 5x5 kernels
    '''

    def __init__(self, channels=16, **kwargs):
        super(ResConv5, self).__init__()
        self.conv5 = nn.Conv2d(kernel_size=(
            5, 5), in_channels=channels, out_channels=2*channels, padding=2)
        self.bn = nn.BatchNorm2d(channels*2)

    def forward(self, x):
        x = self.conv5(x) + torch.cat((x, x), 1)
        x = self.bn(x)
        x = F.relu(x)
        return x


class CNN_Area(nn.Module):
    '''Area attention, Mingke Xu, 2020
    '''

    def __init__(self, height=3, width=3, out_size=4, shape=(26, 63), **kwargs):
        super(CNN_Area, self).__init__()
        self.height = height
        self.width = width
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

        i = 80 * ((shape[0]-1)//2) * ((shape[1]-1)//4)
        self.fc = nn.Linear(in_features=i, out_features=4)

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

        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x


class CNN_AttnPooling(nn.Module):
    '''Head atention, Mingke Xu, 2020
    Attention pooling, Pengcheng Li, Interspeech, 2019
    '''

    def __init__(self, head=4, attn_hidden=64, shape=(26, 63), **kwargs):
        super(CNN_AttnPooling, self).__init__()
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
        self.top_down = nn.Conv2d(kernel_size=(
            1, 1), in_channels=80, out_channels=4)
        self.bottom_up = nn.Conv2d(kernel_size=(
            1, 1), in_channels=80, out_channels=1)
        # i = 80 * ((shape[0]-1)//4) * ((shape[1]-1)//4)
        # self.fc = nn.Linear(in_features=i, out_features=4)

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

        x1 = self.top_down(x)
        x1 = F.softmax(x1, 1)
        x2 = self.bottom_up(x)

        x = x1 * x2

        # x = x.sum((2,3))
        x = x.mean((2, 3))

        return x


class CNN_GAP(nn.Module):
    '''Head atention, Mingke Xu, 2020
    Attention pooling, Pengcheng Li, Interspeech, 2019
    '''

    def __init__(self, head=4, attn_hidden=64, shape=(26, 63), **kwargs):
        super(CNN_GAP, self).__init__()
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
        i = 80 * ((shape[0]-1)//4) * ((shape[1]-1)//4)
        self.fc = nn.Linear(in_features=i, out_features=4)

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


class ResMultiConv3(nn.Module):
    '''
    GLAM - gMLP
    '''

    def __init__(self, shape=(26, 63), **kwargs):
        super(ResMultiConv3, self).__init__()
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

        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x


class ResMultiConv5(nn.Module):
    '''
    GLAM5 - gMLP
    '''

    def __init__(self, shape=(26, 63), **kwargs):
        super(ResMultiConv5, self).__init__()
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

        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x
