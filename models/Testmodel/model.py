
from torch import nn
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F
from torch import Tensor, nn
import torch
from . import components
from transformers import BertModel, BertTokenizer, BertConfig, Wav2Vec2Model, Wav2Vec2Config, Wav2Vec2Processor, Trainer, get_linear_schedule_with_warmup
from .leaf_pytorch import Leaf


class Testmodel(nn.Module):
    """ 
    """

    def __init__(self) -> None:
        super().__init__()
        # 前置特征处理
        self.pre_extract_s1 = components.Conv2dFeatureExtractor(
            in_channel=1,
            shape=[
                [32, {"kernel_size": [3, 3], "stride":[1, 1]},
                 "bn", {
                    "num_features": 32},
                 "", {},
                 "relu", {}],
            ]
        )
        self.pre_extract_s2 = components.Conv2dFeatureExtractor(
            in_channel=1,
            shape=[
                [32, {"kernel_size": [3, 3], "stride":[1, 1]},
                 "bn", {
                    "num_features": 32},
                 "", {},
                 "relu", {}],
            ]
        )
        self.pre_extract_s3 = components.Conv2dFeatureExtractor(
            in_channel=1,
            shape=[
                [32, {"kernel_size": [3, 3], "stride":[1, 1]},
                 "bn", {
                    "num_features": 32},
                 "", {},
                 "relu", {}],
            ]
        )
        # 多 Stream 处理头
        self.slow_extract_head_s1 = components.ReparamLargeKernelConv(
            in_channels=32,
            out_channels=64,
            kernel_size=[3, 7],
            stride=[1, 1],
            groups=1,
            small_kernel=[3, 3])
        self.fast_extract_head_s2 = components.ReparamLargeKernelConv(
            in_channels=32,
            out_channels=64,
            kernel_size=[11, 3],
            stride=[1, 1],
            groups=1,
            small_kernel=[3, 3]
        )
        self.norm_extract_head_s3 = components.Conv2dFeatureExtractor(
            in_channel=32,
            shape=[
                [64, {"kernel_size": [3, 3], "stride":[1, 1], "padding":"same"},
                 "bn", {
                    "num_features": 64},
                 "", {},
                 "relu", {}],
            ]
        )
        # 多 Stream 处理尾
        self.slow_extract_s1 = components.Conv2dFeatureExtractor(
            in_channel=64,
            shape=[
                [64, {"kernel_size": [3, 3], "stride":[1, 1]},
                 "bn", {
                    "num_features": 64},
                 "", {},
                 "relu", {}],
                [64, {"kernel_size": [1, 1], "stride":[1, 1]},
                 "bn", {
                    "num_features": 64},
                 "", {},
                 "relu", {}],
                [64, {"kernel_size": [3, 3], "stride":[1, 1]},
                 "bn", {
                    "num_features": 64},
                 "", {},
                 "relu", {}],
            ]
        )
        self.fast_extract_s2 = components.Conv2dFeatureExtractor(
            in_channel=64,
            shape=[
                [64, {"kernel_size": [3, 3], "stride":[1, 1]},
                 "bn", {
                    "num_features": 64},
                 "", {},
                 "relu", {}],
                [64, {"kernel_size": [1, 1], "stride":[1, 1]},
                 "bn", {
                    "num_features": 64},
                 "", {},
                 "relu", {}],
                [64, {"kernel_size": [3, 3], "stride":[1, 1]},
                 "bn", {
                    "num_features": 64},
                 "", {},
                 "relu", {}],
            ]
        )
        self.norm_extract_s3 = components.Conv2dFeatureExtractor(
            in_channel=64,
            shape=[
                [64, {"kernel_size": [3, 3], "stride":[1, 1]},
                 "bn", {
                    "num_features": 64},
                 "", {},
                 "relu", {}],
                [64, {"kernel_size": [1, 1], "stride":[1, 1]},
                 "bn", {
                    "num_features": 64},
                 "", {},
                 "relu", {}],
                [64, {"kernel_size": [3, 3], "stride":[1, 1]},
                 "bn", {
                    "num_features": 64},
                 "", {},
                 "relu", {}],
            ]
        )
        self.conv_extractor = components.Conv2dFeatureExtractor(
            in_channel=64*3,
            shape=[
                [64, {"kernel_size": [3, 3], "stride":[1, 1]},
                 "bn", {
                    "num_features": 64},
                 "ap", {"kernel_size": [2, 2]},
                 "relu", {}],

                [96, {"kernel_size": [3, 3], "stride":[1, 1]},
                 "bn", {
                    "num_features": 96},
                 "ap", {"kernel_size": [2, 2]},
                 "relu", {}],

                [128, {"kernel_size": [3, 3], "stride":[1, 1]},
                 "bn", {
                    "num_features": 128},
                 "ap", {"kernel_size": [2, 1]},
                 "relu", {}],

                [160, {"kernel_size": [3, 3], "stride":[1, 1]},
                 "bn", {
                    "num_features": 160},
                 "ap", {"kernel_size": [2, 1]},
                 "relu", {}],

                [320, {"kernel_size": [1, 1], "stride":[1, 1]},
                 "bn", {
                    "num_features": 320},
                 "gap", {"kernel_size": [1, 1]},
                 "relu", {}],
            ]
        )
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(in_features=320, out_features=4)

    def forward(self, x: Tensor) -> Tensor:
        if len(x.shape) == 3:
            x = x.unsqueeze(1)

        out1 = self.pre_extract_s1(x)
        out1 = self.slow_extract_head_s1(out1)
        out1 = self.slow_extract_s1(out1)

        out2 = self.pre_extract_s2(x)
        out2 = self.fast_extract_head_s2(out2)
        out2 = self.fast_extract_s2(out2)

        out3 = self.pre_extract_s3(x)
        out3 = self.norm_extract_head_s3(out3)
        out3 = self.norm_extract_s3(out3)

        x = torch.cat([out1, out2, out3], dim=1)
        x = self.conv_extractor(x)
        x = self.dropout(x.squeeze(2).squeeze(2))
        x = self.classifier(x)

        return F.softmax(x, dim=1)


class Testmodel_Reparam(nn.Module):
    """ 
    """

    def __init__(self) -> None:
        super().__init__()
        # 多 Stream 处理头
        self.slow_extract_head_s1 = components.ReparamLargeKernelConv(
            in_channels=1,
            out_channels=32,
            kernel_size=[11, 1],
            stride=[1, 1],
            groups=1,
            small_kernel=[3, 1])
        self.fast_extract_head_s2 = components.ReparamLargeKernelConv(
            in_channels=1,
            out_channels=32,
            kernel_size=[1, 9],
            stride=[1, 1],
            groups=1,
            small_kernel=[1, 3]
        )
        self.norm_extract_head_s3 = components.Conv2dFeatureExtractor(
            in_channel=1,
            shape=[
                [32, {"kernel_size": [3, 3], "stride":[1, 1], "padding":"same"},
                 "bn", {
                    "num_features": 32},
                 "", {},
                 "relu", {}],
            ]
        )
        # 多 Stream 处理尾

        self.conv_extractor = components.Conv2dFeatureExtractor(
            in_channel=32*3,
            shape=[
                [64, {"kernel_size": [3, 3], "stride":[1, 1]},
                 "bn", {
                    "num_features": 64},
                 "ap", {"kernel_size": [2, 2]},
                 "relu", {}],

                [96, {"kernel_size": [3, 3], "stride":[1, 1]},
                 "bn", {
                    "num_features": 96},
                 "ap", {"kernel_size": [2, 2]},
                 "relu", {}],

                [128, {"kernel_size": [3, 3], "stride":[1, 1]},
                 "bn", {
                    "num_features": 128},
                 "ap", {"kernel_size": [2, 1]},
                 "relu", {}],

                [160, {"kernel_size": [3, 3], "stride":[1, 1]},
                 "bn", {
                    "num_features": 160},
                 "ap", {"kernel_size": [2, 1]},
                 "relu", {}],

                [320, {"kernel_size": [1, 1], "stride":[1, 1]},
                 "bn", {
                    "num_features": 320},
                 "gap", {"kernel_size": [1, 1]},
                 "relu", {}],
            ]
        )
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(in_features=320, out_features=4)

    def forward(self, x: Tensor) -> Tensor:
        if len(x.shape) == 3:
            x = x.unsqueeze(1)

        out1 = self.slow_extract_head_s1(x)

        out2 = self.fast_extract_head_s2(x)

        out3 = self.norm_extract_head_s3(x)

        x = torch.cat([out1, out2, out3], dim=1)
        x = self.conv_extractor(x)
        x = self.dropout(x.squeeze(2).squeeze(2))
        x = self.classifier(x)

        return F.softmax(x, dim=1)


class Testmodel_head(nn.Module):
    """
    """

    def __init__(self) -> None:
        super().__init__()

       # 多 Stream 处理头
        self.slow_extract_head_s1 = components.Conv2dFeatureExtractor(
            in_channel=1,
            shape=[
                [32, {"kernel_size": [11, 1], "stride":[1, 1], "padding":"same"},
                 "bn", {
                    "num_features": 32},
                 "", {},
                 "relu", {}],
            ]
        )
        self.fast_extract_head_s2 = components.Conv2dFeatureExtractor(
            in_channel=1,
            shape=[
                [32, {"kernel_size": [1, 9], "stride":[1, 1], "padding":"same"},
                 "bn", {
                    "num_features": 32},
                 "", {},
                 "relu", {}],
            ]
        )
        self.norm_extract_head_s3 = components.Conv2dFeatureExtractor(
            in_channel=1,
            shape=[
                [32, {"kernel_size": [3, 3], "stride":[1, 1], "padding":"same"},
                 "bn", {
                    "num_features": 32},
                 "", {},
                 "relu", {}],
            ]
        )
        self.slow_extract_head_s4 = components.Conv2dFeatureExtractor(
            in_channel=1,
            shape=[
                [32, {"kernel_size": [7, 1], "stride":[1, 1], "padding":"same"},
                 "bn", {
                    "num_features": 32},
                 "", {},
                 "relu", {}],
            ]
        )
        self.fast_extract_head_s5 = components.Conv2dFeatureExtractor(
            in_channel=1,
            shape=[
                [32, {"kernel_size": [1, 5], "stride":[1, 1], "padding":"same"},
                 "bn", {
                    "num_features": 32},
                 "", {},
                 "relu", {}],
            ]
        )
        self.norm_extract_head_s6 = components.Conv2dFeatureExtractor(
            in_channel=1,
            shape=[
                [32, {"kernel_size": [5, 5], "stride":[1, 1], "padding":"same"},
                 "bn", {
                    "num_features": 32},
                 "", {},
                 "relu", {}],
            ]
        )

        self.conv_extractor = components.Conv2dFeatureExtractor(
            in_channel=32*6,
            shape=[
                [64, {"kernel_size": [3, 3], "stride":[1, 1]},
                 "bn", {
                    "num_features": 64},
                 "ap", {"kernel_size": [2, 2]},
                 "relu", {}],

                [96, {"kernel_size": [3, 3], "stride":[1, 1]},
                 "bn", {
                    "num_features": 96},
                 "ap", {"kernel_size": [2, 2]},
                 "relu", {}],

                [128, {"kernel_size": [3, 3], "stride":[1, 1]},
                 "bn", {
                    "num_features": 128},
                 "ap", {"kernel_size": [2, 1]},
                 "relu", {}],

                [160, {"kernel_size": [3, 3], "stride":[1, 1]},
                 "bn", {
                    "num_features": 160},
                 "ap", {"kernel_size": [2, 1]},
                 "relu", {}],

                [320, {"kernel_size": [1, 1], "stride":[1, 1]},
                 "bn", {
                    "num_features": 320},
                 "gap", {"kernel_size": [1, 1]},
                 "relu", {}],
            ]
        )
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(in_features=320, out_features=4)

    def forward(self, x: Tensor) -> Tensor:
        if len(x.shape) == 3:
            x = x.unsqueeze(1)

        out1 = self.slow_extract_head_s1(x)

        out2 = self.fast_extract_head_s2(x)

        out3 = self.norm_extract_head_s3(x)
        out4 = self.slow_extract_head_s4(x)

        out5 = self.fast_extract_head_s5(x)

        out6 = self.norm_extract_head_s6(x)
        x = torch.cat([out1, out2, out3, out4, out5, out6], dim=1)
        x = self.conv_extractor(x)
        x = self.dropout(x.squeeze(2).squeeze(2))
        x = self.classifier(x)

        return F.softmax(x, dim=1)


class Testmodel_STL_head(nn.Module):
    """
    """

    def __init__(self):
        super().__init__()
        self.extractor = components.Conv2dFeatureExtractor(
            in_channel=1,
            shape=[
                [16, {"kernel_size": [2, 2], "stride":[1, 1], "padding":"same"},
                 "bn", {
                    "num_features": 16},
                 "ap", {"kernel_size": [2, 2]},
                 "relu", {}],

                [32, {"kernel_size": [3, 3], "stride":[1, 1], "padding":"same"},
                 "bn", {
                    "num_features": 32},
                 "ap", {"kernel_size": [2, 2]},
                 "relu", {}],

                [64, {"kernel_size": [3, 3], "stride":[2, 2], "padding":"same"},
                 "bn", {
                    "num_features": 64},
                 "ap", {"kernel_size": [2, 2]},
                 "relu", {}],

                [64, {"kernel_size": [3, 3], "stride":[1, 1], "padding":"same"},
                 "bn", {
                    "num_features": 64},
                 "", {},
                 "relu", {}],

                [64, {"kernel_size": [2, 2], "stride":[1, 1], "padding":"same"},
                 "bn", {
                    "num_features": 64},
                 "ap", {"kernel_size": [2, 2]},
                 "relu", {}],
            ]
        )

        self.bilstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=1,
                              bias=False, batch_first=True, dropout=0., bidirectional=True)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(in_features=256, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=4)

    def forward(self, x: Tensor):
        if x.ndim == 3:
            x = x.unsqueeze(1)
        # [batch, 1 , frames, feature]
        x = self.extractor(x)
        # [batch, channel , out_frames, out_feature]
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        # [batch, out_frames, channel*out_feature]
        x, _ = self.bilstm(x)
        # x = [batch, seq_len, num_directions * hidden_size]
        # _[] = [batch, num_layers * num_directions, hidden_size]
        x = x.permute(0, 2, 1)
        # [batch, num_directions * hidden_size, seq_len]
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes *
                 self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(
            spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat(
                [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, width=64):
        super().__init__()
        self.output_dim = output_dim

        # the 3-layer stem
        self.conv1 = nn.Conv2d(1, width // 2, kernel_size=3,
                               stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2,
                               kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(
            width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        # embed_dim = width * 32  # the ResNet feature dimension
        # self.attnpool = AttentionPool2d(
        #     input_resolution // 32, embed_dim, heads, output_dim)
        self.attnpool = nn.AdaptiveAvgPool2d(1)
        self.outputprog = nn.Linear(
            in_features=self._inplanes, out_features=output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)
        x = x.squeeze()
        x = self.outputprog(x)
        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class LightSerNet(nn.Module):
    """
    paper: Light-SERNet: A lightweight fully convolutional neural network for speech emotion recognition
    """

    def __init__(self) -> None:
        super().__init__()
        self.path1 = nn.Sequential(
            components.Conv2dSame(in_channels=1, out_channels=32, kernel_size=(
                11, 1), stride=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2)
        )
        self.path2 = nn.Sequential(
            components.Conv2dSame(in_channels=1, out_channels=32, kernel_size=(
                1, 9), stride=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2)
        )
        self.path3 = nn.Sequential(
            components.Conv2dSame(in_channels=1, out_channels=32, kernel_size=(
                3, 3), stride=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2)
        )
        self.conv_extractor = components._get_Conv2d_extractor(
            in_channels=32*3,
            shapes=[  # [channel:int, kernel:list|int, stride:list|int, norm:str|None, pool:str|None, pool_kernel:list|int]
                [64, [3, 3], 1, "bn", "ap", [2, 2]],
                [128, [3, 3], 1, "bn", "ap", [2, 2]],
                [256, [3, 3], 1, "bn", "ap", [2, 1]],
                [512, [3, 3], 1, "bn", "ap", [2, 1]],
                [1024, [1, 1], 1, "bn", "gap", 1]
            ],
            bias=False
        )
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(in_features=320, out_features=4)

    def forward(self, x: Tensor) -> Tensor:
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        out1 = self.path1(x)
        out2 = self.path2(x)
        out3 = self.path3(x)

        x = torch.cat([out1, out2, out3], dim=1)

        x = self.conv_extractor(x)
        x = self.dropout(x.squeeze(2).squeeze(2))

        return x


class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_dim = 1024  # text_projection

        vision_width = 64  # visual.layer1.0.conv1.weight
        vision_layers = (3, 4, 6, 3)  # 层数
        speech_resolution = 40  # √(attnpool.positional_embedding[0]-1) * 32
        vision_heads = vision_width * 32 // 64
        self.leaf = Leaf(n_filters=40,
                         sample_rate=int(16000),
                         window_len=64.,
                         window_stride=16.,
                         preemp=False,
                         init_min_freq=80.0,
                         init_max_freq=7600.0,
                         mean_var_norm=False,
                         pcen_compression=True,
                         use_legacy_complex=False,
                         initializer="default")
        self.speechencoder = LightSerNet()
        # self.speechencoder = ModifiedResNet(
        #     layers=vision_layers, output_dim=self.embed_dim, width=vision_width)
        # self.speechencoder = components.ResNet(components.tvresnet.Bottleneck, [3, 4, 6, 3],num_classes=self.embed_dim)
        # self.speechencoder = Wav2Vec2Model.from_pretrained(
        #     "facebook/wav2vec2-base-960h")

        self.textencoder = BertModel.from_pretrained(
            "bert-base-uncased",  # 小写的 12 层预训练模型
        )

        self.ln_final = LayerNorm(768)

        self.text_projection = nn.Parameter(
            torch.empty(768, self.embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.initialize_parameters()

    def initialize_parameters(self):
        # 初始化语音编码器参数
        if isinstance(self.speechencoder, ModifiedResNet):
            # if self.speechencoder.attnpool is not None:
            #     std = self.speechencoder.attnpool.c_proj.in_features ** -0.5
            #     nn.init.normal_(
            #         self.speechencoder.attnpool.q_proj.weight, std=std)
            #     nn.init.normal_(
            #         self.speechencoder.attnpool.k_proj.weight, std=std)
            #     nn.init.normal_(
            #         self.speechencoder.attnpool.v_proj.weight, std=std)
            #     nn.init.normal_(
            #         self.speechencoder.attnpool.c_proj.weight, std=std)
            for resnet_block in [self.speechencoder.layer1, self.speechencoder.layer2, self.speechencoder.layer3, self.speechencoder.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)
        # 初始化文本序列映射矩阵参数
        if self.text_projection is not None:
            nn.init.normal_(self.text_projection,
                            std=self.embed_dim ** -0.5)

    @property
    def dtype(self):
        # return self.speechencoder.feature_extractor.conv_layers[0].conv.weight.dtype # wav2vec
        # return self.speechencoder.conv1.weight.dtype  # resnet

        # light
        return self.speechencoder.conv_extractor.conv_blocks[0].conv.weight.dtype

    def encode_speech(self, speech):
        if isinstance(self.speechencoder, ModifiedResNet):
            output = self.speechencoder(speech.type(self.dtype))
        elif isinstance(self.speechencoder, Wav2Vec2Model):
            output = self.speechencoder(
                speech.type(self.dtype)).last_hidden_state
        else:
            output = self.speechencoder(
                speech.type(self.dtype))
        return output

    def encode_text(self, input_ids, attention_mask):
        outputs = self.textencoder(input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=None,
                                   position_ids=None)
        x = outputs["pooler_output"].type(self.dtype)
        x = self.ln_final(x).type(self.dtype)
        x = x @ self.text_projection
        return x

    def forward(self, speechs, text_ids, text_mask, target_text_ids, target_text_mask):
        speech_features = self.leaf(speechs)
        speech_features = self.encode_speech(
            speech_features.permute(0, 1, 2).unsqueeze(1))

        text_features = self.encode_text(text_ids, text_mask)
        target_features = self.encode_text(target_text_ids, target_text_mask)

        # normalized features
        speech_features = speech_features / \
            speech_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        target_features = target_features / \
            target_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_speech = logit_scale * speech_features @ target_features.t()
        logits_per_text = logit_scale * text_features @ target_features.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_speech, logits_per_text


class Net_DSM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_dim = 1024  # text_projection

        vision_width = 64  # visual.layer1.0.conv1.weight
        vision_layers = (3, 4, 6, 3)  # 层数
        speech_resolution = 40  # √(attnpool.positional_embedding[0]-1) * 32
        vision_heads = vision_width * 32 // 64
        self.leaf = Leaf(n_filters=40,
                         sample_rate=int(16000),
                         window_len=64.,
                         window_stride=16.,
                         preemp=False,
                         init_min_freq=80.0,
                         init_max_freq=7600.0,
                         mean_var_norm=False,
                         pcen_compression=True,
                         use_legacy_complex=False,
                         initializer="default")
        self.speechencoder = LightSerNet()
        # self.speechencoder = ModifiedResNet(
        #     layers=vision_layers, output_dim=self.embed_dim, width=vision_width)
        # self.speechencoder = components.ResNet(components.tvresnet.Bottleneck, [3, 4, 6, 3],num_classes=self.embed_dim)
        # self.speechencoder = Wav2Vec2Model.from_pretrained(
        #     "facebook/wav2vec2-base-960h")

        self.textencoder = BertModel.from_pretrained(
            "bert-base-uncased",  # 小写的 12 层预训练模型
        )

        self.ln_final = LayerNorm(768)

        self.text_projection = nn.Parameter(
            torch.empty(768, self.embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.initialize_parameters()

    def initialize_parameters(self):
        # 初始化语音编码器参数
        if isinstance(self.speechencoder, ModifiedResNet):
            # if self.speechencoder.attnpool is not None:
            #     std = self.speechencoder.attnpool.c_proj.in_features ** -0.5
            #     nn.init.normal_(
            #         self.speechencoder.attnpool.q_proj.weight, std=std)
            #     nn.init.normal_(
            #         self.speechencoder.attnpool.k_proj.weight, std=std)
            #     nn.init.normal_(
            #         self.speechencoder.attnpool.v_proj.weight, std=std)
            #     nn.init.normal_(
            #         self.speechencoder.attnpool.c_proj.weight, std=std)
            for resnet_block in [self.speechencoder.layer1, self.speechencoder.layer2, self.speechencoder.layer3, self.speechencoder.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)
        # 初始化文本序列映射矩阵参数
        if self.text_projection is not None:
            nn.init.normal_(self.text_projection,
                            std=self.embed_dim ** -0.5)

    @property
    def dtype(self):
        # return self.speechencoder.feature_extractor.conv_layers[0].conv.weight.dtype # wav2vec
        # return self.speechencoder.conv1.weight.dtype  # resnet

        # light
        return self.speechencoder.conv_extractor.conv_blocks[0].conv.weight.dtype

    def encode_speech(self, speech):
        if isinstance(self.speechencoder, ModifiedResNet):
            output = self.speechencoder(speech.type(self.dtype))
        elif isinstance(self.speechencoder, Wav2Vec2Model):
            output = self.speechencoder(
                speech.type(self.dtype)).last_hidden_state
        else:
            output = self.speechencoder(
                speech.type(self.dtype))
        return output

    def encode_text(self, input_ids, attention_mask):
        outputs = self.textencoder(input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=None,
                                   position_ids=None)
        x = outputs["pooler_output"].type(self.dtype)
        x = self.ln_final(x).type(self.dtype)
        x = x @ self.text_projection
        return x

    def forward(self, speechs, target_text_ids, target_text_mask):
        speech_features = self.leaf(speechs)
        speech_features = self.encode_speech(
            speech_features.permute(0, 1, 2).unsqueeze(1))

        target_features = self.encode_text(target_text_ids, target_text_mask)

        # normalized features
        speech_features = speech_features / \
            speech_features.norm(dim=1, keepdim=True)

        target_features = target_features / \
            target_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_speech = logit_scale * speech_features @ target_features.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_speech


def build_model():
    model = CLIP()
    print(model)
    return model.train()


if __name__ == '__main__':
    pass
    # model = Testmodel_head()
    # model.eval()

    # print('------------------- training-time model -------------')
    # print(model)
    # x = torch.randn(2, 1, 256, 40)
    # origin_y = model(x)
    # print(origin_y.shape)
