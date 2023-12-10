from typing import Type, Any, Callable, Union, List, Optional
import torch
from torch import Tensor, nn
import torch.nn.functional as F


class DenseNet(nn.Module):
    """DenseNet 基本模型, 可通过给定参数(growth_rate, block_config,num_init_features)定制模型. 例如:
        densenet121 = DenseNet(32, (6, 12, 24, 16), 64)
        densenet161 = DenseNet(48, (6, 12, 36, 24), 96)
        densenet169 = DenseNet(32, (6, 12, 32, 32), 64)
        densenet201 = DenseNet(32, (6, 12, 48, 32), 64)
    Args:
        growth_rate (int, optional): _description_. Defaults to 32.
        block_config (Tuple[int, int, int, int], optional): _description_. Defaults to (6, 12, 24, 16).
        num_init_features (int, optional): _description_. Defaults to 64.
        bn_size (int, optional): _description_. Defaults to 4.
        drop_rate (float, optional): _description_. Defaults to 0.
        num_classes (int, optional): _description_. Defaults to 1000.
    """

    def __init__(
        self,
        in_channels=3,
        num_classes=1000,
        growth_rate: int = 32,
        block_config: List[int,] = [6, 12, 24, 16],
        num_init_features: int = 64,
        bn_size: int = 4,
        drop_rate: float = 0,
    ) -> None:
        super().__init__()
        self.num_outputs = num_classes

        # First convolution
        self.conv0 = nn.Conv2d(
            in_channels,
            num_init_features,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.norm0 = nn.BatchNorm2d(num_init_features)
        self.relu0 = nn.ReLU(inplace=True)
        self.pool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Each denseblock

        self.denseblock = nn.Sequential()
        self._make_block(
            num_init_features,
            block_config,
            bn_size,
            growth_rate,
            drop_rate,
            self.num_outputs,
        )

    def _make_block(
        self,
        num_init_features,
        block_config,
        bn_size,
        growth_rate,
        drop_rate,
        num_outputs,
    ):
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            # 构造 block
            block = DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
            )
            self.denseblock.add_module("denseblock%d" % (i + 1), block)

            num_features = num_features + num_layers * growth_rate  # 计算 block 的输出特征通道

            # block 过渡层, 每次过渡, 特征通道少一半
            if i != len(block_config) - 1:
                trans = _transitionblock(
                    num_input_features=num_features,
                    num_output_features=num_features // 2,
                )
                self.denseblock.add_module("transition%d" % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.denseblock.add_module("norm5", nn.BatchNorm2d(num_features))

        # Linear layer
        self.last_linear = nn.Linear(num_features, num_outputs)

        # 初始化参数
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        if len(x.size()) == 3:
            x = x.unsqueeze(1)  # x
        out = self.conv0(x)
        out = self.norm0(out)
        out = self.relu0(out)
        out = self.pool0(out)

        out = self.denseblock(out)

        out = F.relu(out, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.last_linear(out)
        return out


def _transitionblock(num_input_features: int, num_output_features: int) -> None:
    transition = nn.Sequential()
    transition.add_module("trans_norm", nn.BatchNorm2d(num_input_features))
    transition.add_module("trans_relu", nn.ReLU(inplace=True))
    transition.add_module(
        "trans_conv",
        nn.Conv2d(
            num_input_features, num_output_features, kernel_size=1, stride=1, bias=False
        ),
    )
    transition.add_module("trans_pool", nn.AvgPool2d(kernel_size=2, stride=2))
    return transition


class DenseBlock(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_input_features: int,
        bn_size: int,
        growth_rate: int,
        drop_rate: float,
    ) -> None:
        super().__init__()

        self.layers = nn.Sequential()
        for i in range(num_layers):
            layer = DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
            )
            self.layers.add_module("denseLayer%d" % (i + 1), layer)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            y = layer(x)
            x = torch.cat((x, y), dim=1)
        return x


class DenseLayer(nn.Module):
    """进行 Dense 的我每一层 Layer"""

    def __init__(
        self, num_input_features: int, growth_rate: int, bn_size: int, drop_rate: float
    ) -> None:
        super().__init__()
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            num_input_features,
            bn_size * growth_rate,
            kernel_size=1,
            stride=1,
            bias=False,
        )

        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)

        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            bn_size * growth_rate,
            growth_rate,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.drop_rate = float(drop_rate)

    def forward(self, input: Tensor) -> Tensor:
        out = self.conv2(
            self.relu2(self.norm2(self.conv1(self.relu1(self.norm1(input)))))
        )
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return out


if __name__ == "__main__":
    from torchinfo import summary

    input = torch.randn(2, 3, 224, 224)
    # densenet121 mult-adds (G): 5.67 params (M): 7,978,856
    model = DenseNet(
        growth_rate=32,
        block_config=[6, 12, 24, 16],
        num_init_features=64,
    )
    output = model(input)
    print("\n\nmodel: densenet121")
    print("input shape: ", input.shape)
    print("output shape: ", output[0].shape)
    summary(model, input_size=(2, 3, 224, 224), depth=0, device="cuda:1")

    # densenet161 mult-adds (G): 15.46 params (M): 28,681,000
    model = DenseNet(
        growth_rate=48,
        block_config=[6, 12, 36, 24],
        num_init_features=96,
    )
    output = model(input)
    print("\n\nmodel: densenet161")
    print("input shape: ", input.shape)
    print("output shape: ", output[0].shape)
    summary(model, input_size=(2, 3, 224, 224), depth=0, device="cuda:1")

    # densenet169 mult-adds (G): 6.72 params (M): 14,149,480
    model = DenseNet(
        growth_rate=32,
        block_config=[6, 12, 32, 32],
        num_init_features=64,
    )
    output = model(input)
    print("\n\nmodel: densenet169")
    print("input shape: ", input.shape)
    print("output shape: ", output[0].shape)
    summary(model, input_size=(2, 3, 224, 224), depth=0, device="cuda:1")

    # densenet201 mult-adds (G): 8.58 params (M): 20,013,928
    model = DenseNet(
        growth_rate=32,
        block_config=[6, 12, 48, 32],
        num_init_features=64,
    )
    output = model(input)
    print("\n\nmodel: densenet201")
    print("input shape: ", input.shape)
    print("output shape: ", output[0].shape)
    summary(model, input_size=(2, 3, 224, 224), depth=0, device="cuda:1")
