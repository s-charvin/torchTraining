import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F

__all__ = ["Res2Net", "res2net50"]


model_urls = {
    "res2net50_26w_4s": "https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_26w_4s-06e79181.pth",
    "res2net50_48w_2s": "https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_48w_2s-afed724a.pth",
    "res2net50_14w_8s": "https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_14w_8s-6527dddc.pth",
    "res2net50_26w_6s": "https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_26w_6s-19041792.pth",
    "res2net50_26w_8s": "https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_26w_8s-2c7c9f12.pth",
    "res2net101_26w_4s": "https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net101_26w_4s-02a759a1.pth",
}


class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        baseWidth=26,
        scale=4,
        stype="normal",
    ):
        """Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == "stage":
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(
                nn.Conv2d(
                    width, width, kernel_size=3, stride=stride, padding=1, bias=False
                )
            )
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(
            width * scale, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == "stage":
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == "normal":
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == "stage":
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Res2Net(nn.Module):
    def __init__(
        self, block, layers, baseWidth=26, scale=4, in_channels=3, num_classes=1000
    ):
        self.inplanes = 64
        super(Res2Net, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample=downsample,
                stype="stage",
                baseWidth=self.baseWidth,
                scale=self.scale,
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale)
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def res2net50(pretrained=False, **kwargs):
    """Constructs a Res2Net-50 model.
    Res2Net-50 refers to the Res2Net-50_26w_4s.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return res2net50_26w_4s(pretrained=pretrained, **kwargs)


def res2net50_26w_4s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_26w_4s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["res2net50_26w_4s"]))
    return model


def res2net101_26w_4s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_26w_4s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 23, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["res2net101_26w_4s"]))
    return model


def res2net50_26w_6s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_26w_4s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=6, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["res2net50_26w_6s"]))
    return model


def res2net50_26w_8s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_26w_4s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=8, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["res2net50_26w_8s"]))
    return model


def res2net50_48w_2s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_48w_2s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=48, scale=2, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["res2net50_48w_2s"]))
    return model


def res2net50_14w_8s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_14w_8s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=14, scale=8, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["res2net50_14w_8s"]))
    return model


if __name__ == "__main__":
    # 测试前将 from . import components 改为 import components
    # 测试后再改回来

    from torchinfo import summary

    input = torch.randn(2, 3, 224, 224)
    # res2net50 mult-adds (G): 8.51 params (M): 25,699,120
    model = res2net50_26w_4s()
    output = model(input)
    print("model: res2net50_26w_4s")
    print("input shape: ", input.shape)
    print("output shape: ", output[0].shape)
    summary(model, input_size=(2, 3, 224, 224), depth=0, device="cuda:1")

    # res2net50_26w_6s mult-adds (G): 12.59 params (M): 37,051,448
    model = res2net50_26w_6s()
    output = model(input)
    print("\n\nmodel: res2net50_26w_6s")
    print("input shape: ", input.shape)
    print("output shape: ", output[0].shape)
    summary(model, input_size=(2, 3, 224, 224), depth=0, device="cuda:1")

    # res2net50_26w_8s mult-adds (G): 16.67 params (M): 48,403,776
    model = res2net50_26w_8s()
    output = model(input)
    print("\n\nmodel: res2net50_26w_8s")
    print("input shape: ", input.shape)
    print("output shape: ", output[0].shape)
    summary(model, input_size=(2, 3, 224, 224), depth=0, device="cuda:1")

    # res2net50_48w_2s mult-adds (G): 8.32 params (M): 25,287,304
    model = res2net50_48w_2s()
    output = model(input)
    print("\n\nmodel: res2net50_48w_2s")
    print("input shape: ", input.shape)
    print("output shape: ", output[0].shape)
    summary(model, input_size=(2, 3, 224, 224), depth=0, device="cuda:1")

    # res2net50_14w_8s mult-adds (G): 8.36 params (M): 25,059,816
    model = res2net50_14w_8s()
    output = model(input)
    print("\n\nmodel: res2net50_14w_8s")
    print("input shape: ", input.shape)
    print("output shape: ", output[0].shape)
    summary(model, input_size=(2, 3, 224, 224), depth=0, device="cuda:1")

    # res2net101 mult-adds (G): 16.13 params (M): 45,206,688
    model = res2net101_26w_4s()
    output = model(input)
    print("\n\nmodel: res2net101_26w_4s")
    print("input shape: ", input.shape)
    print("output shape: ", output[0].shape)
    summary(model, input_size=(2, 3, 224, 224), depth=0, device="cuda:1")
