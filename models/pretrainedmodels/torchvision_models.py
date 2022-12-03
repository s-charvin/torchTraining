import torchvision.models as models
import torch
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torch import Tensor, nn
import types
import re


__all__ = [
    'alexnet',
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2'
    'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19',
    'squeezenet1_0', 'squeezenet1_1',
    'inceptionv3',
    'densenet121', 'densenet169', 'densenet201', 'densenet161',
    # 'googlenet',
    # 'mobilenet_v2', "mobilenet_v3_large", "mobilenet_v3_small",
    # 'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3',
    # 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0',
]


model_urls = {
    'imagenet':
    {'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
     'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
     'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
     'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
     'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
     'inceptionv3': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
     'googlenet': 'https://download.pytorch.org/models/googlenet-1378be20.pth',
     'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
     'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
     'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
     'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
     'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
     'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
     'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
     'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
     'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
     'squeezenet1_0': 'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth',
     'squeezenet1_1': 'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth',
     'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
     'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
     'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
     'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
     'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
     'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
     'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
     'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
     'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
     "mobilenet_v3_large": "https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth",
     "mobilenet_v3_small": "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth",
     "mnasnet0_5":
     "https://download.pytorch.org/models/mnasnet0.5_top1_67.823-3ffadce67e.pth",
     "mnasnet1_0":
     "https://download.pytorch.org/models/mnasnet1.0_top1_73.512-f206786ef8.pth",
     'shufflenetv2_x0.5': 'https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth',
     'shufflenetv2_x1.0': 'https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth', }


}


def update_state_dict(state_dict):
    # '.'s are no longer allowed in module names, but pervious _DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls. This pattern is used
    # to find such keys.
    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    return state_dict


def load_pretrained(model, model_url):
    state_dict = model_zoo.load_url(model_url)
    state_dict = update_state_dict(state_dict)
    model.load_state_dict(state_dict)
    return model


#################################################################
# AlexNet

def alexnet(pretrained='imagenet'):
    model = models.alexnet(pretrained=False)
    if pretrained is not None:
        model_url = model_urls[pretrained]['alexnet']
        model = load_pretrained(model, model_url)
    model = modify_alexnet(model)
    return model


def modify_alexnet(model):
    # 无需更改属性
    # model.features
    # model.avgpool

    # 需要更改的属性
    model.dropout_c0 = model.classifier[0]
    model.linear_c0 = model.classifier[1]
    model.relu_c0 = model.classifier[2]
    model.dropout_c1 = model.classifier[3]
    model.linear_c1 = model.classifier[4]
    model.relu_c1 = model.classifier[5]
    model.last_linear = model.classifier[6]
    del model.classifier

    def forward(self, x: Tensor) -> Tensor:
        features = self.features(x)
        out = self.avgpool(features)
        out = torch.flatten(out, 1)
        out = self.dropout_c0(out)
        out = self.linear_c0(out)
        out = self.relu_c0(out)
        out = self.dropout_c1(out)
        out = self.linear_c1(out)
        out = self.relu_c1(out)
        out = self.last_linear(out)
        return out

    model.forward = types.MethodType(forward, model)
    return model


###############################################################
# ResNets


def modify_resnets(model):
    # 需要修改的属性
    model.last_linear = model.fc
    del model.fc

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.last_linear(out)
        return out

    model.forward = types.MethodType(forward, model)
    return model


def resnet18(pretrained='imagenet'):
    """Constructs a ResNet-18 model.
    """
    model = models.resnet18(pretrained=False)
    if pretrained is not None:
        model_url = model_urls[pretrained]['resnet18']
        model = load_pretrained(model, model_url)
    model = modify_resnets(model)
    return model


def resnet34(pretrained='imagenet'):
    """Constructs a ResNet-34 model.
    """
    model = models.resnet34(pretrained=False)
    if pretrained is not None:
        model_url = model_urls[pretrained]['resnet34']
        model = load_pretrained(model, model_url)
    model = modify_resnets(model)
    return model


def resnet50(pretrained='imagenet'):
    """Constructs a ResNet-50 model.
    """
    model = models.resnet50(pretrained=False)
    if pretrained is not None:
        model_url = model_urls[pretrained]['resnet50']
        model = load_pretrained(model, model_url)
    model = modify_resnets(model)
    return model


def resnet101(pretrained='imagenet'):
    """Constructs a ResNet-101 model.
    """
    model = models.resnet101(pretrained=False)
    if pretrained is not None:
        model_url = model_urls[pretrained]['resnet101']
        model = load_pretrained(model, model_url)
    model = modify_resnets(model)
    return model


def resnet152(pretrained='imagenet'):
    """Constructs a ResNet-152 model.
    """
    model = models.resnet152(pretrained=False)
    if pretrained is not None:
        model_url = model_urls[pretrained]['resnet152']
        model = load_pretrained(model, model_url)
    model = modify_resnets(model)
    return model


def resnext50_32x4d(pretrained='imagenet'):
    """Constructs a ResNet-152 model.
    """
    model = models.resnext50_32x4d(pretrained=False)
    if pretrained is not None:
        model_url = model_urls[pretrained]['resnext50_32x4d']
        model = load_pretrained(model, model_url)
    model = modify_resnets(model)
    return model


def resnext101_32x8d(pretrained='imagenet'):
    """Constructs a ResNet-152 model.
    """
    model = models.resnext101_32x8d(pretrained=False)
    if pretrained is not None:
        model_url = model_urls[pretrained]['resnext101_32x8d']
        model = load_pretrained(model, model_url)
    model = modify_resnets(model)
    return model


def wide_resnet50_2(pretrained='imagenet'):
    """Constructs a ResNet-152 model.
    """
    model = models.wide_resnet50_2(pretrained=False)
    if pretrained is not None:
        model_url = model_urls[pretrained]['wide_resnet50_2']
        model = load_pretrained(model, model_url)
    model = modify_resnets(model)
    return model


def wide_resnet101_2(pretrained='imagenet'):
    """Constructs a ResNet-152 model.
    """
    model = models.wide_resnet101_2(pretrained=False)
    if pretrained is not None:
        model_url = model_urls[pretrained]['wide_resnet101_2']
        model = load_pretrained(model, model_url)
    model = modify_resnets(model)
    return model


###############################################################
# VGGs

def vgg11(pretrained='imagenet'):
    """VGG 11-layer model (configuration "A")
    """
    model = models.vgg11(pretrained=False)
    if pretrained is not None:
        model_url = model_urls[pretrained]['vgg11']
        model = load_pretrained(model, model_url)
    model = modify_vggs(model)
    return model


def vgg11_bn(pretrained='imagenet'):
    """VGG 11-layer model (configuration "A") with batch normalization
    """
    model = models.vgg11_bn(pretrained=False)
    if pretrained is not None:
        model_url = model_urls[pretrained]['vgg11_bn']
        model = load_pretrained(model, model_url)
    model = modify_vggs(model)
    return model


def vgg13(pretrained='imagenet'):
    """VGG 13-layer model (configuration "B")
    """
    model = models.vgg13(pretrained=False)
    if pretrained is not None:
        model_url = model_urls[pretrained]['vgg13']
        model = load_pretrained(model, model_url)
    model = modify_vggs(model)
    return model


def vgg13_bn(pretrained='imagenet'):
    """VGG 13-layer model (configuration "B") with batch normalization
    """
    model = models.vgg13_bn(pretrained=False)
    if pretrained is not None:
        model_url = model_urls[pretrained]['vgg13_bn']
        model = load_pretrained(model, model_url)
    model = modify_vggs(model)
    return model


def vgg16(pretrained='imagenet'):
    """VGG 16-layer model (configuration "D")
    """
    model = models.vgg16(pretrained=False)
    if pretrained is not None:
        model_url = model_urls[pretrained]['vgg16']
        model = load_pretrained(model, model_url)
    model = modify_vggs(model)
    return model


def vgg16_bn(pretrained='imagenet'):
    """VGG 16-layer model (configuration "D") with batch normalization
    """
    model = models.vgg16_bn(pretrained=False)
    if pretrained is not None:
        model_url = model_urls[pretrained]['vgg16_bn']
        model = load_pretrained(model, model_url)
    model = modify_vggs(model)
    return model


def vgg19(pretrained='imagenet'):
    """VGG 19-layer model (configuration "E")
    """
    model = models.vgg19(pretrained=False)
    if pretrained is not None:
        model_url = model_urls[pretrained]['vgg19']
        model = load_pretrained(model, model_url)
    model = modify_vggs(model)
    return model


def vgg19_bn(pretrained='imagenet'):
    """VGG 19-layer model (configuration 'E') with batch normalization
    """
    model = models.vgg19_bn(pretrained=False)
    if pretrained is not None:
        model_url = model_urls[pretrained]['vgg19_bn']
        model = load_pretrained(model, model_url)
    model = modify_vggs(model)
    return model


def modify_vggs(model):
    # 需要修改的属性

    model.linear0 = model.classifier[0]
    model.relu0 = model.classifier[1]
    model.dropout0 = model.classifier[2]
    model.linear1 = model.classifier[3]
    model.relu1 = model.classifier[4]
    model.dropout1 = model.classifier[5]
    model.last_linear = model.classifier[6]
    del model.classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear0(out)
        out = self.relu0(out)
        out = self.dropout0(out)
        out = self.linear1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.last_linear(out)
        return out

    model.forward = types.MethodType(forward, model)
    return model


###############################################################
# SqueezeNets

def squeezenet1_0(pretrained='imagenet'):

    model = models.squeezenet1_0(pretrained=False)
    if pretrained is not None:
        model_url = model_urls[pretrained]['squeezenet1_0']
        model = load_pretrained(model, model_url)
    model = modify_squeezenets(model)
    return model


def squeezenet1_1(pretrained='imagenet'):

    model = models.squeezenet1_1(pretrained=False)
    if pretrained is not None:
        model_url = model_urls[pretrained]['squeezenet1_1']
        model = load_pretrained(model, model_url)
    model = modify_squeezenets(model)
    return model


def modify_squeezenets(model):

    # 要修改的属性
    model.dropout_c = model.classifier[0]
    model.last_conv_c = model.classifier[1]
    model.relu_c = model.classifier[2]
    model.avgpool_c = model.classifier[3]
    del model.classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.dropout_c(x)
        x = self.last_conv_c(x)
        x = self.relu_c(x)
        x = self.avgpool_c(x)
        return torch.flatten(x, 1)

    model.forward = types.MethodType(forward, model)
    return model

###############################################################
# InceptionV3


def inceptionv3(pretrained='imagenet'):
    model = models.inception_v3(pretrained=False)
    if pretrained is not None:
        model_url = model_urls[pretrained]['inceptionv3']
        model = load_pretrained(model, model_url)
    model = modify_inceptionvs(model)
    return model


def modify_inceptionvs(model):
    model.last_linear = model.fc
    del model.fc

    def forward(self, x: Tensor):
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * \
                (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * \
                (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * \
                (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.maxpool2(x)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        # N x 768 x 17 x 17
        aux = None
        if self.training and self.aux_logits:
            aux = self.AuxLogits(x)
        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = self.avgpool(x)
        # N x 2048 x 1 x 1
        x = self.dropout(x)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 2048
        x = self.last_linear(x)
        # N x 1000 (num_classes)
        if self.training and self.aux_logits:
            return x, aux
        return x


###############################################################
# DenseNets

def densenet121(pretrained='imagenet'):
    model = models.densenet121(pretrained=False)
    if pretrained is not None:
        model_url = model_urls[pretrained]['densenet121']
        model = load_pretrained(model, model_url)
    model = modify_densenets(model)
    return model


def densenet169(pretrained='imagenet'):
    r"""Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    """
    model = models.densenet169(pretrained=False)
    if pretrained is not None:
        model_url = model_urls[pretrained]['densenet169']
        model = load_pretrained(model, model_url)
    model = modify_densenets(model)
    return model


def densenet201(pretrained='imagenet'):
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    """
    model = models.densenet201(pretrained=False)
    if pretrained is not None:
        model_url = model_urls[pretrained]['densenet201']
        model = load_pretrained(model, model_url)
    model = modify_densenets(model)
    return model


def densenet161(pretrained='imagenet'):
    r"""Densenet-161 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    """
    model = models.densenet161(pretrained=False)
    if pretrained is not None:
        model_url = model_urls[pretrained]['densenet161']
        model = load_pretrained(model, model_url)
    model = modify_densenets(model)
    return model


def modify_densenets(model):
    # 无需更改属性
    # model.features

    # 需要更改的属性
    model.last_linear = model.classifier
    del model.classifier

    def forward(self, x: Tensor) -> Tensor:
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.last_linear(out)
        return out

    model.forward = types.MethodType(forward, model)
    return model
