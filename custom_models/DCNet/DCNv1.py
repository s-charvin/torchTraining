
import torch
from torch import  nn
import torch.nn.functional as F
import math
import torchvision

class DCNv1(torchvision.ops.DeformConv2d):
    def __init__(self, *args, **kwargs):
        super(DCNv1, self).__init__(*args, **kwargs)
        self.conv_offset2d = nn.Conv2d(
            self.in_channels, 
            self.groups * 2 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size, 
            stride=self.stride, 
            padding=self.padding, 
            dilation=self.dilation,
            bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        if hasattr(self, 'conv_offset2d'):
            self.conv_offset2d.weight.data.zero_()
            self.conv_offset2d.bias.data.zero_()
        
    def forward(self, x: torch.Tensor):
        offset = self.conv_offset2d(x)
        return torchvision.ops.deform_conv2d(x, offset, self.weight, self.bias, stride=self.stride,
                             padding=self.padding, dilation=self.dilation)