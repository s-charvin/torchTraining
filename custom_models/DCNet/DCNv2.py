
import torch
from torch import  nn
import torch.nn.functional as F
import math
import torchvision

class DCNv2(torchvision.ops.DeformConv2d):
    def __init__(self, *args, **kwargs):
        super(DCNv2, self).__init__(*args, **kwargs)
        self.conv_offset2d = nn.Conv2d(
            self.in_channels, 
            self.groups * 3 * self.kernel_size[0] * self.kernel_size[1],
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
        out = self.conv_offset2d(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return torchvision.ops.deform_conv2d(x, offset, self.weight, self.bias, stride=self.stride,
                             padding=self.padding, dilation=self.dilation, mask=mask)
