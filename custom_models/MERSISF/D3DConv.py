import time
import math
import torch
from torch import nn
from torch.autograd import Function
from torch.nn.modules.utils import _triple
from torch.nn import init
from torch.autograd.function import once_differentiable
from torch.autograd import gradcheck



import D3D

class DeformConvFunction(Function):
    @staticmethod
    def forward(ctx, input, offset, weight, bias,
                stride, padding, dilation, group, deformable_groups, im2col_step):
        ctx.stride = _triple(stride)
        ctx.padding = _triple(padding)
        ctx.dilation = _triple(dilation)
        ctx.kernel_size = _triple(weight.shape[2:5])
        ctx.group = group
        ctx.deformable_groups = deformable_groups
        ctx.im2col_step = im2col_step
        output = D3D.deform_conv_forward(input, weight, bias,
                                         offset,
                                         ctx.kernel_size[0], ctx.kernel_size[1],ctx.kernel_size[2],
                                         ctx.stride[0], ctx.stride[1],ctx.stride[2],
                                         ctx.padding[0], ctx.padding[1],ctx.padding[2],
                                         ctx.dilation[0], ctx.dilation[1],ctx.dilation[2],
                                         ctx.group,
                                         ctx.deformable_groups,
                                         ctx.im2col_step)
        ctx.save_for_backward(input, offset, weight, bias)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, offset, weight, bias = ctx.saved_tensors
        grad_input, grad_offset, grad_weight, grad_bias = \
            D3D.deform_conv_backward(input, weight,
                                     bias,
                                     offset,
                                     grad_output,
                                     ctx.kernel_size[0], ctx.kernel_size[1], ctx.kernel_size[2],
                                     ctx.stride[0], ctx.stride[1], ctx.stride[2],
                                     ctx.padding[0], ctx.padding[1], ctx.padding[2],
                                     ctx.dilation[0], ctx.dilation[1], ctx.dilation[2],
                                     ctx.group,
                                     ctx.deformable_groups,
                                     ctx.im2col_step)

        return grad_input, grad_offset, grad_weight, grad_bias,\
            None, None, None, None, None, None


class DeformConv(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding, dilation=1, groups=1, deformable_groups=1, im2col_step=64, bias=True):
        super(DeformConv, self).__init__()

        if in_channels % groups != 0:
            raise ValueError('in_channels {} must be divisible by groups {}'.format(in_channels, groups))
        if out_channels % groups != 0:
            raise ValueError('out_channels {} must be divisible by groups {}'.format(out_channels, groups))

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.padding = _triple(padding)
        self.dilation = _triple(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.im2col_step = im2col_step
        self.use_bias = bias
        
        self.weight = nn.Parameter(torch.Tensor(
            out_channels, in_channels//groups, *self.kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()
        if not self.use_bias:
            self.bias.requires_grad = False

    def reset_parameters(self):
        n = self.in_channels
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input, offset):
        assert 3 * self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2] == \
            offset.shape[1]
        return DeformConvFunction.apply(input, offset,
                                                   self.weight,
                                                   self.bias,
                                                   self.stride,
                                                   self.padding,
                                                   self.dilation,
                                                   self.groups,
                                                   self.deformable_groups,
                                                   self.im2col_step)

_DeformConv = DeformConvFunction.apply

class DeformConvPack(DeformConv):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding,
                 dilation=1, groups=1, deformable_groups=1, im2col_step=64, bias=True, lr_mult=0.1):
        super(DeformConvPack, self).__init__(in_channels, out_channels,
                                  kernel_size, stride, padding, dilation, groups, deformable_groups, im2col_step, bias)


        out_channels = self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]
        self.conv_offset = nn.Conv3d(self.in_channels,
                                          out_channels,
                                          kernel_size=self.kernel_size,
                                          stride=self.stride,
                                          padding=self.padding,
                                          bias=True)
        self.conv_offset.lr_mult = lr_mult
        self.init_offset()

    def init_offset(self):
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

    def forward(self, input):
        offset = self.conv_offset(input)
        return DeformConvFunction.apply(input, offset,
                                          self.weight, 
                                          self.bias, 
                                          self.stride, 
                                          self.padding, 
                                          self.dilation, 
                                          self.groups,
                                          self.deformable_groups,
                                          self.im2col_step)


class DeformConv_d(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding, dimension='THW', dilation=1, groups=1, deformable_groups=1, im2col_step=64, bias=True):
        super(DeformConv_d, self).__init__()

        if in_channels % groups != 0:
            raise ValueError('in_channels {} must be divisible by groups {}'.format(in_channels, groups))
        if out_channels % groups != 0:
            raise ValueError('out_channels {} must be divisible by groups {}'.format(out_channels, groups))

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.padding = _triple(padding)
        self.dilation = _triple(dilation)
        self.dimension = dimension
        self.length = len(dimension)
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.im2col_step = im2col_step
        self.use_bias = bias

        self.weight = nn.Parameter(torch.Tensor(
            out_channels, in_channels // groups, *self.kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()
        if not self.use_bias:
            self.bias.requires_grad = False

    def reset_parameters(self):
        n = self.in_channels
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input, temp):
        dimension_T = 'T' in self.dimension
        dimension_H = 'H' in self.dimension
        dimension_W = 'W' in self.dimension
        b, c, t, h, w = temp.shape
        if self.length == 2:
            offset = temp.new_zeros(b, 3 * self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2], t, h, w)
            if dimension_T == False:
                for i in range(
                        self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]):
                    offset[:, i * 3, :, :, :] = 0  # T
                    offset[:, i * 3 + 1, :, :, :] = temp[:, i * 2, :, :, :]
                    offset[:, i * 3 + 2, :, :, :] = temp[:, i * 2 + 1, :, :, :]
            if dimension_H == False:
                for i in range(
                        self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]):
                    offset[:, i * 3, :, :, :] = temp[:, i * 2, :, :, :]
                    offset[:, i * 3 + 1, :, :, :] = 0  # H
                    offset[:, i * 3 + 2, :, :, :] = temp[:, i * 2 + 1, :, :, :]
            if dimension_W == False:
                for i in range(
                        self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]):
                    offset[:, i * 3, :, :, :] = temp[:, i * 2, :, :, :]
                    offset[:, i * 3 + 1, :, :, :] = temp[:, i * 2 + 1, :, :, :]
                    offset[:, i * 3 + 2, :, :, :] = 0  # W

        if self.length == 1:
            offset = temp.new_zeros(b, 3 * self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2], t, h, w)
            if dimension_T == True:
                for i in range(
                        self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]):
                    offset[:, i * 3, :, :, :] = temp[:, i, :, :, :]  # T
                    offset[:, i * 3 + 1, :, :, :] = 0
                    offset[:, i * 3 + 2, :, :, :] = 0
            if dimension_H == True:
                for i in range(
                        self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]):
                    offset[:, i * 3, :, :, :] = 0
                    offset[:, i * 3 + 1, :, :, :] = temp[:, i, :, :, :]  # H
                    offset[:, i * 3 + 2, :, :, :] = 0
            if dimension_W == True:
                for i in range(
                        self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]):
                    offset[:, i * 3, :, :, :] = 0
                    offset[:, i * 3 + 1, :, :, :] = 0
                    offset[:, i * 3 + 2, :, :, :] = temp[:, i, :, :, :]  # W

        return DeformConvFunction.apply(input, offset,
                                        self.weight,
                                        self.bias,
                                        self.stride,
                                        self.padding,
                                        self.dilation,
                                        self.groups,
                                        self.deformable_groups,
                                        self.im2col_step)


_DeformConv = DeformConvFunction.apply


class DeformConvPack_d(DeformConv_d):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding, dimension='THW',
                 dilation=1, groups=1, deformable_groups=1, im2col_step=64, bias=True, lr_mult=0.1):
        super(DeformConvPack_d, self).__init__(in_channels, out_channels,
                                             kernel_size, stride, padding, dimension, dilation, groups, deformable_groups,
                                             im2col_step, bias)
        self.dimension = dimension
        self.length = len(dimension)
        out_channels = self.deformable_groups * self.length * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]
        self.conv_offset = nn.Conv3d(self.in_channels,
                                     out_channels,
                                     kernel_size=self.kernel_size,
                                     stride=self.stride,
                                     padding=self.padding,
                                     bias=True)
        self.conv_offset.lr_mult = lr_mult
        self.init_offset()

    def init_offset(self):
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

    def forward(self, input):
        temp = self.conv_offset(input)
        dimension_T = 'T' in self.dimension
        dimension_H = 'H' in self.dimension
        dimension_W = 'W' in self.dimension
        b, c, t, h, w = temp.shape
        if self.length == 2:
            offset = temp.new_zeros(b, 3 * self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2], t, h, w)
            if dimension_T == False:
                for i in range(
                        self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]):
                    offset[:, i * 3, :, :, :] = 0  # T
                    offset[:, i * 3 + 1, :, :, :] = temp[:, i * 2, :, :, :]
                    offset[:, i * 3 + 2, :, :, :] = temp[:, i * 2 + 1, :, :, :]
            if dimension_H == False:
                for i in range(
                        self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]):
                    offset[:, i * 3, :, :, :] = temp[:, i * 2, :, :, :]
                    offset[:, i * 3 + 1, :, :, :] = 0  # H
                    offset[:, i * 3 + 2, :, :, :] = temp[:, i * 2 + 1, :, :, :]
            if dimension_W == False:
                for i in range(
                        self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]):
                    offset[:, i * 3, :, :, :] = temp[:, i * 2, :, :, :]
                    offset[:, i * 3 + 1, :, :, :] = temp[:, i * 2 + 1, :, :, :]
                    offset[:, i * 3 + 2, :, :, :] = 0  # W

        if self.length == 1:
            offset = temp.new_zeros(b, 3 * self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2], t, h, w)
            if dimension_T == True:
                for i in range(
                        self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]):
                    offset[:, i * 3, :, :, :] = temp[:, i, :, :, :]  # T
                    offset[:, i * 3 + 1, :, :, :] = 0
                    offset[:, i * 3 + 2, :, :, :] = 0
            if dimension_H == True:
                for i in range(
                        self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]):
                    offset[:, i * 3, :, :, :] = 0
                    offset[:, i * 3 + 1, :, :, :] = temp[:, i, :, :, :]  # H
                    offset[:, i * 3 + 2, :, :, :] = 0
            if dimension_W == True:
                for i in range(
                        self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]):
                    offset[:, i * 3, :, :, :] = 0
                    offset[:, i * 3 + 1, :, :, :] = 0
                    offset[:, i * 3 + 2, :, :, :] = temp[:, i, :, :, :]  # W

        return DeformConvFunction.apply(input, offset,
                                        self.weight,
                                        self.bias,
                                        self.stride,
                                        self.padding,
                                        self.dilation,
                                        self.groups,
                                        self.deformable_groups,
                                        self.im2col_step)
        
        
        
deformable_groups = 1
B, inC, inT, inH, inW = 2, 8, 16, 16, 16
outC = 8
kT, kH, kW = 3, 3, 3
sT, sH, sW = 1, 1, 1
pT, pH, pW = 1, 1, 1

def example_dconv():
    print('============using its own offsets===========')
    input = torch.randn(B, inC, inT, inH, inW).cuda()
    dcn = DeformConvPack(inC, outC, kernel_size=[kT, kH, kW], stride=[sT, sH, sW],padding=[pT, pH, pW]).cuda()
    print('input.shape: ', input.shape)
    output = dcn(input)
    targert = output.new(*output.size())
    targert.data.uniform_(-0.01, 0.01)
    error = (targert - output).mean()
    error.backward()
    print('output.shape: ', output.shape)

def example_dconv_offset():
    print('=============using extra offsets============')
    input = torch.randn(B, inC, inT, inH, inW).cuda()
    offset = torch.randn(B, kT*kH*kW*3, inT, inH, inW).cuda()
    dcn = DeformConv(inC, outC, kernel_size=[kT, kH, kW], stride=[sT, sH, sW],padding=[pT, pH, pW]).cuda()
    print('input.shape: ', input.shape)
    print('offset.shape: ', offset.shape)
    output = dcn(input, offset)
    targert = output.new(*output.size())
    targert.data.uniform_(-0.01, 0.01)
    error = (targert - output).mean()
    error.backward()
    print('output.shape: ', output.shape)

def example_dconv_d():
    print('============using its own offsets===========')
    input = torch.randn(B, inC, inT, inH, inW).cuda()
    dcn = DeformConvPack_d(inC, outC, kernel_size=[kT, kH, kW], stride=[sT, sH, sW],padding=[pT, pH, pW],dimension='TW').cuda()
    #  dimension = 'T' or 'H' or 'W' or any combination of these three letters
    #  'T' represents the deformation in temporal dimension
    #  'H' represents the deformation in height dimension
    #  'W' represents the deformation in weigh dimension
    print('input.shape: ', input.shape)
    output = dcn(input)
    targert = output.new(*output.size())
    targert.data.uniform_(-0.01, 0.01)
    error = (targert - output).mean()
    error.backward()
    print('output.shape: ', output.shape)

def example_dconv_offset_d():
    print('=============using extra offsets============')
    input = torch.randn(B, inC, inT, inH, inW).cuda()
    #  offset
    dimension = 'HW' # choose any dimension you want to deform
    offset = torch.randn(B, kT*kH*kW*len(dimension), inT, inH, inW).cuda()
    dcn = DeformConv_d(inC, outC, kernel_size=[kT, kH, kW], stride=[sT, sH, sW],padding=[pT, pH, pW],dimension=dimension).cuda()
    #  dimension = 'T' or 'H' or 'W' or any combination of these three letters
    #  'T' represents the deformation in temporal dimension
    #  'H' represents the deformation in height dimension
    #  'W' represents the deformation in weigh dimension
    print('input.shape: ', input.shape)
    print('offset.shape: ', offset.shape)
    output = dcn(input, offset)
    targert = output.new(*output.size())
    targert.data.uniform_(-0.01, 0.01)
    error = (targert - output).mean()
    error.backward()
    print('output.shape: ', output.shape)

if __name__ == '__main__':
    print('==================Deformable 3D convolution==================', '\n')
    # D3D deform in three dimensions
    print('=============D3D deform in three dimensions===========')
    example_dconv() # DCN using its own offsets
    example_dconv_offset() # DCN using extra offsets
    print('\n')

    # D3D available for deformable dimension
    print('============option for deformable dimension===========')
    example_dconv_d()  # DCN using its own offsets
    example_dconv_offset_d()  # DCN using extra offsets