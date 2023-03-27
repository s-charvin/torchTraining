import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import math
from torch import nn


class Conv2dSame(torch.nn.Conv2d):

    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.size()[-2:]

        pad_h = self.calc_same_pad(
            i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0])
        pad_w = self.calc_same_pad(
            i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1])

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2,
                    pad_h // 2, pad_h - pad_h // 2]
            )
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class ResMultiScaleConv2d(nn.Module):
    '''
    Multi-scale block with short-cut connections
    '''

    def __init__(self, channels=16, **kwargs):
        super(ResMultiScaleConv2d, self).__init__()
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


# def exists(val):
#     return val is not None


# class Residual(nn.Module):
#     def __init__(self, fn):
#         super().__init__()
#         self.fn = fn

#     def forward(self, x):
#         return self.fn(x) + x


# class PreNorm(nn.Module):
#     def __init__(self, dim, fn):
#         super().__init__()
#         self.fn = fn
#         self.norm = nn.LayerNorm(dim)

#     def forward(self, x, **kwargs):
#         x = self.norm(x)
#         return self.fn(x, **kwargs)


# class SpatialGatingUnit(nn.Module):
#     def __init__(
#         self,
#         dim,
#         dim_seq,
#         causal=False,
#         act=nn.Identity(),
#         heads=1,
#         init_eps=1e-3,
#         circulant_matrix=False
#     ):
#         super().__init__()
#         dim_out = dim // 2
#         self.heads = heads
#         self.causal = causal
#         self.norm = nn.LayerNorm(dim_out)
#         self.act = act

#         # parameters
#         if circulant_matrix:
#             self.circulant_pos_x = nn.Parameter(torch.ones(heads, dim_seq))
#             self.circulant_pos_y = nn.Parameter(torch.ones(heads, dim_seq))

#         self.circulant_matrix = circulant_matrix
#         shape = (heads, dim_seq,) if circulant_matrix else (
#             heads, dim_seq, dim_seq)
#         weight = torch.zeros(shape)

#         self.weight = nn.Parameter(weight)
#         init_eps /= dim_seq
#         nn.init.uniform_(self.weight, -init_eps, init_eps)
#         self.bias = nn.Parameter(torch.ones(heads, dim_seq))

#     def forward(self, x):
#         device, n, h = x.device, x.shape[1], self.heads

#         res, gate = x.chunk(2, dim=-1)
#         gate = self.norm(gate)

#         weight, bias = self.weight, self.bias

#         if self.circulant_matrix:
#             # build the circulant matrix

#             dim_seq = weight.shape[-1]
#             weight = F.pad(weight, (0, dim_seq), value=0)
#             weight = repeat(weight, '... n -> ... (r n)', r=dim_seq)
#             weight = weight[:, :-dim_seq].reshape(h, dim_seq, 2 * dim_seq - 1)
#             weight = weight[:, :, (dim_seq - 1):]

#             # give circulant matrix absolute position awareness

#             pos_x, pos_y = self.circulant_pos_x, self.circulant_pos_y
#             weight = weight * \
#                 rearrange(pos_x, 'h i -> h i ()') * \
#                 rearrange(pos_y, 'h j -> h () j')

#         if self.causal:
#             weight, bias = weight[:, :n, :n], bias[:, :n]
#             mask = torch.ones(weight.shape[-2:], device=device).triu_(1).bool()
#             mask = rearrange(mask, 'i j -> () i j')
#             weight = weight.masked_fill(mask, 0.)

#         gate = rearrange(gate, 'b n (h d) -> b h n d', h=h)

#         gate = einsum('b h n d, h m n -> b h m d', gate, weight)
#         gate = gate + rearrange(bias, 'h n -> () h n ()')

#         gate = rearrange(gate, 'b h n d -> b n (h d)')

#         return self.act(gate) * res


# class gMLPBlock(nn.Module):
#     def __init__(
#         self,
#         *,
#         dim,
#         dim_ff,
#         seq_len,
#         heads=1,
#         causal=False,
#         act=nn.Identity(),
#         circulant_matrix=False
#     ):
#         super().__init__()
#         self.proj_in = nn.Sequential(
#             nn.Linear(dim, dim_ff),
#             nn.GELU()
#         )

#         self.sgu = SpatialGatingUnit(
#             dim_ff, seq_len, causal, act, heads, circulant_matrix=circulant_matrix)
#         self.proj_out = nn.Linear(dim_ff // 2, dim)

#     def forward(self, x):
#         x = self.proj_in(x)
#         x = self.sgu(x)
#         x = self.proj_out(x)
#         return x


# class gMLP(nn.Module):
#     def __init__(
#         self,
#         *,
#         num_tokens=None,
#         dim,
#         depth,
#         seq_len,
#         heads=1,
#         ff_mult=4,
#         attn_dim=None,
#         prob_survival=1.,
#         causal=False,
#         circulant_matrix=False,

#         act=nn.Identity()
#     ):
#         super().__init__()
#         assert (dim % heads) == 0, 'dimension must be divisible by number of heads'

#         dim_ff = dim * ff_mult
#         self.seq_len = seq_len
#         self.prob_survival = prob_survival

#         self.layers = nn.ModuleList([
#             Residual(
#                 PreNorm(
#                     dim,
#                     gMLPBlock(
#                         dim=dim,
#                         heads=heads,
#                         dim_ff=dim_ff,
#                         seq_len=seq_len,
#                         attn_dim=attn_dim,
#                         causal=causal,
#                         act=act,
#                         circulant_matrix=circulant_matrix
#                     )
#                 )
#             ) for i in range(depth)])

#     def forward(self, x):
#         layers = self.layers
#         out = nn.Sequential(*layers)(x)
#         return out
