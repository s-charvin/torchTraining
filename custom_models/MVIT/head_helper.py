#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""ResNe(X)t Head helper."""
import torch
import torch.nn as nn
from detectron2.layers import ROIAlign
from .batchnorm_helper import (
    NaiveSyncBatchNorm1d as NaiveSyncBatchNorm1d,
)


class MLPHead(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        mlp_dim,
        num_layers,
        bn_on=False,
        bias=True,
        flatten=False,
        xavier_init=True,
        bn_sync_num=1,
        global_sync=False,
    ):
        super(MLPHead, self).__init__()
        self.flatten = flatten
        b = False if bn_on else bias
        # assert bn_on or bn_sync_num=1
        mlp_layers = [nn.Linear(dim_in, mlp_dim, bias=b)]
        mlp_layers[-1].xavier_init = xavier_init
        for i in range(1, num_layers):
            if bn_on:
                if global_sync or bn_sync_num > 1:
                    mlp_layers.append(
                        NaiveSyncBatchNorm1d(
                            num_sync_devices=bn_sync_num,
                            global_sync=global_sync,
                            num_features=mlp_dim,
                        )
                    )
                else:
                    mlp_layers.append(nn.BatchNorm1d(num_features=mlp_dim))
            mlp_layers.append(nn.ReLU(inplace=True))
            if i == num_layers - 1:
                d = dim_out
                b = bias
            else:
                d = mlp_dim
            mlp_layers.append(nn.Linear(mlp_dim, d, bias=b))
            mlp_layers[-1].xavier_init = xavier_init
        self.projection = nn.Sequential(*mlp_layers)

    def forward(self, x):
        if x.ndim == 5:
            x = x.permute((0, 2, 3, 4, 1))
        if self.flatten:
            x = x.reshape(-1, x.shape[-1])

        return self.projection(x)


class ResNetRoIHead(nn.Module):
    """
    ResNe(X)t RoI head.
    """

    def __init__(
        self,
        dim_in,
        num_classes,
        pool_size,
        resolution,
        scale_factor,
        dropout_rate=0.0,
        act_func="softmax",
        aligned=True,
        detach_final_fc=False,
    ):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        ResNetRoIHead takes p pathways as input where p in [1, infty].

        Args:
            dim_in (list): the list of channel dimensions of the p inputs to the
                ResNetHead.
            num_classes (int): the channel dimensions of the p outputs to the
                ResNetHead.
            pool_size (list): the list of kernel sizes of p spatial temporal
                poolings, temporal pool kernel size, spatial pool kernel size,
                spatial pool kernel size in order.
            resolution (list): the list of spatial output size from the ROIAlign.
            scale_factor (list): the list of ratio to the input boxes by this
                number.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
            aligned (bool): if False, use the legacy implementation. If True,
                align the results more perfectly.
            detach_final_fc (bool): if True, detach the final fc layer from the
                gradient graph. By doing so, only the final fc layer will be
                trained.
        Note:
            Given a continuous coordinate c, its two neighboring pixel indices
            (in our pixel model) are computed by floor (c - 0.5) and ceil
            (c - 0.5). For example, c=1.3 has pixel neighbors with discrete
            indices [0] and [1] (which are sampled from the underlying signal at
            continuous coordinates 0.5 and 1.5). But the original roi_align
            (aligned=False) does not subtract the 0.5 when computing neighboring
            pixel indices and therefore it uses pixels with a slightly incorrect
            alignment (relative to our pixel model) when performing bilinear
            interpolation.
            With `aligned=True`, we first appropriately scale the ROI and then
            shift it by -0.5 prior to calling roi_align. This produces the
            correct neighbors; It makes negligible differences to the model's
            performance if ROIAlign is used together with conv layers.
        """
        super(ResNetRoIHead, self).__init__()
        assert (
            len({len(pool_size), len(dim_in)}) == 1
        ), "pathway dimensions are not consistent."
        self.num_pathways = len(pool_size)
        self.detach_final_fc = detach_final_fc

        for pathway in range(self.num_pathways):
            temporal_pool = nn.AvgPool3d(
                [pool_size[pathway][0], 1, 1], stride=1
            )
            self.add_module("s{}_tpool".format(pathway), temporal_pool)

            roi_align = ROIAlign(
                resolution[pathway],
                spatial_scale=1.0 / scale_factor[pathway],
                sampling_ratio=0,
                aligned=aligned,
            )
            self.add_module("s{}_roi".format(pathway), roi_align)
            spatial_pool = nn.MaxPool2d(resolution[pathway], stride=1)
            self.add_module("s{}_spool".format(pathway), spatial_pool)

        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)

        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        self.projection = nn.Linear(sum(dim_in), num_classes, bias=True)

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=1)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

    def forward(self, inputs, bboxes):
        assert (
            len(inputs) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        pool_out = []
        for pathway in range(self.num_pathways):
            t_pool = getattr(self, "s{}_tpool".format(pathway))
            out = t_pool(inputs[pathway])
            assert out.shape[2] == 1
            out = torch.squeeze(out, 2)

            roi_align = getattr(self, "s{}_roi".format(pathway))
            out = roi_align(out, bboxes)

            s_pool = getattr(self, "s{}_spool".format(pathway))
            pool_out.append(s_pool(out))

        # B C H W.
        x = torch.cat(pool_out, 1)

        # Perform dropout.
        if hasattr(self, "dropout"):
            x = self.dropout(x)

        x = x.view(x.shape[0], -1)
        if self.detach_final_fc:
            x = x.detach()
        x = self.projection(x)
        x = self.act(x)
        return x


class TransformerBasicHead(nn.Module):
    """
    BasicHead. No pool.
    """

    def __init__(
        self,
        dim_in,
        num_classes,
        dropout_rate=0.0,
        act_func="softmax",

        num_mlp_layers=1,
        mlp_dim=None,
        bn_mlp=False,
        bn_sync_mlp=False,
        num_sync_devices=1,
        global_sync=False,
        detach_final_fc=False,
    ):
        """
        Perform linear projection and activation as head for tranformers.
        Args:
            dim_in (int): the channel dimension of the input to the head.
            num_classes (int): the channel dimensions of the output to the head.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
        """
        super(TransformerBasicHead, self).__init__()
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        self.projection = nn.Linear(dim_in, num_classes, bias=True)

        if num_mlp_layers == 1:
            self.projection = nn.Linear(dim_in, num_classes, bias=True)
        else:
            self.projection = MLPHead(
                dim_in,
                num_classes,
                mlp_dim,
                num_mlp_layers,
                bn_on=bn_mlp,
                bn_sync_num=num_sync_devices
                if bn_sync_mlp
                else 1,
                global_sync=(
                    bn_sync_mlp and global_sync
                ),
            )
        self.detach_final_fc = detach_final_fc

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=1)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        elif act_func == "none":
            self.act = None
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

    def forward(self, x):
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        if self.detach_final_fc:
            x = x.detach()
        x = self.projection(x)

        if not self.training:
            if self.act is not None:
                x = self.act(x)
            # Performs fully convlutional inference.
            if x.ndim == 5 and x.shape[1:4] > torch.Size([1, 1, 1]):
                x = x.mean([1, 2, 3])

        x = x.view(x.shape[0], -1)

        return x
