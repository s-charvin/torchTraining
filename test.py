
import os
import sys
import torch
import time
from fvcore.nn import FlopCountAnalysis
from fvcore.nn import flop_count_table
import numpy as np
from torchinfo import summary

if sys.platform.startswith("linux"):
    os.chdir("/home/user0/SCW/torchTraining")
elif sys.platform.startswith("win"):
    pass


from custom_models.resnet import ResNet
from custom_models.densenet import DenseNet
from custom_models.MAADA import AACNN
from custom_models.SlowFast import slowfast101
from custom_models.MFFNet import getModel, LightSerNet_Encoder, LightResMultiSerNet_Encoder
from custom_models.MERSISF import VideoEncoder
from custom_datasets import *

from torchinfo import summary
import yaml  # yaml文件解析模块

if __name__ == "__main__":


    # uniformerv2_b16 mult-adds (G): 8.51 params (M): 114,925,288
    x = torch.rand(2, 30, 3, 150, 150).to(device="cuda:0")
    model = VideoEncoder(
            in_channels = 3,
            out_features=320,
            model_name="resnet50",
            input_size=[150, 150],
            num_frames=30,
            before_dropout=0,
            before_softmax=False,
            pretrained = True).to(device="cuda:0")
    output = model(x)
    print("model: .")
    print("input shape: ", x.shape)
    print("output shape: ", output.shape)
    summary(model, input_data={"x": x}, depth=6, device="cuda:0")
    flops = FlopCountAnalysis(model, x)
    print(flop_count_table(flops, max_depth=1))
