
import os
import sys
import torch
import torchaudio


if sys.platform.startswith('linux'):
    os.chdir("/home/visitors2/SCW/deeplearning/")
elif sys.platform.startswith('win'):
    pass


from custom_models.resnet import ResNet
from custom_models.densenet import DenseNet
from custom_models.MAADA import AACNN
from custom_models.SlowFast import slowfast101
from custom_models.MFFNet import getModel, LightSerNet_Encoder, LightResMultiSerNet_Encoder
from custom_datasets import *

from torchinfo import summary
import yaml  # yaml文件解析模块


if __name__ == '__main__':

    a = MOSEI("/sdb/visitors2/SCW/data/CMU_MOSEI",
             transform={"COVAREP": {}, "OpenFace2": {}, "Glove": {}}, mode="avt")
    feature = a.__getitem__(100)

    # data = torch.rand((8, 1,  256, 40))
    # model = LightResMultiSerNet_Encoder()
    # print(model)
    # print("")
    # out = model(data)
    # print(out.shape)
