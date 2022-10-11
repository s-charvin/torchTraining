

import os
import sys

import torchaudio


if sys.platform.startswith('linux'):
    os.chdir("/home/user4/SCW/deeplearning/")
elif sys.platform.startswith('win'):
    pass

# from models.wav2vec2 import wav2vec2_model
# from models.resnet import ResNet
# from models.densenet import DenseNet
# from models.emotion_baseline_attention_lstm import LSTM
# from models.resnext import ResNeXt
# from models.lstm_res_test import LstmResTest
# from models.AFC_ALSTM_ACNN import CLFC
# from models.SAE import STLSER
# from models.MAADA import AACNN
from models.SlowFast import slowfast101
from models.Testmodel import *
from torchinfo import summary
import yaml  # yaml文件解析模块
from SCW.deeplearningcopy.custom_datasets.IEMOCAP import IEMOCAP
import soundfile as sf
from transformers import get_linear_schedule_with_warmup
if __name__ == '__main__':

 # 模型输出测试
    config = yaml.safe_load(
        open("./config.yaml", 'r', encoding='utf-8').read())
    # a = IEMOCAPDataset(config)
    # feature, label = a.__getitem__(0)
    # testodel = Wav2Vec2_CLS(config)
    # print(feature.shape)
    # summary(testodel, (268, 40), batch_size=64, device="cpu")

    # wav2vec2_base = wav2vec2_model(config)
    # resnet_50 = ResNet(config)
    # densenet_121 = DenseNet(config)

    # net = slowfast101()
    # size1 = (2, 1, 100, 128)
    # size2 = (2, 1, 400, 128)
    # x1 = torch.randn(size1)
    # x2 = torch.randn(size2)
    # x = [x1, x2]
    # x = torch.randn(size2)
    # y = net(x)
    # print(net)
    # print(y)
    # summary(net, input_size=size2, depth=4)

    model = build_model()
    tokenizer = BertTokenizer.from_pretrained(
        "bert-base-uncased", do_lower_case=True)
    processor = Wav2Vec2Processor.from_pretrained(
        "facebook/wav2vec2-base-960h")
    print(model)
    text = ["I love you. Annie, I love you."]*2
    target = ["This person is very happy."]*2
    speechfiles = [
        "./data/IEMOCAP/Session1/sentences/wav/Ses01F_script01_3/Ses01F_script01_3_M008.wav"]*2

    encoded_text = tokenizer.batch_encode_plus(
        batch_text_or_text_pairs=text,   # 输入文本 batch
        # is_split_into_words=True,
        add_special_tokens=True,  # 添加 '[CLS]' 和 '[SEP]'
        # max_length=128,           # 填充 & 截断长度
        padding='longest',  # 设置填充到本 batch 中的最长长度
        return_attention_mask=True,   # 返回 attn. masks.
        return_tensors='pt',     # 返回 pytorch tensors 格式的数据
    )
    encoded_target = tokenizer.batch_encode_plus(
        batch_text_or_text_pairs=target,   # 输入文本 batch
        # is_split_into_words=True,
        add_special_tokens=True,  # 添加 '[CLS]' 和 '[SEP]'
        # max_length=128,           # 填充 & 截断长度
        padding='longest',  # 设置填充到本 batch 中的最长长度
        return_attention_mask=True,   # 返回 attn. masks.
        return_tensors='pt',     # 返回 pytorch tensors 格式的数据
    )

    def batch_encode_plus(self, text):
        return self.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=text,   # 输入文本 batch
            # is_split_into_words=True,
            add_special_tokens=True,  # 添加 '[CLS]' 和 '[SEP]'
            # max_length=128,           # 填充 & 截断长度
            padding='longest',  # 设置填充到本 batch 中的最长长度
            return_attention_mask=True,   # 返回 attn. masks.
            return_tensors='pt',     # 返回 pytorch tensors 格式的数据
        )
    # input_values = processor(
    #     [sf.read(file)[0] for file in speechfiles], return_tensors="pt").input_values

    transform = torchaudio.transforms.MFCC(16000, n_mfcc=40)
    input_values = torch.cat(
        [transform(torchaudio.load(file, normalize=True)[0]).permute(0, 2, 1)
         for file in speechfiles], dim=0).unsqueeze(dim=1)

    logits = model(speechs=input_values,
                   text_ids=encoded_text['input_ids'], text_mask=encoded_text['attention_mask'],
                   target_text_ids=encoded_target['input_ids'], target_text_mask=encoded_target['attention_mask'])
    pass
