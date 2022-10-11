# from matplotlib import pyplot as plt
# import numpy as np
# from sqlalchemy import false
# import torch
# from torch import nn
# from d2l import torch as d2l
import torchaudio
import torch
import soundfile as sf

# from models.emotion_baseline_attention_lstm import LSTM

# batch_size, num_steps = 32, 35
# train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

# vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
# num_epochs, lr = 500, 1
# num_inputs = vocab_size
# lstm_layer = LSTM(num_inputs, num_hiddens, num_layers=1, bidirectional=False)
# model = d2l.RNNModel(lstm_layer, len(vocab))
# model = model.to(device)
# ppl_all, num_epochs = d2l.train_ch8(
#     model, train_iter, vocab, lr, num_epochs, device)
# plt.plot(np.arange(10, 501, 10), ppl_all)

# plt.savefig("Text.png")

file_path = r"/home/user4/SCW/deeplearning/data/EMODB/wav/03a02Ta.wav"
a = sf.info(file_path)
labels = ["a", "a", "a", "b", "b", "b", "c", "d", "d"]
frame_num = [3, 3, 1, 2]
begin = 0
true_labels = []

for i, num in enumerate(frame_num):
    x = labels[begin:begin+num]
    true_labels.append(x[0])
    begin = begin+num

pass
# y0, sr = torchaudio.load(file_path, normalize=True)
# transform = torchaudio.transforms.Spectrogram(
#     n_fft=560, win_length=560, hop_length=140,
#     pad=0, window_fn=torch.hamming_window, power=2.,
#     normalized=True, wkwargs=None, center=True,
#     pad_mode="reflect", onesided=True,)
# spectrogram = transform(y0)

# transform = torchaudio.transforms.MelSpectrogram(
#     sample_rate=sr, n_fft=560, win_length=560,
#     hop_length=140, f_min=0., f_max=None, pad=0,
#     n_mels=281, window_fn=torch.hamming_window,
#     power=2., normalized=True, wkwargs=None, center=True,
#     pad_mode="reflect", onesided=True, norm=None,)
# melspectrogram = transform(y0)
