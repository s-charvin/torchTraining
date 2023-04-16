import os
import sys
from typing import Sequence
from pathlib import Path
from IPython import display
import torch
import numpy as np
import librosa.display
import torchaudio
import decord
from custom_transforms.video_transforms import *
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.manifold import TSNE


def plot_spectrogram(input, outdir="./output/image/"):
    try:
        outdir = Path(outdir)
        outdir.mkdir(parents=True)
    except FileExistsError:
        pass
    spectrogram = torchaudio.transforms.Spectrogram(
        n_fft=1024, win_length=1024, hop_length=256, pad=0, power=2, normalized=True, center=True, pad_mode="reflect", onesided=True)
    if isinstance(input, str) and os.path.isfile(input):
        wav, sr = torchaudio.load(input)
        spec = spectrogram(wav)
        save_spectrogram(spec[0], path=os.path.join(
            outdir.resolve(), Path(input).stem+".png"), title="Spectrogram (db)", ylabel="freq_bin")
    elif isinstance(input, Sequence) and os.path.isfile(input[0]):
        for it in input:
            path = Path(it)
            wav, sr = torchaudio.load(path.resolve())
            spec = spectrogram(wav)
            save_spectrogram(spec[0], path=os.path.join(
                outdir.resolve(), path.stem+".png"), title="Spectrogram (db)", ylabel="freq_bin")


def save_spectrogram(specgram, path, title, ylabel):

    fig, axs = plt.subplots(1, 1)
    axs.set_title(title)
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    # cmap =  ["inferno", "gnuplot2", "CMRmap", "jet", "turbo"]
    im = axs.imshow(librosa.power_to_db(specgram),
                    origin="lower", aspect="auto", cmap=plt.cm.jet)
    fig.colorbar(im, ax=axs)
    plt.savefig(path, dpi=600, transparent=True,
                bbox_inches="tight", pad_inches=0.,)
    plt.close()


def plot_melspectrogram(input, outdir="./output/image/"):
    try:
        outdir = Path(outdir)
        outdir.mkdir(parents=True)
    except FileExistsError:
        pass
    melspectrogram = torchaudio.transforms.MelSpectrogram(
        n_mels=128, n_fft=1024, win_length=1024, hop_length=256, f_min=80, f_max=7600, pad=0, power=2, normalized=True, center=True, pad_mode="", onesided=True)
    if isinstance(input, str) and os.path.isfile(input):
        wav, sr = torchaudio.load(input)
        spec = melspectrogram(wav)
        save_spectrogram(spec[0], path=os.path.join(
            outdir.resolve(), Path(input).stem+".png"), title="MelSpectrogram (db)", ylabel="mel freq")
    elif isinstance(input, Sequence) and os.path.isfile(input[0]):
        for it in input:
            path = Path(it)
            wav, sr = torchaudio.load(path.resolve())
            spec = melspectrogram(wav)
            save_spectrogram(spec[0], path=os.path.join(
                outdir.resolve(), path.stem+".png"), title="MelSpectrogram (db)", ylabel="mel freq")


def plot_RGBDifference(input, outdir="./output/Video/"):
    try:
        outdir = Path(outdir)
        outdir.mkdir(parents=True)
    except FileExistsError:
        pass
    difference = calcRGBDifference(keepRGB=False)
    if isinstance(input, str) and os.path.isfile(input):
        with open(input, "rb") as fh:
            video_file = io.BytesIO(fh.read())
        _av_reader = decord.VideoReader(
            uri=video_file,
            ctx=decord.cpu(0),
            width=-1,
            height=-1,
            num_threads=8,
            fault_tol=-1,)
        width, height = video.shape[2], video.shape[1]
        _fps = _av_reader.get_avg_fps()
        video = _av_reader.get_batch(frame_idxs)
        video = video.asnumpy()
        video = difference(video).numpy()
        save_video(video, path=os.path.join(
            outdir.resolve(), Path(input).stem+".avi"), width=width, height=height, _fps=_fps)

    elif isinstance(input, Sequence) and os.path.isfile(input[0]):
        for it in input:
            path = Path(it)
            with open(path.resolve(), "rb") as fh:
                video_file = io.BytesIO(fh.read())
            _av_reader = decord.VideoReader(
                uri=video_file,
                ctx=decord.cpu(0),
                width=-1,
                height=-1,
                num_threads=8,
                fault_tol=-1,
            )

            _fps = _av_reader.get_avg_fps()
            frame_idxs = list(range(0, len(_av_reader)))
            video = _av_reader.get_batch(frame_idxs)
            width, height = video.shape[2], video.shape[1]
            video = video.asnumpy()
            video = difference(video).numpy().astype(np.uint8)
            save_video(video, out_path=os.path.join(
                outdir.resolve(), Path(path).stem+".avi"), width=width, height=height, _fps=_fps)


def save_video(video, out_path, width, height, _fps):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 视频的编码
    # 定义视频对象输出
    writer = cv2.VideoWriter(out_path, fourcc, _fps,
                             (width, height))  # , isColor=False
    for img in video:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # RGB2GRAY
        writer.write(img)  # 视频保存
    writer.release()


class Plot:
    def __init__(self, xlabel=None, ylabel=None, legend=None, grid=True, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('b-', 'm--', 'g-.', 'r:', 'c:', 'y:', 'k:', 'w:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5), ion=None):
        """使用matplotlib绘制数据点。
        para xlabel: 子窗X轴标签
        para ylabel: 子窗Y轴标签
        para legend: 设置左上角图例,例:legend(handles=[line, ...], labels=['label',...], loc='best(# location)')
        para grid: 确认是否生成网格
        para xlim: X轴初始值
        para ylim: Y轴初始值y
        para xscale: X轴值变化尺度,可选{"function", "functionlog", "linear", "log", "logit", "symlog"}
        para yscale: Y轴值变化尺度,可选{"function", "functionlog", "linear", "log", "logit", "symlog"}
        para fmts: 定义[颜色][标记][线条](蓝色实线,紫红色虚线,绿色交错点线,红色点线)
        para nrows: 图窗行数
        para ncols: 图窗列数
        para figsize: 图窗大小
        para ion: 是否支持动态更新，默认否
        """
        # 增量地绘制多条线
        if legend is None:
            self.legend = []  # 不设置图例
        else:
            self.legend = legend
        if ion:
            plt.ion()
        self.use_svg_display()
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数，设置matplotlib的轴等属性。
        self.config_axes = lambda: self.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend, grid)
        self.X, self.Y, self.fmts = None, None, fmts

    def use_svg_display(self):
        """使用svg格式显示绘图。"""
        display.set_matplotlib_formats('svg')

    def set_axes(self, axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend, grid):
        """设置matplotlib的轴。
        para axes: 传入的绘图子窗
        para xlabel: 子窗X轴标签
        para ylabel: 子窗Y轴标签
        para xlim: X轴初始值
        para ylim: Y轴初始值
        para xscale: X轴值变化尺度,可选{"function", "functionlog", "linear", "log", "logit", "symlog"}
        para yscale: Y轴值变化尺度,可选{"function", "functionlog", "linear", "log", "logit", "symlog"}
        para legend: 设置左上角图例,例:legend(handles=[line, ...], labels=['label',...], loc='best(# location)')
        para grid: 确认是否生成网格
        """
        axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel)
        axes.set_xscale(xscale)
        axes.set_yscale(yscale)
        axes.set_xlim(xlim)
        axes.set_ylim(ylim)
        if legend:
            axes.legend(legend)
        if grid:
            axes.grid()  # 生成网格

    def draw(self, X, Y=None):
        """使用matplotlib绘制数据点。
        para X: 给定Y为坐标否则为要绘制的数据
        para Y: 给定X条件下,Y为数据
        """
        # 如果输入的 `X` 只有有一个轴，输出True(表示只输入了一维数据,即一条线)
        def has_one_axis(X):
            return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                    and not hasattr(X[0], "__len__"))
        if has_one_axis(X):
            X = [X]
        if Y is None:  # Y为空,赋值X给Y,坐标为空
            X, Y = [[]] * len(X), X
        elif has_one_axis(Y):  # X,Y都给定了,X为坐标,Y为数据,且Y只有一条线的数据
            Y = [Y]
        if len(X) != len(Y):
            X = X * len(Y)  # 坐标长度不够就复制扩充
        self.axes[0].cla()  # 清除当前图形中的当前活动轴。其他轴不受影响。
        # 将三种元素一一对应打包成元组列表,多余的会自动删除,长度为最短的那个.
        for x, y, fmt in zip(X, Y, self.fmts):
            if len(x):
                self.axes[0].plot(x, y, fmt)  # 如果X,Y都给定了
            else:
                self.axes[0].plot(y, fmt)  # 如果只给定了X
        self.config_axes()

    def show_images(self, imgs, scale=1.5, n=1, titles=None):  # @save
        """展示一组二维灰度图片.
        para imgs: 图片集(tensor、numpy、bites)
        para titles: 图片标题，默认无
        para scale: 调整图像显示大小
        return 设置好的所有图窗（一维）
        """
        # self.fig.set_size_inches(self.fig.figure.bbox_inches.height*scale,self.fig.figure.bbox_inches.width*scale)
        # 得到绘图窗口(Figure)和绘图窗口上的坐标系(axis)
        self.axes = self.axes.flatten()  # 把rows*cols个子窗构成的array展平成一维的，方便遍历
        # 1. 每个子窗与对应的img对应，并打包成元组构成的列表 2. 将此列表组合为一个索引序列（下标，zip元组）
        for i, (ax, img) in enumerate(zip(self.axes, imgs)):
            if torch.is_tensor(img):  # 图片张量
                # img = unnormalize(img,[0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                if len(img.shape) == 2:
                    ax.imshow(img.cpu().numpy())
                elif len(img.shape) == 3:
                    ax.imshow(np.transpose(img.cpu().numpy(), (1, 2, 0)))
            else:  # PIL图片
                ax.imshow(img)
            ax.axes.get_xaxis().set_visible(False)  # 取消x坐标轴显示
            ax.axes.get_yaxis().set_visible(False)  # 取消y坐标轴显示
            if titles:
                ax.set_title(titles[i])  # 设置子窗标题
        return self.axes

    def show_spectrogram(self, specs, sr, scale=1.5, n=1, titles=None):  # @save
        """展示语谱图.
        para imgs: 功率谱数据集(tensor、numpy、bites)
        para titles: 图片标题，默认无
        para scale: 调整图像显示大小
        return 设置好的所有图窗（一维）
        """
        # self.fig.set_size_inches(self.fig.figure.bbox_inches.height*scale,self.fig.figure.bbox_inches.width*scale)
        # 得到绘图窗口(Figure)和绘图窗口上的坐标系(axis)
        self.axes = self.axes.flatten()  # 把rows*cols个子窗构成的array展平成一维的，方便遍历
        # 1. 每个子窗与对应的spec对应，并打包成元组构成的列表 2. 将此列表组合为一个索引序列（下标，zip元组）
        for i, (ax, spec) in enumerate(zip(self.axes, specs)):
            if torch.is_tensor(spec):  # 张量数据
                spec = librosa.display.specshow(
                    spec.cpu().numpy().T, y_axis='mel', sr=sr, x_axis='time', ax=ax)
                # self.fig.colorbar(spec, ax=ax, format="%+2.f dB")
            ax.axes.get_xaxis().set_visible(False)  # 取消x坐标轴显示
            ax.axes.get_yaxis().set_visible(False)  # 取消y坐标轴显示
            if titles:
                ax.set_title(titles[i])  # 设置子窗标题
        return self.axes

    def animator_add(self, x, y):
        """在动画中绘制数据。
        para x: x轴坐标
        para y: y轴多个的线条数据（点）
        """
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]  # 判断是否只有一个数据,是就加到列表里，也意味着只有一条线
        n = len(y)  # 获取需要绘制的线条数目
        if not hasattr(x, "__len__"):
            x = [x] * n  # 将坐标组复制n个,对应n条线
        if not self.X:  # 确定是否为第一次画线
            self.X = [[] for _ in range(n)]  # 定义n个空坐标组,对应n条线
        if not self.Y:
            self.Y = [[] for _ in range(n)]  # 定义n个空数据组,对应n条线
        for i, (a, b) in enumerate(zip(x, y)):  # 将坐标组与数据组一一对应，打包成可迭代对象，并给出每个(坐标,数据)的下标，下标对应哪条线段
            if a is not None and b is not None:
                self.X[i].append(a)  # 将坐标扩展到到self.X的第i条线中
                self.Y[i].append(b)  # 将数据扩展到到self.Y的第i条线中
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)  # 绘制已存储的点
        self.config_axes()  # 初始化窗口
        plt.pause(0.1)
        plt.ioff()
        # display.display(self.fig)
        # display.clear_output(wait=True)


def unnormalize(tensor, mean, std, inplace=False):
    """Unnormalize a tensor image with mean and standard deviation.

    Args:
        tensor (Tensor): Tensor image of size (C, H, W) or (B, C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation inplace.

    Returns:
        Tensor: Unnormalize Tensor image.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(
            'Input tensor should be a torch tensor. Got {}.'.format(type(tensor)))

    if tensor.ndim < 3:
        raise ValueError('Expected tensor to be a tensor image of size (..., C, H, W). Got tensor.size() = '
                         '{}.'.format(tensor.size()))

    if not inplace:
        tensor = tensor.clone()

    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    if (std == 0).any():
        raise ValueError(
            'std evaluated to zero after conversion to {}, leading to division by zero.'.format(dtype))
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    tensor.mul_(std).add_(mean)
    return tensor


def plot_tsne(model: torch.nn.Module, dataloader, outdir="./output/t-sne/", info="", device=torch.device('cpu')):
    """
    Visualizes the feature embeddings of a PyTorch model using t-SNE.

    Args:
        model (callable): A PyTorch model or any other callable that accepts a batch of data as input and returns feature embeddings.
        dataloader (torch.utils.data.DataLoader): A PyTorch DataLoader object that provides the data to be visualized.
    """
    try:
        outdir = Path(outdir)
        outdir.mkdir(parents=True)
    except FileExistsError:
        pass

    # Get feature embeddings for all data points
    # embeddings = [[], [], []]
    embeddings = None
    labels = []
    print("获取特征向量")
    with torch.no_grad():
        if model is not None:
            model = model.to(device)
            for data in dataloader:
                if "audio" in data:
                    audio_features = data["audio"]
                    audio_features = audio_features.float().to(device)
                if "video" in data:
                    video_features = data["video"]
                    video_features = video_features.float().to(device)
                labels = labels+data["label"]

                # embed = model(audio_features)
                embed = model(audio_features, video_features)
                # embed = model(video_features)
                if not isinstance(embed, tuple):
                    embed = (embed,)
                if embeddings is not None:
                    embed = [i.cpu().numpy() for i in embed]
                    for i in range(len(embeddings)):
                        embeddings[i].append(embed[i])
                else:
                    embed = [i.cpu().numpy() for i in embed]
                    embeddings = [[embed[i]] for i in range(len(embed))]

    embeddings = [np.concatenate(embedding, axis=0)
                  for embedding in embeddings]
    label_encoder = sklearn.preprocessing.LabelEncoder()
    label_encoder.fit(y=labels)
    labels = label_encoder.transform(labels)
    classes_ = label_encoder.classes_

    print("t-SNE 降维")
    tsne = TSNE(n_components=2, random_state=0, init='pca', perplexity=30)

    embeddings_tsne = [tsne.fit_transform(
        embedding) for embedding in embeddings]

    print("展示图片")
    for j, embedding_tsne in enumerate(embeddings_tsne):
        plt.clf()
        plt.cla()
        plt.figure(figsize=(10, 10))
        unique_labels = np.unique(labels)
        for i in unique_labels:
            mask = labels == i
            plt.scatter(embedding_tsne[mask, 0],
                        embedding_tsne[mask, 1], label=classes_[i])
        plt.legend()
        # Save the figure if an output path is specified
        if outdir is not None:
            plt.savefig(os.path.join(
                outdir.resolve(), f"{info}_t-sne_{j}.png"))
        else:
            plt.show()


def plot_rt(model: torch.nn.Module, dataloader, outdir="./output/real-time/", info="", device=torch.device('cpu')):
    """
    Visualizes the real-time classification.

    Args:
        model (callable): A PyTorch model or any other callable that accepts a batch of data as input and returns feature embeddings.
        dataloader (torch.utils.data.DataLoader): A PyTorch DataLoader object that provides the data to be visualized.
    """
    try:
        outdir = Path(outdir)
        outdir.mkdir(parents=True)
    except FileExistsError:
        pass

    # Get preds for all data points
    all_preds = []
    classes_ = {"angry": 0, "happy": 1, "neutral": 2, "sad": 3}
    print("获取预测值")
    with torch.no_grad():

        if model is not None:
            model = model.to(device)
            for data in dataloader:
                if "audio" in data:
                    audio_features = data["audio"]
                    audio_features = audio_features.float().to(device)
                    # audio_features = torch.split(audio_features, 441, dim=1)
                    af_seq_len = 441
                    af_seq_step = 63
                    audio_features = [audio_features[:, af_seq_step*i:af_seq_step*i+af_seq_len, :, :]
                                      for i in range(int(((audio_features.shape[1]-af_seq_len) / af_seq_step)+1))]

                if "video" in data:
                    video_features = data["video"]
                    video_features = video_features.float().to(device)
                    # video_features = torch.split(video_features, 70, dim=1)

                    vf_seq_len = 70
                    vf_seq_step = 10
                    video_features = [video_features[:, vf_seq_step*i:vf_seq_step*i+vf_seq_len, :, :, :]
                                      for i in range(int(((video_features.shape[1]-vf_seq_len) / vf_seq_step)+1))]

                # out = [model(a, v)[2]
                #        for a, v in zip(audio_features, video_features)]
                # out = [model(a)[0] for a in audio_features]
                out = [model(v)[0] for v in video_features]
                if out:
                    out = torch.stack(out).permute(1, 0, 2)
                    pred_prob = F.softmax(out, dim=2).tolist()
                    target = [classes_[t] for t in data["label"]]

                    for i_b, b_pre in enumerate(pred_prob):
                        t = target[i_b]
                        for i_p, pre in enumerate(b_pre):
                            pred_prob[i_b][i_p] = pred_prob[i_b][i_p][t]
                    a = len(pred_prob)
                    all_preds = all_preds+pred_prob

    print("展示图片")
    plt.clf()
    plt.cla()
    fig, ax = plt.subplots(figsize=(10, 10))
    for pre in all_preds:
        x = np.arange(len(pre))
        pre = np.array(pre)
        ax.plot(x, pre, linestyle='-', marker='.')  # , color='coral'

    # Save the figure if an output path is specified
    ax.legend()
    if outdir is not None:
        fig.savefig(os.path.join(
            outdir.resolve(), f"{info}_rtPred_{0}.png"))
    else:
        fig.show()
