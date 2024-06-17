import os
import sys
from typing import Sequence
from pathlib import Path
from IPython import display
from sklearn import preprocessing
import torch
import numpy as np
import librosa.display
import torchaudio
import decord
# from custom_transforms.video_transforms import *
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import sklearn
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from matplotlib.font_manager import FontProperties
import matplotlib as mpl
from matplotlib.font_manager import FontManager
from matplotlib import font_manager
from torch.functional import F

font_dirs = ["/home/visitors2/.fonts"]
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for font_file in font_files:
    font_manager.fontManager.addfont(font_file)

font_path = '/home/visitors2/.fonts/Noto_Sans_SC/static/NotoSansSC_Regular.ttf'
font_properties = FontProperties(fname=font_path)
font = font_manager.fontManager.findfont(font_properties)
print("Font name to find:", font)
mpl.rcParams['font.family'] = font_properties.get_name()
print("Font name to use in matplotlib:", mpl.rcParams['font.family'])

def plot_spectrogram(input, outdir="./output/image/"):
    try:
        outdir = Path(outdir)
        outdir.mkdir(parents=True)
    except FileExistsError:
        pass
    spectrogram = torchaudio.transforms.Spectrogram(
        n_fft=1024,
        win_length=1024,
        hop_length=256,
        pad=0,
        power=2,
        normalized=True,
        center=True,
        pad_mode="reflect",
        onesided=True,
    )
    if isinstance(input, str) and os.path.isfile(input):
        wav, sr = torchaudio.load(input)
        spec = spectrogram(wav)
        save_spectrogram(
            spec[0],
            path=os.path.join(outdir.resolve(), Path(input).stem + ".svg"),
            title="Spectrogram (db)",
            ylabel="freq_bin",
        )
    elif isinstance(input, Sequence) and os.path.isfile(input[0]):
        for it in input:
            path = Path(it)
            wav, sr = torchaudio.load(path.resolve())
            spec = spectrogram(wav)
            save_spectrogram(
                spec[0],
                path=os.path.join(outdir.resolve(), path.stem + ".svg"),
                title="Spectrogram (db)",
                ylabel="freq_bin",
            )


def save_spectrogram(specgram, path, title, ylabel):
    fig, axs = plt.subplots(1, 1)
    fig.set_dpi(600)  # 这里将 DPI 设置为 600
    axs.set_title(title)
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    # cmap =  ["inferno", "gnuplot2", "CMRmap", "jet", "turbo"]
    im = axs.imshow(
        librosa.power_to_db(specgram), origin="lower", aspect="auto", cmap=plt.cm.jet
    )
    fig.colorbar(im, ax=axs)
    plt.savefig(
        path,
        dpi=600,
        transparent=True,
        bbox_inches="tight",
        pad_inches=0.0,
    )
    plt.close()


def plot_melspectrogram(input, outdir="./output/image/"):
    try:
        outdir = Path(outdir)
        outdir.mkdir(parents=True)
    except FileExistsError:
        pass
    melspectrogram = torchaudio.transforms.MelSpectrogram(
        n_mels=128,
        n_fft=1024,
        win_length=1024,
        hop_length=256,
        f_min=80,
        f_max=7600,
        pad=0,
        power=2,
        normalized=True,
        center=True,
        pad_mode="",
        onesided=True,
    )
    if isinstance(input, str) and os.path.isfile(input):
        wav, sr = torchaudio.load(input)
        spec = melspectrogram(wav)
        save_spectrogram(
            spec[0],
            path=os.path.join(outdir.resolve(), Path(input).stem + ".svg"),
            title="MelSpectrogram (db)",
            ylabel="mel freq",
        )
    elif isinstance(input, Sequence) and os.path.isfile(input[0]):
        for it in input:
            path = Path(it)
            wav, sr = torchaudio.load(path.resolve())
            spec = melspectrogram(wav)
            save_spectrogram(
                spec[0],
                path=os.path.join(outdir.resolve(), path.stem + ".svg"),
                title="MelSpectrogram (db)",
                ylabel="mel freq",
            )


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
            fault_tol=-1,
        )
        width, height = video.shape[2], video.shape[1]
        _fps = _av_reader.get_avg_fps()
        video = _av_reader.get_batch(frame_idxs)
        video = video.asnumpy()
        video = difference(video).numpy()
        save_video(
            video,
            path=os.path.join(outdir.resolve(), Path(input).stem + ".avi"),
            width=width,
            height=height,
            _fps=_fps,
        )

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
            save_video(
                video,
                out_path=os.path.join(outdir.resolve(), Path(path).stem + ".avi"),
                width=width,
                height=height,
                _fps=_fps,
            )


def save_video(video, out_path, width, height, _fps):
    fourcc = cv2.VideoWriter_fourcc(*"XVID")  # 视频的编码
    # 定义视频对象输出
    writer = cv2.VideoWriter(out_path, fourcc, _fps, (width, height))  # , isColor=False
    for img in video:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # RGB2GRAY
        writer.write(img)  # 视频保存
    writer.release()


class Plot:
    def __init__(
        self,
        xlabel=None,
        ylabel=None,
        legend=None,
        grid=True,
        xlim=None,
        ylim=None,
        xscale="linear",
        yscale="linear",
        fmts=("b-", "m--", "g-.", "r:", "c:", "y:", "k:", "w:"),
        nrows=1,
        ncols=1,
        figsize=(3.5, 2.5),
        ion=None,
    ):
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
            self.axes = [
                self.axes,
            ]
        # 使用lambda函数捕获参数，设置matplotlib的轴等属性。
        self.config_axes = lambda: self.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend, grid
        )
        self.X, self.Y, self.fmts = None, None, fmts

    def use_svg_display(self):
        """使用svg格式显示绘图。"""
        display.set_matplotlib_formats("svg")

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
            return (
                hasattr(X, "ndim")
                and X.ndim == 1
                or isinstance(X, list)
                and not hasattr(X[0], "__len__")
            )

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
                    spec.cpu().numpy().T, y_axis="mel", sr=sr, x_axis="time", ax=ax
                )
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
        for i, (a, b) in enumerate(
            zip(x, y)
        ):  # 将坐标组与数据组一一对应，打包成可迭代对象，并给出每个(坐标,数据)的下标，下标对应哪条线段
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
            "Input tensor should be a torch tensor. Got {}.".format(type(tensor))
        )

    if tensor.ndim < 3:
        raise ValueError(
            "Expected tensor to be a tensor image of size (..., C, H, W). Got tensor.size() = "
            "{}.".format(tensor.size())
        )

    if not inplace:
        tensor = tensor.clone()

    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    if (std == 0).any():
        raise ValueError(
            "std evaluated to zero after conversion to {}, leading to division by zero.".format(
                dtype
            )
        )
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    tensor.mul_(std).add_(mean)
    return tensor


def plot_avc_tsne(
    model: torch.nn.Module,
    dataloader,
    outdir="./output/t-sne/",
    info="",
    device=torch.device("cpu"),
):
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
    embeddings = [[], [], []]  # a_embed, v_embed, c_embed
    labels = []
    print("获取特征向量")
    with torch.no_grad():
        model = model.to(device)

        # count = 0
        for data in dataloader:
            # if count == 20:
            #     break
            audio_features, video_features = data["audio"], data["video"]
            audio_features = audio_features.float().to(device)
            video_features = video_features.float().to(device)
            labels = labels + data["label"]

            a_embed, v_embed, c_embed = model(audio_features, video_features)

            embeddings[0].append(a_embed.cpu().numpy())
            embeddings[1].append(v_embed.cpu().numpy())
            embeddings[2].append(c_embed.cpu().numpy())
            # count += 1

    embeddings = [np.concatenate(embedding, axis=0) for embedding in embeddings]

    print("t-SNE 降维")
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        random_state=42,
        init="pca",  # random
        early_exaggeration=12.0,
        learning_rate=500,
        n_iter=1000,
        n_iter_without_progress=300,
        min_grad_norm=1e-7,
        metric="euclidean",
        metric_params=None,
        verbose=0,
        method="barnes_hut",
        angle=0.6,
        n_jobs=-1,
    )

    embeddings_tsne = [
        preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit_transform(
            tsne.fit_transform(embedding)
        )
        for embedding in embeddings
    ]
    # [3, B, 2]

    print("展示图片")
    titles = ["a_embed", "v_embed", "c_embed"]
    plt.clf()
    plt.cla()
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(10, 10))
    fig.set_dpi(600)  # 这里将 DPI 设置为 600
    for i in range(3):
        x = embeddings_tsne[i][:, 0]
        y = embeddings_tsne[i][:, 1]
        ax.scatter(x, y, label=titles[i])
    # 设置图示字体属性
    legend_font = {"family": "serif", "size": 20, "weight": "bold"}  # 定义字体属性
    ax.legend(prop=legend_font)
    # Save the figure if an output path is specified
    ax.tick_params(axis="both", labelsize=20, which="both", width=5)
    ax.yaxis.set_ticks([-1, -0.5, 0, 0.5, 1])
    ax.xaxis.set_ticks([-1, -0.5, 0, 0.5, 1])
    if outdir is not None:
        fig.tight_layout()
        fig.savefig(os.path.join(outdir.resolve(), f"{info}_t-sne.png"), dpi=600)
    else:
        fig.show()


def plot_tsne(
    model: torch.nn.Module,
    dataloader,
    outdir="./output/t-sne/",
    info="",
    device=torch.device("cpu"),
):
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
                audio_features = None
                video_features = None
                if "audio" in data:
                    audio_features = data["audio"]
                    audio_features = audio_features.float().to(device)
                if "video" in data:
                    video_features = data["video"]
                    video_features = video_features.float().to(device)
                labels = labels + data["label"]

                if audio_features is not None and video_features is not None:
                    embed = model(audio_features, video_features)
                elif audio_features is not None:
                    embed = model(audio_features)
                elif video_features is not None:
                    embed = model(video_features)

                if not isinstance(embed, tuple):
                    embed = (embed,)
                if embeddings is not None:
                    embed = [i.cpu().numpy() for i in embed]
                    for i in range(len(embeddings)):
                        embeddings[i].append(embed[i])
                else:
                    embed = [i.cpu().numpy() for i in embed]
                    embeddings = [[embed[i]] for i in range(len(embed))]
        else:
            for data in dataloader:
                audio_features = None
                video_features = None
                if "audio" in data:
                    audio_features = data["audio"]
                    audio_features = audio_features.float()
                if "video" in data:
                    video_features = data["video"]
                    video_features = video_features.float()
                labels = labels + data["label"]
                if audio_features is not None and video_features is not None:
                    embed = (
                        audio_features.view(audio_features.shape[0], -1),
                        video_features.view(video_features.shape[0], -1),
                    )
                elif audio_features is not None:
                    embed = (audio_features.view(audio_features.shape[0], -1),)
                elif video_features is not None:
                    embed = (video_features.view(audio_features.shape[0], -1),)

                if embeddings is not None:
                    embed = [i.cpu().numpy() for i in embed]
                    for i in range(len(embeddings)):
                        embeddings[i].append(embed[i])
                else:
                    embed = [i.cpu().numpy() for i in embed]
                    embeddings = [[embed[i]] for i in range(len(embed))]

    embeddings = [np.concatenate(embedding, axis=0) for embedding in embeddings]
    label_encoder = sklearn.preprocessing.LabelEncoder()
    label_encoder.fit(y=labels)
    labels = label_encoder.transform(labels)
    classes_ = label_encoder.classes_

    print("t-SNE 降维")
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        early_exaggeration=12.0,
        learning_rate=200,
        n_iter=1000,
        n_iter_without_progress=300,
        min_grad_norm=1e-7,
        metric="euclidean",
        init="random",
        random_state=None,
        method="barnes_hut",
        angle=0.5,
        metric_params=None,
        verbose=0,
        n_jobs=-1,
    )

    embeddings_tsne = [
        preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit_transform(
            tsne.fit_transform(embedding)
        )
        for embedding in embeddings
    ]

    print("展示图片")
    for j, embedding_tsne in enumerate(embeddings_tsne):
        plt.clf()
        plt.cla()
        fig, ax = plt.subplots(1, 1, sharex=True, figsize=(10, 10))
        fig.set_dpi(600)  # 这里将 DPI 设置为 600
        unique_labels = np.unique(labels)
        for i in unique_labels:
            mask = labels == i
            ax.scatter(
                embedding_tsne[mask, 0], embedding_tsne[mask, 1], label=classes_[i]
            )
        # 设置图示字体属性
        legend_font = {"family": "serif", "size": 20, "weight": "bold"}  # 定义字体属性
        ax.legend(prop=legend_font)
        # Save the figure if an output path is specified
        ax.tick_params(axis="both", labelsize=20, which="both", width=5)
        ax.yaxis.set_ticks([-1, -0.5, 0, 0.5, 1])
        ax.xaxis.set_ticks([-1, -0.5, 0, 0.5, 1])
        if outdir is not None:
            fig.tight_layout()
            fig.savefig(
                os.path.join(outdir.resolve(), f"{info}_t-sne_{j}.svg"), dpi=600
            )
        else:
            fig.show()


def plot_rt(
    model: torch.nn.Module,
    dataloader,
    outdir="./output/real-time/",
    info="",
    device=torch.device("cpu"),
):
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
    smoothness_scores = []
    classes_ = {"angry": 0, "happy": 1, "neutral": 2, "sad": 3}
    print("获取预测值")
    with torch.no_grad():
        if model is not None:
            model = model.to(device)
            for ind, data in enumerate(dataloader):
                if (ind not in [0, 3, 6, 7, 15, (16), 17, 21, 22] ):
                    continue
                audio_features = None
                video_features = None
                if "audio" in data:
                    audio_features = data["audio"]
                    audio_features = audio_features.float().to(device)
                    # audio_features = torch.split(audio_features, 441, dim=1)
                    af_seq_len = 63 * 7
                    af_seq_step = 63 * 3
                    total_length = audio_features.shape[1]
                    
                    remainder = (total_length - af_seq_len) % af_seq_step
                    # 计算需要填充的数量
                    if remainder != 0:
                        padding_length = af_seq_step - remainder
                    else:
                        padding_length = 0
                    padding = torch.zeros((audio_features.shape[0], padding_length, audio_features.shape[2], audio_features.shape[3]), device=device)
                    audio_features = torch.cat((audio_features, padding), dim=1)
                    
                    
                    
                    audio_features = [
                        audio_features[
                            :, af_seq_step * i : af_seq_step * i + af_seq_len, :, :
                        ]
                        for i in range(
                            int(
                                ((audio_features.shape[1] - af_seq_len) / af_seq_step)
                                + 1
                            )
                        )
                    ]

                if "video" in data:
                    video_features = data["video"]
                    video_features = video_features.float().to(device)
                    # video_features = torch.split(video_features, 70, dim=1)

                    vf_seq_len = 10 * 7
                    vf_seq_step = 10 * 3
                    total_length = video_features.shape[1]
                    
                    remainder = (total_length - vf_seq_len) % vf_seq_step
                    # 计算需要填充的数量
                    if remainder != 0:
                        padding_length = vf_seq_step - remainder
                    else:
                        padding_length = 0
                        
                    padding = torch.zeros((video_features.shape[0], padding_length, video_features.shape[2], video_features.shape[3], video_features.shape[4]), device=device)
                    video_features = torch.cat((video_features, padding), dim=1)

                    video_features = [
                        video_features[
                            :, vf_seq_step * i : vf_seq_step * i + vf_seq_len, :, :, :
                        ]
                        for i in range(
                            int(
                                ((video_features.shape[1] - vf_seq_len) / vf_seq_step)
                                + 1
                            )
                        )
                    ]
                if audio_features is not None and video_features is not None:
                    pred_prob = [
                        F.softmax(model(a, v)[0], dim=1)
                        for a, v in zip(audio_features, video_features)
                    ]
                elif audio_features is not None:
                    pred_prob = [F.softmax(model(a)[0], dim=1) for a in audio_features]
                elif video_features is not None:
                    pred_prob = [F.softmax(model(v)[0], dim=1) for v in video_features]
                # S * [B,C]
                if pred_prob:
                    pred_prob = torch.stack(pred_prob).permute(1, 0, 2).tolist()
                    #  [B, S, C]
                    target = [classes_[t] for t in data["label"]]
                    #  [B]
                    for i_b, b_pre in enumerate(pred_prob):
                        t = target[i_b]
                        for i_p, pre in enumerate(b_pre):
                            pred_prob[i_b][i_p] = pred_prob[i_b][i_p][t]
                    all_preds = all_preds + pred_prob
    
    # Calculate smoothness
    for preds in all_preds:
        angles = np.arctan(np.diff(preds) / 3)  # 3 seconds per step, assume y/x
        angle_changes = np.abs(np.diff(angles))
        smoothness = np.mean(angle_changes)
        smoothness_scores.append(smoothness)
    average_smoothness = np.mean(smoothness_scores)
    print("Average Smoothness of All Curves:", average_smoothness)
    
    colors = [
        "#FFB6C1",
        "#66CDAA",
        "#FFD700",
        "#6A5ACD",
        "#FFA07A",
        "#98FB98",
        "#2E8B57",
        "#CD853F",
        "#20B2AA",
        "#FF4500",
        "#DB7093",
        "#32CD32",
        "#8B008B",
        "#ADFF2F",
        "#4169E1",
        "#FF69B4",
        "#7FFF00",
        "#0000FF",
        "#BA55D3",
        "#3CB371",
        "#1E90FF",
        "#8A2BE2",
        "#00FF7F",
        "#6B8E23",
        "#9370DB",
        "#2F4F4F",
        "#FFD700",
        "#00BFFF",
        "#FF8C00",
        "#8B4513",
    ]
    print("展示图片")
    plt.clf()
    plt.cla()
    fig, ax = plt.subplots(len(all_preds), 1, sharex=True, figsize=(10, 13))
    fig.set_dpi(600)  # 这里将 DPI 设置为 600
    for i, preds in enumerate(all_preds):
        x = 7 + np.arange(0, len(preds)) * 3  # Convert index to seconds (3 seconds per step)
        ax[i].plot(
            x,
            preds,
            linestyle="-",
            marker=".",
            color=colors[i % len(colors)],
            label=f"{i}",
            linewidth=4,
        )
        # 设置图示字体属性
        ax[i].legend(loc='upper right', prop={"family": "serif", "size": 20, "weight": "bold"})
        ax[i].tick_params(axis="both", labelsize=20, which="both", width=5)
        ax[i].yaxis.set_ticks([-0.1, 1.1])
        # 设置 y 轴刻度显示整数位
        ax[i].yaxis.set_major_locator(MultipleLocator(base=1.0))# 设置刻度间隔为 1
    if outdir is not None:
        fig.tight_layout()
        fig.savefig(os.path.join(outdir.resolve(), f"{info}_rtPred.png"), dpi=600)
    else:
        fig.show()


def plot_confusionMatrix(
    model: torch.nn.Module,
    dataloader,
    outdir="./output/confusionMatrix/",
    info="",
    device=torch.device("cpu"),
):
    """
    绘制混淆矩阵
    :param model: 已训练好的分类模型
    :param dataloader: 数据集的dataloader
    :param outdir: 保存混淆矩阵图像的输出目录
    :param info: 给混淆矩阵图像添加的注释信息
    :param device: 运行模型的设备, 默认为CPU
    """
    try:
        outdir = Path(outdir)
        outdir.mkdir(parents=True)
    except FileExistsError:
        pass
    model.eval()
    # Get preds for all data points
    all_preds = []
    all_labels = []

    print("获取预测值")
    with torch.no_grad():
        if model is not None:
            model = model.to(device)
            for data in dataloader:
                audio_features = None
                video_features = None
                if "audio" in data:
                    audio_features = data["audio"]
                    audio_features = audio_features.float().to(device)

                if "video" in data:
                    video_features = data["video"]
                    video_features = video_features.float().to(device)

                if audio_features is not None and video_features is not None:
                    pred_prob = model(audio_features, video_features)[0]
                elif audio_features is not None:
                    pred_prob = model(audio_features)[0]
                elif video_features is not None:
                    pred_prob = model(video_features)[0]

                _, preds = torch.max(pred_prob, 1)
                all_preds.extend(preds.tolist())
                all_labels.extend(data["label"])

    label_encoder = sklearn.preprocessing.LabelEncoder()
    label_encoder.fit(y=all_labels)
    all_labels = label_encoder.transform(all_labels)
    classes_ = label_encoder.classes_
    # classes_ = {"angry": 0, "happy": 1, "neutral": 2, "sad": 3}
    classes = {"angry", "happy", "neutral", "sad"}

    cm = confusion_matrix(all_labels, all_preds)
    # 绘制混淆矩阵
    # classes = np.arange(len(cm))
    fig, ax = plt.subplots()
    fig.set_dpi(600)  # 这里将 DPI 设置为 600
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title="Confusion matrix" + info,
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fmt = "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()

    if outdir is not None:
        fig.savefig(
            os.path.join(outdir.resolve(), f"{info}_confusionMatrix.svg"), dpi=600
        )
    else:
        fig.show()


from torch.nn.functional import softmax, kl_div

def plot_features(
    model: torch.nn.Module,
    data,
    outdir="./output/features/",
    info="",
    device=torch.device("cpu"),
):
    try:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        pass
    
    print("获取特征向量")
    with torch.no_grad():
        if model is not None:
            model = model.to(device)
            audio_features = None
            video_features = None
            if "audio" in data:
                audio_features = data["audio"]
                audio_features = audio_features.float().to(device)
            if "video" in data:
                video_features = data["video"]
                video_features = video_features.float().to(device)

            if audio_features is not None and video_features is not None:
                embed = model(audio_features, video_features)
            elif audio_features is not None:
                embed = model(audio_features)
            elif video_features is not None:
                embed = model(video_features)
            else:
                raise ValueError("No valid audio or video features provided.")
                
    [common_feature, af_fea, vf_fea, common_p, af_p, vf_p] = embed[1]
    features = [af_fea, common_feature, vf_fea]
    prob_features = [af_p, common_p, vf_p]
    feature_labels = ['音频特征', '相似性特征', '视频特征']
    prob_labels = ['音频概率分布特征', '相似性概率分布特征', '视频概率分布特征']
    
    # features =[af_fea, common_feature, vf_fea, af_p, common_p, vf_p]
    # feature_labels = ['音频特征', '相似性特征', '视频特征', '音频概率分布特征', '相似性概率分布特征', '视频概率分布特征']

    # Plot feature vectors
    plt.figure(figsize=(15, 5))
    for i, (feature, label) in enumerate(zip(features, feature_labels)):
        feature = feature[0]  # Assume first item in batch
        normalized_feature = (feature - feature.min()) / (feature.max() - feature.min())
        ax = plt.subplot(1, 3, i+1)
        # 为每一行中间的图形选择更深的颜色
        if i == 2:  # 第二个即中间的位置
            color_intensity = 0.7  # 更深的颜色
        else:
            color_intensity = 0.5  # 相对较浅的颜色
        color_value = plt.get_cmap('Oranges')(color_intensity)

        ax.plot(normalized_feature.cpu().numpy(), color=color_value)
        ax.set_title(f'{label}', fontsize=20)
        ax.set_xlabel('特征维度', fontsize=18)
        ax.set_ylabel('标准化特征值', fontsize=18)
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
    plt.tight_layout()
    plt.savefig(outdir / f"{info}_feature_vectors.png", dpi = 1200)
    plt.close()

    # Plot probability distributions and compute KL divergences
    plt.figure(figsize=(15, 5))
    kl_divergences = []
    for i, (prob_feature, label) in enumerate(zip(prob_features, prob_labels)):
        prob_feature = prob_feature[0]  # Apply softmax and assume first item in batch
        normalized_feature = (prob_feature - prob_feature.min()) / (prob_feature.max() - prob_feature.min())
        ax = plt.subplot(1, 3, i+1)
        # 为每一行中间的图形选择更深的颜色
        if i == 2:  # 第二个即中间的位置
            color_intensity = 0.7  # 更深的颜色
        else:
            color_intensity = 0.5  # 相对较浅的颜色
        color_value = plt.get_cmap('Blues')(color_intensity)
        # ax.bar(range(len(normalized_feature)), normalized_feature.numpy())
        ax.plot(normalized_feature.cpu().numpy(), color=color_value)
        ax.set_title(f'{label}', fontsize=20)
        ax.set_xlabel('特征维度', fontsize=18)
        ax.set_ylabel('标准化特征值', fontsize=18)
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        
        if i != 1:  # Compute KL divergence with the "common_p" as reference
            kl_divergences.append(kl_div(prob_features[1].log(), prob_features[i], reduction="batchmean"))
    plt.tight_layout()
    plt.savefig(outdir / f"{info}_prob_distributions.png", dpi = 1200)
    plt.close()

    # Print KL divergences
    for i, kl_div_value in enumerate(kl_divergences):
        print(f"{info} KL divergence between common and {'audio' if i == 0 else 'video'} features: {kl_div_value:.6f}")
    
    # plt.figure(figsize=(15, 10))
    # for i, (feature, label) in enumerate(zip(features, feature_labels), 1):
    #     feature = feature[0]
    #     normalized_feature = (feature - feature.min()) / (feature.max() - feature.min())
    #     ax = plt.subplot(2, 3, i)
    #     # 为每一行中间的图形选择更深的颜色
    #     if i % 3 == 2:  # 第二个即中间的位置
    #         color_intensity = 0.7  # 更深的颜色
    #     else:
    #         color_intensity = 0.5  # 相对较浅的颜色

    #     # 选择颜色系列基于行号
    #     color_map = warm_colors if i <= 3 else cool_colors
    #     color_value = color_map(color_intensity)
    #     ax.plot(normalized_feature.cpu().numpy(), color=color_value)  # 使用线图展示
    #     ax.set_title(f'{label}', fontsize=20)
    #     ax.set_xlabel('特征维度', fontsize=18)
    #     ax.set_ylabel('标准化特征值', fontsize=18)
    #     ax.tick_params(axis='x', labelsize=16)
    #     ax.tick_params(axis='y', labelsize=16)
    # # 调整子图间隔
    # plt.subplots_adjust(hspace=0.5)  # 调整垂直间隔
    # plt.tight_layout()
    # plt.savefig(outdir / f"{info}_features_line.png")
    # plt.close()

def plot_image(model: torch.nn.Module, data, outdir="./output/features/", info="", device=torch.device("cpu")):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    print("获取处理结果")
    with torch.no_grad():
        model = model.to(device)
        if "video" in data:
            video_features = data["video"].float().to(device)  # [B, Seq, W, H, C]
            video_features = video_features.permute(0, 1, 4, 2, 3)  # Rearrange to [B, Seq, C, W, H]
            
            T = video_features.shape[1]  # 获取总帧数
            frame_indices = np.random.choice(T, size=5, replace=False)
            for idx in frame_indices:
                # 处理每个随机选择的帧
                sample_frame = video_features[:, idx, :, :, :].unsqueeze(1)  # [B, 1, C, H, W]
                deform_input = sample_frame.reshape(-1, sample_frame.shape[2], sample_frame.shape[3], sample_frame.shape[4])  # [B * 1, C, H, W]
                deform_output = model.video_feature_extractor.deform_conv(deform_input)  # 应用 deform_conv 层
                # deform_output = model.video_feature_extractor.model.conv1(deform_output)
                # deform_output = model.video_feature_extractor.model.bn1(deform_output)
                # deform_output = model.video_feature_extractor.model.relu(deform_output)

                deform_output_normalized = torch.zeros_like(deform_output)
                for i in range(deform_output.shape[1]):  # 遍历所有通道
                    channel_min = deform_output[:, i, :, :].min()
                    channel_max = deform_output[:, i, :, :].max()
                    deform_output_normalized[:, i, :, :] = (deform_output[:, i, :, :] - channel_min) / (channel_max - channel_min)
                # 分别可视化每个通道
                fig, axes = plt.subplots(1, 4, figsize=(12, 6))
                for i in range(3):
                    channel_data = deform_output_normalized[0][i].cpu().unsqueeze(-1).numpy()
                    axes[i].imshow(channel_data, cmap='gray')  # 以灰度图形式显示
                    axes[i].set_title(f'Channel {i+1}')
                    axes[i].axis('off')  # 关闭坐标轴
            
                axes[3].imshow(deform_input[0].cpu().permute(1, 2, 0).numpy().astype('uint8'), cmap='gray')  # 以灰度图形式显示
                axes[3].set_title(f'原图 {i+1}')
                axes[3].axis('off')  # 关闭坐标轴
                
                plt.tight_layout()
                plt.savefig(outdir / f"{info}_frame_{idx}.png")
                plt.close()