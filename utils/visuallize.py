from IPython import display
from matplotlib import pyplot as plt
import torch
import numpy as np
import librosa.display


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
