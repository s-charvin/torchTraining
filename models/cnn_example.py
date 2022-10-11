from layers import Inception, DenseBlock, Residual, GraphConvolution
import torch.nn.functional as F  # nn.Conv2是一个类，而F.conv2d是一个函数
import torch.nn as nn


class CNNNetExemple(nn.Module):

    def __init__(self, config) -> None:
        """简单的CNN网络示例
        Args:
            config (_type_): _description_
        """
        super().__init__()
        self.config = config
        # LinearNN、MultilayerNN、LeNet、AlexNet、VggNet、NiNNet
        # GoogleNet、ResNet、DenseNet、
        self.input_chanel = self.config['model']['input_chanel']  # 输入特征维度
        self.num_outputs = self.config['model']['num_outputs']  # 输出分类数
        # 网络的构成的Sequential序列
        self.net = self.ResNet(self.input_chanel, self.num_outputs)

    def init_weights(self, m):
        """初始化输入结构 m 的参数"""
        if type(m) == nn.Linear or type(m) == nn.Conv2d:  # 判断需要初始化的网络层是否是全连接层或卷积层
            nn.init.xavier_uniform_(m.weight)
            # nn.init.normal_(m.weight, std=0.01)  # 重写参数值w,为正太分布（0，0.01）中的随机取值
            # m.bias.data.fill_(0)  # 重写参数值b为0
            # print(f"weight:{m.weight.data},bias:{m.bias.data}")

    def forward(self, X):
        # 模型前向计算，输入X得到Y
        X = X.unsqueeze(1)
        Y = self.net(X)
        return F.softmax(Y, dim=1)  # 得到最终分类概率值

    def LinearNN(self, num_inputs=2, num_outputs=1):
        # Sequential定义了一个容器，包含多个串联在一起的层，前一层输出会作为下一层的输入
        # 给定线性模型输入特征形状：2，输出特征形状：1。
        return nn.Sequential(
            nn.Linear(num_inputs, 10),
            nn.ReLU(),
            nn.Linear(10, num_outputs)
        )

    def MultilayerNN(self, num_inputs, num_outputs):
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_inputs, 256),
            nn.ReLU(),
            nn.Linear(256, num_outputs)
        )

    def LeNet(self, dim_inputs=1, num_outputs=10):
        """
        LeNet-5由两个卷积层、三个全连接层组成。
        """
        return nn.Sequential(
            # 如果要做彩色图像的任务,换成nn.Conv2d(3, 6, kernel_size=5)即可
            nn.Conv2d(dim_inputs, 6, kernel_size=5, padding=2),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(6, 16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Flatten(),

            nn.Linear(16 * 5 * 5, 120),
            nn.BatchNorm1d(120),
            nn.ReLU(),

            nn.Linear(120, 84),
            nn.BatchNorm1d(84),
            nn.ReLU(),

            nn.Linear(84, num_outputs)
        )

    def AlexNet(self, dim_inputs, num_outputs):
        """
        AlexNet由八层组成：五个卷积层、两个全连接隐藏层和一个全连接输出层。
        AlexNet结构与LeNet相似
        """
        return nn.Sequential(
            # 这里，我们使用一个5*5的窗口来捕捉对象。
            # 同时，步幅为4，以减少输出的高度和宽度。
            # 另外，输出通道的数目远大于LeNet
            # 论文中kernel_size = 11,stride = 4,padding = 2
            nn.Conv2d(dim_inputs, 96, kernel_size=5, stride=2, padding=2),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            # 使用三个连续的卷积层和较小的卷积窗口。
            # 除了最后的卷积层，输出通道的数量进一步增加。
            # 在前两个卷积层之后，汇聚层不用于减少输入的高度和宽度
            nn.Conv2d(256, 384, kernel_size=3, stride=1,  padding=1),
            nn.Sigmoid(),

            nn.Conv2d(384, 384, kernel_size=3, stride=1,  padding=1),
            nn.Sigmoid(),

            nn.Conv2d(384, 384, kernel_size=3, stride=1,  padding=1),
            nn.Sigmoid(),

            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Flatten(),

            # 这里，全连接层的输出数量是LeNet中的好几倍。使用dropout层来减轻过度拟合
            nn.Linear(14592, 4096),
            nn.Sigmoid(),
            nn.Dropout(p=0.5),

            nn.Linear(4096, 4096),
            nn.Sigmoid(),
            nn.Dropout(p=0.5),

            # 最后是输出层
            nn.Linear(4096, num_outputs))

    def VggNet(self, dim_inputs, num_outputs, conv_arch):
        """
        思想：构建卷积块，每块包含指定数量的卷积层和最后的最大池化层
        VggNet-11由8个卷积层和3个全连接层组成。
        conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
        """
        conv_blks = []
        in_channels = dim_inputs

        for (num_convs, out_channels) in conv_arch:
            # 获取每一个卷积块的卷积层数量num_convs和最终输出通道数out_channels
            layers = []  # 存储每块所需的计算层，每次块开始构建时清零
            for _ in range(num_convs):
                # 构建卷积块，每个块含有num_convs个卷积激活层，和最后一个最大池化层
                layers.append(nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1))  # 添加卷积层
                layers.append(nn.ReLU())  # 添加激活层
                in_channels = out_channels  # 本层输出通道数作为构建的下一层的输入通道数
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            conv_blks.append(
                nn.Sequential(*layers)
            )
            in_channels = out_channels  # 本块输出通道数作为构建的下一个卷积块的输入通道数

        return nn.Sequential(
            *conv_blks,  # *在输入变量前面表示解包，把列表里面的东西解包出来
            nn.Flatten(),
            # 全连接层部分
            nn.Linear(out_channels * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, num_outputs))

    def NiNNet(self, dim_inputs, num_outputs):
        """
        使用1X1卷积层替换全连接层，参数减小但是训练时间增加了。
        """
        return nn.Sequential(

            nn.Conv2d(dim_inputs, 96, kernel_size=11, strides=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(96, 96, kernel_size=1),  # 共享参数，用kernel_size为1的卷积层充当全连接层
            nn.ReLU(),
            nn.Conv2d(96, 96, kernel_size=1),
            nn.ReLU(),

            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(96, 256, kernel_size=11, strides=4, padding=0),
            nn.ReLU(),
            # 共享参数，用kernel_size为1的卷积层充当全连接层
            nn.Conv2d(256, 256, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.ReLU(),

            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(256, 384, kernel_size=11, strides=4, padding=0),
            nn.ReLU(),
            # 共享参数，用kernel_size为1的卷积层充当全连接层
            nn.Conv2d(384, 384, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=1),
            nn.ReLU(),

            nn.MaxPool2d(3, stride=2),

            nn.Dropout(0.5),
            # 标签类别数是10

            nn.Conv2d(384, num_outputs, kernel_size=11, strides=4, padding=0),
            nn.ReLU(),
            # 共享参数，用kernel_size为1的卷积层充当全连接层
            nn.Conv2d(num_outputs, num_outputs, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(num_outputs, num_outputs, kernel_size=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),
            # 将四维的输出转成二维的输出，其形状为(批量大小, 10)
            nn.Flatten())

    def GoogleNet(self, dim_inputs, num_outputs):
        """ 并行使用多个不同核大小的卷积层
        不同大小的滤波器可以有效地识别不同范围的图像细节。
        """
        return nn.Sequential(
            # 第一个模块使用 64 个通道、  7×7  卷积层。
            nn.Conv2d(in_channels=dim_inputs, out_channels=64,
                      kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # 第二个模块使用两个卷积层：64个通道、  1×1  卷积层；通道数量增加三倍的  3×3  卷积层。
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=192,
                      kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=3, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # 第三个模块串联两个完整的Inception块。
            # 第一个 Inception 块的输出通道数为  64+128+32+32=256 ，四个路径之间的输出通道数量比为  64:128:32:32=2:4:1:1 。第二个和第三个路径首先将输入通道的数量分别减少到  96/192=1/2  和  16/192=1/12 ，然后连接第二个卷积层。
            Inception(192, 64, (96, 128), (16, 32), 32),
            # 第二个 Inception 块的输出通道数增加到  128+192+96+64=480 ，四个路径之间的输出通道数量比为  128:192:96:64=4:6:3:2 。 第二条和第三条路径首先将输入通道的数量分别减少到  128/256=1/2  和  32/256=1/8 。
            Inception(256, 128, (128, 192), (32, 96), 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # 第四模块
            Inception(480, 192, (96, 208), (16, 48), 64),
            Inception(512, 160, (112, 224), (24, 64), 64),
            Inception(512, 128, (128, 256), (24, 64), 64),
            Inception(512, 112, (144, 288), (32, 64), 64),
            Inception(528, 256, (160, 320), (32, 128), 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # 第五模块
            Inception(832, 256, (160, 320), (32, 128), 128),
            Inception(832, 384, (192, 384), (48, 128), 128),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            # 最终的全连接层
            nn.Linear(1024, num_outputs)
        )

    def ResNet(self, dim_inputs=1, num_outputs=10):
        """
        残差思想 F=X+f(X)
        """
        return nn.Sequential(
            nn.Conv2d(dim_inputs, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            Residual(64, 64, use_1x1conv=False, strides=1),
            Residual(64, 64, use_1x1conv=False, strides=1),

            Residual(64, 128, use_1x1conv=True, strides=2),
            Residual(128, 128),

            Residual(128, 256, use_1x1conv=True, strides=2),
            Residual(256, 256),

            Residual(256, 512, use_1x1conv=True, strides=2),
            Residual(512, 512),

            nn.AdaptiveAvgPool2d((1, 1)),  # 全局平均汇聚层
            nn.Flatten(),
            nn.Linear(512, num_outputs)  # 全连接层
        )

    def DenseNet(self, dim_inputs, num_outputs, growth_rate=64, num_convs_in_dense_blocks=[4, 4, 4, 4]):
        """多项展开残差思想 [x,f1(x),f2([x,f1(x)]),f3([x,f1(x),f2([x,f1(x)])]),…].
        para growth_rate: 通道数增长速率
        para num_convs_in_dense_blocks: # 定义稠密块个数及其内部含有的卷积块个数 4 * 32 +64 
            四层稠密块，每个稠密块含有四个卷积块 4 * 32 +64 
        """
        num_channels = 64  # 此处为blks的起始输入维度
        blks = []
        for i, num_convs in enumerate(num_convs_in_dense_blocks):
            blks.append(DenseBlock(num_convs, num_channels, growth_rate))
            num_channels += num_convs * growth_rate  # 上一个稠密块的输出通道数
            # 在稠密块之间添加一个转换层，使通道数量减半
            if i != len(num_convs_in_dense_blocks) - 1:  # 如果不是最后一层
                blks.append(
                    nn.Sequential(
                        # 对输入Batch内数据的每个channel其中所有特征数据进行归一化。
                        nn.BatchNorm2d(num_channels),
                        nn.ReLU(),
                        nn.Conv2d(num_channels, num_channels //
                                  2, kernel_size=1),
                        nn.AvgPool2d(kernel_size=2, stride=2)))
                num_channels = num_channels // 2

        return nn.Sequential(

            nn.Conv2d(dim_inputs, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            *blks,  # 输入维度64，输出维度为迭代后得到的num_channels

            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((1, 1)),

            nn.Flatten(),
            nn.Linear(num_channels, num_outputs))
