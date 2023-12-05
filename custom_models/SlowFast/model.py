from . import components
import torch.nn as nn
import torch

# 模型深度, 类似于ResNet
_MODEL_STAGE_DEPTH = {50: (3, 4, 6, 3), 101: (3, 4, 23, 3)}

# kernel sizes for each of the stage.
_TKERNEL_ = {
    "slow": [
        [[1]],  # conv1 kernel.
        [[1]],  # res2 kernel.
        [[1]],  # res3 kernel.
        [[3]],  # res4 tkernel.
        [[3]],  # res5 kernel.
    ],
    "fast": [
        [[5]],  # conv1 kernel.
        [[3]],  # res2 kernel.
        [[3]],  # res3 kernel.
        [[3]],  # res4 kernel.
        [[3]],  # res5 kernel.
    ],
    "slowfast": [
        [[1], [5]],  # conv1  kernel for slow and fast pathway.
        [[1], [3]],  # res2 kernel for slow and fast pathway.
        [[1], [3]],  # res3 kernel for slow and fast pathway.
        [[3], [3]],  # res4 kernel for slow and fast pathway.
        [[3], [3]],  # res5 kernel for slow and fast pathway.
    ],
}

_POOL1 = {
    "slow": [[1, 1]],
    "fast": [[1, 1]],
    "slowfast": [[1, 1], [1, 1]],
}


class SlowFast(nn.Module):
    """
    SlowFast model builder for SlowFast network.
    page: https://arxiv.org/pdf/1812.03982.pdf
    """

    def __init__(self, modelarch, stagelayers, pathway):
        """ """
        super(SlowFast, self).__init__()
        self.norm_module = components.get_norm(norm_type="batchnorm")

        # 确认当前模型使用路径
        if modelarch == "slowfast":
            self.num_pathways = 2
        elif modelarch in ["slow", "fast"]:
            self.num_pathways = 1

        self._construct_network(modelarch, stagelayers)

        # 设置FC层初始化的预期标准偏差参数。
        FC_INIT_STD = 0.01
        #  如果为 true, 初始化每一个block里面最后一层BN层的 gamma 参数为0.
        ZERO_INIT_FINAL_BN = False
        components.init_weights(self, FC_INIT_STD, ZERO_INIT_FINAL_BN)

    def _construct_network(self, modelarch, stagelayers):
        """
        Builds a SlowFast model.
        双路径时,
            0: Slow pathway;
            1: Fast pathway.
        """
        # 池化层核大小
        assert modelarch in _POOL1
        pool_size = _POOL1[modelarch]  #

        # 判断池化层核维度是否与网络路径数量是否一致
        assert len({len(pool_size), self.num_pathways}) == 1

        # 获取对应网络层层数
        (d2, d3, d4, d5) = (i for i in stagelayers)

        num_groups = 1  # 分组卷积组数 1为不分组, 即正常的 ResNet; 大于1为ResNeXt
        width_per_group = 64  # 每组的宽度, ResNet不分组则为原大小
        dim_inner = num_groups * width_per_group  # 计算输入总维度

        # 对应于Slow路径和Fast路径之间的通道缩减比的倒数。
        BETA_INV = 8
        # Slow路径和Fast路径之间的通道尺寸之比。
        FUSION_CONV_CHANNEL_RATIO = 2

        out_dim_ratio = BETA_INV // FUSION_CONV_CHANNEL_RATIO  # 计算得到输出特征图维度比  # = 4
        # 获取时域卷积核大小, 对应[conv1,res2,res3,...]
        kernel_size = _TKERNEL_[modelarch]

        # 输入谱图通道维度
        INPUT_CHANNEL_NUM = [1, 1]
        # 用于融合Fast路径和Slow路径信息的卷积核维度。
        FUSION_KERNEL_SZ = 5
        # 对应于慢速路径和快速路径之间的帧速率降低比率 $\Alpha$。
        ALPHA = 8

        # 不同的 res stages 的步长大小
        FREQUENCY_STRIDES = [[1], [2], [2], [2]]
        # 每一个block中卷积块的数量
        NUM_BLOCK_TEMP_KERNEL = [[3], [4], [6], [3]]
        # Transformation function.
        TRANS_FUNC = "bottleneck_transform"
        # Size of dilation on different res stages.
        FREQUENCY_DILATIONS = [[1], [1], [1], [1]]

        # 最终分类输出数量
        NUM_CLASSES = 4
        # 输入谱图的时间帧数
        NUM_FRAMES = 256
        # 输入频谱图的特征数
        NUM_FREQUENCIES = 128
        # 在最终投射映射成分类之前的中间DROPOUT比率。
        DROPOUT_RATE = 0.5
        # 输出的激活层。
        HEAD_ACT = "softmax"

        # 建立多路前端语音特征提取模型
        self.s1 = components.AudioModelStem(
            dim_in=INPUT_CHANNEL_NUM,  # [1,1]
            dim_out=[width_per_group, width_per_group // BETA_INV],  # [64,8]
            kernel=[kernel_size[0][0], 7],  # [[1,7], [5,7]]
            stride=[[2, 2], [2, 2]] * 2,
            padding=[  # padding设置为卷积核//2
                [kernel_size[0][0][0] // 2, 3],  # [[0, 3], [0, 3]]
                [kernel_size[0][1][0] // 2, 3],
            ],
            norm_module=self.norm_module,
        )

        # 将Fast特征融合进Flow路径中的计算模型
        # # ??? 卷积核大小可以受控制. 步长ALPHA可以受控制
        self.s1_fuse = components.FuseFastToSlow(
            width_per_group // BETA_INV,  # 8
            FUSION_CONV_CHANNEL_RATIO,  # 2
            FUSION_KERNEL_SZ,  # 5
            ALPHA,
            norm_module=self.norm_module,
        )

        self.s2 = components.ResStage(
            dim_in=[
                width_per_group + width_per_group // out_dim_ratio,
                width_per_group // BETA_INV,
            ],
            dim_out=[
                width_per_group * 4,
                width_per_group * 4 // BETA_INV,
            ],
            dim_inner=[dim_inner, dim_inner // BETA_INV],
            temp_kernel_sizes=kernel_size[1],
            stride=FREQUENCY_STRIDES[0],
            num_blocks=[d2] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=NUM_BLOCK_TEMP_KERNEL[0],
            trans_func_name=TRANS_FUNC,
            dilation=FREQUENCY_DILATIONS[0],
            norm_module=self.norm_module,
        )
        self.s2_fuse = components.FuseFastToSlow(
            width_per_group * 4 // BETA_INV,
            FUSION_CONV_CHANNEL_RATIO,
            FUSION_KERNEL_SZ,
            ALPHA,
            norm_module=self.norm_module,
        )

        for pathway in range(self.num_pathways):
            pool = nn.MaxPool2d(
                kernel_size=pool_size[pathway],
                stride=pool_size[pathway],
                padding=[0, 0],
            )
            self.add_module("pathway{}_pool".format(pathway), pool)

        self.s3 = components.ResStage(
            dim_in=[
                width_per_group * 4 + width_per_group * 4 // out_dim_ratio,
                width_per_group * 4 // BETA_INV,
            ],
            dim_out=[
                width_per_group * 8,
                width_per_group * 8 // BETA_INV,
            ],
            dim_inner=[dim_inner * 2, dim_inner * 2 // BETA_INV],
            temp_kernel_sizes=kernel_size[2],
            stride=FREQUENCY_STRIDES[1],
            num_blocks=[d3] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=NUM_BLOCK_TEMP_KERNEL[1],
            trans_func_name=TRANS_FUNC,
            dilation=FREQUENCY_DILATIONS[1],
            norm_module=self.norm_module,
        )
        self.s3_fuse = components.FuseFastToSlow(
            width_per_group * 8 // BETA_INV,
            FUSION_CONV_CHANNEL_RATIO,
            FUSION_KERNEL_SZ,
            ALPHA,
            norm_module=self.norm_module,
        )

        self.s4 = components.ResStage(
            dim_in=[
                width_per_group * 8 + width_per_group * 8 // out_dim_ratio,
                width_per_group * 8 // BETA_INV,
            ],
            dim_out=[
                width_per_group * 16,
                width_per_group * 16 // BETA_INV,
            ],
            dim_inner=[dim_inner * 4, dim_inner * 4 // BETA_INV],
            temp_kernel_sizes=kernel_size[3],
            stride=FREQUENCY_STRIDES[2],
            num_blocks=[d4] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=NUM_BLOCK_TEMP_KERNEL[2],
            trans_func_name=TRANS_FUNC,
            dilation=FREQUENCY_DILATIONS[2],
            norm_module=self.norm_module,
        )
        self.s4_fuse = components.FuseFastToSlow(
            width_per_group * 16 // BETA_INV,
            FUSION_CONV_CHANNEL_RATIO,
            FUSION_KERNEL_SZ,
            ALPHA,
            norm_module=self.norm_module,
        )

        self.s5 = components.ResStage(
            dim_in=[
                width_per_group * 16 + width_per_group * 16 // out_dim_ratio,
                width_per_group * 16 // BETA_INV,
            ],
            dim_out=[
                width_per_group * 32,
                width_per_group * 32 // BETA_INV,
            ],
            dim_inner=[dim_inner * 8, dim_inner * 8 // BETA_INV],
            temp_kernel_sizes=kernel_size[4],
            stride=FREQUENCY_STRIDES[3],
            num_blocks=[d5] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=NUM_BLOCK_TEMP_KERNEL[3],
            trans_func_name=TRANS_FUNC,
            dilation=FREQUENCY_DILATIONS[3],
            norm_module=self.norm_module,
        )

        self.head = components.ResNetBasicHead(
            dim_in=[
                width_per_group * 32,
                width_per_group * 32 // BETA_INV,
            ],
            num_classes=NUM_CLASSES if len(NUM_CLASSES) > 1 else NUM_CLASSES[0],
            pool_size=[
                [
                    NUM_FRAMES // ALPHA // 4 // pool_size[0][0],
                    NUM_FREQUENCIES // 32 // pool_size[0][1],
                ],
                [
                    NUM_FRAMES // 4 // pool_size[1][0],
                    NUM_FREQUENCIES // 32 // pool_size[1][1],
                ],
            ],
            dropout_rate=DROPOUT_RATE,
            act_func=HEAD_ACT,
        )

    def forward(self, x, bboxes=None):
        x = self.s1(x)
        x = self.s1_fuse(x)
        x = self.s2(x)
        x = self.s2_fuse(x)
        for pathway in range(self.num_pathways):
            pool = getattr(self, "pathway{}_pool".format(pathway))
            x[pathway] = pool(x[pathway])
        x = self.s3(x)
        x = self.s3_fuse(x)
        x = self.s4(x)
        x = self.s4_fuse(x)
        x = self.s5(x)
        x = self.head(x)
        return x

    def freeze_fn(self, freeze_mode):
        if freeze_mode == "bn_parameters":
            print("Freezing all BN layers' parameters except the first one.")
            for n, m in self.named_modules():
                if (
                    isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm)
                ) and (
                    "s1.pathway0_stem.bn" not in n
                    and "s1.pathway1_stem.bn" not in n
                    and "s1_fuse.bn" not in n
                ):
                    # shutdown parameters update in frozen mode
                    m.weight.requires_grad_(False)
                    m.bias.requires_grad_(False)
        elif freeze_mode == "bn_statistics":
            print("Freezing all BN layers' statistics except the first one.")
            for n, m in self.named_modules():
                if (
                    isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm)
                ) and (
                    "s1.pathway0_stem.bn" not in n
                    and "s1.pathway1_stem.bn" not in n
                    and "s1_fuse.bn" not in n
                ):
                    # shutdown running statistics update in frozen mode
                    m.eval()


def slowfast50(pretrain=False) -> SlowFast:
    model = SlowFast("slowfast", [3, 4, 6, 3])
    #
    if pretrain:
        # load_state_dict_from_url
        URL = "https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/SLOWFAST_4x16_R50.pkl"
        state_dict = torch.hub.load_state_dict_from_url(URL, map_location="cpu")
        model.load_state_dict(state_dict["model"])


def slowfast101() -> SlowFast:
    return SlowFast(
        "slowfast",
        [3, 4, 23, 3],
    )
