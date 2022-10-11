from typing import List
from . import componentscopy as components
import torch.nn as nn

audiomodelstem_shape = [
    [
        [64, {"kernel_size": [1, 7], "stride":[2, 2], "padding":[0, 3]},
         "bn", {"num_features": 64},
         "mp", {"kernel_size": [3, 3], "stride":[2, 2], "padding":[1, 1]},
         "relu", {}],
    ],
    [
        [8, {"kernel_size": [5, 7], "stride":[2, 2], "padding":[2, 3]},
         "bn", {"num_features": 8},
         "mp", {"kernel_size": [3, 3], "stride":[2, 2], "padding":[1, 1]},
         "relu", {}],
    ]
]

num_blocks_50 = [
    [3, 4, 6, 3],
    [3, 4, 6, 3]
]

num_blocks_101 = [
    [3, 4, 23, 3],
    [3, 4, 23, 3]
]

pool_size = [
    [1, 1],
    [1, 1]
]
bottlenecktransform_shape = [
    [
        [
            [64, {"kernel_size": [1, 1], "stride":[1, 1], "padding":[0, 0]},
             "bn", {"num_features": 64},
             "", {},
             "relu", {}],
            [64, {"kernel_size": [1, 3], "stride":[1, 2], "padding":[0, 1], "dilation":[1, 1]},
             "bn", {"num_features": 64},
             "", {},
             "relu", {}],
            [256, {"kernel_size": [1, 1], "stride":[1, 1], "padding":[0, 0]},
             "bn", {"num_features": 256},
             "", {},
             "", {}],
        ],
        [
            [128, {"kernel_size": [1, 1], "stride":[1, 1], "padding":[0, 0]},
             "bn", {"num_features": 128},
             "", {},
             "relu", {}],
            [128, {"kernel_size": [1, 3], "stride":[1, 2], "padding":[0, 1], "dilation":[1, 1]},
             "bn", {"num_features": 128},
             "", {},
             "relu", {}],
            [512, {"kernel_size": [1, 1], "stride":[1, 1], "padding":[0, 0]},
             "bn", {"num_features": 512},
             "", {},
             "", {}],
        ],
        [
            [256, {"kernel_size": [3, 1], "stride":[1, 1], "padding":[1, 0]},
             "bn", {"num_features": 256},
             "", {},
             "relu", {}],
            [256, {"kernel_size": [1, 3], "stride":[1, 2], "padding":[0, 1], "dilation":[1, 1]},
             "bn", {"num_features": 256},
             "", {},
             "relu", {}],
            [1024, {"kernel_size": [1, 1], "stride":[1, 1], "padding":[0, 0]},
             "bn", {"num_features": 1024},
             "", {},
             "", {}],
        ],
        [
            [512, {"kernel_size": [3, 1], "stride":[1, 1], "padding":[1, 0]},
             "bn", {"num_features": 512},
             "", {},
             "relu", {}],
            [512, {"kernel_size": [1, 3], "stride":[1, 2], "padding":[0, 1], "dilation":[1, 1]},
             "bn", {"num_features": 512},
             "", {},
             "relu", {}],
            [2048, {"kernel_size": [1, 1], "stride":[1, 1], "padding":[0, 0]},
             "bn", {"num_features": 2048},
             "", {},
             "", {}],
        ],
    ],
    [
        [
            [8, {"kernel_size": [3, 1], "stride":[1, 1], "padding":[1, 0]},
             "bn", {"num_features": 8},
             "", {},
             "relu", {}],
            [8, {"kernel_size": [1, 3], "stride":[1, 2], "padding":[0, 1], "dilation":[1, 1]},
             "bn", {"num_features": 8},
             "", {},
             "relu", {}],
            [32, {"kernel_size": [1, 1], "stride":[1, 1], "padding":[0, 0]},
             "bn", {"num_features": 32},
             "", {},
             "", {}],
        ],
        [
            [16, {"kernel_size": [3, 1], "stride":[1, 1], "padding":[1, 0]},
             "bn", {"num_features": 16},
             "", {},
             "relu", {}],
            [16, {"kernel_size": [1, 3], "stride":[1, 2], "padding":[0, 1], "dilation":[1, 1]},
             "bn", {"num_features": 16},
             "", {},
             "relu", {}],
            [64, {"kernel_size": [1, 1], "stride":[1, 1], "padding":[0, 0]},
             "bn", {"num_features": 64},
             "", {},
             "", {}],
        ],
        [
            [32, {"kernel_size": [3, 1], "stride":[1, 1], "padding":[1, 0]},
             "bn", {"num_features": 32},
             "", {},
             "relu", {}],
            [32, {"kernel_size": [1, 3], "stride":[1, 2], "padding":[0, 1], "dilation":[1, 1]},
             "bn", {"num_features": 32},
             "", {},
             "relu", {}],
            [128, {"kernel_size": [1, 1], "stride":[1, 1], "padding":[0, 0]},
             "bn", {"num_features": 128},
             "", {},
             "", {}],
        ],
        [
            [64, {"kernel_size": [3, 1], "stride":[1, 1], "padding":[1, 0]},
             "bn", {"num_features": 64},
             "", {},
             "relu", {}],
            [64, {"kernel_size": [1, 3], "stride":[1, 2], "padding":[0, 1], "dilation":[1, 1]},
             "bn", {"num_features": 64},
             "", {},
             "relu", {}],
            [256, {"kernel_size": [1, 1], "stride":[1, 1], "padding":[0, 0]},
             "bn", {"num_features": 256},
             "", {},
             "", {}],
        ]
    ]
]

basictransform_shape = [
    [
        [
            [64, {"kernel_size": [1, 1], "stride":[1, 1]},
             "bn", {"num_features": 64},
             "", {},
             "relu", {}],
            [64, {"kernel_size": [1, 3], "stride":[1, 1]},
             "bn", {"num_features": 64},
             "", {},
             "", {}]
        ],
        [
            [128, {"kernel_size": [1, 1], "stride":[1, 1]},
             "bn", {"num_features": 128},
             "", {},
             "relu", {}],
            [128, {"kernel_size": [1, 3], "stride":[1, 1]},
             "bn", {"num_features": 128},
             "", {},
             "", {}],
        ],
        [
            [256, {"kernel_size": [3, 1], "stride":[1, 1]},
             "bn", {"num_features": 256},
             "", {},
             "relu", {}],
            [256, {"kernel_size": [1, 3], "stride":[1, 1]},
             "bn", {"num_features": 256},
             "", {},
             "", {}]
        ],
        [
            [512, {"kernel_size": [3, 1], "stride":[1, 1]},
             "bn", {"num_features": 512},
             "", {},
             "relu", {}],
            [512, {"kernel_size": [1, 3], "stride":[1, 1]},
             "bn", {"num_features": 512},
             "", {},
             "", {}]
        ],
    ],
    [
        [
            [8, {"kernel_size": [3, 1], "stride":[1, 1]},
             "", {},
             "", {},
             "relu", {}],
            [8, {"kernel_size": [1, 3], "stride":[1, 1]},
             "", {},
             "", {},
             "", {}]
        ],
        [
            [16, {"kernel_size": [3, 1], "stride":[1, 1]},
             "", {},
             "", {},
             "relu", {}],
            [16, {"kernel_size": [1, 3], "stride":[1, 1]},
             "", {},
             "", {},
             "", {}]
        ],
        [
            [32, {"kernel_size": [3, 1], "stride":[1, 1]},
             "", {},
             "", {},
             "relu", {}],
            [32, {"kernel_size": [1, 3], "stride":[1, 1]},
             "", {},
             "", {},
             "", {}]
        ],
        [
            [64, {"kernel_size": [3, 1], "stride":[1, 1]},
             "", {},
             "", {},
             "relu", {}],
            [64, {"kernel_size": [1, 3], "stride":[1, 1]},
             "", {},
             "", {},
             "", {}]
        ]
    ]
]


class SlowFast(nn.Module):
    def __init__(self, num_classes, training, stem_shapes, num_blocks, stage_shapes):
        """
        """
        super(SlowFast, self).__init__()
        # 确认当前模型使用路径
        assert (len(stem_shapes) == len(stage_shapes) ==
                len(num_blocks)), "SlowFast输入参数的 Stream 数量不统一"
        self.num_pathways = len(stem_shapes)
        self._construct_network(num_classes, training,
                                stem_shapes, num_blocks, stage_shapes)

    def _construct_network(self, num_classes, training, stem_shapes, num_blocks, stage_shapes):
        "Builds a SlowFast model. "
        # 建立多路前端语音特征提取模型
        # 获取对应网络层层数
        self.s1 = components.MultiStream_Conv2d_extractor(
            in_channels=[1, 1],
            shapes=stem_shapes)
        self.s1_fuse = components.FuseS1ToS0(
            dim_in=stem_shapes[1][-1][0],  # 8
            dim_out=stem_shapes[0][-1][0],  # 64
            kernel=[7, 1],
            stride=[4, 1],  # 时序初始下采样四倍
        )

        stage2_shape = [stage_shapes[0][0], stage_shapes[1][0]]
        self.s2 = components.MultiStream_ResStage(
            num_blocks=[num_blocks[0][0], num_blocks[1][0]],
            in_channels=[stem_shapes[0][-1][0]*2, stem_shapes[1][-1][0]],
            shapes=stage2_shape)
        self.s2_fuse = components.FuseS1ToS0(
            dim_in=stage2_shape[1][-1][0],
            dim_out=stage2_shape[0][-1][0],
            kernel=[7, 1],
            stride=[4, 1],
        )

        for pathway in range(self.num_pathways):
            pool = nn.MaxPool2d(
                kernel_size=pool_size[pathway],
                stride=pool_size[pathway],
                padding=[0, 0],
            )
            self.add_module("pathway{}_pool".format(pathway), pool)

        stage3_shape = [stage_shapes[0][1], stage_shapes[1][1]]
        self.s3 = components.MultiStream_ResStage(
            num_blocks=[num_blocks[0][1], num_blocks[1][1]],
            in_channels=[stage2_shape[0][-1][0]*2, stage2_shape[1][-1][0]],
            shapes=stage3_shape)
        self.s3_fuse = components.FuseS1ToS0(
            dim_in=stage3_shape[1][-1][0],
            dim_out=stage3_shape[0][-1][0],
            kernel=[7, 1],
            stride=[4, 1],
        )
        stage4_shape = [stage_shapes[0][2], stage_shapes[1][2]]
        self.s4 = components.MultiStream_ResStage(
            num_blocks=[num_blocks[0][2], num_blocks[1][2]],
            in_channels=[stage3_shape[0][-1][0]*2, stage3_shape[1][-1][0]],
            shapes=stage4_shape)
        self.s4_fuse = components.FuseS1ToS0(
            dim_in=stage4_shape[1][-1][0],
            dim_out=stage4_shape[0][-1][0],
            kernel=[7, 1],
            stride=[4, 1],
        )
        stage5_shape = [stage_shapes[0][3], stage_shapes[1][3]]
        self.s5 = components.MultiStream_ResStage(
            num_blocks=[num_blocks[0][3], num_blocks[1][3]],
            in_channels=[stage4_shape[0][-1][0]*2, stage4_shape[1][-1][0]],
            shapes=stage5_shape)
        self.head = components.BasicHead(
            dim_in=[stage5_shape[0][-1][0], stage5_shape[1][-1][0]],
            num_classes=num_classes,
            act_func="softmax",
            dropout_rate=0.1,
            training=training)

    def forward(self, x):
        if len(x.size()) == 3:
            x = x.unsqueeze(1)  # x
        if not isinstance(x, List) or len(x) == 1:
            x = [nn.AvgPool2d(kernel_size=[4, 1], stride=[4, 1])(x), x]

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


def slowfast50() -> SlowFast:
    return SlowFast(num_classes=4, training=True, stem_shapes=audiomodelstem_shape, num_blocks=num_blocks_50, stage_shapes=bottlenecktransform_shape)


def slowfast101() -> SlowFast:
    return SlowFast(num_classes=4, training=True, stem_shapes=audiomodelstem_shape, num_blocks=num_blocks_101, stage_shapes=bottlenecktransform_shape)
