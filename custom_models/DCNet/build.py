# --------------------------------------------------------
# InternImage
# Copyright (c) 2022 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from .intern_image import InternImage


def build_model(model_type: str='intern_image'):
    if model_type == 'intern_image':
        model = InternImage(
                 core_op='DCNv3',
                 channels=80,
                 depths=[4, 4, 21, 4],
                 groups=[5, 10, 20, 40],
                 num_classes=4,
                 mlp_ratio=4.,
                 drop_rate=0.4,
                 drop_path_rate=0.2,
                 drop_path_type='linear',
                 act_layer='GELU',
                 norm_layer='LN',
                 layer_scale=1e-5,
                 offset_scale=1.0,
                 post_norm=True,
                 cls_scale=1.5,
                 with_cp=False,
                 dw_kernel_size=None, # for InternImage-H/G
                 use_clip_projector=False, # for InternImage-H/G
                 level2_post_norm=False, # for InternImage-H/G
                 level2_post_norm_block_ids=None, # for InternImage-H/G
                 res_post_norm=False, # for InternImage-H/G
                 center_feature_scale=False, # for InternImage-H/G
                 remove_center=False, # for InternImage-H/G
        )
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model


if __name__ == '__main__':
    import torch
    model = build_model('intern_image')
    print(model)
    input = torch.randn(1, 3, 224, 224)
    output = model(input)
    print(output.shape)