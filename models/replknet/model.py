
from . import components
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint


class RepLKNet(nn.Module):

    def __init__(self, large_kernel_sizes, layers, channels, drop_path_rate, small_kernel,
                 dw_ratio=1, ffn_ratio=4, in_channels=1, num_classes=4, out_indices=None,
                 use_checkpoint=False,
                 small_kernel_merged=False,
                 use_sync_bn=False,
                 # for RepLKNet-XL on COCO and ADE20K, use an extra BN to normalize the intermediate feature maps then feed them into the heads
                 norm_intermediate_features=False
                 ):
        super().__init__()

        if num_classes is None and out_indices is None:
            raise ValueError(
                'must specify one of num_classes (for pretraining) and out_indices (for downstream tasks)')
        elif num_classes is not None and out_indices is not None:
            raise ValueError(
                'cannot specify both num_classes (for pretraining) and out_indices (for downstream tasks)')
        elif num_classes is not None and norm_intermediate_features:
            raise ValueError(
                'for pretraining, no need to normalize the intermediate feature maps')
        self.out_indices = out_indices
        if use_sync_bn:
            components.enable_sync_bn()

        base_width = channels[0]
        self.use_checkpoint = use_checkpoint
        self.norm_intermediate_features = norm_intermediate_features
        self.num_stages = len(layers)

        self.stem = nn.ModuleList([
            components.conv_bn_relu(in_channels=in_channels, out_channels=base_width,
                                    kernel_size=3, stride=2, padding=1, groups=1),
            components.conv_bn_relu(in_channels=base_width, out_channels=base_width,
                                    kernel_size=3, stride=1, padding=1, groups=base_width),
            components.conv_bn_relu(in_channels=base_width, out_channels=base_width,
                                    kernel_size=1, stride=1, padding=0, groups=1),
            components.conv_bn_relu(in_channels=base_width, out_channels=base_width, kernel_size=3, stride=2, padding=1, groups=base_width)])
        # stochastic depth. We set block-wise drop-path rate. The higher level blocks are more likely to be dropped. This implementation follows Swin.

        dpr = [x.item() for x in torch.linspace(
            0, drop_path_rate, sum(layers))]
        self.stages = nn.ModuleList()
        self.transitions = nn.ModuleList()

        for stage_idx in range(self.num_stages):
            layer = components.RepLKNetStage(channels=channels[stage_idx], num_blocks=layers[stage_idx],
                                             stage_lk_size=large_kernel_sizes[stage_idx],
                                             drop_path=dpr[sum(layers[:stage_idx]):sum(
                                                 layers[:stage_idx + 1])],
                                             small_kernel=small_kernel, dw_ratio=dw_ratio, ffn_ratio=ffn_ratio,
                                             use_checkpoint=use_checkpoint, small_kernel_merged=small_kernel_merged,
                                             norm_intermediate_features=norm_intermediate_features)
            self.stages.append(layer)
            if stage_idx < len(layers) - 1:
                transition = nn.Sequential(
                    components.conv_bn_relu(
                        channels[stage_idx], channels[stage_idx + 1], 1, 1, 0, groups=1),
                    components.conv_bn_relu(channels[stage_idx + 1], channels[stage_idx + 1], 3, stride=2, padding=1, groups=channels[stage_idx + 1]))
                self.transitions.append(transition)

        if num_classes is not None:
            self.norm = components.get_bn(channels[-1])
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.head = nn.Linear(channels[-1], num_classes)

    def forward_features(self, x):
        x = self.stem[0](x)
        for stem_layer in self.stem[1:]:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(stem_layer, x)     # save memory
            else:
                x = stem_layer(x)

        if self.out_indices is None:
            #   Just need the final output
            for stage_idx in range(self.num_stages):
                x = self.stages[stage_idx](x)
                if stage_idx < self.num_stages - 1:
                    x = self.transitions[stage_idx](x)
            return x
        else:
            #   Need the intermediate feature maps
            outs = []
            for stage_idx in range(self.num_stages):
                x = self.stages[stage_idx](x)
                if stage_idx in self.out_indices:
                    # For RepLKNet-XL normalize the features before feeding them into the heads
                    outs.append(self.stages[stage_idx].norm(x))
                if stage_idx < self.num_stages - 1:
                    x = self.transitions[stage_idx](x)
            return outs

    def forward(self, x):
        if len(x.size()) == 3:
            x = x.unsqueeze(1)  # x
        x = self.forward_features(x)
        if self.out_indices:
            return x
        else:
            x = self.norm(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.head(x)
            return x

    def structural_reparam(self):
        for m in self.modules():
            if hasattr(m, 'merge_kernel'):
                m.merge_kernel()

    #   If your framework cannot automatically fuse BN for inference, you may do it manually.
    #   The BNs after and before conv layers can be removed.
    #   No need to call this if your framework support automatic BN fusion.
    def deep_fuse_BN(self):
        for m in self.modules():
            if not isinstance(m, nn.Sequential):
                continue
            if not len(m) in [2, 3]:  # Only handle conv-BN or conv-BN-relu
                continue
            #   If you use a custom Conv2d impl, assume it also has 'kernel_size' and 'weight'
            if hasattr(m[0], 'kernel_size') and hasattr(m[0], 'weight') and isinstance(m[1], nn.BatchNorm2d):
                conv = m[0]
                bn = m[1]
                fused_kernel, fused_bias = components.fuse_bn(conv, bn)
                fused_conv = components.get_conv2d(conv.in_channels, conv.out_channels, kernel_size=conv.kernel_size,
                                                   stride=conv.stride,
                                                   padding=conv.padding, dilation=conv.dilation, groups=conv.groups, bias=True)
                fused_conv.weight.data = fused_kernel
                fused_conv.bias.data = fused_bias
                m[0] = fused_conv
                m[1] = nn.Identity()


def create_RepLKNet31B(drop_path_rate=0.3, num_classes=1000, use_checkpoint=True, small_kernel_merged=False):
    return RepLKNet(large_kernel_sizes=[31, 29, 27, 13], layers=[2, 2, 18, 2], channels=[128, 256, 512, 1024],
                    drop_path_rate=drop_path_rate, small_kernel=5, num_classes=num_classes, use_checkpoint=use_checkpoint,
                    small_kernel_merged=small_kernel_merged)


def create_RepLKNet31L(drop_path_rate=0.3, num_classes=1000, use_checkpoint=True, small_kernel_merged=False):
    return RepLKNet(large_kernel_sizes=[31, 29, 27, 13], layers=[2, 2, 18, 2], channels=[192, 384, 768, 1536],
                    drop_path_rate=drop_path_rate, small_kernel=5, num_classes=num_classes, use_checkpoint=use_checkpoint,
                    small_kernel_merged=small_kernel_merged)


def create_RepLKNetXL(drop_path_rate=0.3, num_classes=1000, use_checkpoint=True, small_kernel_merged=False):
    return RepLKNet(large_kernel_sizes=[27, 27, 27, 13], layers=[2, 2, 18, 2], channels=[256, 512, 1024, 2048],
                    drop_path_rate=drop_path_rate, small_kernel=None, dw_ratio=1.5,
                    num_classes=num_classes, use_checkpoint=use_checkpoint,
                    small_kernel_merged=small_kernel_merged)


if __name__ == '__main__':
    model = create_RepLKNet31B(small_kernel_merged=False)
    model.eval()
    print('------------------- training-time model -------------')
    print(model)
    x = torch.randn(2, 3, 224, 224)
    origin_y = model(x)
    model.structural_reparam()
    print('------------------- after re-param -------------')
    print(model)
    reparam_y = model(x)
    print('------------------- the difference is ------------------------')
    print((origin_y - reparam_y).abs().sum())
