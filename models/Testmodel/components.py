from typing import Optional, Tuple, List, Type, Any, Callable, Union
import torch
from torch import Tensor, nn
from torch.nn import Module
import torch.nn.functional as F
import math
import torchvision.models.resnet as tvresnet


class Conv2dSame(torch.nn.Conv2d):

    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.size()[-2:]

        pad_h = self.calc_same_pad(
            i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0])
        pad_w = self.calc_same_pad(
            i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1])

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2,
                    pad_h // 2, pad_h - pad_h // 2]
            )
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class LayerNorm(nn.LayerNorm):
    """重写带 transpose 的LayerNorm
        normalized_shape (int or list or torch.Size): 预期输入形状
            如果是整数, 则需是输入数据的最后维度的大小, 否则应是最后多个维度的大小
        eps: 为了防止标准差为零时分母为零, 设置的极小值。Default: 1e-5
        elementwise_affine:  设置是否需要仿射变换, 如果为True, 
            添加可学习的的逐元素仿射参数, 这些参数会被初始化为1(对于 weights )和0(对于 biases )。
            Default: ``True``.
        math: 向量减去均值后除以标准差
    """

    def forward(self, input: Tensor) -> Tensor:
        x = input.transpose(-2, -1)  # 将输入的倒数两个维度调换位置, 即转置
        x = nn.functional.layer_norm(
            x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.transpose(-2, -1)  # 维度还原回来
        return x


class Conv2dFeatureExtractor(nn.Module):
    """构建连续的2D卷积网络结构主Module
    Args:
        conv_layers (nn.ModuleList): 模型所需的2D卷积网络结构列表
    """

    def __init__(self, conv_layers: nn.ModuleList,):
        super().__init__()
        self.conv_blocks = conv_layers

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor):
                shape: ``[batch,in_channels, frame, feature]``.
        Returns:
            Tensor:
                输出结果, shape: ``[batch,out_channels, frame, feature]``
        """
        # 检查参数
        if x.ndim != 4:
            raise ValueError(
                f"期望输入的维度为 4D (batch,in_channels, frame, feature), 但得到的输入数据维度为 {list(x.shape)}")
        for block in self.conv_blocks:
            x = block(x)
        return x


def _get_Conv2d_extractor(
        in_channels,
        # [channel:int, kernel:list|int, stride:list|int, norm:str|None, pool:str|None, pool_kernel:list|int]
        shapes,
        bias: bool,) -> Conv2dFeatureExtractor:

    blocks = []
    in_channels = in_channels  # 输入数据的通道

    for i, (out_channels, kernel_size, stride, norm_mode, pool_mode, pool_kernel) in enumerate(shapes):

        # Norm设置
        assert norm_mode in ["gn", "ln", "bn", "None"]
        assert pool_mode in ["mp", "ap", "gap", "None"]
        if norm_mode == "gn":
            normalization = nn.GroupNorm(
                num_groups=out_channels,
                num_channels=out_channels,
                affine=True,
            )
        elif norm_mode == "ln":
            normalization = nn.LayerNorm(
                normalized_shape=out_channels,
                elementwise_affine=True,
            )
        elif norm_mode == "bn":
            normalization = nn.BatchNorm2d(out_channels)
        else:
            normalization = None

        if pool_mode == "mp":
            poolling = nn.MaxPool2d(pool_kernel)
        elif pool_mode == "ap":
            poolling = nn.AvgPool2d(pool_kernel)
        elif pool_mode == "gap":  # GlobalAveragePooling2D
            pool_kernel = 1
            poolling = nn.AdaptiveAvgPool2d(1)
        else:
            poolling = None
        # 卷积块设置
        blocks.append(
            Conv2dLayerBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                bias=bias,
                layer_norm=normalization,
                layer_poolling=poolling,
                activate_func=nn.functional.relu
            )
        )

        in_channels = out_channels  # 修改输入通道为 Extractor 的输出通道
    return Conv2dFeatureExtractor(nn.ModuleList(blocks))


class Conv2dLayerBlock(nn.Module):
    """单个卷积块,内部含有 Conv2d_layer, Norm_layer(可选), Activation_layer(可选, 默认GELU)"""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size,
            stride: int,
            bias: bool,
            layer_norm: Optional[nn.Module],
            layer_poolling: Optional[nn.Module],
            activate_func=nn.functional.relu,):

        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

        self.conv = Conv2dSame(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,  # 步长
            bias=bias,  # 是否使用偏置数
            padding_mode='zeros'
        )
        self.layer_norm = layer_norm
        self.activate_func = activate_func
        self.leyer_poolling = layer_poolling

    def forward(
            self,
            x: Tensor,) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            x (Tensor): Shape ``[batch, in_channels, in_frames, in_features]``.
        Returns:
            Tensor: Shape ``[batch, out_channels, out_frames, in_features]``.
        """
        x = self.conv(x)
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        if self.activate_func is not None:
            x = self.activate_func(x)
        if self.leyer_poolling is not None:
            x = self.leyer_poolling(x)
        return x


class Conv1dLayerBlock(Module):
    """单个卷积块,内部含有 Conv1d_layer, Norm_layer(可选), Activation_layer(可选, 默认GELU)"""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int,
            bias: bool,
            layer_norm: Optional[Module],
            activate_func=nn.functional.gelu,):

        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.layer_norm = layer_norm
        self.conv = nn.Conv1d(  # 卷积核是一维的，卷积操作沿着输入[batch, in_channels, in_frame]第二维在第三维上进行。
            in_channels=in_channels,
            out_channels=out_channels,  # 输出通道数,即使用几个卷积核
            kernel_size=kernel_size,  # 竖着扫的范围
            stride=stride,  # 步长
            bias=bias,  # 是否使用偏置数
        )
        self.activate_func = activate_func

    def forward(
            self,
            x: Tensor,
            length: Optional[Tensor],
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            x (Tensor): Shape ``[batch, in_channels, in_frames]``.
            length (Tensor or None, optional): Shape ``[batch, ]``.
        Returns:
            Tensor: Shape ``[batch, out_channels, out_frames]``.
            Optional[Tensor]: Shape ``[batch, ]``.
        """
        x = self.conv(x)
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        if self.activate_func is not None:
            x = self.activate_func(x)

        if length is not None:
            length = torch.div(length - self.kernel_size,  # 张量和标量做逐元素除法
                               self.stride, rounding_mode='floor') + 1  # 计算输入序列真实的输出数量
            # 当输入的 length 含 0 时, 计算结果可能是负的, 将其替换成0
            length = torch.max(torch.zeros_like(length), length)
        return x, length


class ConvFeatureExtractor(Module):
    """定义 Wav2Vec 2.0 中 ConvFeatureExtractor Model 前向计算

    Args:
        conv_layers (nn.ModuleList): 模型所需的网络结构列表
    """

    def __init__(self, conv_layers: nn.ModuleList,):
        super().__init__()
        self.conv_blocks = conv_layers

    def forward(self, x: Tensor, length: Optional[Tensor],) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            x (Tensor):
                一个batch的音频Tensor, shape: ``[batch, time]``.
            length (Tensor or None, optional):
                batch中每个音频的有效长度, shape: ``[batch, ]``.

        Returns:
            Tensor:
                输出结果, shape: ``[batch, frame, feature]``
            Optional[Tensor]:
                batch中每个输出的的有效长度, shape: ``[batch, ]``.
        """
        # 检查参数
        if x.ndim != 2:
            raise ValueError(
                f"期望输入的维度为 2D (batch, time), 但得到的输入数据维度为 {list(x.shape)}")

        x = x.unsqueeze(1)  # 调整输入数据维度为(batch, channel==1, frame)
        for block in self.conv_blocks:
            x, length = block(x, length)  # (batch, feature, frame)
        x = x.transpose(1, 2)  # (batch, frame, feature)
        return x, length


class FeatureProjection(Module):
    """通过 LayerNorm, Linear, Dropout 将输入数据的特征维度映射成所需的维度

    Args:
        in_features (int): Input feature dim.
        out_features (int): Output feature dim.
        dropout (float): Dropout probability.
    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            dropout: float,
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_features)
        self.projection = nn.Linear(in_features, out_features,)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x (Tensor): shape: ``[batch, frame, in_feature]``
        Returns:
            Tensor: shape: ``[batch, frame, out_feature]``.
        """
        x = self.layer_norm(x)
        x = self.projection(x)
        x = self.dropout(x)
        return x


class ConvolutionalPositionalEmbedding(Module):
    """放置在 TransformerEncoder 前面的位置嵌入

    Args:
        embed_dim (int): 输入 Tensor 的 Feature 维度.
        kernel_size (int): The number of frames to be use.
        groups (int): 进行卷积操作时, 输入channel和输出channel进行分组, 
            默认为1, 即所有输入channel和输出channel一一进行计算.
    """

    def __init__(
            self,
            embed_dim: int,
            kernel_size: int,
            groups: int,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.conv = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=groups,  # 将输入和输出进行分组卷积
        )
        self.conv = nn.utils.weight_norm(self.conv, name="weight", dim=2)
        self.num_remove: int = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x):
        """
        Args:
            x (Tensor): shape ``[batch, frame, feature]``.

        Returns:
            Tensor: The resulting feature. Shape ``[batch, frame, feature]``.
        """
        x = x.transpose(-2, -1)  # [batch, feature, frame]
        x = self.conv(x)
        if self.num_remove > 0:
            # [batch, feature, frame-self.num_remove]
            x = x[..., :-self.num_remove]
        x = torch.nn.functional.gelu(x)
        x = x.transpose(-2, -1)  # [batch, frame-self.num_remove, feature]
        return x


class SelfAttention(Module):
    """多头注意力机制 (Multihead Self Attention module )

    Args:
        embed_dim (int): Total dimension of the model.
        num_heads (int): The number of heads.
        dropout (float, optional):
            Dropout probabiliry on attn_output_weights. Default: ``0.0``
    """

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dropout: float = 0.0,):
        super().__init__()
        head_dim = embed_dim // num_heads  # 计算每个 Head 对应的 Q, K, V 的输出特征维度
        if head_dim * num_heads != embed_dim:
            raise ValueError(
                f"`输入嵌入向量为度 embed_dim({embed_dim})`不能整除 `num_heads ({num_heads})`")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = head_dim

        self.scaling = self.head_dim ** -0.5

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(
            self,
            x: Tensor,
            attention_mask: Optional[Tensor] = None,) -> Tensor:
        """
        Args:
            x (Tensor): shape: ``[batch_size, sequence_length, embed_dim]``.
            attention_mask (Tensor or None, optional):
                shape: ``[batch_size, 1, sequence_length, sequence_length]``

        Returns:
            Tensor: The resulting tensor. shape: ``[batch, sequence_length, embed_dim]``
        """
        # 检查参数
        if x.ndim != 3 or x.shape[2] != self.embed_dim:
            raise ValueError(
                f"期望输入维度大小为 (batch, sequence, embed_dim=={self.embed_dim}), 但得到的却是 {x.shape}."
            )
        batch_size, length, embed_dim = x.size()
        if attention_mask is not None:
            shape_ = (batch_size, 1, length, length)
            if attention_mask.size() != shape_:
                raise ValueError(
                    f"期望的 attention mask 维度大小为 {shape_}, 但得到的却是 {attention_mask.size()}."
                )

        shape = (batch_size, length, self.num_heads, self.head_dim)
        # [batch_size, num_heads, sequence_length, head_dim], 这里 num_heads*head_dim =embed_dim
        q = self.q_proj(x).view(*shape).transpose(2, 1)  # B, nH, L, Hd
        # [batch_size, num_heads, head_dim, sequence_length]
        k = self.k_proj(x).view(*shape).permute(0, 2, 3, 1)  # B, nH, Hd, L
        # [batch_size, num_heads, sequence_length, head_dim]
        v = self.v_proj(x).view(*shape).transpose(2, 1)  # B, nH, L, Hd

        # 求平均, 进行特征缩放
        weights = self.scaling * (q @ k)  # B, nH, L, L , 注意 @ = torch.matmul
        if attention_mask is not None:
            weights += attention_mask

        weights = torch.nn.functional.softmax(weights, dim=-1)
        weights = torch.nn.functional.dropout(
            weights, p=self.dropout, training=self.training)
        # [batch_size, num_heads, sequence_length, head_dim]
        output = weights @ v  # B, nH, L, Hd
        # [batch_size, sequence_length, embed_dim]
        output = output.transpose(2, 1).reshape(batch_size, length, embed_dim)

        output = self.out_proj(output)
        return output


class FeedForward(Module):
    """Muti-attention Layer 后面的 FeedForward 层, 由两层 Linear、Dropout 和 GELU 组成
    """

    def __init__(
            self,
            io_features: int,
            intermediate_features: int,
            intermediate_dropout: float,
            output_dropout: float,):
        super().__init__()
        self.intermediate_dense = nn.Linear(io_features, intermediate_features)
        self.intermediate_dropout = nn.Dropout(intermediate_dropout)
        self.output_dense = nn.Linear(intermediate_features, io_features)
        self.output_dropout = nn.Dropout(output_dropout)

    def forward(self, x):
        """
        Args:
            x (Tensor): shape: `(batch, sequence_length, io_features)`
        Returns:
            x (Tensor): shape: `(batch, sequence_length, io_features)`
        """
        x = self.intermediate_dense(x)
        x = torch.nn.functional.gelu(x)
        x = self.intermediate_dropout(x)

        x = self.output_dense(x)
        x = self.output_dropout(x)
        return x


class EncoderLayer(Module):
    """
    TransformerEncoder 中的 Encoder 结构
    内含 multihead self attention 和 feed forward 结构
    """

    def __init__(
            self,
            attention: Module,
            dropout: float,
            layer_norm_first: bool,
            feed_forward: Module,):
        super().__init__()
        self.attention = attention
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(attention.embed_dim)
        self.layer_norm_first = layer_norm_first
        self.feed_forward = feed_forward
        self.final_layer_norm = nn.LayerNorm(attention.embed_dim)

    def forward(
            self,
            x: Tensor,
            attention_mask: Optional[Tensor] = None,
    ):
        """
        Args:
            x (Tensor): shape: `(batch, sequence_length, embed_dim)`
            attention_mask (Tensor or None, optional):
                shape: `(batch, 1, sequence_length, sequence_length)`
        """
        residual = x

        if self.layer_norm_first:
            x = self.layer_norm(x)

        x = self.attention(x, attention_mask)
        x = self.dropout(x)
        x = residual + x

        if self.layer_norm_first:
            x = x + self.feed_forward(self.final_layer_norm(x))
        else:
            x = self.layer_norm(x)
            x = self.final_layer_norm(x + self.feed_forward(x))
        return x


class TransformerEncoder(Module):
    """附带 pos_conv_embed 的 TransformerEncoder 结构
    支持获取中间层功能
    """

    def __init__(
            self,
            pos_conv_embed: Module,
            dropout: float,
            layers: Module,
            layer_norm_first: bool,
            layer_drop: float,
    ):
        super().__init__()
        self.pos_conv_embed = pos_conv_embed
        self.layer_norm = nn.LayerNorm(pos_conv_embed.embed_dim)
        self.layer_norm_first = layer_norm_first
        self.layer_drop = layer_drop
        self.dropout = nn.Dropout(dropout)
        self.layers = layers

    def _preprocess(self, x: Tensor):
        x = x + self.pos_conv_embed(x)

        if self.layer_norm_first:
            x = self.layer_norm(x)

        x = self.dropout(x)
        return x

    def forward(
            self,
            x: Tensor,
            attention_mask: Optional[Tensor] = None,
    ):
        x = self._preprocess(x)

        for layer in self.layers:
            # 似乎是训练过程中添加随机性, 忽略中间网络?
            if not (self.training and torch.rand(1).item() <= self.layer_drop):
                x = layer(x, attention_mask)

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        return x

    def get_intermediate_outputs(
            self,
            x: Tensor,
            attention_mask: Optional[Tensor] = None,
            num_layers: Optional[int] = None,) -> List[Tensor]:
        """获取模型输入数据后的中间层输出

        Args:
            x (Tensor): 
            attention_mask (Optional[Tensor], optional):  Defaults to None.
            num_layers (Optional[int], optional):  Defaults to None.
        Returns:
            List[Tensor]: 
        """
        if num_layers is not None:
            if not 0 < num_layers <= len(self.layers):
                raise ValueError(
                    f'`num_layers` 必须在 [1, {len(self.layers)}] 之间')

        ret: List[Tensor] = []
        x = self._preprocess(x)
        for layer in self.layers:
            x = layer(x, attention_mask)
            ret.append(x)  # 记录每一个 layer 的输出
            if num_layers is not None and len(ret) >= num_layers:
                return ret
        return ret


def _get_feature_extractor(
        norm_mode: str,
        shapes: List[Tuple[int, int, int]],
        bias: bool,) -> ConvFeatureExtractor:
    """通过给定参数构造 ConvFeatureExtractor 网络结构的函数

    Args:
        norm_mode (str): 特征提取器的操作模式
        shapes (List[Tuple[int, int, int]]): 特征提取器卷积层的配置参数
        bias (bool): 是否在每次卷积运算中包含偏置项

    Returns:
        ConvFeatureExtractor: ConvFeatureExtractor Model
    """
    # 检查参数
    assert norm_mode in ["group_norm", "layer_norm"]

    # 定义 Extractor
    blocks = []
    in_channels = 1  # 输入数据的通道
    for i, (out_channels, kernel_size, stride) in enumerate(shapes):

        # Norm设置
        normalization = None
        if norm_mode == "group_norm" and i == 0:  # 只在第一个卷积块使用GroupNorm

            normalization = nn.GroupNorm(
                num_groups=out_channels,
                num_channels=out_channels,
                affine=True,
            )
        elif norm_mode == "layer_norm":  # 所有卷积块都有一个layer_norm层
            normalization = LayerNorm(
                normalized_shape=out_channels,
                elementwise_affine=True,
            )

        # 卷积块设置
        blocks.append(
            Conv1dLayerBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                bias=bias,
                layer_norm=normalization,
                activate_func=nn.functional.gelu
            )
        )

        in_channels = out_channels  # 修改输入通道为 Extractor 的输出通道
    return ConvFeatureExtractor(nn.ModuleList(blocks))


class Encoder(Module):
    """ Wav2Vec 2.0 的 Encoder 结构, 含 _preprocess 和 transformerEncoder
        支持获取输入数据的中间层特征功能

    """

    def __init__(
            self,
            feature_projection: Module,
            transformerencoder: Module,):
        super().__init__()
        self.feature_projection = feature_projection
        self.transformerencoder = transformerencoder

    def _preprocess(
            self,
            features: Tensor,
            lengths: Optional[Tensor] = None,) -> Tuple[Tensor, Optional[Tensor]]:
        x = self.feature_projection(features)

        mask: Optional[Tensor] = None
        if lengths is not None:
            batch_size, max_len, _ = x.shape
            # 为真实 length 之外被填充的元素创建 mask 并将其置零
            # mask 的大小为 [batch_size, max_len], 相当于得到每个序列对应1-max_len的位置
            # 并将每个序列中大于真实 length 的位置设为 True, 其余位置为 False
            mask = torch.arange(max_len, device=lengths.device
                                ).expand(batch_size, max_len) >= lengths[:, None]
            x[mask] = 0.0  # 根据 mask 所给的位置坐标系, 对 x 中数据进行处理

            # 将 mask 扩展到 attention 形状[batch_size, 1, 1, max_len]并设置权重
            mask = -10000.0 * mask[:, None, None, :].to(dtype=features.dtype)
            # 沿着第2维度, 复制扩展 mask , [batch_size, 1, max_len, max_len]
            mask = mask.expand(batch_size, 1, max_len, max_len)
        return x, mask

    def forward(
            self,
            features: Tensor,
            lengths: Optional[Tensor] = None,) -> Tensor:

        x, mask = self._preprocess(features, lengths)
        x = self.transformerencoder(x, attention_mask=mask)
        return x

    def extract_features(
            self,
            features: Tensor,
            lengths: Optional[Tensor] = None,
            num_layers: Optional[int] = None,) -> List[Tensor]:

        x, masks = self._preprocess(features, lengths)
        return self.transformerencoder.get_intermediate_outputs(x, attention_mask=masks, num_layers=num_layers)


def _get_encoder(
        in_features: int,
        embed_dim: int,
        dropout_input: float,
        pos_conv_kernel: int,
        pos_conv_groups: int,
        num_layers: int,
        num_heads: int,
        attention_dropout: float,
        ff_interm_features: int,
        ff_interm_dropout: float,
        dropout: float,
        layer_norm_first: bool,
        layer_drop: float,) -> Encoder:
    """ 通过给定参数, 建立 Wav2Vec 2.0 的 Encoder 结构
    Args:
        in_features (int): 输入特征的维度
        embed_dim (int): TransformerEncoder中的中间层和输出特征维度, 一般是 768(Base) 或 1024(Large)
            -->fairseq.encoder_embed_dim
        dropout_input (float): 将输入特征投影到``embed_dim``之后应用的 dropout probability
            -->fairseq.dropout_input
            一般是 0.1
        pos_conv_kernel (int): convolutional positional embeddings 中的卷积核大小
            -->fairseq.conv_pos
            一般是 128
        pos_conv_groups (int): convolutional positional embeddings 的 groups 数目
            -->fairseq.conv_pos_groups
            一般是16
        num_layers (int): transformerEncoder block 中的 self attention layers 数目
            -->fairseq.encoder_layers
            一般是 12(Base) 或 24(Large)
        num_heads (int):self attention layers 的 heads 数目
            -->fairseq.encoder_attention_heads
            一般是 12(Base) 或 16(Large)
        attention_dropout (float): self-attention layer 中, softmax 之后的 dropout probability
            -->fairseq.attention_dropout
            一般是 0.1(Base) 或 0.0(Large)
        ff_interm_features (int): feed forward 层中, hidden features 的维度
            -->fairseq.encoder_ffn_embed_dim
            一般是 3072(Base) 或 4096(Large)
        ff_interm_dropout (float):
            -->fairseq.activation_dropout
            feed forward 层中使用的第一个 dropout probability
            一般是 0.1
        dropout (float):
            -->fairseq.dropout
            feed forward 层最后使用的 dropout probability
            一般是 0.1(Base) 或 0.0(Large)
        layer_norm_first (bool): 控制 transformerEncoder 层和各 encoder 层的 norm 层顺序
            -->fairseq.layer_norm_first
            如果为True, 则在 transformerEncoder 层中, 在将特征输入到 encoder layer 之前应用 norm 层。
            在 encoder 中的 self attention 前后均添加norm 层。
            如果为False, 则在 transformerEncoder 层中, 在将特征输入到 encoder layer 之后应用 norm 层。
            在 encoder 中的 self attention 后添加两个 norm 层, 其分别在前馈层(feed forward)的前后。
            一般是 False(Base) 或 True(Large)
        layer_drop (float): 训练期间 drop 每个 encoder layer 的概率
            -->fairseq.layerdrop
            一般是 0.1
    """
    feature_projection = FeatureProjection(
        in_features, embed_dim, dropout_input)  # 将特征维度投影到 encoder 所需的大小

    pos_conv = ConvolutionalPositionalEmbedding(
        embed_dim, pos_conv_kernel, pos_conv_groups)  # 卷积位置编码

    encoder_layers = nn.ModuleList()
    for _ in range(num_layers):
        attention = SelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
        )
        feed_forward = FeedForward(
            io_features=embed_dim,
            intermediate_features=ff_interm_features,
            intermediate_dropout=ff_interm_dropout,
            output_dropout=dropout,
        )
        encoder_layers.append(
            EncoderLayer(
                attention=attention,
                dropout=dropout,
                layer_norm_first=layer_norm_first,
                feed_forward=feed_forward,
            )
        )

    transformerencoder = TransformerEncoder(
        pos_conv_embed=pos_conv,
        dropout=dropout,
        layers=encoder_layers,
        layer_norm_first=not layer_norm_first,
        layer_drop=layer_drop,
    )
    return Encoder(feature_projection, transformerencoder)


class ResNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[tvresnet.BasicBlock, tvresnet.Bottleneck]],
        layers: List[int],
        num_classes: int = 1024,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, tvresnet.Bottleneck):
                    # type: ignore[arg-type]
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, tvresnet.BasicBlock):
                    # type: ignore[arg-type]
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block: Type[Union[tvresnet.BasicBlock, tvresnet.Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                tvresnet.conv1x1(self.inplanes, planes *
                                 block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
