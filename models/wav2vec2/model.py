from . import components
from typing import Optional, Tuple, List
import torch
from torch import Tensor, nn
from torch.nn import Module
import torch.nn.functional as F
# from torchaudio.models import wav2vec2


class Wav2Vec2Model(Module):
    """定义 Wav2Vec2Model 的前向计算, 并提供单独的 extract_features 功能
    Args:
        feature_extractor (torch.nn.Module):
            Feature_Extractor 能从原始音频tensor中提取特征向量。

        encoder (torch.nn.Module):
            Encoder 能将音频特征转换为标签上的概率分布序列(negative log-likelihood)。

        aux (torch.nn.Module or None, optional):
            如果提供辅助模型, 则Encoder的输出将传递到此模块。
    """

    def __init__(
            self,
            feature_extractor: Module,
            encoder: Module,
            aux: Optional[Module] = None,):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.encoder = encoder
        self.aux = aux

    @torch.jit.export  # 定义此方法默认要被编译,即使没有被前向函数调用,暂时无用
    def extract_features(
            self,
            waveforms: Tensor,
            lengths: Optional[Tensor] = None,
            num_layers: Optional[int] = None,) -> Tuple[List[Tensor], Optional[Tensor]]:
        """从原始语音波形中提取特征向量
        返回encoder中 transformer block 中间层的输出列表。

        Args:
            waveforms (Tensor): 音频tensor, 大小为 `(batch, frames)`.
            lengths (Tensor or None, optional):
                batch中每个音频的有效长度, 大小为`(batch, )`.
                当``waveforms``包含不同时间长度的音频时，
                通过提供``lengths``参数, 模型将计算相应的有效输出长度
                并在transformer attention layer中应用适当的mask。
                如果``None``, 则假定整个音频波形长度有效。
            num_layers (int or None, optional):
                如果给出, 限制要通过的中间层的数量。
                提供`1`将在经过一个中间层后停止计算。
                如果``None``，则返回所有中间层的输出。

        Returns:
            (List[Tensor], Optional[Tensor]):
            List of Tensors
                requested layers中的特征Tensor列表。
                其中每个Tensor的大小状为：`(batch, time frame, feature dimension)`
            Tensor or None
                如果提供了``Lengths``参数，则返回大小为`(Batch, )`的Tensor。
                它表示每个特征Tensor在时间轴上的有效长度。
        """
        x, lengths = self.feature_extractor(waveforms, lengths)
        x = self.encoder.extract_features(x, lengths, num_layers)
        return x, lengths

    def forward(
            self,
            waveforms: Tensor,
            lengths: Optional[Tensor] = None,) -> Tuple[Tensor, Optional[Tensor]]:
        """计算标签上的概率分布序列。

        Args:
            waveforms (Tensor): 音频tensor, 大小为 `(batch, frames)`.
            lengths (Tensor or None, optional):
                batch中每个音频的有效长度, 大小为`(batch, )`.
                当``waveforms``包含不同时间长度的音频时，
                通过提供``lengths``参数, 模型将计算相应的有效输出长度
                并在transformer attention layer中应用适当的mask。
                如果``None``, 则假定整个音频波形长度有效。

        Returns:
            (Tensor, Optional[Tensor]):
            Tensor
                标签上的概率分布序列(以logit表示)。
                大小为: `(batch, frames, num labels)`.
            Tensor or None
                如果提供了``Lengths``参数，则返回大小状为`(Batch, )`的Tensor。
                它表示输出Tensor在时间轴上的有效长度。
        """
        out, lengths = self.feature_extractor(waveforms, lengths)
        out = self.encoder(out, lengths)
        if self.aux is not None:
            for layer in self.aux:
                if isinstance(layer, nn.LSTM):
                    # [num_layers *num_directions,batch,hidden_size]
                    output, (h_n, c_n) = layer(out)
                    out = h_n.reshape((h_n.shape[1], -1))
                elif isinstance(layer, nn.Linear):
                    out = layer(out)
                    # # return x, lengths
                    #     x = F.softmax(x, dim=2)
                    #     x = torch.mean(x, dim=1)
        return out


def wav2vec2_model(config,) -> Wav2Vec2Model:
    """给定(Feature_Extractor, Encoder, aux)的参数, 建立一个自定义的 Wav2Vec2Model 网络结构, 
    Args:
        extractor_mode (str): 特征提取器的操作模式
            -->fairse.extractor_mode
            输入"group_norm"或"layer_norm"
            如果为"group_norm", 则只在第一个卷积块使用 normalization. 
            否则所有卷积块都有一个 normalization 层
        extractor_conv_layer_config (Optional[List[Tuple[int, int, int]]]): 特征提取器卷积层的配置
            -->fairse.conv_feature_layers
            输入卷积层参数的列表, 默认为(即输入None)
            .. code-block:: python

               [
                 (512, 10, 5),
                 (512, 3, 2),
                 (512, 3, 2),
                 (512, 3, 2),
                 (512, 3, 2),
                 (512, 2, 2),
                 (512, 2, 2),
               ]
        extractor_conv_bias (bool): 是否在每次卷积运算中包含偏置项
            -->fairseq.conv_bias
        encoder_embed_dim (int): encoder中嵌入向量的维度
            -->encoder_embed_dim
        encoder_projection_dropout (float): 输入特征被投影后应用的dropout比例
            -->dropout_input
        encoder_pos_conv_kernel (int): 卷积位置嵌入的kernel大小
            -->conv_pos
        encoder_pos_conv_groups (int): 卷积位置嵌入的组数
            -->conv_pos_groups
        encoder_num_layers (int): transformer block 中 self attention layers 的数目
            -->encoder_layers
        encoder_num_heads (int): self attention layers 中 heads的数目
            -->encoder_attention_heads
        encoder_attention_dropout (float): elf-attention layer 中 softmax 后的 dropout 比例
            -->attention_dropout
        encoder_ff_interm_features (int): 前馈层(feed forward)中隐藏层特征向量的维度。
            -->encoder_ffn_embed_dim
        encoder_ff_interm_dropout (float): 前馈层(feed forward)中中间的 dropout 比例
            -->activation_dropout
        encoder_dropout (float): 前馈层(feed forward)最后的 dropout 比例
            -->dropout
        encoder_layer_norm_first (bool): 控制 transformer 层和各 encoder 层的 norm 层顺序。
            -->layer_norm_first
            如果为True, 则在 transformer 层中, 在将特征输入到 encoder layer 之前应用 norm 层。
            在 encoder 中的 self attention 前后均添加norm 层。
            如果为False, 则在 transformer 层中, 在将特征输入到 encoder layer 之后应用 norm 层。
            在 encoder 中的 self attention 后添加两个 norm 层, 分别在前馈层(feed forward)的前后。
        encoder_layer_drop (float): 训练期间 drop 每个 encoder layer 的概率
            -->layerdrop
        aux_num_out (Optional[int]): 如果提供，在 encoder layer 最后附加一个额外的线性网络层，可用于微调。
            -->

    Returns:
        Wav2Vec2Model: 最终的Wav2Vec2模型, nn.model类型
    """
    # 定义默认的卷积层参数, 共七层(output_channel, kernel_size, stride)

    extractor_mode = config["wev2vec_encoder"]["extractor_mode"]
    extractor_conv_layer_config = config["wev2vec_encoder"]["extractor_conv_layer_config"]
    extractor_conv_bias = config["wev2vec_encoder"]["extractor_conv_bias"]
    encoder_embed_dim = config["wev2vec_encoder"]["encoder_embed_dim"]
    encoder_projection_dropout = config["wev2vec_encoder"]["encoder_projection_dropout"]
    encoder_pos_conv_kernel = config["wev2vec_encoder"]["encoder_pos_conv_kernel"]
    encoder_pos_conv_groups = config["wev2vec_encoder"]["encoder_pos_conv_groups"]
    encoder_num_layers = config["wev2vec_encoder"]["encoder_num_layers"]
    encoder_num_heads = config["wev2vec_encoder"]["encoder_num_heads"]
    encoder_attention_dropout = config["wev2vec_encoder"]["encoder_attention_dropout"]
    encoder_ff_interm_features = config["wev2vec_encoder"]["encoder_ff_interm_features"]
    encoder_ff_interm_dropout = config["wev2vec_encoder"]["encoder_ff_interm_dropout"]
    encoder_dropout = config["wev2vec_encoder"]["encoder_dropout"]
    encoder_layer_norm_first = config["wev2vec_encoder"]["encoder_layer_norm_first"]
    encoder_layer_drop = config["wev2vec_encoder"]["encoder_layer_drop"]
    aux_num_out = config["wev2vec_encoder"]["aux_num"]

    if extractor_conv_layer_config == "None":
        extractor_conv_layer_config = [
            (512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2
    # 定义 Feature Extractor
    feature_extractor = components._get_feature_extractor(
        extractor_mode, extractor_conv_layer_config, extractor_conv_bias)
    # 定义 Wav2Vec Encoder
    encoder = components._get_encoder(
        in_features=extractor_conv_layer_config[-1][0],
        embed_dim=encoder_embed_dim,
        dropout_input=encoder_projection_dropout,
        pos_conv_kernel=encoder_pos_conv_kernel,
        pos_conv_groups=encoder_pos_conv_groups,
        num_layers=encoder_num_layers,
        num_heads=encoder_num_heads,
        attention_dropout=encoder_attention_dropout,
        ff_interm_features=encoder_ff_interm_features,
        ff_interm_dropout=encoder_ff_interm_dropout,
        dropout=encoder_dropout,
        layer_norm_first=encoder_layer_norm_first,
        layer_drop=encoder_layer_drop,
    )
    # [batch, frame, feature]
    # 定义输出分类
    # aux = None
    # if aux_num_out is not None:
    #     aux = nn.Sequential(
    #         nn.Flatten(),
    #         torch.nn.Linear(in_features=encoder_embed_dim*448,
    #                         out_features=aux_num_out),)

    aux = None
    if aux_num_out != "None":
        aux = torch.nn.Linear(in_features=encoder_embed_dim,
                              out_features=aux_num_out)
    return Wav2Vec2Model(feature_extractor, encoder, aux)
