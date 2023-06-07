from torch import Tensor, nn
from custom_models import *

class CooperationNet(nn.Module):
    def __init__(self, num_classes, model_names, input_size, seq_len, last_hidden_dim, seq_step) -> None:
        super().__init__()
        for i, name in enumerate(model_names):
            setattr(self, f"coopnet{i}", AudioNet(
                num_classes=num_classes[i], model_name=name,  last_hidden_dim=last_hidden_dim[i], input_size=input_size[i], pretrained=False))

    def forward(self, x: Tensor, **args) -> Tensor:
        raise "CooperationNet 只是一个对多个模型的包装, 而非实际要训练的网络. 请使用内部的 coopnet{id} 进行训练"
        return x
