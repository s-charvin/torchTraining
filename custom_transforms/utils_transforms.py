
import torch
from torch import Tensor


class Compose():
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data


class Permute_Channel(torch.nn.Module):

    def __init__(self,) -> None:
        super(Permute_Channel, self).__init__()

    def forward(self, feature: Tensor) -> Tensor:
        r"""
        Args:
            feature (Tensor): Tensor of feature of dimension (C,...).

        Returns:
            Tensor: Tensor of feature of dimension (..., C).
        """
        return feature.permute([i for i in range(1, len(feature.shape))]+[0])
