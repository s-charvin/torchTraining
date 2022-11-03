
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

    def __init__(self, dims) -> None:
        super(Permute_Channel, self).__init__()
        self.dims = dims

    def forward(self, feature: Tensor) -> Tensor:
        r"""
        Args:
            feature (Tensor): Tensor of feature of dimension (...).

        Returns:
            Tensor: Tensor of feature of dimension (dims).
        """
        return feature.permute(*self.dims)
