
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

    def __init__(self, dims, names) -> None:
        super(Permute_Channel, self).__init__()
        self.dims = dims
        self.names = names

    def transform(self, feature: Tensor, dims):
        return feature.permute(*dims)

    def forward(self, datadict: dict) -> Tensor:
        r"""
        Args:
            datadict (dict): 字典格式存储的数据.

        Returns:
            datadict: .
        """
        for i, name in enumerate(self.names):
            index = range(len(datadict[name]))
            for j in index:
                datadict[name][j] = self.transform(
                    datadict[name][j], self.dims[i])  # 处理当前行的语音数据
        return datadict
