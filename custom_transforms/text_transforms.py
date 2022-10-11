

import sklearn
import torch


class Permute_Channel(torch.nn.Module):

    def __init__(self,) -> None:
        super(Permute_Channel, self).__init__()

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            feature (Tensor): Tensor of feature of dimension (C,...).

        Returns:
            Tensor: Tensor of feature of dimension (..., C).
        """
        return feature.permute([i for i in range(1, len(feature.shape))]+[0])


le = sklearn.preprocessing.LabelEncoder()

self.indexx = le.fit(y=self.labels)  # 定义文字标签与数值的对应关系
# self.classes = self.indexx.classes_
self.config["data"]["wav_maxlength"] = self.wav_maxlength
self.config["data"]["wav_minlength"] = self.wav_minlength
self.config["data"]["indexx"] = self.indexx
# self.config["data"]["classes"] = self.classes
