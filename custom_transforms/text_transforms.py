

import sklearn
import torch
# preprocessing.OneHotEncoder()
# preprocessing.LabelEncoder()


class Label2Id(torch.nn.Module):
    r""" 
    """

    def __init__(self) -> None:
        super().__init__()

    def transform(self, label_list):
        label_encoder = sklearn.preprocessing.LabelEncoder()
        label_encoder.fit(y=label_list)
        # id2label = {f"{id}": label for id, label in enumerate()}
        return label_encoder.transform(label_list)

    def forward(self, datadict: dict):
        r"""
        Args:
            datadict (dict): 字典格式存储的数据.

        Returns:
            datadict: .
        """
        #
        datadict["labelid"] = self.transform(datadict["label"])
        return datadict
