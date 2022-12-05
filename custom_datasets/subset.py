import collections
import numpy as np
import torch  # 矩阵运算模块
from torch.utils.data.dataset import Dataset
from typing import List, Union
from pathlib import Path


class CustomDataset(Dataset):
    def __init__(self,
                 root: Union[str, Path],
                 name,
                 filter: dict,
                 transform: dict,
                 enhance: dict,
                 ):
        super(Dataset, self).__init__()


class Subset():
    r"""
    Subset of a dataset at specified indices.

    Args:
        dataset (Dataset): The whole Dataset
        indices (list): Indices in the whole set selected for subset
    """

    def __init__(self, dataset, indices) -> None:
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]]
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


def random_split_index_K(n_samples: int, K: int = 5, shuffle=True):
    r"""
    Args:
        n_samples (Dataset): 数据集总数据数量
        K (int): 分割份数.
        shuffle (seed): 是否打乱数据.
    """
    if shuffle:
        indices = np.random.permutation(n_samples)  # 打乱数据index
    else:
        indices = np.arange(0, n_samples)
    test_indices = np.array_split(indices, K)
    # fold_sizes = np.full(K, n_samples // K, dtype=int)
    # fold_sizes[:n_samples % K] += 1  # 获取完整的分割index
    for test_indice in test_indices:
        test_index = np.zeros(n_samples, dtype=bool)
        test_index[test_indice] = True
        train_index = indices[np.logical_not(test_index)]
        test_index = indices[test_index]
        yield (train_index, test_index)


def collate_fn(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    # 如果 batch 已经是一个 torch.Tensor, 直接合并到一个共享内存张量，以避免额外的复制
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:  # 如果不是在主进程跑的
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        try:
            return torch.stack(batch, 0, out=out)
        except RuntimeError as e:
            if "equal size" in e.args[0]:
                batch_lengths = [i.shape[0] for i in batch]
                batch = torch.nn.utils.rnn.pad_sequence(
                    batch, batch_first=True, padding_value=0.)
                packed_batch = torch.nn.utils.rnn.pack_padded_sequence(
                    batch, batch_lengths, batch_first=True, enforce_sorted=False)
                return packed_batch
            raise e
        # 如果 batch 是一个 numpy, 且内部元素类型不是字符串类型, array["str",...]
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        # 如果 batch 内部的每个元素依旧是一个 ndarray, 则将其转换成 torch.Tensor 列表
        # array[array[...],...] -> [Tensor[...],...]
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            return collate_fn([torch.as_tensor(b) for b in batch])
        # 如果 batch 内部的元素是常量, 直接将 batch 转换为 Tensor.
        elif elem.shape == ():
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        # 如果 batch 元素为 float, 直接将 batch 转换为 Tensor.
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        # 如果 batch 元素为 int, 直接将 batch 转换为 Tensor.
        return torch.tensor(batch)
    elif isinstance(elem, (str, bytes)):
        # 如果 batch 元素为字符类型, 不做转换
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        # 如果 batch 元素为 Mapping 类型, 如字典类型
        # 提取每个元素 key 对应的所有值到一个 list 中, 重新构造字典
        # batch[{"key1":value, ...}, ...] ->["key1":[value,...],"key2":[value,...],...]
        # 其中 value 会进行一次类型转换, 判断是否可以继续通过 collate_fn 重构格式, 或将值转换为 Tensor
        return {key: collate_fn([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        # 如果 batch 元素为具名元组, 即 tuple 中的每个元素有其类型名称, namedtuple(name=[value, value, ...], ...)
        # 则将其每个 name 的值打包在一起, 且判断是否可以继续通过 collate_fn 重构格式, 或将值转换为 Tensor
        # 然后返回一个 namedtuple 类型数据
        # namedtuple(name=[value, value, ...], ...)
        # ->
        # namedtuple(name=([value, value, ...],...), ...)
        return elem_type(*(collate_fn(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):

        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError(
                'each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [collate_fn(samples) for samples in transposed]

    raise TypeError(elem_type)
