import collections
import numpy as np  # 矩阵运算模块
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from typing import List, Union
from pathlib import Path
import os
from abc import ABC, abstractmethod
import pandas as pd
from custom_enhance import *
from custom_transforms import *
from collections import OrderedDict
from utils.filefolderTool import create_folder


class BaseDataset(Dataset):
    def __init__(self,
                 root: Union[str, Path],  # 原始数据集路径
                 info: str,  # 使用数据集的额外信息
                 transform: dict,  # 数据后期处理及其参数
                 enhance: dict,  # 数据前期处理函数及其参数
                 # 过滤数据的条件
                 filter: dict,
                 savePath: str,
                 split: tuple = ([0.8, 0.2, ], 1),  # 要分割的数据比例和分割折数
                 ):
        super(Dataset, self).__init__()
        self.dataset_name = "BaseDataset"
        self.root = os.fspath(root)
        self.info = info
        self.datadict = []
        self.enhance = enhance
        self.transform = transform
        self.filter = filter
        self.savePath = savePath
        self.split = split

    @abstractmethod
    def build_datadict(self, **args):
        pass

    def build_enhance(self):
        pl = []
        if self.enhance:
            for processor, param in self.enhance.items():
                if param:
                    pl.append((processor, globals()[processor](**param)))
                else:
                    pl.append((processor, globals()[processor]()))
        processors = nn.Sequential(OrderedDict(pl))
        print("# 数据增强...")
        self.datadict = processors(self.datadict)

    def build_transform(self):
        pl = []
        if self.transform:
            for processor, param in self.transform.items():
                if param:
                    pl.append((processor, globals()[processor](**param)))
                else:
                    pl.append((processor, globals()[processor]()))
        processors = nn.Sequential(OrderedDict(pl))
        print("# 数据转换...")
        self.datadict = processors(self.datadict)

    def save_data(self):
        namelist = []
        if self.savePath:
            torch.save(self.datadict, self.savePath)
            return True
        if self.enhance:
            for k, v in self.enhance.items():
                namelist.append(k)
        if self.transform:
            for k, v in self.transform.items():
                namelist.append(k)

        path = os.path.join(self.root, "Feature", )
        create_folder(path)
        path = os.path.join(
            path, "-".join([self.dataset_name, self.info, *namelist, ".npy"]))
        torch.save(self.datadict, path)
        self.root = path
        self.savePath = path
        print(f"存储数据: {path}")
        return True

    def build_filter(self):
        print("数据筛选...")
        # 根据提供的字典替换值, key 决定要替换的列, value 是 字符串字典, 决定被替换的值
        self.datadict = pd.DataFrame(self.datadict)
        if "replace" in self.filter:
            self.datadict.replace(
                None if self.filter["replace"] else self.filter["replace"], inplace=True)
            # 根据提供的字典删除指定值所在行, key 决定要替换的列, value 是被替换的值列表
        if "dropna" in self.filter:
            for k, v in self.filter["dropna"].items():
                self.datadict = self.datadict[~self.datadict[k].isin(v)]
            self.datadict.dropna(axis=0, how='any', thresh=None,
                                 subset=None, inplace=True)
        if "contains" in self.filter:
            # 根据提供的字典删除包含指定值的所在行, key 决定要替换的列, value 是被替换的值列表
            for k, v in self.filter["contains"].items():
                self.datadict = self.datadict[~self.datadict[k].str.contains(
                    v, case=True)]
        if "query" in self.filter:
            if self.filter["query"]:
                self.datadict.query(self.filter["query"], inplace=True)
        if "sort_values" in self.filter:
            self.datadict.sort_values(
                by=self.filter["sort_values"], inplace=True, ascending=False)

        self.datadict.reset_index(drop=True, inplace=True)
        self.datadict = self.datadict.to_dict(orient="list")

    @abstractmethod
    def __getitem__(self, n: int):
        pass

    def load_data(self):
        print("加载数据...")
        self.datadict = torch.load(self.root)


class MediaDataset(BaseDataset):
    def __init__(self,
                 mode: str,  # 设置数据集要包含的数据模态(文本,视频,音频)
                 root: Union[str, Path],
                 **kwargs
                 ):
        super(MediaDataset, self).__init__(root=root, **kwargs)
        assert (mode in ["v", "a", "t", "av", "at", "vt", "avt"])
        self.dataset_name = "MediaDataset"
        self.mode = mode


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
        indices = np.random.permutation(n_samples)  # 打乱数据 index
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
