# 官方库
import os
from typing import Dict, Union, Optional
from pathlib import Path
import logging
# 第三方库
import librosa
import pandas as pd
import torch
import soundfile as sf
import h5py
import requests
import math
# 自定库
from custom_transforms import *
from custom_enhance import *
from custom_datasets.subset import MediaDataset
from tqdm import tqdm


_URL_ = {}

_URL_["TimestampedWords"] = 'http://immortal.multicomp.cs.cmu.edu/CMU-MOSI/language/CMU_MOSI_TimestampedWords.csd'
_URL_["TimestampedPhones"] = 'http://immortal.multicomp.cs.cmu.edu/CMU-MOSI/language/CMU_MOSI_TimestampedPhones.csd'

# acoustic
_URL_["COVAREP"] = 'http://immortal.multicomp.cs.cmu.edu/CMU-MOSI/acoustic/CMU_MOSI_COVAREP.csd'
_URL_["openSMILE_IS09"] = 'http://immortal.multicomp.cs.cmu.edu/CMU-MOSI/acoustic/CMU_MOSI_openSMILE_IS09.csd'
_URL_["OpenSmile_EB10"] = 'http://immortal.multicomp.cs.cmu.edu/CMU-MOSI/acoustic/CMU_MOSI_OpenSmile_EB10.csd'

# language
# _URL_["WordVectors"] = 'http://immortal.multicomp.cs.cmu.edu/CMU-MOSI/language/CMU_MOSI_TimestampedWordVectors.csd'
_URL_["GloveWordVectors"] = 'http://immortal.multicomp.cs.cmu.edu/CMU-MOSI/language/CMU_MOSI_TimestampedWordVectors_1.1.csd'

# visual
_URL_["Facet_42"] = 'http://immortal.multicomp.cs.cmu.edu/CMU-MOSI/visual/CMU_MOSI_Visual_Facet_42.csd'
_URL_["OpenFace_2"] = 'http://immortal.multicomp.cs.cmu.edu/CMU-MOSI/visual/CMU_MOSI_Visual_OpenFace_2.csd'
_URL_["labels"] = "http://immortal.multicomp.cs.cmu.edu/CMU-MOSI/labels/CMU_MOSI_Opinion_Labels.csd"

standard_train_fold = ['2iD-tVS8NPw', '8d-gEyoeBzc', 'Qr1Ca94K55A', 'Ci-AH39fi3Y', '8qrpnFRGt2A', 'Bfr499ggo-0', 'QN9ZIUWUXsY', '9T9Hf74oK10', '7JsX8y1ysxY', '1iG0909rllw', 'Oz06ZWiO20M', 'BioHAh1qJAQ', '9c67fiY0wGQ', 'Iu2PFX3z_1s', 'Nzq88NnDkEk', 'Clx4VXItLTE', '9J25DZhivz8', 'Af8D0E4ZXaw', 'TvyZBvOMOTc', 'W8NXH0Djyww', '8OtFthrtaJM', '0h-zjBukYpk', 'Vj1wYRQjB-o', 'GWuJjcEuzt8', 'BI97DNYfe5I',
                       'PZ-lDQFboO8', '1DmNV9C1hbY', 'OQvJTdtJ2H4', 'I5y0__X72p0', '9qR7uwkblbs', 'G6GlGvlkxAQ', '6_0THN4chvY', 'Njd1F0vZSm4', 'BvYR0L6f2Ig', '03bSnISJMiM', 'Dg_0XKD0Mf4', '5W7Z1C_fDaE', 'VbQk4H8hgr0', 'G-xst2euQUc', 'MLal-t_vJPM', 'BXuRRbG0Ugk', 'LSi-o-IrDMs', 'Jkswaaud0hk', '2WGyTLYerpo', '6Egk_28TtTM', 'Sqr0AcuoNnk', 'POKffnXeBds', '73jzhE8R1TQ', 'OtBXNcAL_lE', 'HEsqda8_d0Q', 'VCslbP0mgZI', 'IumbAb8q2dM']
standard_valid_fold = ['WKA5OygbEKI', 'c5xsKMxpXnc', 'atnd_PF-Lbs', 'bvLlb-M3UXU',
                       'bOL9jKpeJRs', '_dI--eQ6qVU', 'ZAIRrfG22O0', 'X3j2zQgwYgE', 'aiEXnCPZubE', 'ZUXBRvtny7o']
standard_test_fold = ['tmZoasNr4rU', 'zhpQhgha_KU', 'lXPQBPVc5Cw', 'iiK8YX8oH1E', 'tStelxIAHjw', 'nzpVDcQ0ywM', 'etzxEpPuc6I', 'cW1FSBF59ik', 'd6hH302o4v8', 'k5Y_838nuGo', 'pLTX3ipuDJI', 'jUzDDGyPkXU', 'f_pcplsH_V0', 'yvsjCA6Y5Fc', 'nbWiPyCm4g0',
                      'rnaNMUZpvvg', 'wMbj6ajWbic', 'cM3Yna7AavY', 'yDtzw_Y-7RU', 'vyB00TXsimI', 'dq3Nf_lMPnE', 'phBUpBr1hSo', 'd3_k5Xpfmik', 'v0zCBqDeKcE', 'tIrG4oNLFzE', 'fvVhgmXxadc', 'ob23OKe5a9Q', 'cXypl4FnoZo', 'vvZ4IcEtiZc', 'f9O3YtZ2VfI', 'c7UH_rxdZv4']


def metadataToDict(mtdmetadata):
    if (type(mtdmetadata) is dict):
        return mtdmetadata
    else:
        metadata = {}
        for key in mtdmetadata.keys():
            metadata[key] = mtdmetadata[key][0]
        return metadata

# reading MTD files directly from HDD


def readHDF5(resource, destination=None):
    try:
        h5handle = h5py.File('%s' % resource, 'r')
    except:
        raise ValueError(f"{resource} 文件地址不对")
    logging.info(f"读取 {resource} 文件成功")
    return h5handle, dict(h5handle[list(h5handle.keys())[0]]["data"]), metadataToDict(h5handle[list(h5handle.keys())[0]]["metadata"])


def download_file(url, destination):

    r = requests.get(url, stream=True)
    if r.status_code != 200:
        raise ValueError(f"{url} 网络地址不对")
    total_size = int(r.headers.get('content-length', 0))
    block_size = 1024
    wrote = 0
    with open(destination, 'wb') as f:
        for data in tqdm(r.iter_content(block_size), total=math.ceil(total_size//block_size), unit='KB', unit_scale=True, leave=False):
            wrote = wrote + len(data)
            f.write(data)
    f.close()
    if total_size != 0 and wrote != total_size:
        raise ValueError(f"下载数据失败")
    logging.info(f"下载 {destination} 文件成功")


def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


class MOSI(MediaDataset):
    """Create a Dataset for MOSI.
        Args:
        Returns:
            tuple: ``(waveform, sample_rate, transcript, val, act, dom, speaker)``

    """

    def __init__(self,
                 root: Union[str, Path],
                 info: str = "",
                 filter: dict = {"replace": {}, "dropna": {'emotion': [
                     "other", "xxx"]}, "contains": "", "query": "", "sort_values": ""},
                 transform: dict = {},
                 enhance: dict = {},
                 savePath: str = "",
                 mode: str = "",
                 ):
        super(MOSI, self).__init__(
            root=root, info=info, filter=filter,
            transform=transform, enhance=enhance, mode=mode,
            savePath=savePath,
        )
        self.dataset_name = "MOSI"

        if os.path.isfile(root):
            self.load_data()
            if filter:
                self.build_filter()
        elif os.path.isdir(root):
            self.build_datadict()
            # self.build_enhance()
            # self.build_transform()
            self.save_data()
            if filter:
                self.build_filter()
        else:
            raise ValueError("数据库地址不对")

    def build_datadict(self):
        """
        读取 MOSI 数据集官方提供的原始数据和预处理数据。
        """
        logging.info("构建初始数据")
        # 文件情感标签与序号的对应关系
        assert (True in [i in self.transform for i in [
                "EmoBase_2010", "IS09"]])

        # 加载特征文件(本地没有则重新从网络下载)
        destination = os.path.join(self.root, "processed")
        create_folder(destination)
        audioFeature = None
        if "EmoBase_2010" in self.transform and "a" in self.mode:
            # 判断 CMU_MOSI_OpenSmile_EB10.csd 文件是否存在
            destination = os.path.join(
                self.root, "processed", "CMU_MOSI_OpenSmile_EB10.csd")
            if os.path.isfile(os.path.join(self.root, "processed", "CMU_MOSI_OpenSmile_EB10.csd")):
                h5handle, audioFeature, metadata = readHDF5(
                    os.path.join(self.root, "processed", "CMU_MOSI_OpenSmile_EB10.csd"))
                logging.info("加载 CMU_MOSI_OpenSmile_EB10.csd 文件成功")
            else:
                logging.info("下载 CMU_MOSI_OpenSmile_EB10.csd 文件")
                download_file(_URL_["OpenSmile_EB10"], destination)
                h5handle, audioFeature, metadata = readHDF5(destination)
                logging.info("加载 CMU_MOSI_OpenSmile_EB10.csd 文件成功")
        elif "IS09" in self.transform and "a" in self.mode:
            # 判断 CMU_MOSI_openSMILE_IS09.csd 文件是否存在
            destination = os.path.join(
                self.root, "processed", "CMU_MOSI_openSMILE_IS09.csd")
            if os.path.isfile(os.path.join(self.root, "processed", "CMU_MOSI_openSMILE_IS09.csd")):
                h5handle, audioFeature, metadata = readHDF5(
                    os.path.join(self.root, "processed", "CMU_MOSI_openSMILE_IS09.csd"))
                logging.info("加载 CMU_MOSI_openSMILE_IS09.csd 文件成功")
            else:
                logging.info("下载 CMU_MOSI_openSMILE_IS09.csd 文件")
                download_file(_URL_["openSMILE_IS09"], destination)
                h5handle, audioFeature, metadata = readHDF5(destination)
                logging.info("加载 CMU_MOSI_openSMILE_IS09.csd 文件成功")
        elif "a" in self.mode:
            assert (False)

        videoFeature = None
        if "Facet42" in self.transform and "v" in self.mode:
            destination = os.path.join(
                self.root, "processed", "CMU_MOSI_Visual_Facet_42.csd")
            if os.path.isfile(os.path.join(self.root, "processed", "CMU_MOSI_Visual_Facet_42.csd")):
                h5handle, videoFeature, metadata = readHDF5(
                    os.path.join(self.root, "processed", "CMU_MOSI_Visual_Facet_42.csd"))
                logging.info("加载 CMU_MOSI_Visual_Facet_42.csd 文件成功")
            else:
                logging.info("下载 CMU_MOSI_Visual_Facet_42.csd 文件")
                download_file(_URL_["Facet_42"], destination)
                h5handle, videoFeature, metadata = readHDF5(destination)
                logging.info("加载 CMU_MOSI_Visual_Facet_42.csd 文件成功")
        elif "OpenFace2" in self.transform and "v" in self.mode:
            destination = os.path.join(
                self.root, "processed", "CMU_MOSI_Visual_OpenFace_2.csd")
            if os.path.isfile(os.path.join(self.root, "processed", "CMU_MOSI_Visual_OpenFace_2.csd")):
                h5handle, videoFeature, metadata = readHDF5(
                    os.path.join(self.root, "processed", "CMU_MOSI_Visual_OpenFace_2.csd"))
                logging.info("加载 CMU_MOSI_Visual_OpenFace_2.csd 文件成功")
            else:
                logging.info("下载 CMU_MOSI_Visual_OpenFace_2.csd 文件")
                download_file(_URL_["OpenFace_2"], destination)
                h5handle, videoFeature, metadata = readHDF5(destination)
                logging.info("加载 CMU_MOSI_Visual_OpenFace_2.csd 文件成功")
        elif "v" in self.mode:
            assert (False)

        textFeature = None
        if "Glove" in self.transform and "t" in self.mode:
            destination = os.path.join(
                self.root, "processed", "CMU_MOSI_TimestampedWordVectors_1.1.csd")
            if os.path.isfile(os.path.join(self.root, "processed", "CMU_MOSI_TimestampedWordVectors_1.1.csd")):
                h5handle, textFeature, metadata = readHDF5(
                    os.path.join(self.root, "processed", "CMU_MOSI_TimestampedWordVectors_1.1.csd"))
                logging.info("加载 CMU_MOSI_TimestampedWordVectors_1.1.csd 文件成功")
            else:
                logging.info("下载 CMU_MOSI_TimestampedWordVectors_1.1.csd 文件")
                download_file(_URL_["GloveWordVectors"], destination)
                h5handle, textFeature, metadata = readHDF5(destination)
                logging.info("加载 CMU_MOSI_TimestampedWordVectors_1.1.csd 文件成功")
        elif "t" in self.mode:
            assert (False)

        label = None
        destination = os.path.join(
            self.root, "processed", "CMU_MOSI_Opinion_Labels.csd")
        if os.path.isfile(os.path.join(self.root, "processed", "CMU_MOSI_Opinion_Labels.csd")):
            h5handle, label, metadata = readHDF5(
                os.path.join(self.root, "processed", "CMU_MOSI_Opinion_Labels.csd"))
            logging.info("加载 CMU_MOSI_Opinion_Labels.csd 文件成功")
        else:
            logging.info("下载 CMU_MOSI_Opinion_Labels.csd 文件")
            download_file(_URL_["labels"], destination)
            h5handle, label, metadata = readHDF5(destination)
            logging.info("加载 CMU_MOSI_Opinion_Labels.csd 文件成功")
        # assert (len(textFeature) == len(audioFeature) == len(videoFeature) == len(label))
        audiolist = []
        videolist = []
        textlist = []
        labelList = []
        self.indices = {}
        self.indices["train"] = []
        self.indices["valid"] = []
        self.indices["test"] = []
        i = 0
        for k, inds in {"train": standard_train_fold, "valid": standard_valid_fold, "test": standard_test_fold}.items():
            for m in inds:
                if m == "c5xsKMxpXnc":
                    continue
                if textFeature is not None:
                    tmp_data = textFeature[m]["features"][:]
                    tmp_data[tmp_data == -np.inf] = 0
                    tmp_data = tmp_data.astype(np.float32)
                    textlist.append(tmp_data)
                if audioFeature is not None:
                    tmp_data = audioFeature[m]["features"][:]
                    tmp_data[tmp_data == -np.inf] = 0
                    tmp_data = tmp_data.astype(np.float32)
                    audiolist.append(tmp_data)
                if videoFeature is not None:
                    tmp_data = videoFeature[m]["features"][:]
                    tmp_data[tmp_data == -np.inf] = 0
                    tmp_data = tmp_data.astype(np.float32)
                    videolist.append(tmp_data)
                if label is not None:
                    labelList.append(label[m]["features"][:].item(0))
                self.indices[k].append(i)
                i += 1

        self.datadict = {
            "label": labelList,
        }

        if textFeature is not None:
            self.datadict["text"] = [torch.from_numpy(i) for i in textlist]
        if audioFeature is not None:
            self.datadict["audio"] = [torch.from_numpy(i) for i in audiolist]
        if videoFeature is not None:
            self.datadict["video"] = [torch.from_numpy(i) for i in videolist]

        # self.datadict = {
        #     "path": self._pathList,
        #     "duration": self._periodList,
        #     "text": _textList,
        #     "gender": _genderList,
        #     "label": _labelList,
        #     "sample_rate": _sample_rate
        # }

    def __getitem__(self, n: int) -> Dict[str, Optional[Union[int, torch.Tensor, int, str, int, str, str]]]:
        """加载 MOSI 数据库的第 n 个样本数据
        Args:
            n (int): 要加载的样本的索引
        Returns:
            Dict[str, Optional[Union[int, torch.Tensor, int, str, int, str, str]]]:
        """
        return {
            "id": n,
            "audio": self.datadict["audio"][n],
            "sample_rate": self.datadict["sample_rate"][n],
            "text": self.datadict["text"][n],
            "labelid": self.datadict["labelid"][n],
            "gender": self.datadict["gender"][n],
            "label": self.datadict["label"][n]
        }

    def __len__(self) -> int:
        """为 MOSI 数据库类型重写 len 默认行为.
        Returns:
            int: MOSI 数据库数据数量
        """
        return len(self.datadict["path"])


if __name__ == '__main__':
    a = MOSI("/sdb/user0/SCW/data/CMU_MOSI",
             transform={"EmoBase_2010": {}, "Facet42": {}, "Glove": {}}, mode="avt")
    feature = a.__getitem__(100)
    pass
