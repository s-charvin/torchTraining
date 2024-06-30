from custom_datasets.CREMA_D import CREMA_D

if __name__ == "__main__":
    # 将当前文件夹的上级目录加入到系统路径中
    import sys

    sys.path.append("/home/visitors2/SCW/torchTraining/")

    dataset = CREMA_D(
        root="/sdb/visitors2/SCW/data/CREMA-D",
        mode="a",
        threads=8,
        enhance={
            "AudioSplit": {
                "sampleNum": 112000,
            }
        },
        transform={
            "MFCC": {
                "sample_rate": 16000,
                "n_mfcc": 40,
                "log_mels": True,
                "melkwargs": {
                    "n_fft": 1024,
                    "win_length": 1024,
                    "hop_length": 256,
                    "f_min": 80.0,
                    "f_max": 7600.0,
                    "pad": 0,
                    "n_mels": 128,
                    "power": 2.0,
                    "normalized": True,
                    "center": True,
                    "pad_mode": "reflect",
                    "onesided": True,
                },
            },
            "Permute_Channel": {"dims": [[2, 1, 0]], "names": ["audio"]},
        },
        filter={
            "replace": {},
            "dropna": {"label": ["xxx", "disgusted", "fearful"]},
        },
    )
    print(len(dataset))
    print(dataset[0]["audio"].shape)
    # print(dataset[0]["video"].shape)
    # print(dataset[0]["text"])
    print(dataset[0]["label"])
    print(dataset[0]["speaker"])
    print(dataset[0]["sentence_code"])
    print(dataset[0]["level"])
