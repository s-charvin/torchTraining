import os
import sys

if sys.platform.startswith("linux"):
    os.chdir("/home/user0/SCW/torchTraining")
elif sys.platform.startswith("win"):
    pass
from utils.visuallize import *
from custom_models import *
from custom_datasets import *
from custom_datasets.subset import Subset


def plot_rt_MIFFNet_Conv2D_InterFusion_Joint():
    print("加载模型")
    dictionary = torch.load(
        "/sdb/user0/SCW/checkpoint/MIFFNet_Conv2D_InterFusion_Joint_Conv2D-inlength(7s)-model(lightresmultisernet2|resnet50)-batch(8)-batch_delay(1)-epoch(200)-lr(0.00025)-42-8-200-Audio_Video_Joint_Classification-AdamW0.00025-IEMOCAP//000194.ckpt",
        map_location=torch.device("cpu"),
    )
    data = IEMOCAP(
        root="/sdb/user0/SCW/data/IEMOCAP/Feature/IEMOCAP-pad-MFCC-KeyFrameGetterBasedIntervalSampling-Permute_Channel-.npy",
        mode="av",
        videomode="crop",
        cascPATH="/home/user0/SCW/torchTraining/utils/haarcascade_frontalface_alt2.xml",
        filter={
            "replace": {"label": {"excited": "happy"}},
            "dropna": {
                "label": [
                    "other",
                    "xxx",
                    "frustrated",
                    "disgusted",
                    "fearful",
                    "surprised",
                ],
            },
            "query": {"duration": "duration>22.0"},
        },
    )
    indices = [i for i in range(len(data)) if i in dictionary["train_indices"]]
    # data = Subset(data, indices=[0, 1, 5, 6, 9, 10, 11, 12, 13, 14, 15])
    print(len(data))

    dataloader = torch.utils.data.DataLoader(
        data, batch_size=1, shuffle=False, num_workers=0
    )

    net = MIFFNet_Conv2D_InterFusion_Joint(
        num_classes=4,
        input_size=[[189, 40], [150, 150]],
        model_name=["lightmultisernet", "resnet50"],
        last_hidden_dim=[320, 320],
        seq_len=[189, 30],
        enable_classifier=True,
    )
    weights_dict = {}
    for k, v in dictionary["net"].items():
        new_k = k.replace("module.", "") if "module" in k else k
        weights_dict[new_k] = v

    net.load_state_dict(weights_dict)

    plot_rt(
        net,
        dataloader,
        outdir="./output/real-time/",
        info="MIFFNet_Conv2D_InterFusion_Joint",
        device=device,
    )


def plot_rt_MIFFNet_Conv2D_SVC_InterFusion_Joint():
    print("加载模型")
    dictionary = torch.load(
        "/sdb/user0/SCW/checkpoint/MIFFNet_Joint-step(1s)-window(3s)-inlength(7s)-model(lightresmultisernet2|resnet50)-batch(8)-batch_delay(1)-epoch(200)-lr(0.00025)-42-8-200-Audio_Video_Joint_Classification-AdamW0.00025-IEMOCAP/000150.ckpt",
        map_location=torch.device("cpu"),
    )
    data = IEMOCAP(
        root="/sdb/user0/SCW/data/IEMOCAP/Feature/IEMOCAP-pad-MFCC-KeyFrameGetterBasedIntervalSampling-Permute_Channel-.npy",
        mode="av",
        videomode="crop",
        cascPATH="/home/user0/SCW/torchTraining/utils/haarcascade_frontalface_alt2.xml",
        filter={
            "replace": {"label": {"excited": "happy"}},
            "dropna": {
                "label": [
                    "other",
                    "xxx",
                    "frustrated",
                    "disgusted",
                    "fearful",
                    "surprised",
                ],
            },
            "query": {"duration": "duration>22.0"},
        },
    )
    indices = [i for i in range(len(data)) if i in dictionary["train_indices"]]
    # data = Subset(data, indices=[0, 1, 5, 6, 9, 10, 11, 12, 13, 14, 15])
    print(len(data))
    dataloader = torch.utils.data.DataLoader(
        data, batch_size=1, shuffle=False, num_workers=0
    )

    net = MIFFNet_Conv2D_SVC_InterFusion_Joint(
        num_classes=4,
        input_size=[[189, 40], [150, 150]],
        model_name=["lightmultisernet", "resnet50"],
        last_hidden_dim=[320, 320],
        seq_len=[189, 30],
        enable_classifier=True,
    )
    weights_dict = {}
    for k, v in dictionary["net"].items():
        new_k = k.replace("module.", "") if "module" in k else k
        weights_dict[new_k] = v

    net.load_state_dict(weights_dict)

    plot_rt(
        net,
        dataloader,
        outdir="./output/real-time/",
        info="MIFFNet_Conv2D_SVC_InterFusion_Joint",
        device=device,
    )


def plot_rt_VideoNet_Conv2D():
    print("加载模型")
    dictionary = torch.load(
        "/sdb/user0/SCW/checkpoint/VideoNet_Conv2D-inlength(7s)-model(resnet50)-batch(8)-batch_delay(1)-epoch(200)-lr(0.00025)-42-8-200-Video_Classification-AdamW0.00025-IEMOCAP/000152.ckpt",
        map_location=torch.device("cpu"),
    )

    data = IEMOCAP(
        root="/sdb/user0/SCW/data/IEMOCAP/Feature/IEMOCAP-pad-MFCC-KeyFrameGetterBasedIntervalSampling-Permute_Channel-.npy",
        mode="v",
        videomode="crop",
        cascPATH="/home/user0/SCW/torchTraining/utils/haarcascade_frontalface_alt2.xml",
        filter={
            "replace": {"label": {"excited": "happy"}},
            "dropna": {
                "label": [
                    "other",
                    "xxx",
                    "frustrated",
                    "disgusted",
                    "fearful",
                    "surprised",
                ],
            },
            "query": {"duration": "duration>20.0"},
        },
    )

    print(len(data))
    dataloader = torch.utils.data.DataLoader(
        data, batch_size=1, shuffle=False, num_workers=0
    )

    net = MER_VideoNet(
        num_classes=4,
        model_name="resnet50",
        last_hidden_dim=320,
        enable_classifier=True,
    )

    weights_dict = {}
    for k, v in dictionary["net"].items():
        new_k = k.replace("module.", "") if "module" in k else k
        weights_dict[new_k] = v

    net.load_state_dict(weights_dict)

    plot_rt(
        net,
        dataloader,
        outdir="./output/real-time/",
        info="VideoNet_Conv2D",
        device=device,
    )


def plot_rt_AudioNet():
    print("加载模型")

    dictionary = torch.load(
        "/sdb/user0/SCW/checkpoint/AudioNet-inlength(7s)-model(lightresmultisernet2)-batch(64)-batch_delay(1)-epoch(200)-lr(0.00025)-42-64-200-Audio_Classification-AdamW0.00025-IEMOCAP/000131.ckpt",
        map_location=torch.device("cpu"),
    )

    data = IEMOCAP(
        root="/sdb/user0/SCW/data/IEMOCAP/Feature/IEMOCAP-pad-audio_MFCC_Permute_Channel.npy",
        mode="a",
        videomode="crop",
        cascPATH="/home/user0/SCW/torchTraining/utils/haarcascade_frontalface_alt2.xml",
        filter={
            "replace": {"label": {"excited": "happy"}},
            "dropna": {
                "label": [
                    "other",
                    "xxx",
                    "frustrated",
                    "disgusted",
                    "fearful",
                    "surprised",
                ],
            },
            "query": {"duration": "duration>20.0"},
        },
    )

    print(len(data))
    dataloader = torch.utils.data.DataLoader(
        data, batch_size=1, shuffle=False, num_workers=0
    )

    net = MER_AudioNet(
        num_classes=4,
        model_name="lightresmultisernet2",
        last_hidden_dim=320,
        enable_classifier=True,
    )

    weights_dict = {}
    for k, v in dictionary["net"].items():
        new_k = k.replace("module.", "") if "module" in k else k
        weights_dict[new_k] = v

    net.load_state_dict(weights_dict)

    plot_rt(
        net, dataloader, outdir="./output/real-time/", info="AudioNet", device=device
    )


def plot_tsne_MIFFNet_Conv2D_InterFusion_Joint_all():
    print("加载模型")
    dictionary = torch.load(
        "/sdb/user0/SCW/checkpoint/MIFFNet_Conv2D_InterFusion_Joint_Conv2D-inlength(7s)-model(lightresmultisernet2|resnet50)-batch(8)-batch_delay(1)-epoch(200)-lr(0.00025)-42-8-200-Audio_Video_Joint_Classification-AdamW0.00025-IEMOCAP//000194.ckpt",
        map_location=torch.device("cpu"),
    )

    data = IEMOCAP(
        root="/sdb/user0/SCW/data/IEMOCAP/Feature/IEMOCAP-crop-7s-AVSplit-MFCC-KeyFrameGetterBasedIntervalSampling-Permute_Channel-.npy",
        mode="av",
        videomode="crop",
        cascPATH="/home/user0/SCW/torchTraining/utils/haarcascade_frontalface_alt2.xml",
        filter={
            "replace": {"label": {"excited": "happy"}},
            "dropna": {
                "label": [
                    "other",
                    "xxx",
                    "frustrated",
                    "disgusted",
                    "fearful",
                    "surprised",
                ],
            },
        },
    )
    # data = Subset(data, indices=dictionary["train_indices"])

    print(len(data))
    dataloader = torch.utils.data.DataLoader(
        data, batch_size=32, shuffle=False, num_workers=0
    )

    net = MIFFNet_Conv2D_InterFusion_Joint(
        num_classes=4,
        input_size=[[189, 40], [150, 150]],
        model_name=["lightmultisernet", "resnet50"],
        last_hidden_dim=[320, 320],
        seq_len=[189, 30],
        enable_classifier=False,
    )

    weights_dict = {}
    for k, v in dictionary["net"].items():
        new_k = k.replace("module.", "") if "module" in k else k
        weights_dict[new_k] = v

    net.load_state_dict(weights_dict)

    plot_tsne(
        net,
        dataloader,
        outdir="./output/t-sne/",
        info="MIFFNet_Conv2D_InterFusion_Joint",
        device=device,
    )


def plot_tsne_MIFFNet_Conv2D_SVC_InterFusion_Joint_all():
    print("加载模型")
    dictionary = torch.load(
        "/sdb/user0/SCW/checkpoint/MIFFNet_Joint-step(1s)-window(3s)-inlength(7s)-model(lightresmultisernet2|resnet50)-batch(8)-batch_delay(1)-epoch(200)-lr(0.00025)-42-8-200-Audio_Video_Joint_Classification-AdamW0.00025-IEMOCAP/000150.ckpt",
        map_location=torch.device("cpu"),
    )

    data = IEMOCAP(
        root="/sdb/user0/SCW/data/IEMOCAP/Feature/IEMOCAP-crop-7s-AVSplit-MFCC-KeyFrameGetterBasedIntervalSampling-Permute_Channel-.npy",
        mode="av",
        videomode="crop",
        cascPATH="/home/user0/SCW/torchTraining/utils/haarcascade_frontalface_alt2.xml",
        filter={
            "replace": {"label": {"excited": "happy"}},
            "dropna": {
                "label": [
                    "other",
                    "xxx",
                    "frustrated",
                    "disgusted",
                    "fearful",
                    "surprised",
                ],
            },
        },
    )
    # data = Subset(data, indices=dictionary["train_indices"])

    print(len(data))
    dataloader = torch.utils.data.DataLoader(
        data, batch_size=32, shuffle=False, num_workers=0
    )

    net = MIFFNet_Conv2D_SVC_InterFusion_Joint(
        num_classes=4,
        input_size=[[189, 40], [150, 150]],
        model_name=["lightmultisernet", "resnet50"],
        last_hidden_dim=[320, 320],
        seq_len=[189, 30],
        enable_classifier=False,
    )

    weights_dict = {}
    for k, v in dictionary["net"].items():
        new_k = k.replace("module.", "") if "module" in k else k
        weights_dict[new_k] = v

    net.load_state_dict(weights_dict)

    plot_tsne(
        net,
        dataloader,
        outdir="./output/t-sne/",
        info="MIFFNet_Conv2D_SVC_InterFusion_Joint",
        device=device,
    )


def plot_tsne_VideoNet_Conv2D_all():
    print("加载模型")
    dictionary = torch.load(
        "/sdb/user0/SCW/checkpoint/VideoNet_Conv2D-inlength(7s)-model(resnet50)-batch(8)-batch_delay(1)-epoch(200)-lr(0.00025)-42-8-200-Video_Classification-AdamW0.00025-IEMOCAP/000152.ckpt",
        map_location=torch.device("cpu"),
    )

    data = IEMOCAP(
        root="/sdb/user0/SCW/data/IEMOCAP/Feature/IEMOCAP-crop-7s-AVSplit-MFCC-KeyFrameGetterBasedIntervalSampling-Permute_Channel-.npy",
        mode="v",
        videomode="crop",
        cascPATH="/home/user0/SCW/torchTraining/utils/haarcascade_frontalface_alt2.xml",
        filter={
            "replace": {"label": {"excited": "happy"}},
            "dropna": {
                "label": [
                    "other",
                    "xxx",
                    "frustrated",
                    "disgusted",
                    "fearful",
                    "surprised",
                ],
            },
        },
    )
    # data = Subset(data, indices=dictionary["train_indices"])

    print(len(data))
    dataloader = torch.utils.data.DataLoader(
        data, batch_size=32, shuffle=False, num_workers=0
    )

    net = MER_VideoNet(
        num_classes=4,
        model_name="resnet50",
        last_hidden_dim=320,
        enable_classifier=False,
    )

    weights_dict = {}
    for k, v in dictionary["net"].items():
        new_k = k.replace("module.", "") if "module" in k else k
        weights_dict[new_k] = v

    net.load_state_dict(weights_dict)

    plot_tsne(
        net, dataloader, outdir="./output/t-sne/", info="VideoNet_Conv2D", device=device
    )


def plot_tsne_AudioNet_all():
    print("加载模型")

    dictionary = torch.load(
        "/sdb/user0/SCW/checkpoint/AudioNet-inlength(7s)-model(lightresmultisernet2)-batch(64)-batch_delay(1)-epoch(200)-lr(0.00025)-42-64-200-Audio_Classification-AdamW0.00025-IEMOCAP/000131.ckpt",
        map_location=torch.device("cpu"),
    )

    data = IEMOCAP(
        root="/sdb/user0/SCW/data/IEMOCAP/Feature/IEMOCAP-7s-audio_AudioSplit-audio_MFCC_Permute_Channel-.npy",
        mode="a",
        videomode="crop",
        cascPATH="/home/user0/SCW/torchTraining/utils/haarcascade_frontalface_alt2.xml",
        filter={
            "replace": {"label": {"excited": "happy"}},
            "dropna": {
                "label": [
                    "other",
                    "xxx",
                    "frustrated",
                    "disgusted",
                    "fearful",
                    "surprised",
                ],
            },
        },
    )
    # data = Subset(data, indices=dictionary["train_indices"])

    print(len(data))
    dataloader = torch.utils.data.DataLoader(
        data, batch_size=64, shuffle=False, num_workers=0
    )

    net = MER_AudioNet(
        num_classes=4,
        model_name="lightresmultisernet2",
        last_hidden_dim=320,
        enable_classifier=False,
    )

    weights_dict = {}
    for k, v in dictionary["net"].items():
        new_k = k.replace("module.", "") if "module" in k else k
        weights_dict[new_k] = v

    net.load_state_dict(weights_dict)

    plot_tsne(net, dataloader, outdir="./output/t-sne/", info="AudioNet", device=device)


def plot_tsne_MIFFNet_Conv2D_InterFusion_Joint_train():
    print("加载模型")
    dictionary = torch.load(
        "/sdb/user0/SCW/checkpoint/MIFFNet_Conv2D_InterFusion_Joint_Conv2D-inlength(7s)-model(lightresmultisernet2|resnet50)-batch(8)-batch_delay(1)-epoch(200)-lr(0.00025)-42-8-200-Audio_Video_Joint_Classification-AdamW0.00025-IEMOCAP//000194.ckpt",
        map_location=torch.device("cpu"),
    )

    data = IEMOCAP(
        root="/sdb/user0/SCW/data/IEMOCAP/Feature/IEMOCAP-crop-7s-AVSplit-MFCC-KeyFrameGetterBasedIntervalSampling-Permute_Channel-.npy",
        mode="av",
        videomode="crop",
        cascPATH="/home/user0/SCW/torchTraining/utils/haarcascade_frontalface_alt2.xml",
        filter={
            "replace": {"label": {"excited": "happy"}},
            "dropna": {
                "label": [
                    "other",
                    "xxx",
                    "frustrated",
                    "disgusted",
                    "fearful",
                    "surprised",
                ],
            },
        },
    )
    data = Subset(data, indices=dictionary["train_indices"])

    print(len(data))
    dataloader = torch.utils.data.DataLoader(
        data, batch_size=32, shuffle=False, num_workers=0
    )

    net = MIFFNet_Conv2D_InterFusion_Joint(
        num_classes=4,
        input_size=[[189, 40], [150, 150]],
        model_name=["lightmultisernet", "resnet50"],
        last_hidden_dim=[320, 320],
        seq_len=[189, 30],
        enable_classifier=False,
    )

    weights_dict = {}
    for k, v in dictionary["net"].items():
        new_k = k.replace("module.", "") if "module" in k else k
        weights_dict[new_k] = v

    net.load_state_dict(weights_dict)

    plot_tsne(
        net,
        dataloader,
        outdir="./output/t-sne/",
        info="MIFFNet_Conv2D_InterFusion_Joint_train_",
        device=device,
    )


def plot_tsne_MIFFNet_Conv2D_SVC_InterFusion_Joint_train():
    print("加载模型")
    dictionary = torch.load(
        "/sdb/user0/SCW/checkpoint/MIFFNet_Joint-step(1s)-window(3s)-inlength(7s)-model(lightresmultisernet2|resnet50)-batch(8)-batch_delay(1)-epoch(200)-lr(0.00025)-42-8-200-Audio_Video_Joint_Classification-AdamW0.00025-IEMOCAP/000150.ckpt",
        map_location=torch.device("cpu"),
    )

    data = IEMOCAP(
        root="/sdb/user0/SCW/data/IEMOCAP/Feature/IEMOCAP-crop-7s-AVSplit-MFCC-KeyFrameGetterBasedIntervalSampling-Permute_Channel-.npy",
        mode="av",
        videomode="crop",
        cascPATH="/home/user0/SCW/torchTraining/utils/haarcascade_frontalface_alt2.xml",
        filter={
            "replace": {"label": {"excited": "happy"}},
            "dropna": {
                "label": [
                    "other",
                    "xxx",
                    "frustrated",
                    "disgusted",
                    "fearful",
                    "surprised",
                ],
            },
        },
    )
    data = Subset(data, indices=dictionary["train_indices"])

    print(len(data))
    dataloader = torch.utils.data.DataLoader(
        data, batch_size=32, shuffle=False, num_workers=0
    )

    net = MIFFNet_Conv2D_SVC_InterFusion_Joint(
        num_classes=4,
        input_size=[[189, 40], [150, 150]],
        model_name=["lightmultisernet", "resnet50"],
        last_hidden_dim=[320, 320],
        seq_len=[189, 30],
        enable_classifier=False,
    )

    weights_dict = {}
    for k, v in dictionary["net"].items():
        new_k = k.replace("module.", "") if "module" in k else k
        weights_dict[new_k] = v

    net.load_state_dict(weights_dict)

    plot_tsne(
        net,
        dataloader,
        outdir="./output/t-sne/",
        info="MIFFNet_Conv2D_SVC_InterFusion_Joint_train_",
        device=device,
    )


def plot_tsne_VideoNet_Conv2D_train():
    print("加载模型")
    dictionary = torch.load(
        "/sdb/user0/SCW/checkpoint/VideoNet_Conv2D-inlength(7s)-model(resnet50)-batch(8)-batch_delay(1)-epoch(200)-lr(0.00025)-42-8-200-Video_Classification-AdamW0.00025-IEMOCAP/000152.ckpt",
        map_location=torch.device("cpu"),
    )

    data = IEMOCAP(
        root="/sdb/user0/SCW/data/IEMOCAP/Feature/IEMOCAP-crop-7s-AVSplit-MFCC-KeyFrameGetterBasedIntervalSampling-Permute_Channel-.npy",
        mode="v",
        videomode="crop",
        cascPATH="/home/user0/SCW/torchTraining/utils/haarcascade_frontalface_alt2.xml",
        filter={
            "replace": {"label": {"excited": "happy"}},
            "dropna": {
                "label": [
                    "other",
                    "xxx",
                    "frustrated",
                    "disgusted",
                    "fearful",
                    "surprised",
                ],
            },
        },
    )
    data = Subset(data, indices=dictionary["train_indices"])

    print(len(data))
    dataloader = torch.utils.data.DataLoader(
        data, batch_size=32, shuffle=False, num_workers=0
    )

    net = MER_VideoNet(
        num_classes=4,
        model_name="resnet50",
        last_hidden_dim=320,
        enable_classifier=False,
    )

    weights_dict = {}
    for k, v in dictionary["net"].items():
        new_k = k.replace("module.", "") if "module" in k else k
        weights_dict[new_k] = v

    net.load_state_dict(weights_dict)

    plot_tsne(
        net,
        dataloader,
        outdir="./output/t-sne/",
        info="VideoNet_Conv2D_train_",
        device=device,
    )


def plot_tsne_AudioNet_train():
    print("加载模型")

    dictionary = torch.load(
        "/sdb/user0/SCW/checkpoint/AudioNet-inlength(7s)-model(lightresmultisernet2)-batch(64)-batch_delay(1)-epoch(200)-lr(0.00025)-42-64-200-Audio_Classification-AdamW0.00025-IEMOCAP/000131.ckpt",
        map_location=torch.device("cpu"),
    )

    data = IEMOCAP(
        root="/sdb/user0/SCW/data/IEMOCAP/Feature/IEMOCAP-7s-audio_AudioSplit-audio_MFCC_Permute_Channel-.npy",
        mode="a",
        videomode="crop",
        cascPATH="/home/user0/SCW/torchTraining/utils/haarcascade_frontalface_alt2.xml",
        filter={
            "replace": {"label": {"excited": "happy"}},
            "dropna": {
                "label": [
                    "other",
                    "xxx",
                    "frustrated",
                    "disgusted",
                    "fearful",
                    "surprised",
                ],
            },
        },
    )
    data = Subset(data, indices=dictionary["train_indices"])

    print(len(data))
    dataloader = torch.utils.data.DataLoader(
        data, batch_size=64, shuffle=False, num_workers=0
    )

    net = MER_AudioNet(
        num_classes=4,
        model_name="lightresmultisernet2",
        last_hidden_dim=320,
        enable_classifier=False,
    )

    weights_dict = {}
    for k, v in dictionary["net"].items():
        new_k = k.replace("module.", "") if "module" in k else k
        weights_dict[new_k] = v

    net.load_state_dict(weights_dict)

    plot_tsne(
        net, dataloader, outdir="./output/t-sne/", info="AudioNet_train_", device=device
    )


def plot_tsne_MIFFNet_Conv2D_InterFusion_Joint_test():
    print("加载模型")
    dictionary = torch.load(
        "/sdb/user0/SCW/checkpoint/MIFFNet_Conv2D_InterFusion_Joint_Conv2D-inlength(7s)-model(lightresmultisernet2|resnet50)-batch(8)-batch_delay(1)-epoch(200)-lr(0.00025)-42-8-200-Audio_Video_Joint_Classification-AdamW0.00025-IEMOCAP//000194.ckpt",
        map_location=torch.device("cpu"),
    )

    data = IEMOCAP(
        root="/sdb/user0/SCW/data/IEMOCAP/Feature/IEMOCAP-crop-7s-AVSplit-MFCC-KeyFrameGetterBasedIntervalSampling-Permute_Channel-.npy",
        mode="av",
        videomode="crop",
        cascPATH="/home/user0/SCW/torchTraining/utils/haarcascade_frontalface_alt2.xml",
        filter={
            "replace": {"label": {"excited": "happy"}},
            "dropna": {
                "label": [
                    "other",
                    "xxx",
                    "frustrated",
                    "disgusted",
                    "fearful",
                    "surprised",
                ],
            },
        },
    )
    data = Subset(data, indices=dictionary["test_indices"])

    print(len(data))
    dataloader = torch.utils.data.DataLoader(
        data, batch_size=32, shuffle=False, num_workers=0
    )

    net = MIFFNet_Conv2D_InterFusion_Joint(
        num_classes=4,
        input_size=[[189, 40], [150, 150]],
        model_name=["lightmultisernet", "resnet50"],
        last_hidden_dim=[320, 320],
        seq_len=[189, 30],
        enable_classifier=False,
    )

    weights_dict = {}
    for k, v in dictionary["net"].items():
        new_k = k.replace("module.", "") if "module" in k else k
        weights_dict[new_k] = v

    net.load_state_dict(weights_dict)

    plot_tsne(
        net,
        dataloader,
        outdir="./output/t-sne/",
        info="MIFFNet_Conv2D_InterFusion_Joint_test_",
        device=device,
    )


def plot_tsne_MIFFNet_Conv2D_SVC_InterFusion_Joint_test():
    print("加载模型")
    dictionary = torch.load(
        "/sdb/user0/SCW/checkpoint/MIFFNet_Joint-step(1s)-window(3s)-inlength(7s)-model(lightresmultisernet2|resnet50)-batch(8)-batch_delay(1)-epoch(200)-lr(0.00025)-42-8-200-Audio_Video_Joint_Classification-AdamW0.00025-IEMOCAP/000150.ckpt",
        map_location=torch.device("cpu"),
    )

    data = IEMOCAP(
        root="/sdb/user0/SCW/data/IEMOCAP/Feature/IEMOCAP-crop-7s-AVSplit-MFCC-KeyFrameGetterBasedIntervalSampling-Permute_Channel-.npy",
        mode="av",
        videomode="crop",
        cascPATH="/home/user0/SCW/torchTraining/utils/haarcascade_frontalface_alt2.xml",
        filter={
            "replace": {"label": {"excited": "happy"}},
            "dropna": {
                "label": [
                    "other",
                    "xxx",
                    "frustrated",
                    "disgusted",
                    "fearful",
                    "surprised",
                ],
            },
        },
    )
    data = Subset(data, indices=dictionary["test_indices"])

    print(len(data))
    dataloader = torch.utils.data.DataLoader(
        data, batch_size=32, shuffle=False, num_workers=0
    )

    net = MIFFNet_Conv2D_SVC_InterFusion_Joint(
        num_classes=4,
        input_size=[[189, 40], [150, 150]],
        model_name=["lightmultisernet", "resnet50"],
        last_hidden_dim=[320, 320],
        seq_len=[189, 30],
        enable_classifier=False,
    )

    weights_dict = {}
    for k, v in dictionary["net"].items():
        new_k = k.replace("module.", "") if "module" in k else k
        weights_dict[new_k] = v

    net.load_state_dict(weights_dict)

    plot_tsne(
        net,
        dataloader,
        outdir="./output/t-sne/",
        info="MIFFNet_Conv2D_SVC_InterFusion_Joint_test_",
        device=device,
    )


def plot_tsne_VideoNet_Conv2D_test():
    print("加载模型")
    dictionary = torch.load(
        "/sdb/user0/SCW/checkpoint/VideoNet_Conv2D-inlength(7s)-model(resnet50)-batch(8)-batch_delay(1)-epoch(200)-lr(0.00025)-42-8-200-Video_Classification-AdamW0.00025-IEMOCAP/000152.ckpt",
        map_location=torch.device("cpu"),
    )

    data = IEMOCAP(
        root="/sdb/user0/SCW/data/IEMOCAP/Feature/IEMOCAP-crop-7s-AVSplit-MFCC-KeyFrameGetterBasedIntervalSampling-Permute_Channel-.npy",
        mode="v",
        videomode="crop",
        cascPATH="/home/user0/SCW/torchTraining/utils/haarcascade_frontalface_alt2.xml",
        filter={
            "replace": {"label": {"excited": "happy"}},
            "dropna": {
                "label": [
                    "other",
                    "xxx",
                    "frustrated",
                    "disgusted",
                    "fearful",
                    "surprised",
                ],
            },
        },
    )
    data = Subset(data, indices=dictionary["test_indices"])

    print(len(data))
    dataloader = torch.utils.data.DataLoader(
        data, batch_size=32, shuffle=False, num_workers=0
    )

    net = MER_VideoNet(
        num_classes=4,
        model_name="resnet50",
        last_hidden_dim=320,
        enable_classifier=False,
    )

    weights_dict = {}
    for k, v in dictionary["net"].items():
        new_k = k.replace("module.", "") if "module" in k else k
        weights_dict[new_k] = v

    net.load_state_dict(weights_dict)

    plot_tsne(
        net,
        dataloader,
        outdir="./output/t-sne/",
        info="VideoNet_Conv2D_test_",
        device=device,
    )


def plot_tsne_AudioNet_test():
    print("加载模型")

    dictionary = torch.load(
        "/sdb/user0/SCW/checkpoint/AudioNet-inlength(7s)-model(lightresmultisernet2)-batch(64)-batch_delay(1)-epoch(200)-lr(0.00025)-42-64-200-Audio_Classification-AdamW0.00025-IEMOCAP/000131.ckpt",
        map_location=torch.device("cpu"),
    )

    data = IEMOCAP(
        root="/sdb/user0/SCW/data/IEMOCAP/Feature/IEMOCAP-7s-audio_AudioSplit-audio_MFCC_Permute_Channel-.npy",
        mode="a",
        videomode="crop",
        cascPATH="/home/user0/SCW/torchTraining/utils/haarcascade_frontalface_alt2.xml",
        filter={
            "replace": {"label": {"excited": "happy"}},
            "dropna": {
                "label": [
                    "other",
                    "xxx",
                    "frustrated",
                    "disgusted",
                    "fearful",
                    "surprised",
                ],
            },
        },
    )
    data = Subset(data, indices=dictionary["test_indices"])

    print(len(data))
    dataloader = torch.utils.data.DataLoader(
        data, batch_size=64, shuffle=False, num_workers=0
    )

    net = MER_AudioNet(
        num_classes=4,
        model_name="lightresmultisernet2",
        last_hidden_dim=320,
        enable_classifier=False,
    )

    weights_dict = {}
    for k, v in dictionary["net"].items():
        new_k = k.replace("module.", "") if "module" in k else k
        weights_dict[new_k] = v

    net.load_state_dict(weights_dict)

    plot_tsne(
        net, dataloader, outdir="./output/t-sne/", info="AudioNet_test_", device=device
    )


def plot_tsne_MIFFNet_Conv2D_SVC_InterFusion_Joint_AVA_test():
    print("加载模型")
    dictionary = torch.load(
        "/sdb/user0/SCW/checkpoint/MIFFNet_Joint-step(1s)-window(3s)-inlength(7s)-model(lightresmultisernet2|resnet50)-batch(8)-batch_delay(1)-epoch(200)-lr(0.00025)-42-8-200-Audio_Video_Joint_Classification-AdamW0.00025-IEMOCAP/000150.ckpt",
        map_location=torch.device("cpu"),
    )

    data = IEMOCAP(
        root="/sdb/user0/SCW/data/IEMOCAP/Feature/IEMOCAP-crop-7s-AVSplit-MFCC-KeyFrameGetterBasedIntervalSampling-Permute_Channel-.npy",
        mode="av",
        videomode="crop",
        cascPATH="/home/user0/SCW/torchTraining/utils/haarcascade_frontalface_alt2.xml",
        filter={
            "replace": {"label": {"excited": "happy"}},
            "dropna": {
                "label": [
                    "other",
                    "xxx",
                    "frustrated",
                    "disgusted",
                    "fearful",
                    "surprised",
                ],
            },
        },
    )
    data = Subset(data, indices=dictionary["test_indices"])

    print(len(data))
    dataloader = torch.utils.data.DataLoader(
        data, batch_size=32, shuffle=False, num_workers=0
    )

    net = MIFFNet_Conv2D_SVC_InterFusion_Joint(
        num_classes=4,
        input_size=[[189, 40], [150, 150]],
        model_name=["lightmultisernet", "resnet50"],
        last_hidden_dim=[320, 320],
        seq_len=[189, 30],
        enable_classifier=False,
    )

    weights_dict = {}
    for k, v in dictionary["net"].items():
        new_k = k.replace("module.", "") if "module" in k else k
        weights_dict[new_k] = v

    net.load_state_dict(weights_dict)

    plot_tsne(
        net,
        dataloader,
        outdir="./output/t-sne/",
        info="MIFFNet_Conv2D_SVC_InterFusion_Joint_AVA_test",
        device=device,
    )


def plot_tsne_MIFFNet_Conv2D_SVC_InterFusion_Joint_AVA_all():
    print("加载模型")
    dictionary = torch.load(
        "/sdb/user0/SCW/checkpoint/MIFFNet_Joint-step(1s)-window(3s)-inlength(7s)-model(lightresmultisernet2|resnet50)-batch(8)-batch_delay(1)-epoch(200)-lr(0.00025)-42-8-200-Audio_Video_Joint_Classification-AdamW0.00025-IEMOCAP/000150.ckpt",
        map_location=torch.device("cpu"),
    )

    data = IEMOCAP(
        root="/sdb/user0/SCW/data/IEMOCAP/Feature/IEMOCAP-crop-7s-AVSplit-MFCC-KeyFrameGetterBasedIntervalSampling-Permute_Channel-.npy",
        mode="av",
        videomode="crop",
        cascPATH="/home/user0/SCW/torchTraining/utils/haarcascade_frontalface_alt2.xml",
        filter={
            "replace": {"label": {"excited": "happy"}},
            "dropna": {
                "label": [
                    "other",
                    "xxx",
                    "frustrated",
                    "disgusted",
                    "fearful",
                    "surprised",
                ],
            },
        },
    )
    # data = Subset(data, indices=dictionary["test_indices"])

    print(len(data))
    dataloader = torch.utils.data.DataLoader(
        data, batch_size=32, shuffle=False, num_workers=0
    )

    net = MIFFNet_Conv2D_SVC_InterFusion_Joint(
        num_classes=4,
        input_size=[[189, 40], [150, 150]],
        model_name=["lightmultisernet", "resnet50"],
        last_hidden_dim=[320, 320],
        seq_len=[189, 30],
        enable_classifier=False,
    )

    weights_dict = {}
    for k, v in dictionary["net"].items():
        new_k = k.replace("module.", "") if "module" in k else k
        weights_dict[new_k] = v

    net.load_state_dict(weights_dict)

    plot_tsne(
        net,
        dataloader,
        outdir="./output/t-sne/",
        info="MIFFNet_Conv2D_SVC_InterFusion_Joint_AVA_all",
        device=device,
    )


def plot_tsne_MIFFNet_Conv2D_SVC_InterFusion_Joint_AVA_train():
    print("加载模型")
    dictionary = torch.load(
        "/sdb/user0/SCW/checkpoint/MIFFNet_Joint-step(1s)-window(3s)-inlength(7s)-model(lightresmultisernet2|resnet50)-batch(8)-batch_delay(1)-epoch(200)-lr(0.00025)-42-8-200-Audio_Video_Joint_Classification-AdamW0.00025-IEMOCAP/000150.ckpt",
        map_location=torch.device("cpu"),
    )

    data = IEMOCAP(
        root="/sdb/user0/SCW/data/IEMOCAP/Feature/IEMOCAP-crop-7s-AVSplit-MFCC-KeyFrameGetterBasedIntervalSampling-Permute_Channel-.npy",
        mode="av",
        videomode="crop",
        cascPATH="/home/user0/SCW/torchTraining/utils/haarcascade_frontalface_alt2.xml",
        filter={
            "replace": {"label": {"excited": "happy"}},
            "dropna": {
                "label": [
                    "other",
                    "xxx",
                    "frustrated",
                    "disgusted",
                    "fearful",
                    "surprised",
                ],
            },
        },
    )
    data = Subset(data, indices=dictionary["train_indices"])

    print(len(data))
    dataloader = torch.utils.data.DataLoader(
        data, batch_size=32, shuffle=False, num_workers=0
    )

    net = MIFFNet_Conv2D_SVC_InterFusion_Joint(
        num_classes=4,
        input_size=[[189, 40], [150, 150]],
        model_name=["lightmultisernet", "resnet50"],
        last_hidden_dim=[320, 320],
        seq_len=[189, 30],
        enable_classifier=False,
    )

    weights_dict = {}
    for k, v in dictionary["net"].items():
        new_k = k.replace("module.", "") if "module" in k else k
        weights_dict[new_k] = v

    net.load_state_dict(weights_dict)

    plot_tsne(
        net,
        dataloader,
        outdir="./output/t-sne/",
        info="MIFFNet_Conv2D_SVC_InterFusion_Joint_AVA_train",
        device=device,
    )


def plot_tsne_MIFFNet_Conv2D_SVC_InterFusion_Joint_Attention_train():
    print("加载模型")
    dictionary = torch.load(
        "/sdb/user0/SCW/checkpoint/MIFFNet_Conv2D_SVC_InterFusion_Joint_Attention-inlength(7s)-model(lightresmultisernet2|resnet50)-batch(8)-batch_delay(1)-epoch(200)-lr(0.00025)-42-10-200-Audio_Video_Joint_Classification-AdamW0.00025-IEMOCAP/000150.ckpt",
        map_location=torch.device("cpu"),
    )

    data = IEMOCAP(
        root="/sdb/user0/SCW/data/IEMOCAP/Feature/IEMOCAP-crop-7s-AVSplit-MFCC-KeyFrameGetterBasedIntervalSampling-Permute_Channel-.npy",
        mode="av",
        videomode="crop",
        cascPATH="/home/user0/SCW/torchTraining/utils/haarcascade_frontalface_alt2.xml",
        filter={
            "replace": {"label": {"excited": "happy"}},
            "dropna": {
                "label": [
                    "other",
                    "xxx",
                    "frustrated",
                    "disgusted",
                    "fearful",
                    "surprised",
                ],
            },
        },
    )
    data = Subset(data, indices=dictionary["train_indices"])

    print(len(data))
    dataloader = torch.utils.data.DataLoader(
        data, batch_size=32, shuffle=False, num_workers=0
    )

    net = MIFFNet_Conv2D_SVC_InterFusion_Joint_Attention(
        num_classes=4,
        input_size=[[189, 40], [150, 150]],
        model_name=["lightmultisernet", "resnet50"],
        last_hidden_dim=[320, 320],
        seq_len=[189, 30],
        enable_classifier=False,
    )

    weights_dict = {}
    for k, v in dictionary["net"].items():
        new_k = k.replace("module.", "") if "module" in k else k
        weights_dict[new_k] = v

    net.load_state_dict(weights_dict)

    plot_tsne(
        net,
        dataloader,
        outdir="./output/t-sne/",
        info="MIFFNet_Conv2D_SVC_InterFusion_Joint_Attention_train",
        device=device,
    )


def plot_tsne_MIFFNet_Conv2D_SVC_InterFusion_Joint_Attention_test():
    print("加载模型")
    dictionary = torch.load(
        "/sdb/user0/SCW/checkpoint/MIFFNet_Conv2D_SVC_InterFusion_Joint_Attention-inlength(7s)-model(lightresmultisernet2|resnet50)-batch(8)-batch_delay(1)-epoch(200)-lr(0.00025)-42-10-200-Audio_Video_Joint_Classification-AdamW0.00025-IEMOCAP/000150.ckpt",
        map_location=torch.device("cpu"),
    )

    data = IEMOCAP(
        root="/sdb/user0/SCW/data/IEMOCAP/Feature/IEMOCAP-crop-7s-AVSplit-MFCC-KeyFrameGetterBasedIntervalSampling-Permute_Channel-.npy",
        mode="av",
        videomode="crop",
        cascPATH="/home/user0/SCW/torchTraining/utils/haarcascade_frontalface_alt2.xml",
        filter={
            "replace": {"label": {"excited": "happy"}},
            "dropna": {
                "label": [
                    "other",
                    "xxx",
                    "frustrated",
                    "disgusted",
                    "fearful",
                    "surprised",
                ],
            },
        },
    )
    data = Subset(data, indices=dictionary["test_indices"])

    print(len(data))
    dataloader = torch.utils.data.DataLoader(
        data, batch_size=32, shuffle=False, num_workers=0
    )

    net = MIFFNet_Conv2D_SVC_InterFusion_Joint_Attention(
        num_classes=4,
        input_size=[[189, 40], [150, 150]],
        model_name=["lightmultisernet", "resnet50"],
        last_hidden_dim=[320, 320],
        seq_len=[189, 30],
        enable_classifier=False,
    )

    weights_dict = {}
    for k, v in dictionary["net"].items():
        new_k = k.replace("module.", "") if "module" in k else k
        weights_dict[new_k] = v

    net.load_state_dict(weights_dict)

    plot_tsne(
        net,
        dataloader,
        outdir="./output/t-sne/",
        info="MIFFNet_Conv2D_SVC_InterFusion_Joint_Attention_test",
        device=device,
    )


def plot_tsne_MIFFNet_Conv2D_SVC_InterFusion_Joint_Attention_all():
    print("加载模型")
    dictionary = torch.load(
        "/sdb/user0/SCW/checkpoint/MIFFNet_Conv2D_SVC_InterFusion_Joint_Attention-inlength(7s)-model(lightresmultisernet2|resnet50)-batch(8)-batch_delay(1)-epoch(200)-lr(0.00025)-42-10-200-Audio_Video_Joint_Classification-AdamW0.00025-IEMOCAP/000150.ckpt",
        map_location=torch.device("cpu"),
    )

    data = IEMOCAP(
        root="/sdb/user0/SCW/data/IEMOCAP/Feature/IEMOCAP-crop-7s-AVSplit-MFCC-KeyFrameGetterBasedIntervalSampling-Permute_Channel-.npy",
        mode="av",
        videomode="crop",
        cascPATH="/home/user0/SCW/torchTraining/utils/haarcascade_frontalface_alt2.xml",
        filter={
            "replace": {"label": {"excited": "happy"}},
            "dropna": {
                "label": [
                    "other",
                    "xxx",
                    "frustrated",
                    "disgusted",
                    "fearful",
                    "surprised",
                ],
            },
        },
    )
    # data = Subset(data, indices=dictionary["test_indices"])

    print(len(data))
    dataloader = torch.utils.data.DataLoader(
        data, batch_size=32, shuffle=False, num_workers=0
    )

    net = MIFFNet_Conv2D_SVC_InterFusion_Joint_Attention(
        num_classes=4,
        input_size=[[189, 40], [150, 150]],
        model_name=["lightmultisernet", "resnet50"],
        last_hidden_dim=[320, 320],
        seq_len=[189, 30],
        enable_classifier=False,
    )

    weights_dict = {}
    for k, v in dictionary["net"].items():
        new_k = k.replace("module.", "") if "module" in k else k
        weights_dict[new_k] = v

    net.load_state_dict(weights_dict)

    plot_tsne(
        net,
        dataloader,
        outdir="./output/t-sne/",
        info="MIFFNet_Conv2D_SVC_InterFusion_Joint_Attention_all",
        device=device,
    )


def plot_rt_MIFFNet_Conv2D_SVC_InterFusion_Joint_Attention():
    print("加载模型")
    dictionary = torch.load(
        "/sdb/user0/SCW/checkpoint/MIFFNet_Conv2D_SVC_InterFusion_Joint_Attention-inlength(7s)-model(lightresmultisernet2|resnet50)-batch(8)-batch_delay(1)-epoch(200)-lr(0.00025)-42-10-200-Audio_Video_Joint_Classification-AdamW0.00025-IEMOCAP/000150.ckpt",
        map_location=torch.device("cpu"),
    )

    data = IEMOCAP(
        root="/sdb/user0/SCW/data/IEMOCAP/Feature/IEMOCAP-pad-MFCC-KeyFrameGetterBasedIntervalSampling-Permute_Channel-.npy",
        mode="av",
        videomode="crop",
        cascPATH="/home/user0/SCW/torchTraining/utils/haarcascade_frontalface_alt2.xml",
        filter={
            "replace": {"label": {"excited": "happy"}},
            "dropna": {
                "label": [
                    "other",
                    "xxx",
                    "frustrated",
                    "disgusted",
                    "fearful",
                    "surprised",
                ],
            },
            "query": {"duration": "duration>22.0"},
        },
    )
    indices = [i for i in range(len(data)) if i in dictionary["train_indices"]]
    # data = Subset(data, indices=[0, 1, 5, 6, 9, 10, 11, 12, 13, 14, 15])
    print(len(data))
    dataloader = torch.utils.data.DataLoader(
        data, batch_size=1, shuffle=False, num_workers=0
    )

    net = MIFFNet_Conv2D_SVC_InterFusion_Joint_Attention(
        num_classes=4,
        input_size=[[189, 40], [150, 150]],
        model_name=["lightmultisernet", "resnet50"],
        last_hidden_dim=[320, 320],
        seq_len=[189, 30],
        enable_classifier=True,
    )
    weights_dict = {}
    for k, v in dictionary["net"].items():
        new_k = k.replace("module.", "") if "module" in k else k
        weights_dict[new_k] = v

    net.load_state_dict(weights_dict)

    plot_rt(
        net,
        dataloader,
        outdir="./output/real-time/",
        info="MIFFNet_Conv2D_SVC_InterFusion_Joint_Attention",
        device=device,
    )


def plot_confusionMatrix_MIFFNet_Conv2D_SVC_InterFusion_Joint_Attention():
    print("加载模型")
    dictionary = torch.load(
        "/sdb/user0/SCW/checkpoint/MIFFNet_Conv2D_SVC_InterFusion_Joint_Attention-inlength(7s)-model(lightresmultisernet2|resnet50)-batch(8)-batch_delay(1)-epoch(200)-lr(0.00025)-42-10-200-Audio_Video_Joint_Classification-AdamW0.00025-IEMOCAP/000150.ckpt",
        map_location=torch.device("cpu"),
    )

    data = IEMOCAP(
        root="/sdb/user0/SCW/data/IEMOCAP/Feature/IEMOCAP-crop-7s-AVSplit-MFCC-KeyFrameGetterBasedIntervalSampling-Permute_Channel-.npy",
        mode="av",
        videomode="crop",
        cascPATH="/home/user0/SCW/torchTraining/utils/haarcascade_frontalface_alt2.xml",
        filter={
            "replace": {"label": {"excited": "happy"}},
            "dropna": {
                "label": [
                    "other",
                    "xxx",
                    "frustrated",
                    "disgusted",
                    "fearful",
                    "surprised",
                ],
            },
            "sort_values": "duration",
        },
    )
    data = Subset(data, indices=dictionary["test_indices"])

    print(len(data))
    dataloader = torch.utils.data.DataLoader(
        data, batch_size=32, shuffle=False, num_workers=0
    )

    net = MIFFNet_Conv2D_SVC_InterFusion_Joint_Attention(
        num_classes=4,
        input_size=[[189, 40], [150, 150]],
        model_name=["lightmultisernet", "resnet50"],
        last_hidden_dim=[320, 320],
        seq_len=[189, 30],
        enable_classifier=True,
    )

    weights_dict = {}
    for k, v in dictionary["net"].items():
        new_k = k.replace("module.", "") if "module" in k else k
        weights_dict[new_k] = v

    net.load_state_dict(weights_dict)

    plot_confusionMatrix(
        net,
        dataloader,
        outdir="./output/confusionMatrix/",
        info="MIFFNet_Conv2D_SVC_InterFusion_Joint_Attention_all",
        device=device,
    )


"""

"""


def plot_similarity_MIFFNet_Conv2D_SVC_InterFusion_Joint_Attention():
    dataset = IEMOCAP(
        root="/sdb/user0/SCW/data/IEMOCAP/Feature/IEMOCAP-crop-7s-AVSplit-MFCC-KeyFrameGetterBasedIntervalSampling-Permute_Channel-.npy",
        mode="av",
        videomode="crop",
        cascPATH="/home/user0/SCW/torchTraining/utils/haarcascade_frontalface_alt2.xml",
        filter={
            "replace": {"label": {"excited": "happy"}},
            "dropna": {
                "label": [
                    "other",
                    "xxx",
                    "frustrated",
                    "disgusted",
                    "fearful",
                    "surprised",
                ],
            },
        },
    )

    print("加载模型")
    # 000001 | 000002 | 000003 | 000004 | 000005 | 000008 | 000012 | 000021 | 000027 | 000035 | 000038 | 000041 | 000057 | 000067 | 000077 | 000100 | 000144 | 000150 | 000199 |

    for i in [ 2, 3, 4, 12, 21, 35, 41, 57, 67, 77, 100, 144, 150]:
        dictionary = torch.load(
            "/sdb/user0/SCW/checkpoint/MIFFNet_Conv2D_SVC_InterFusion_Joint_Attention-inlength(7s)-model(lightresmultisernet2|resnet50)-batch(8)-batch_delay(1)-epoch(200)-lr(0.00025)-42-10-200-Audio_Video_Joint_Classification-AdamW0.00025-IEMOCAP/%06d.ckpt"
            % i,
            map_location=torch.device("cpu"),
        )

        weights_dict = {}
        for k, v in dictionary["net"].items():
            new_k = k.replace("module.", "") if "module" in k else k
            weights_dict[new_k] = v

        subDataset = Subset(dataset, indices=dictionary["train_indices"])

        dataloader = torch.utils.data.DataLoader(
            subDataset, batch_size=10, shuffle=False, num_workers=0
        )
        net = MIFFNet_Conv2D_SVC_InterFusion_Joint_Attention(
            num_classes=4,
            input_size=[[189, 40], [150, 150]],
            model_name=["lightmultisernet2", "resnet50"],
            last_hidden_dim=[320, 320],
            seq_len=[189, 30],
            enable_classifier=False,
        )
        net.load_state_dict(weights_dict)
        net.to(device)
        net.eval()
        plot_avc_tsne(model=net, dataloader=dataloader, outdir="./output/similarity/", info="MIFFNet_Conv2D_SVC_InterFusion_Joint_Attention_%06d" % i, device=device)


device = torch.device("cuda:1")
if __name__ == "__main__":
    # plot_rt_MIFFNet_Conv2D_InterFusion_Joint()
    # plot_rt_MIFFNet_Conv2D_SVC_InterFusion_Joint()
    # plot_rt_VideoNet_Conv2D()
    # plot_rt_AudioNet()
    # plot_tsne_MIFFNet_Conv2D_InterFusion_Joint_all()
    # plot_tsne_MIFFNet_Conv2D_SVC_InterFusion_Joint_all()
    # plot_tsne_VideoNet_Conv2D_all()
    # plot_tsne_AudioNet_all()
    # plot_tsne_MIFFNet_Conv2D_InterFusion_Joint_train()
    # plot_tsne_MIFFNet_Conv2D_SVC_InterFusion_Joint_train()
    # plot_tsne_VideoNet_Conv2D_train()
    # plot_tsne_AudioNet_train()
    # plot_tsne_MIFFNet_Conv2D_InterFusion_Joint_test()
    # plot_tsne_MIFFNet_Conv2D_SVC_InterFusion_Joint_test()
    # plot_tsne_VideoNet_Conv2D_test()
    # plot_tsne_AudioNet_test()
    # plot_tsne_MIFFNet_Conv2D_SVC_InterFusion_Joint_AVA_test()
    # plot_tsne_MIFFNet_Conv2D_SVC_InterFusion_Joint_AVA_all()
    # plot_tsne_MIFFNet_Conv2D_SVC_InterFusion_Joint_AVA_train()

    # plot_tsne_MIFFNet_Conv2D_SVC_InterFusion_Joint_Attention_train()
    # plot_tsne_MIFFNet_Conv2D_SVC_InterFusion_Joint_Attention_test()
    # plot_tsne_MIFFNet_Conv2D_SVC_InterFusion_Joint_Attention_all()
    # plot_rt_MIFFNet_Conv2D_SVC_InterFusion_Joint_Attention()
    # plot_confusionMatrix_MIFFNet_Conv2D_SVC_InterFusion_Joint_Attention()
    plot_similarity_MIFFNet_Conv2D_SVC_InterFusion_Joint_Attention()
