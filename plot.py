import os
import sys

if sys.platform.startswith("linux"):
    os.chdir("/home/visitors2/SCW/torchTraining")
elif sys.platform.startswith("win"):
    pass
from utils.visuallize import *
from custom_models import *
from custom_datasets import *
from custom_datasets.subset import Subset


def plot_rt_MIFFNet_Conv2D_InterFusion_Joint():
    print("加载模型")
    dictionary = torch.load(
        "/sdb/visitors2/SCW/checkpoint/MIFFNet_Conv2D_InterFusion_Joint_Conv2D-inlength(7s)-model(lightresmultisernet2|resnet50)-batch(8)-batch_delay(1)-epoch(200)-lr(0.00025)-42-8-200-Audio_Video_Joint_Classification-AdamW0.00025-IEMOCAP//000194.ckpt",
        map_location=torch.device("cpu"),
    )
    data = IEMOCAP(
        root="/sdb/visitors2/SCW/data/IEMOCAP/Feature/IEMOCAP-pad-MFCC-KeyFrameGetterBasedIntervalSampling-Permute_Channel-.npy",
        mode="av",
        videomode="crop",
        cascPATH="/home/visitors2/SCW/torchTraining/utils/haarcascade_frontalface_alt2.xml",
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
        "/sdb/visitors2/SCW/checkpoint/MIFFNet_Joint-step(1s)-window(3s)-inlength(7s)-model(lightresmultisernet2|resnet50)-batch(8)-batch_delay(1)-epoch(200)-lr(0.00025)-42-8-200-Audio_Video_Joint_Classification-AdamW0.00025-IEMOCAP/000150.ckpt",
        map_location=torch.device("cpu"),
    )
    data = IEMOCAP(
        root="/sdb/visitors2/SCW/data/IEMOCAP/Feature/IEMOCAP-pad-MFCC-KeyFrameGetterBasedIntervalSampling-Permute_Channel-.npy",
        mode="av",
        videomode="crop",
        cascPATH="/home/visitors2/SCW/torchTraining/utils/haarcascade_frontalface_alt2.xml",
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
        "/sdb/visitors2/SCW/checkpoint/VideoNet_Conv2D-inlength(7s)-model(resnet50)-batch(8)-batch_delay(1)-epoch(200)-lr(0.00025)-42-8-200-Video_Classification-AdamW0.00025-IEMOCAP/000152.ckpt",
        map_location=torch.device("cpu"),
    )

    data = IEMOCAP(
        root="/sdb/visitors2/SCW/data/IEMOCAP/Feature/IEMOCAP-pad-MFCC-KeyFrameGetterBasedIntervalSampling-Permute_Channel-.npy",
        mode="v",
        videomode="crop",
        cascPATH="/home/visitors2/SCW/torchTraining/utils/haarcascade_frontalface_alt2.xml",
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
        "/sdb/visitors2/SCW/checkpoint/AudioNet-inlength(7s)-model(lightresmultisernet2)-batch(64)-batch_delay(1)-epoch(200)-lr(0.00025)-42-64-200-Audio_Classification-AdamW0.00025-IEMOCAP/000131.ckpt",
        map_location=torch.device("cpu"),
    )

    data = IEMOCAP(
        root="/sdb/visitors2/SCW/data/IEMOCAP/Feature/IEMOCAP-pad-audio_MFCC_Permute_Channel.npy",
        mode="a",
        videomode="crop",
        cascPATH="/home/visitors2/SCW/torchTraining/utils/haarcascade_frontalface_alt2.xml",
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
        "/sdb/visitors2/SCW/checkpoint/MIFFNet_Conv2D_InterFusion_Joint_Conv2D-inlength(7s)-model(lightresmultisernet2|resnet50)-batch(8)-batch_delay(1)-epoch(200)-lr(0.00025)-42-8-200-Audio_Video_Joint_Classification-AdamW0.00025-IEMOCAP//000194.ckpt",
        map_location=torch.device("cpu"),
    )

    data = IEMOCAP(
        root="/sdb/visitors2/SCW/data/IEMOCAP/Feature/IEMOCAP-crop-7s-AVSplit-MFCC-KeyFrameGetterBasedIntervalSampling-Permute_Channel-.npy",
        mode="av",
        videomode="crop",
        cascPATH="/home/visitors2/SCW/torchTraining/utils/haarcascade_frontalface_alt2.xml",
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
        "/sdb/visitors2/SCW/checkpoint/MIFFNet_Joint-step(1s)-window(3s)-inlength(7s)-model(lightresmultisernet2|resnet50)-batch(8)-batch_delay(1)-epoch(200)-lr(0.00025)-42-8-200-Audio_Video_Joint_Classification-AdamW0.00025-IEMOCAP/000150.ckpt",
        map_location=torch.device("cpu"),
    )

    data = IEMOCAP(
        root="/sdb/visitors2/SCW/data/IEMOCAP/Feature/IEMOCAP-crop-7s-AVSplit-MFCC-KeyFrameGetterBasedIntervalSampling-Permute_Channel-.npy",
        mode="av",
        videomode="crop",
        cascPATH="/home/visitors2/SCW/torchTraining/utils/haarcascade_frontalface_alt2.xml",
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
        "/sdb/visitors2/SCW/checkpoint/VideoNet_Conv2D-inlength(7s)-model(resnet50)-batch(8)-batch_delay(1)-epoch(200)-lr(0.00025)-42-8-200-Video_Classification-AdamW0.00025-IEMOCAP/000152.ckpt",
        map_location=torch.device("cpu"),
    )

    data = IEMOCAP(
        root="/sdb/visitors2/SCW/data/IEMOCAP/Feature/IEMOCAP-crop-7s-AVSplit-MFCC-KeyFrameGetterBasedIntervalSampling-Permute_Channel-.npy",
        mode="v",
        videomode="crop",
        cascPATH="/home/visitors2/SCW/torchTraining/utils/haarcascade_frontalface_alt2.xml",
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
        "/sdb/visitors2/SCW/checkpoint/AudioNet-inlength(7s)-model(lightresmultisernet2)-batch(64)-batch_delay(1)-epoch(200)-lr(0.00025)-42-64-200-Audio_Classification-AdamW0.00025-IEMOCAP/000131.ckpt",
        map_location=torch.device("cpu"),
    )

    data = IEMOCAP(
        root="/sdb/visitors2/SCW/data/IEMOCAP/Feature/IEMOCAP-7s-audio_AudioSplit-audio_MFCC_Permute_Channel-.npy",
        mode="a",
        videomode="crop",
        cascPATH="/home/visitors2/SCW/torchTraining/utils/haarcascade_frontalface_alt2.xml",
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
        "/sdb/visitors2/SCW/checkpoint/MIFFNet_Conv2D_InterFusion_Joint_Conv2D-inlength(7s)-model(lightresmultisernet2|resnet50)-batch(8)-batch_delay(1)-epoch(200)-lr(0.00025)-42-8-200-Audio_Video_Joint_Classification-AdamW0.00025-IEMOCAP//000194.ckpt",
        map_location=torch.device("cpu"),
    )

    data = IEMOCAP(
        root="/sdb/visitors2/SCW/data/IEMOCAP/Feature/IEMOCAP-crop-7s-AVSplit-MFCC-KeyFrameGetterBasedIntervalSampling-Permute_Channel-.npy",
        mode="av",
        videomode="crop",
        cascPATH="/home/visitors2/SCW/torchTraining/utils/haarcascade_frontalface_alt2.xml",
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
        "/sdb/visitors2/SCW/checkpoint/MIFFNet_Joint-step(1s)-window(3s)-inlength(7s)-model(lightresmultisernet2|resnet50)-batch(8)-batch_delay(1)-epoch(200)-lr(0.00025)-42-8-200-Audio_Video_Joint_Classification-AdamW0.00025-IEMOCAP/000150.ckpt",
        map_location=torch.device("cpu"),
    )

    data = IEMOCAP(
        root="/sdb/visitors2/SCW/data/IEMOCAP/Feature/IEMOCAP-crop-7s-AVSplit-MFCC-KeyFrameGetterBasedIntervalSampling-Permute_Channel-.npy",
        mode="av",
        videomode="crop",
        cascPATH="/home/visitors2/SCW/torchTraining/utils/haarcascade_frontalface_alt2.xml",
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
        "/sdb/visitors2/SCW/checkpoint/VideoNet_Conv2D-inlength(7s)-model(resnet50)-batch(8)-batch_delay(1)-epoch(200)-lr(0.00025)-42-8-200-Video_Classification-AdamW0.00025-IEMOCAP/000152.ckpt",
        map_location=torch.device("cpu"),
    )

    data = IEMOCAP(
        root="/sdb/visitors2/SCW/data/IEMOCAP/Feature/IEMOCAP-crop-7s-AVSplit-MFCC-KeyFrameGetterBasedIntervalSampling-Permute_Channel-.npy",
        mode="v",
        videomode="crop",
        cascPATH="/home/visitors2/SCW/torchTraining/utils/haarcascade_frontalface_alt2.xml",
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
        "/sdb/visitors2/SCW/checkpoint/AudioNet-inlength(7s)-model(lightresmultisernet2)-batch(64)-batch_delay(1)-epoch(200)-lr(0.00025)-42-64-200-Audio_Classification-AdamW0.00025-IEMOCAP/000131.ckpt",
        map_location=torch.device("cpu"),
    )

    data = IEMOCAP(
        root="/sdb/visitors2/SCW/data/IEMOCAP/Feature/IEMOCAP-7s-audio_AudioSplit-audio_MFCC_Permute_Channel-.npy",
        mode="a",
        videomode="crop",
        cascPATH="/home/visitors2/SCW/torchTraining/utils/haarcascade_frontalface_alt2.xml",
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
        "/sdb/visitors2/SCW/checkpoint/MIFFNet_Conv2D_InterFusion_Joint_Conv2D-inlength(7s)-model(lightresmultisernet2|resnet50)-batch(8)-batch_delay(1)-epoch(200)-lr(0.00025)-42-8-200-Audio_Video_Joint_Classification-AdamW0.00025-IEMOCAP//000194.ckpt",
        map_location=torch.device("cpu"),
    )

    data = IEMOCAP(
        root="/sdb/visitors2/SCW/data/IEMOCAP/Feature/IEMOCAP-crop-7s-AVSplit-MFCC-KeyFrameGetterBasedIntervalSampling-Permute_Channel-.npy",
        mode="av",
        videomode="crop",
        cascPATH="/home/visitors2/SCW/torchTraining/utils/haarcascade_frontalface_alt2.xml",
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
        "/sdb/visitors2/SCW/checkpoint/MIFFNet_Joint-step(1s)-window(3s)-inlength(7s)-model(lightresmultisernet2|resnet50)-batch(8)-batch_delay(1)-epoch(200)-lr(0.00025)-42-8-200-Audio_Video_Joint_Classification-AdamW0.00025-IEMOCAP/000150.ckpt",
        map_location=torch.device("cpu"),
    )

    data = IEMOCAP(
        root="/sdb/visitors2/SCW/data/IEMOCAP/Feature/IEMOCAP-crop-7s-AVSplit-MFCC-KeyFrameGetterBasedIntervalSampling-Permute_Channel-.npy",
        mode="av",
        videomode="crop",
        cascPATH="/home/visitors2/SCW/torchTraining/utils/haarcascade_frontalface_alt2.xml",
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
        "/sdb/visitors2/SCW/checkpoint/VideoNet_Conv2D-inlength(7s)-model(resnet50)-batch(8)-batch_delay(1)-epoch(200)-lr(0.00025)-42-8-200-Video_Classification-AdamW0.00025-IEMOCAP/000152.ckpt",
        map_location=torch.device("cpu"),
    )

    data = IEMOCAP(
        root="/sdb/visitors2/SCW/data/IEMOCAP/Feature/IEMOCAP-crop-7s-AVSplit-MFCC-KeyFrameGetterBasedIntervalSampling-Permute_Channel-.npy",
        mode="v",
        videomode="crop",
        cascPATH="/home/visitors2/SCW/torchTraining/utils/haarcascade_frontalface_alt2.xml",
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
        "/sdb/visitors2/SCW/checkpoint/AudioNet-inlength(7s)-model(lightresmultisernet2)-batch(64)-batch_delay(1)-epoch(200)-lr(0.00025)-42-64-200-Audio_Classification-AdamW0.00025-IEMOCAP/000131.ckpt",
        map_location=torch.device("cpu"),
    )

    data = IEMOCAP(
        root="/sdb/visitors2/SCW/data/IEMOCAP/Feature/IEMOCAP-7s-audio_AudioSplit-audio_MFCC_Permute_Channel-.npy",
        mode="a",
        videomode="crop",
        cascPATH="/home/visitors2/SCW/torchTraining/utils/haarcascade_frontalface_alt2.xml",
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
        "/sdb/visitors2/SCW/checkpoint/MIFFNet_Joint-step(1s)-window(3s)-inlength(7s)-model(lightresmultisernet2|resnet50)-batch(8)-batch_delay(1)-epoch(200)-lr(0.00025)-42-8-200-Audio_Video_Joint_Classification-AdamW0.00025-IEMOCAP/000150.ckpt",
        map_location=torch.device("cpu"),
    )

    data = IEMOCAP(
        root="/sdb/visitors2/SCW/data/IEMOCAP/Feature/IEMOCAP-crop-7s-AVSplit-MFCC-KeyFrameGetterBasedIntervalSampling-Permute_Channel-.npy",
        mode="av",
        videomode="crop",
        cascPATH="/home/visitors2/SCW/torchTraining/utils/haarcascade_frontalface_alt2.xml",
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
        "/sdb/visitors2/SCW/checkpoint/MIFFNet_Joint-step(1s)-window(3s)-inlength(7s)-model(lightresmultisernet2|resnet50)-batch(8)-batch_delay(1)-epoch(200)-lr(0.00025)-42-8-200-Audio_Video_Joint_Classification-AdamW0.00025-IEMOCAP/000150.ckpt",
        map_location=torch.device("cpu"),
    )

    data = IEMOCAP(
        root="/sdb/visitors2/SCW/data/IEMOCAP/Feature/IEMOCAP-crop-7s-AVSplit-MFCC-KeyFrameGetterBasedIntervalSampling-Permute_Channel-.npy",
        mode="av",
        videomode="crop",
        cascPATH="/home/visitors2/SCW/torchTraining/utils/haarcascade_frontalface_alt2.xml",
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
        "/sdb/visitors2/SCW/checkpoint/MIFFNet_Joint-step(1s)-window(3s)-inlength(7s)-model(lightresmultisernet2|resnet50)-batch(8)-batch_delay(1)-epoch(200)-lr(0.00025)-42-8-200-Audio_Video_Joint_Classification-AdamW0.00025-IEMOCAP/000150.ckpt",
        map_location=torch.device("cpu"),
    )

    data = IEMOCAP(
        root="/sdb/visitors2/SCW/data/IEMOCAP/Feature/IEMOCAP-crop-7s-AVSplit-MFCC-KeyFrameGetterBasedIntervalSampling-Permute_Channel-.npy",
        mode="av",
        videomode="crop",
        cascPATH="/home/visitors2/SCW/torchTraining/utils/haarcascade_frontalface_alt2.xml",
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
        "/sdb/visitors2/SCW/checkpoint/MIFFNet_Conv2D_SVC_InterFusion_Joint_Attention-inlength(7s)-model(lightresmultisernet2|resnet50)-batch(8)-batch_delay(1)-epoch(200)-lr(0.00025)-42-10-200-Audio_Video_Joint_Classification-AdamW0.00025-IEMOCAP/000150.ckpt",
        map_location=torch.device("cpu"),
    )

    data = IEMOCAP(
        root="/sdb/visitors2/SCW/data/IEMOCAP/Feature/IEMOCAP-crop-7s-AVSplit-MFCC-KeyFrameGetterBasedIntervalSampling-Permute_Channel-.npy",
        mode="av",
        videomode="crop",
        cascPATH="/home/visitors2/SCW/torchTraining/utils/haarcascade_frontalface_alt2.xml",
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
        "/sdb/visitors2/SCW/checkpoint/MIFFNet_Conv2D_SVC_InterFusion_Joint_Attention-inlength(7s)-model(lightresmultisernet2|resnet50)-batch(8)-batch_delay(1)-epoch(200)-lr(0.00025)-42-10-200-Audio_Video_Joint_Classification-AdamW0.00025-IEMOCAP/000150.ckpt",
        map_location=torch.device("cpu"),
    )

    data = IEMOCAP(
        root="/sdb/visitors2/SCW/data/IEMOCAP/Feature/IEMOCAP-crop-7s-AVSplit-MFCC-KeyFrameGetterBasedIntervalSampling-Permute_Channel-.npy",
        mode="av",
        videomode="crop",
        cascPATH="/home/visitors2/SCW/torchTraining/utils/haarcascade_frontalface_alt2.xml",
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
        "/sdb/visitors2/SCW/checkpoint/MIFFNet_Conv2D_SVC_InterFusion_Joint_Attention-inlength(7s)-model(lightresmultisernet2|resnet50)-batch(8)-batch_delay(1)-epoch(200)-lr(0.00025)-42-10-200-Audio_Video_Joint_Classification-AdamW0.00025-IEMOCAP/000150.ckpt",
        map_location=torch.device("cpu"),
    )

    data = IEMOCAP(
        root="/sdb/visitors2/SCW/data/IEMOCAP/Feature/IEMOCAP-crop-7s-AVSplit-MFCC-KeyFrameGetterBasedIntervalSampling-Permute_Channel-.npy",
        mode="av",
        videomode="crop",
        cascPATH="/home/visitors2/SCW/torchTraining/utils/haarcascade_frontalface_alt2.xml",
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
        "/sdb/visitors2/SCW/checkpoint/MIFFNet_Conv2D_SVC_InterFusion_Joint_Attention-inlength(7s)-model(lightresmultisernet2|resnet50)-batch(8)-batch_delay(1)-epoch(200)-lr(0.00025)-42-10-200-Audio_Video_Joint_Classification-AdamW0.00025-IEMOCAP/000150.ckpt",
        map_location=torch.device("cpu"),
    )

    data = IEMOCAP(
        root="/sdb/visitors2/SCW/data/IEMOCAP/Feature/IEMOCAP-pad-MFCC-KeyFrameGetterBasedIntervalSampling-Permute_Channel-.npy",
        mode="av",
        videomode="crop",
        cascPATH="/home/visitors2/SCW/torchTraining/utils/haarcascade_frontalface_alt2.xml",
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
        "/sdb/visitors2/SCW/checkpoint/MIFFNet_Conv2D_SVC_InterFusion_Joint_Attention-inlength(7s)-model(lightresmultisernet2|resnet50)-batch(8)-batch_delay(1)-epoch(200)-lr(0.00025)-42-10-200-Audio_Video_Joint_Classification-AdamW0.00025-IEMOCAP/000150.ckpt",
        map_location=torch.device("cpu"),
    )

    data = IEMOCAP(
        root="/sdb/visitors2/SCW/data/IEMOCAP/Feature/IEMOCAP-crop-7s-AVSplit-MFCC-KeyFrameGetterBasedIntervalSampling-Permute_Channel-.npy",
        mode="av",
        videomode="crop",
        cascPATH="/home/visitors2/SCW/torchTraining/utils/haarcascade_frontalface_alt2.xml",
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


def plot_similarity_MIFFNet_Conv2D_SVC_InterFusion_Joint_Attention():
    dataset = IEMOCAP(
        root="/sdb/visitors2/SCW/data/IEMOCAP/Feature/IEMOCAP-crop-7s-AVSplit-MFCC-KeyFrameGetterBasedIntervalSampling-Permute_Channel-.npy",
        mode="av",
        videomode="crop",
        cascPATH="/home/visitors2/SCW/torchTraining/utils/haarcascade_frontalface_alt2.xml",
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
            "/sdb/visitors2/SCW/checkpoint/MIFFNet_Conv2D_SVC_InterFusion_Joint_Attention-inlength(7s)-model(lightresmultisernet2|resnet50)-batch(8)-batch_delay(1)-epoch(200)-lr(0.00025)-42-10-200-Audio_Video_Joint_Classification-AdamW0.00025-IEMOCAP/%06d.ckpt"
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



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def plot_tsne_Base_AudioNet_MDEA_4():
    print("加载模型")

    dictionary = torch.load(
        "/sdb/visitors2/SCW/checkpoint/Base_AudioNet_MDEA_4-inlength(7s)-batch(8)-batch_delay(1)-epoch(200)-lr(0.00025)-42-8-200-Audio_Classification-AdamW0.00025-IEMOCAP/000125.ckpt",
        map_location=torch.device("cpu"),
    )

    data = IEMOCAP(
        root="/sdb/visitors2/SCW/data/IEMOCAP/Feature/IEMOCAP-7s-audio_AudioSplit-audio_MFCC_Permute_Channel-.npy",
        mode="a",
        videomode="crop",
        cascPATH="/home/visitors2/SCW/torchTraining/utils/haarcascade_frontalface_alt2.xml",
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
    # 这里我们将使用全部数据来进行可视化
    print(len(data))
    dataloader = torch.utils.data.DataLoader(
        data, batch_size=8, shuffle=False, num_workers=0
    )

    net = Base_AudioNet_MDEA(
        num_classes=4,
        last_hidden_dim=320,  # 根据模型的具体配置进行调整
        enable_classifier=False,
    )

    weights_dict = {}
    for k, v in dictionary["net"].items():
        new_k = k.replace("module.", "") if "module" in k else k
        weights_dict[new_k] = v

    net.load_state_dict(weights_dict)

    plot_tsne(
        net, dataloader, outdir="./output/t-sne/", info="Base_AudioNet_MDEA_4", device=device
    )

def plot_tsne_Base_AudioNet_MDEA_6():
    print("加载模型")

    dictionary = torch.load(
        "/sdb/visitors2/SCW/checkpoint/Base_AudioNet_MDEA_6-inlength(7s)-batch(10)-batch_delay(1)-epoch(200)-lr(0.00025)-42-10-200-Audio_Classification-AdamW0.00025-IEMOCAP/000135.ckpt",
        map_location=torch.device("cpu"),
    )

    data = IEMOCAP(
        root="/sdb/visitors2/SCW/data/IEMOCAP/Feature/IEMOCAP-7s-audio_AudioSplit-audio_MFCC_Permute_Channel-.npy",
        mode="a",
        videomode="crop",
        cascPATH="/home/user0/SCW/torchTraining/utils/haarcascade_frontalface_alt2.xml",
        filter={
            "replace": {},
            "dropna": {
                "label": [
                    "other",
                    "xxx",
                    "disgusted",
                    "fearful",
                    "surprised",
                ],
            },
        },
    )
    print(len(data))
    dataloader = torch.utils.data.DataLoader(
        data, batch_size=10, shuffle=False, num_workers=0
    )

    net = Base_AudioNet_MDEA(
        num_classes=6,
        last_hidden_dim=320, 
        enable_classifier=False,
    )

    weights_dict = {}
    for k, v in dictionary["net"].items():
        new_k = k.replace("module.", "") if "module" in k else k
        weights_dict[new_k] = v

    net.load_state_dict(weights_dict)

    plot_tsne(
        net, dataloader, outdir="./output/t-sne/", info="Base_AudioNet_MDEA_6", device=device
    )

def plot_tsne_Base_AudioNet_Resnet50_4():
    print("加载模型")

    dictionary = torch.load(
        "/sdb/visitors2/SCW/checkpoint/Base_AudioNet_Resnet50_4-inlength(7s)-batch(8)-batch_delay(1)-epoch(200)-lr(0.00025)-42-8-200-Audio_Classification-AdamW0.00025-IEMOCAP/000060.ckpt",
        map_location=torch.device("cpu"),
    )

    data = IEMOCAP(
        root="/sdb/visitors2/SCW/data/IEMOCAP/Feature/IEMOCAP-7s-audio_AudioSplit-audio_MFCC_Permute_Channel-.npy",
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
    print(len(data))
    dataloader = torch.utils.data.DataLoader(
        data, batch_size=8, shuffle=False, num_workers=0
    )

    net = Base_AudioNet_Resnet50(
        num_classes=4,
        input_size = [189, 40],  # 根据模型的具体配置调整
        enable_classifier=False,
    )

    weights_dict = {}
    for k, v in dictionary["net"].items():
        new_k = k.replace("module.", "") if "module" in k else k
        weights_dict[new_k] = v

    net.load_state_dict(weights_dict)

    plot_tsne(
        net, dataloader, outdir="./output/t-sne/", info="Base_AudioNet_Resnet50_4", device=device
    )


def plot_tsne_Base_AudioNet_Resnet50_6():
    print("加载模型")

    dictionary = torch.load(
        "/sdb/visitors2/SCW/checkpoint/Base_AudioNet_Resnet50_6-inlength(7s)-batch(8)-batch_delay(1)-epoch(200)-lr(0.00025)-42-8-200-Audio_Classification-AdamW0.00025-IEMOCAP/000140.ckpt",
        map_location=torch.device("cpu"),
    )

    data = IEMOCAP(
        root="/sdb/visitors2/SCW/data/IEMOCAP/Feature/IEMOCAP-7s-audio_AudioSplit-audio_MFCC_Permute_Channel-.npy",
        mode="a",
        videomode="crop",
        cascPATH="/home/visitors2/SCW/torchTraining/utils/haarcascade_frontalface_alt2.xml",
        filter={
            "replace": {},
            "dropna": {
                "label": [
                    "other",
                    "xxx",
                    "disgusted",
                    "fearful",
                    "surprised",
                ],
            },
        },
    )
    print(len(data))
    dataloader = torch.utils.data.DataLoader(
        data, batch_size=8, shuffle=False, num_workers=0
    )

    net = Base_AudioNet_Resnet50(
        num_classes=6,
        input_size = [189, 40],  # 根据模型的具体配置调整
        enable_classifier=False,
    )

    weights_dict = {}
    for k, v in dictionary["net"].items():
        new_k = k.replace("module.", "") if "module" in k else k
        weights_dict[new_k] = v

    net.load_state_dict(weights_dict)

    plot_tsne(
        net, dataloader, outdir="./output/t-sne/", info="Base_AudioNet_Resnet50_6", device=device
    )

def plot_tsne_Base_VideoNet_MSDC_4():
    print("加载模型")

    dictionary = torch.load(
        "/sdb/visitors2/SCW/checkpoint/Base_VideoNet_MSDC_4-inlength(7s)-batch(8)-batch_delay(1)-epoch(200)-lr(0.00025)-42-8-200-Video_Classification-AdamW0.00025-IEMOCAP/000104.ckpt",
        map_location=torch.device("cpu"),
    )

    data = IEMOCAP(
        root="/sdb/visitors2/SCW/data/IEMOCAP/Feature/IEMOCAP-crop-7s-AVSplit-MFCC-KeyFrameGetterBasedIntervalSampling-Permute_Channel-.npy",
        mode="v",
        videomode="crop",
        cascPATH="/home/visitors2/SCW/torchTraining/utils/haarcascade_frontalface_alt2.xml",
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
    print(len(data))
    dataloader = torch.utils.data.DataLoader(
        data, batch_size=8, shuffle=False, num_workers=0
    )

    net = Base_VideoNet_MSDC(
        num_classes=4,
        input_size = [150, 150],
        num_frames = 71,
        enable_classifier=False,
    )

    weights_dict = {}
    for k, v in dictionary["net"].items():
        new_k = k.replace("module.", "") if "module" in k else k
        weights_dict[new_k] = v

    net.load_state_dict(weights_dict)

    plot_tsne(
        net, dataloader, outdir="./output/t-sne/", info="Base_VideoNet_MSDC_4", device=device
    )


def plot_tsne_Base_VideoNet_MSDC_6():
    print("加载模型")

    dictionary = torch.load(
        "/sdb/visitors2/SCW/checkpoint/Base_VideoNet_MSDC_6-inlength(7s)-batch(8)-batch_delay(1)-epoch(200)-lr(0.00025)-42-8-200-Video_Classification-AdamW0.00025-IEMOCAP/000093.ckpt",
        map_location=torch.device("cpu"),
    )

    data = IEMOCAP(
        root="/sdb/visitors2/SCW/data/IEMOCAP/Feature/IEMOCAP-crop-7s-AVSplit-MFCC-KeyFrameGetterBasedIntervalSampling-Permute_Channel-.npy",
        mode="v",
        videomode="crop",
        cascPATH="/home/user0/SCW/torchTraining/utils/haarcascade_frontalface_alt2.xml",
        filter={
            "replace": {},
            "dropna": {
                "label": [
                    "other",
                    "xxx",
                    "disgusted",
                    "fearful",
                    "surprised",
                ],
            },
        },
    )
    print(len(data))
    dataloader = torch.utils.data.DataLoader(
        data, batch_size=8, shuffle=False, num_workers=0
    )

    net = Base_VideoNet_MSDC(
        num_classes=6,
        input_size = [150, 150],
        num_frames = 71,
        enable_classifier=False,
    )

    weights_dict = {}
    for k, v in dictionary["net"].items():
        new_k = k.replace("module.", "") if "module" in k else k
        weights_dict[new_k] = v

    net.load_state_dict(weights_dict)

    plot_tsne(
        net, dataloader, outdir="./output/t-sne/", info="Base_VideoNet_MSDC_6", device=device
    )



def plot_tsne_MER_VideoNet_six_resnet50():
    print("加载模型")

    dictionary = torch.load(
        "/sdb/visitors2/SCW/checkpoint/MER_VideoNet_six_resnet50-inlength(7s)-batch(8)-batch_delay(1)-epoch(200)-lr(0.00025)-42-8-200-Video_Classification-AdamW0.00025-IEMOCAP/000124.ckpt",
        map_location=torch.device("cpu"),
    )

    data = IEMOCAP(
        root="/sdb/visitors2/SCW/data/IEMOCAP/Feature/IEMOCAP-crop-7s-AVSplit-MFCC-KeyFrameGetterBasedIntervalSampling-Permute_Channel-.npy",
        mode="v",
        videomode="crop",
        cascPATH="/home/user0/SCW/torchTraining/utils/haarcascade_frontalface_alt2.xml",
        filter={
            "replace": {},
            "dropna": {
                "label": [
                    "other",
                    "xxx",
                    "disgusted",
                    "fearful",
                    "surprised",
                ],
            },
        },
    )
    print(len(data))
    dataloader = torch.utils.data.DataLoader(
        data, batch_size=8, shuffle=False, num_workers=0
    )

    net = MER_VideoNet(
        num_classes=6,
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
        net, dataloader, outdir="./output/t-sne/", info="MER_VideoNet_six_resnet50", device=device
    )


def plot_tsne_BER_SISF_4():
    print("加载模型")

    dictionary = torch.load(
        "/sdb/visitors2/SCW/checkpoint/BER_SISF_4-inlength(7s)-batch(10)-batch_delay(1)-epoch(200)-lr(0.00025)-42-10-200-Audio_Video_Joint_Classification_Customloss-AdamW0.00025-IEMOCAP/000103.ckpt",
        map_location=torch.device("cpu"),
    )

    data = IEMOCAP(
        root="/sdb/visitors2/SCW/data/IEMOCAP/Feature/IEMOCAP-crop-7s-AVSplit-MFCC-KeyFrameGetterBasedIntervalSampling-Permute_Channel-.npy",
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
    print(len(data))
    dataloader = torch.utils.data.DataLoader(
        data, batch_size=10, shuffle=False, num_workers=0
    )

    net = BER_SISF(
        num_classes=4,
        input_size = [[189, 40], [150, 150]],
        last_hidden_dim = [320, 320],
        seq_len = [189, 30] ,
        num_frames = 30,
        enable_classifier=False,
    )

    weights_dict = {}
    for k, v in dictionary["net"].items():
        new_k = k.replace("module.", "") if "module" in k else k
        weights_dict[new_k] = v

    net.load_state_dict(weights_dict)

    plot_tsne(
        net, dataloader, outdir="./output/t-sne/", info="BER_SISF_4", device=device
    )


def plot_tsne_BER_SISF_6():
    print("加载模型")

    dictionary = torch.load(
        "/sdb/visitors2/SCW/checkpoint/BER_SISF_6-inlength(7s)-batch(10)-batch_delay(1)-epoch(200)-lr(0.00025)-42-10-200-Audio_Video_Joint_Classification_Customloss-AdamW0.00025-IEMOCAP/000072.ckpt",
        map_location=torch.device("cpu"),
    )

    data = IEMOCAP(
        root="/sdb/visitors2/SCW/data/IEMOCAP/Feature/IEMOCAP-crop-7s-AVSplit-MFCC-KeyFrameGetterBasedIntervalSampling-Permute_Channel-.npy",
        mode="av",
        videomode="crop",
        cascPATH="/home/user0/SCW/torchTraining/utils/haarcascade_frontalface_alt2.xml",
        filter={
            "replace": {},
            "dropna": {
                "label": [
                    "other",
                    "xxx",
                    "disgusted",
                    "fearful",
                    "surprised",
                ],
            },
        },
    )
    print(len(data))
    dataloader = torch.utils.data.DataLoader(
        data, batch_size=10, shuffle=False, num_workers=0
    )

    net = BER_SISF(
        num_classes=6,
        input_size = [[189, 40], [150, 150]],
        last_hidden_dim = [320, 320],
        seq_len = [189, 30] ,
        num_frames = 30,
        enable_classifier=False,
    )

    weights_dict = {}
    for k, v in dictionary["net"].items():
        new_k = k.replace("module.", "") if "module" in k else k
        weights_dict[new_k] = v

    net.load_state_dict(weights_dict)

    plot_tsne(
        net, dataloader, outdir="./output/t-sne/", info="BER_SISF_6", device=device
    )


def plot_tsne_MERSISF_av_SVC_InterFusion_Joint_Attention_SLoss_pretrained_D2DUniG2_four():
    print("加载模型")

    dictionary = torch.load(
        "/sdb/visitors2/SCW/checkpoint/MERSISF_av_SVC_InterFusion_Joint_Attention_SLoss_pretrained_D2DUniG2_four-inlength(7s)-batch(10)-batch_delay(1)-epoch(200)-lr(0.00025)-42-10-200-Audio_Video_Joint_Classification_Customloss-AdamW0.00025-IEMOCAP/000122.ckpt",
        map_location=torch.device("cpu"),
    )

    data = IEMOCAP(
        root="/sdb/visitors2/SCW/data/IEMOCAP/Feature/IEMOCAP-crop-7s-AVSplit-MFCC-KeyFrameGetterBasedIntervalSampling-Permute_Channel-.npy",
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
    print(len(data))
    dataloader = torch.utils.data.DataLoader(
        data, batch_size=10, shuffle=False, num_workers=0
    )

    net = MER_SISF_Conv2D_SVC_InterFusion_Joint_Attention_SLoss_Luck_UniG_pretrained(
        num_classes=4,
        input_size = [[189, 40], [150, 150]],
        model_name = ["md-ease", "resnet50"],
        last_hidden_dim = [320, 320],
        seq_len = [189, 30],
        enable_classifier=False,
    )

    weights_dict = {}
    for k, v in dictionary["net"].items():
        new_k = k.replace("module.", "") if "module" in k else k
        weights_dict[new_k] = v

    net.load_state_dict(weights_dict)

    plot_tsne(
        net, dataloader, outdir="./output/t-sne/", info="MERSISF_av_SVC_InterFusion_Joint_Attention_SLoss_pretrained_D2DUniG2_four", device=device
    )


def plot_tsne_MERSISF_av_SVC_InterFusion_Joint_Attention_SLoss_pretrained_D2DUniG2_six():
    print("加载模型")

    dictionary = torch.load(
        "/sdb/visitors2/SCW/checkpoint/MERSISF_av_SVC_InterFusion_Joint_Attention_SLoss_pretrained_D2DUniG2_six-inlength(7s)-batch(8)-batch_delay(1)-epoch(200)-lr(0.00025)-42-8-200-Audio_Video_Joint_Classification_Customloss-AdamW0.00025-IEMOCAP/000145.ckpt",
        map_location=torch.device("cpu"),
    )

    data = IEMOCAP(
        root="/sdb/visitors2/SCW/data/IEMOCAP/Feature/IEMOCAP-crop-7s-AVSplit-MFCC-KeyFrameGetterBasedIntervalSampling-Permute_Channel-.npy",
        mode="av",
        videomode="crop",
        cascPATH="/home/visitors2/SCW/torchTraining/utils/haarcascade_frontalface_alt2.xml",
        filter={
            "replace": {},
            "dropna": {
                "label": [
                    "other",
                    "xxx",
                    "disgusted",
                    "fearful",
                    "surprised",
                ],
            },
        },
    )
    print(len(data))
    dataloader = torch.utils.data.DataLoader(
        data, batch_size=8, shuffle=False, num_workers=0
    )

    net = MER_SISF_Conv2D_SVC_InterFusion_Joint_Attention_SLoss_Luck_UniG_pretrained(
        num_classes=6,
        input_size = [[189, 40], [150, 150]],
        model_name = ["md-ease", "resnet50"],
        last_hidden_dim = [320, 320],
        seq_len = [189, 30],
        enable_classifier=False,
    )

    weights_dict = {}
    for k, v in dictionary["net"].items():
        new_k = k.replace("module.", "") if "module" in k else k
        weights_dict[new_k] = v

    net.load_state_dict(weights_dict)

    plot_tsne(
        net, dataloader, outdir="./output/t-sne/", info="MERSISF_av_SVC_InterFusion_Joint_Attention_SLoss_pretrained_D2DUniG2_six", device=device
    )



def plot_rt_BER_SISF_4():
    print("加载模型")
    dictionary = torch.load(
        "/sdb/visitors2/SCW/checkpoint/BER_SISF_4-inlength(7s)-batch(10)-batch_delay(1)-epoch(200)-lr(0.00025)-42-10-200-Audio_Video_Joint_Classification_Customloss-AdamW0.00025-IEMOCAP/000103.ckpt",
        map_location=torch.device("cpu"),
    )
    data = IEMOCAP(
        root="/sdb/visitors2/SCW/data/IEMOCAP/Feature/IEMOCAP-pad-MFCC-KeyFrameGetterBasedIntervalSampling-Permute_Channel-.npy",
        mode="av",
        videomode="crop",
        cascPATH="/home/visitors2/SCW/torchTraining/utils/haarcascade_frontalface_alt2.xml",
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
            "query": {"duration": "duration>18.0"},
        },
    )

    print(len(data))
    dataloader = torch.utils.data.DataLoader(
        data, batch_size=1, shuffle=False, num_workers=0
    )

    net = BER_SISF(
        num_classes=4,
        input_size = [[189, 40], [150, 150]],
        last_hidden_dim = [320, 320],
        seq_len = [189, 30] ,
        num_frames = 30,
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
        info="BER_SISF_4",
        device=device,
    )
    
    
def plot_rt_BER_FF_4():
    print("加载模型")
    dictionary = torch.load(
        "/sdb/user0/SCW/checkpoint/BER_ISF_4-inlength(7s)-batch(8)-batch_delay(1)-epoch(200)-lr(0.00025)-42-8-200-Audio_Video_Joint_Classification-AdamW0.00025-IEMOCAP/000091.ckpt",
        map_location=torch.device("cpu"),
    )
    data = IEMOCAP(
        root="/sdb/visitors2/SCW/data/IEMOCAP/Feature/IEMOCAP-pad-MFCC-KeyFrameGetterBasedIntervalSampling-Permute_Channel-.npy",
        mode="av",
        videomode="crop",
        cascPATH="/home/visitors2/SCW/torchTraining/utils/haarcascade_frontalface_alt2.xml",
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
            "query": {"duration": "duration>18.0"},
        },
    )
    print(len(data))
    dataloader = torch.utils.data.DataLoader(
        data, batch_size=1, shuffle=False, num_workers=0
    )

    net = BER_ISF(
        num_classes=4,
        input_size = [[189, 40], [150, 150]],
        seq_len = [189, 30] ,
        num_frames = 30,
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
        info="BER_FF_4",
        device=device,
    )

def plot_features_BER_SISF_4():
    model_paths = [
        # "/sdb/visitors2/SCW/checkpoint/BER_SISF_4-inlength(7s)-batch(10)-batch_delay(1)-epoch(200)-lr(0.00025)-42-10-200-Audio_Video_Joint_Classification_Customloss-AdamW0.00025-IEMOCAP/000000.ckpt",
        "/sdb/visitors2/SCW/checkpoint/BER_SISF_4-inlength(7s)-batch(10)-batch_delay(1)-epoch(200)-lr(0.00025)-42-10-200-Audio_Video_Joint_Classification_Customloss-AdamW0.00025-IEMOCAP/000001.ckpt",
        # "/sdb/visitors2/SCW/checkpoint/BER_SISF_4-inlength(7s)-batch(10)-batch_delay(1)-epoch(200)-lr(0.00025)-42-10-200-Audio_Video_Joint_Classification_Customloss-AdamW0.00025-IEMOCAP/000002.ckpt",
        # "/sdb/visitors2/SCW/checkpoint/BER_SISF_4-inlength(7s)-batch(10)-batch_delay(1)-epoch(200)-lr(0.00025)-42-10-200-Audio_Video_Joint_Classification_Customloss-AdamW0.00025-IEMOCAP/000003.ckpt",
        # "/sdb/visitors2/SCW/checkpoint/BER_SISF_4-inlength(7s)-batch(10)-batch_delay(1)-epoch(200)-lr(0.00025)-42-10-200-Audio_Video_Joint_Classification_Customloss-AdamW0.00025-IEMOCAP/000005.ckpt",
        # "/sdb/visitors2/SCW/checkpoint/BER_SISF_4-inlength(7s)-batch(10)-batch_delay(1)-epoch(200)-lr(0.00025)-42-10-200-Audio_Video_Joint_Classification_Customloss-AdamW0.00025-IEMOCAP/000006.ckpt",
        # "/sdb/visitors2/SCW/checkpoint/BER_SISF_4-inlength(7s)-batch(10)-batch_delay(1)-epoch(200)-lr(0.00025)-42-10-200-Audio_Video_Joint_Classification_Customloss-AdamW0.00025-IEMOCAP/000007.ckpt",
        "/sdb/visitors2/SCW/checkpoint/BER_SISF_4-inlength(7s)-batch(10)-batch_delay(1)-epoch(200)-lr(0.00025)-42-10-200-Audio_Video_Joint_Classification_Customloss-AdamW0.00025-IEMOCAP/000012.ckpt",
        "/sdb/visitors2/SCW/checkpoint/BER_SISF_4-inlength(7s)-batch(10)-batch_delay(1)-epoch(200)-lr(0.00025)-42-10-200-Audio_Video_Joint_Classification_Customloss-AdamW0.00025-IEMOCAP/000026.ckpt",
        "/sdb/visitors2/SCW/checkpoint/BER_SISF_4-inlength(7s)-batch(10)-batch_delay(1)-epoch(200)-lr(0.00025)-42-10-200-Audio_Video_Joint_Classification_Customloss-AdamW0.00025-IEMOCAP/000034.ckpt",
        "/sdb/visitors2/SCW/checkpoint/BER_SISF_4-inlength(7s)-batch(10)-batch_delay(1)-epoch(200)-lr(0.00025)-42-10-200-Audio_Video_Joint_Classification_Customloss-AdamW0.00025-IEMOCAP/000057.ckpt",
        "/sdb/visitors2/SCW/checkpoint/BER_SISF_4-inlength(7s)-batch(10)-batch_delay(1)-epoch(200)-lr(0.00025)-42-10-200-Audio_Video_Joint_Classification_Customloss-AdamW0.00025-IEMOCAP/000068.ckpt",
        "/sdb/visitors2/SCW/checkpoint/BER_SISF_4-inlength(7s)-batch(10)-batch_delay(1)-epoch(200)-lr(0.00025)-42-10-200-Audio_Video_Joint_Classification_Customloss-AdamW0.00025-IEMOCAP/000071.ckpt",
        "/sdb/visitors2/SCW/checkpoint/BER_SISF_4-inlength(7s)-batch(10)-batch_delay(1)-epoch(200)-lr(0.00025)-42-10-200-Audio_Video_Joint_Classification_Customloss-AdamW0.00025-IEMOCAP/000089.ckpt",
        "/sdb/visitors2/SCW/checkpoint/BER_SISF_4-inlength(7s)-batch(10)-batch_delay(1)-epoch(200)-lr(0.00025)-42-10-200-Audio_Video_Joint_Classification_Customloss-AdamW0.00025-IEMOCAP/000103.ckpt",
    ]
    data = IEMOCAP(
        root="/sdb/visitors2/SCW/data/IEMOCAP/Feature/IEMOCAP-crop-7s-AVSplit-MFCC-KeyFrameGetterBasedIntervalSampling-Permute_Channel-.npy",
        mode="av",
        videomode="crop",
        cascPATH="/home/visitors2/SCW/torchTraining/utils/haarcascade_frontalface_alt2.xml",
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
    
    
    dictionary = torch.load(
        model_paths[0],
        map_location=torch.device("cpu"),
    )
    data = Subset(data, indices=dictionary["train_indices"])
    print(len(data))
    num_samples = 5
    indices = np.random.choice(len(data), num_samples, replace=False)
    print(f"数据索引：{indices}")
    sampler = torch.utils.data.SubsetRandomSampler(indices)
    dataloader = torch.utils.data.DataLoader(data, batch_size=1, sampler=sampler, num_workers=0)
    
    for ind, data in enumerate(dataloader):
        for j, path in enumerate(model_paths):
            # 加载模型
            dictionary = torch.load(
                path,
                map_location=torch.device("cpu"),
            )
            net = BER_SISF(
                num_classes=4,
                input_size = [[189, 40], [150, 150]],
                last_hidden_dim = [320, 320],
                seq_len = [189, 30] ,
                num_frames = 30,
                enable_classifier=False,
            )
            weights_dict = {}
            for k, v in dictionary["net"].items():
                new_k = k.replace("module.", "") if "module" in k else k
                weights_dict[new_k] = v

            net.load_state_dict(weights_dict)
            # 随机从数据加载器中获取数据
            
            # 可视化并保存特征
            plot_features(net, data, outdir=f"./output/features/data{indices[ind]}", info=f"checkpoint{path[-8:]}_features", device=device)


def plot_image_BER_SISF_E():
    model_paths = [
        # "/sdb/visitors2/SCW/checkpoint/MERSISF_av_SVC_InterFusion_Joint_Attention_SLoss_pretrained_D2DUniG2_four-inlength(7s)-batch(10)-batch_delay(1)-epoch(200)-lr(0.00025)-42-10-200-Audio_Video_Joint_Classification_Customloss-AdamW0.00025-IEMOCAP/000000.ckpt",
        # "/sdb/visitors2/SCW/checkpoint/MERSISF_av_SVC_InterFusion_Joint_Attention_SLoss_pretrained_D2DUniG2_four-inlength(7s)-batch(10)-batch_delay(1)-epoch(200)-lr(0.00025)-42-10-200-Audio_Video_Joint_Classification_Customloss-AdamW0.00025-IEMOCAP/000001.ckpt",
        # "/sdb/visitors2/SCW/checkpoint/MERSISF_av_SVC_InterFusion_Joint_Attention_SLoss_pretrained_D2DUniG2_four-inlength(7s)-batch(10)-batch_delay(1)-epoch(200)-lr(0.00025)-42-10-200-Audio_Video_Joint_Classification_Customloss-AdamW0.00025-IEMOCAP/000002.ckpt",
        # "/sdb/visitors2/SCW/checkpoint/MERSISF_av_SVC_InterFusion_Joint_Attention_SLoss_pretrained_D2DUniG2_four-inlength(7s)-batch(10)-batch_delay(1)-epoch(200)-lr(0.00025)-42-10-200-Audio_Video_Joint_Classification_Customloss-AdamW0.00025-IEMOCAP/000003.ckpt",
        # "/sdb/visitors2/SCW/checkpoint/MERSISF_av_SVC_InterFusion_Joint_Attention_SLoss_pretrained_D2DUniG2_four-inlength(7s)-batch(10)-batch_delay(1)-epoch(200)-lr(0.00025)-42-10-200-Audio_Video_Joint_Classification_Customloss-AdamW0.00025-IEMOCAP/000004.ckpt",
        # "/sdb/visitors2/SCW/checkpoint/MERSISF_av_SVC_InterFusion_Joint_Attention_SLoss_pretrained_D2DUniG2_four-inlength(7s)-batch(10)-batch_delay(1)-epoch(200)-lr(0.00025)-42-10-200-Audio_Video_Joint_Classification_Customloss-AdamW0.00025-IEMOCAP/000005.ckpt",
        # "/sdb/visitors2/SCW/checkpoint/MERSISF_av_SVC_InterFusion_Joint_Attention_SLoss_pretrained_D2DUniG2_four-inlength(7s)-batch(10)-batch_delay(1)-epoch(200)-lr(0.00025)-42-10-200-Audio_Video_Joint_Classification_Customloss-AdamW0.00025-IEMOCAP/000008.ckpt",
        # "/sdb/visitors2/SCW/checkpoint/MERSISF_av_SVC_InterFusion_Joint_Attention_SLoss_pretrained_D2DUniG2_four-inlength(7s)-batch(10)-batch_delay(1)-epoch(200)-lr(0.00025)-42-10-200-Audio_Video_Joint_Classification_Customloss-AdamW0.00025-IEMOCAP/000014.ckpt",
        # "/sdb/visitors2/SCW/checkpoint/MERSISF_av_SVC_InterFusion_Joint_Attention_SLoss_pretrained_D2DUniG2_four-inlength(7s)-batch(10)-batch_delay(1)-epoch(200)-lr(0.00025)-42-10-200-Audio_Video_Joint_Classification_Customloss-AdamW0.00025-IEMOCAP/000044.ckpt",
        # "/sdb/visitors2/SCW/checkpoint/MERSISF_av_SVC_InterFusion_Joint_Attention_SLoss_pretrained_D2DUniG2_four-inlength(7s)-batch(10)-batch_delay(1)-epoch(200)-lr(0.00025)-42-10-200-Audio_Video_Joint_Classification_Customloss-AdamW0.00025-IEMOCAP/000045.ckpt",
        # "/sdb/visitors2/SCW/checkpoint/MERSISF_av_SVC_InterFusion_Joint_Attention_SLoss_pretrained_D2DUniG2_four-inlength(7s)-batch(10)-batch_delay(1)-epoch(200)-lr(0.00025)-42-10-200-Audio_Video_Joint_Classification_Customloss-AdamW0.00025-IEMOCAP/000060.ckpt",
        # "/sdb/visitors2/SCW/checkpoint/MERSISF_av_SVC_InterFusion_Joint_Attention_SLoss_pretrained_D2DUniG2_four-inlength(7s)-batch(10)-batch_delay(1)-epoch(200)-lr(0.00025)-42-10-200-Audio_Video_Joint_Classification_Customloss-AdamW0.00025-IEMOCAP/000067.ckpt",
        # "/sdb/visitors2/SCW/checkpoint/MERSISF_av_SVC_InterFusion_Joint_Attention_SLoss_pretrained_D2DUniG2_four-inlength(7s)-batch(10)-batch_delay(1)-epoch(200)-lr(0.00025)-42-10-200-Audio_Video_Joint_Classification_Customloss-AdamW0.00025-IEMOCAP/000074.ckpt",
        # "/sdb/visitors2/SCW/checkpoint/MERSISF_av_SVC_InterFusion_Joint_Attention_SLoss_pretrained_D2DUniG2_four-inlength(7s)-batch(10)-batch_delay(1)-epoch(200)-lr(0.00025)-42-10-200-Audio_Video_Joint_Classification_Customloss-AdamW0.00025-IEMOCAP/000087.ckpt",
        # "/sdb/visitors2/SCW/checkpoint/MERSISF_av_SVC_InterFusion_Joint_Attention_SLoss_pretrained_D2DUniG2_four-inlength(7s)-batch(10)-batch_delay(1)-epoch(200)-lr(0.00025)-42-10-200-Audio_Video_Joint_Classification_Customloss-AdamW0.00025-IEMOCAP/000109.ckpt",
        "/sdb/visitors2/SCW/checkpoint/MERSISF_av_SVC_InterFusion_Joint_Attention_SLoss_pretrained_D2DUniG2_four-inlength(7s)-batch(10)-batch_delay(1)-epoch(200)-lr(0.00025)-42-10-200-Audio_Video_Joint_Classification_Customloss-AdamW0.00025-IEMOCAP/000122.ckpt",
    ]
    data = IEMOCAP(
        root="/sdb/visitors2/SCW/data/IEMOCAP/Feature/IEMOCAP-crop-7s-AVSplit-MFCC-KeyFrameGetterBasedIntervalSampling-Permute_Channel-.npy",
        mode="av",
        videomode="crop",
        cascPATH="/home/visitors2/SCW/torchTraining/utils/haarcascade_frontalface_alt2.xml",
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
    
    
    dictionary = torch.load(
        model_paths[0],
        map_location=torch.device("cpu"),
    )
    data = Subset(data, indices=dictionary["train_indices"])
    print(len(data))
    num_samples = 50
    indices = np.random.choice(len(data), num_samples, replace=False)
    print(f"数据索引：{indices}")
    sampler = torch.utils.data.SubsetRandomSampler(indices)
    dataloader = torch.utils.data.DataLoader(data, batch_size=1, sampler=sampler, num_workers=0)
    
    for ind, data in enumerate(dataloader):
        for j, path in enumerate(model_paths):
            # 加载模型
            dictionary = torch.load(
                path,
                map_location=torch.device("cpu"),
            )
            net = MER_SISF_Conv2D_SVC_InterFusion_Joint_Attention_SLoss_Luck_UniG_pretrained(
                num_classes=4,
                input_size = [[189, 40], [150, 150]],
                model_name = ["md-ease", "resnet50"],
                last_hidden_dim = [320, 320],
                seq_len = [189, 30],
                enable_classifier=False,
            )

            weights_dict = {}
            for k, v in dictionary["net"].items():
                new_k = k.replace("module.", "") if "module" in k else k
                weights_dict[new_k] = v

            net.load_state_dict(weights_dict)

            # 随机从数据加载器中获取数据
            
            # 可视化并保存特征
            plot_image(net, data, outdir=f"./output/videoImage/data{indices[ind]}", info=f"video{j}_image", device=device)

def plot_ClassificationReport_BER_SISF_4_CREMA_D():
    print("加载模型")
    dictionary = torch.load(
        "/sdb/visitors2/SCW/checkpoint/BER_SISF_4-inlength(7s)-batch(10)-batch_delay(1)-epoch(200)-lr(0.00025)-42-10-200-Audio_Video_Joint_Classification_Customloss-AdamW0.00025-IEMOCAP/000103.ckpt",
        map_location=torch.device("cpu"),
    )
    data = CREMA_D(
        root="/sdb/visitors2/SCW/data/CREMA-D/Feature/CREMA-D--AVSplit-MFCC-KeyFrameGetterBasedIntervalSampling-Permute_Channel-.npy",
        mode="av",
        filter={
            "dropna": {"label": ["xxx", "disgusted", "fearful"]},
        },
    )
    print(len(data))
    dataloader = torch.utils.data.DataLoader(
        data, batch_size=10, shuffle=False, num_workers=0
    )

    net = BER_SISF(
        num_classes=4,
        input_size = [[189, 40], [150, 150]],
        last_hidden_dim = [320, 320],
        seq_len = [189, 30] ,
        num_frames = 30,
        enable_classifier=True,
    )
    weights_dict = {}
    for k, v in dictionary["net"].items():
        new_k = k.replace("module.", "") if "module" in k else k
        weights_dict[new_k] = v

    net.load_state_dict(weights_dict)

    plot_ClassificationReport(
        net,
        dataloader,
        outdir="./output/classification_report/",
        info="BER_FF_4",
        device=device,
    )


def plot_ClassificationReport_BER_SISF_4_IEMOCAP():
    print("加载模型")
    dictionary = torch.load(
        "/sdb/visitors2/SCW/checkpoint/BER_SISF_4_CREMA_D-inlength(7s)-batch(8)-batch_delay(1)-epoch(200)-lr(0.00025)-42-8-200-Audio_Video_Joint_Classification_Customloss-AdamW0.00025-CREMA_D/000093.ckpt",
        map_location=torch.device("cpu"),
    )
    data = IEMOCAP(
        root="/sdb/visitors2/SCW/data/IEMOCAP/Feature/IEMOCAP-crop-7s-AVSplit-MFCC-KeyFrameGetterBasedIntervalSampling-Permute_Channel-.npy",
        mode="av",
        videomode="crop",
        cascPATH="/home/visitors2/SCW/torchTraining/utils/haarcascade_frontalface_alt2.xml",
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
    print(len(data))
    dataloader = torch.utils.data.DataLoader(
        data, batch_size=10, shuffle=False, num_workers=0
    )

    net = BER_SISF(
        num_classes=4,
        input_size = [[189, 40], [150, 150]],
        last_hidden_dim = [320, 320],
        seq_len = [189, 30] ,
        num_frames = 30,
        enable_classifier=True,
    )
    weights_dict = {}
    for k, v in dictionary["net"].items():
        new_k = k.replace("module.", "") if "module" in k else k
        weights_dict[new_k] = v

    net.load_state_dict(weights_dict)

    plot_ClassificationReport(
        net,
        dataloader,
        outdir="./output/classification_report/",
        info="BER_SISF",
        device=device,
    )
device = torch.device("cuda:0")


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
    # plot_similarity_MIFFNet_Conv2D_SVC_InterFusion_Joint_Attention()
    
    
    # # # # # # # # # # # # # # # # # # # # # # # # 
    
    # plot_tsne_Base_AudioNet_MDEA_4()
    # plot_tsne_Base_AudioNet_MDEA_6()
    # plot_tsne_Base_AudioNet_Resnet50_4()
    # plot_tsne_Base_AudioNet_Resnet50_6()
    # plot_tsne_Base_VideoNet_MSDC_4()
    # plot_tsne_Base_VideoNet_MSDC_6()
    
    # plot_tsne_MER_VideoNet_six_resnet50()
    
    # plot_tsne_BER_SISF_4()
    # plot_tsne_BER_SISF_6()
    plot_tsne_MERSISF_av_SVC_InterFusion_Joint_Attention_SLoss_pretrained_D2DUniG2_four()
    plot_tsne_MERSISF_av_SVC_InterFusion_Joint_Attention_SLoss_pretrained_D2DUniG2_six()
    
    # plot_rt_BER_SISF_4()
    # plot_rt_BER_FF_4()
    
    plot_features_BER_SISF_4()
    # plot_image_BER_SISF_E()
    
    plot_ClassificationReport_BER_SISF_4_CREMA_D()
    # plot_ClassificationReport_BER_SISF_4_IEMOCAP()
    pass