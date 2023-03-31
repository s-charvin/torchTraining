import os
import sys
if sys.platform.startswith('linux'):
    os.chdir("/home/visitors2/SCW/torchTraining")
elif sys.platform.startswith('win'):
    pass
from utils.visuallize import *
from custom_models import *
from custom_datasets import *
from custom_datasets.subset import Subset
if __name__ == '__main__':
    # plot_spectrogram, plot_melspectrogram, plot_RGBDifference
    # audio_list = [
    #     "/sdb/visitors2/SCW/data/IEMOCAP/Session1/sentences/wav/Ses01F_script01_3/Ses01F_script01_3_F005.wav"]
    # video_list = [
    #     "/sdb/visitors2/SCW/data/IEMOCAP/Session1/sentences/video/Ses01F_script01_3/Ses01F_script01_3_F005_crop.avi"]
    # plot_spectrogram(audio_list)
    # plot_RGBDifference(video_list)

    net = MIFFNet_Conv2D_SVC_InterFusion(
        num_classes=4,
        input_size=[[189, 40], [150, 150]],
        model_name=["lightresmultisernet", "resnet50"],
        last_hidden_dim=[320, 320],
        seq_len=[189, 30],
        enable_classifier=False
    )

    dictionary = torch.load(
        "/sdb/visitors2/SCW/checkpoint/test3-42-10-200-Audio_Video_Fusion_Classification-AdamW0.0001-IEMOCAP/000124.ckpt",)
    weights_dict = {}
    for k, v in dictionary['net'].items():
        new_k = k.replace('module.', '') if 'module' in k else k
        weights_dict[new_k] = v

    net.load_state_dict(weights_dict)
    data = IEMOCAP(
        root="/sdb/visitors2/SCW/data/IEMOCAP/Feature/IEMOCAP-crop-7s-AVSplit-MFCC-KeyFrameGetterBasedIntervalSampling-Permute_Channel-.npy",
        mode="av",
        videomode="crop",
        cascPATH="/home/visitors2/SCW/torchTraining/utils/haarcascade_frontalface_alt2.xml",
        filter={
            "replace": {"label": {"excited": "happy"}},
            "dropna":
                {
                    "label":
                        [
                            "other",
                            "xxx",
                            "frustrated",
                            "disgusted",
                            "fearful",
                            "surprised",
                        ],
            },
            "sort_values": "duration",
        })
    data = Subset(data, dictionary["test_indices"])
    dataloader = torch.utils.data.DataLoader(
        data, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

    plot_tsne(None, dataloader, outdir='./output/tsne.png')
