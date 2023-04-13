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

    data = IEMOCAP(
        # root="/sdb/visitors2/SCW/data/IEMOCAP/Feature/IEMOCAP-7s-audio_AudioSplit-audio_MFCC_Permute_Channel-.npy",
        root="/sdb/visitors2/SCW/data/IEMOCAP/Feature/IEMOCAP-crop-7s-audio_AVSplit-audio_MFCC_Permute_Channel-video_KeyFrameGetterBasedIntervalSampling_Permute_Channel-.npy",
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
    dataloader = torch.utils.data.DataLoader(
        data, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

    # net = AudioNet(
    #     num_classes=4, model_name="lightresmultisernet2", last_hidden_dim=320, enable_classifier=False)

    net = MIFFNet_Conv2D_SVC_InterFusion(
        num_classes=4,
        input_size=[[189, 40], [150, 150]],
        model_name=["lightmultisernet", "resnet50"],
        last_hidden_dim=[320, 320],
        seq_len=[189, 30],
        enable_classifier=False
    )
    # net = VideoNet_Conv2D(
    #     num_classes=4, model_name="resnet50", last_hidden_dim=320, enable_classifier=False)
    print("加载模型")
    dictionary = torch.load(
        "/sdb/visitors2/SCW/checkpoint/MIFFNet-step(1s)-window(3s)-inlength(7s)-model(lightmultiressernet|resnet50)-batch(8)-batch_delay(1)-epoch(889)-lr(0.00025)-test3-42-8-200-Audio_Video_Fusion_Classification-AdamW0.00025-IEMOCAP/000175.ckpt.")

    weights_dict = {}
    for k, v in dictionary['net'].items():
        new_k = k.replace('module.', '') if 'module' in k else k
        weights_dict[new_k] = v

    net.load_state_dict(weights_dict)

    plot_tsne(net, dataloader, outdir='./output/t-sne/',
              info="MIFFNet")
