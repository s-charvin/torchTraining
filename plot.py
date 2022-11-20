import os
import sys
if sys.platform.startswith('linux'):
    os.chdir("/home/visitors2/SCW/torchTraining")
elif sys.platform.startswith('win'):
    pass
from utils.visuallize import *

if __name__ == '__main__':
    # plot_spectrogram, plot_melspectrogram, plot_RGBDifference
    audio_list = [
        "/sdb/visitors2/SCW/data/IEMOCAP/Session1/sentences/wav/Ses01F_script01_3/Ses01F_script01_3_F005.wav"]
    video_list = [
        "/sdb/visitors2/SCW/data/IEMOCAP/Session1/sentences/video/Ses01F_script01_3/Ses01F_script01_3_F005_crop.avi"]
    plot_spectrogram(audio_list)
    plot_RGBDifference(video_list)
