from torch import Tensor
import torch
import torch.nn.functional as F


class VideoSplit(torch.nn.Module):
    r"""输入字典格式存储的数据, 对其中 "video" 模态进行处理, 按照指定时长进行截取

    Args:
        frame_rate (int, optional): Sample rate of audio signal. (Default: ``16000``)
    """

    def __init__(self,
                 frame_rate,
                 durations: float = 3.) -> None:
        super(VideoSplit, self).__init__()
        self.video_samples = int(frame_rate * durations)

    def forward(self, datadict: dict) -> Tensor:
        r"""
        Args:
            datadict (dict): 字典格式存储的数据.
        Returns:
            dataframe: .
        """
        columns = datadict.keys()
        index = range(len(datadict["path"]))
        newdatadict = {c: [] for c in columns}

        for i in index:

            video = datadict["video"][i]  # 当前行的视频数据 [seq,h,w,c] RGB

            if video.shape[0] > self.video_samples:
                samples = int(  # 处理比指定时间段长的数据, 将多余的数据截取掉
                    (video.shape[0] // self.video_samples) * self.video_samples)
                video = video[:samples, :, :, :]
            else:  # 填充比指定时间段短的数据
                video = F.pad(
                    video, (0, 0, 0, 0, 0, 0, 0, self.video_samples-video.shape[0]))
            video = torch.split(video, self.video_samples, dim=0)

            for vid in video:
                for col in columns:
                    if col == "video":
                        newdatadict[col].append(vid)
                    else:
                        try:
                            newdatadict[col].append(datadict[col][i])
                        except:
                            newdatadict[col].append(datadict[col])
        return newdatadict
