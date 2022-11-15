import io
from turtle import shape
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import decord


class Video_pad_cut(torch.nn.Module):
    r""" 根据提供的视频帧数数量, 填充或剪切输入视频帧.
    Args:
        samplingNum (int, optional):  -1, 不做处理. Defaults to -1.
        mode (str, optional): "left": 左填充, "right": 右填充, "biside": 两侧填充. Defaults to "right".
    """

    def __init__(self, FramNum: int = -1, mode: str = "right") -> None:
        super(Video_pad_cut, self).__init__()
        self.FramNum = FramNum
        self.mode = mode
        if self.FramNum < -1:
            raise ValueError("采样点数量必须为正数或-1")

    def forward(self, Video: torch.Tensor) -> torch.Tensor:
        r"""

        Args:
            waveform (torch.Tensor): Video of dimension of `(3,N,H,W)`.

        Returns:
            torch.Tensor: Video of dimension of `(3,N,H,W)`.
        """

        if self.FramNum == -1:
            return Video
        length = Video.shape[1]

        if length <= self.FramNum:
            if self.mode == "left":
                Video = F.pad(
                    Video, (0, 0, 0, 0, self.FramNum-length, 0), 'constant', 0)
            if self.mode == "right":
                Video = F.pad(
                    Video, (0, 0, 0, 0, 0, self.FramNum-length), 'constant', 0)
            if self.mode == "biside":
                left_pad = int((self.FramNum-length)/2.0)
                right_pad = self.FramNum-length-left_pad
                Video = F.pad(
                    Video, (0, 0, 0, 0, left_pad, right_pad), 'constant', 0)
        elif length > self.FramNum:
            if self.mode == "left":
                Video = Video[:, -self.FramNum:, :, :]
            if self.mode == "right":
                Video = Video[:, :self.FramNum, :, :]
            if self.mode == "biside":
                left_cut = int((length-self.FramNum)/2.0)
                right_cut = length-self.FramNum-left_cut
                Video = Video[:, left_cut:-right_cut, :, :]
        return Video.clone()


class KeyFrameGetterBasedInterDifference(torch.nn.Module):
    r""" 基于帧间差分的视频关键帧提取
    link:
        https://github.com/monkeyDemon/AI-Toolbox/tree/master/computer_vision/image_classification_keras/multi_label_classification
        https://github.com/monkeyDemon/AI-Toolbox/blob/master/preprocess%20ToolBox/keyframes_extract_tool/keyframes_extract_diff.py
        https://github.com/MiaoMiaoKuangFei/VideoFaceDetector/blob/master/KeyFrameGetter.py
    Args:

    """

    def __init__(self, window=10, smooth=False, alpha=0.07) -> None:
        super().__init__()
        self.window = window  # 关键帧数量
        self.smooth = smooth  # 最终帧的平滑效果, 减小差异性
        self.alpha = alpha  # 差分系数

    def graying_image(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # 灰度化
        gray_image = cv2.GaussianBlur(gray_image, (3, 3), 0)  # 高斯滤波
        return gray_image

    def abs_diff(self, pre_image, curr_image):
        '''
        计算两个图像之间的绝对差分
        :param pre_image:前一帧中的图像
        :param curr_image:当前帧中的图像
        :return:
        '''
        gray_pre_image = self.graying_image(pre_image)
        gray_curr_image = self.graying_image(curr_image)
        diff = cv2.absdiff(gray_pre_image, gray_curr_image)
        res, diff = cv2.threshold(
            diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #  fixme：这里先写成简单加和的形式
        cnt_diff = np.sum(np.sum(diff))
        return cnt_diff

    def load_diff(self, video):

        diff = []
        frm = 0
        pre_image = np.array([])
        curr_image = np.array([])
        for img in video:
            frm = frm + 1
            #  这里是主体
            #  记录每次的数据
            if frm == 1:  # 第一帧因为没有上一帧, 因此使用与其自身做差分
                pre_image = img
                curr_image = img
            else:
                pre_image = curr_image
                curr_image = img

            # 进行差分, 并保存每两连续帧间的差分值
            diff.append(self.abs_diff(pre_image, curr_image))

        if self.smooth:  # 进行帧平滑
            diff = self.exponential_smoothing(diff)

        #  标准化差异数据
        diff = np.array(diff)
        diff_mean = np.mean(diff)
        dev = np.std(diff)
        diff = (diff - diff_mean) / dev

        return diff

    def pick_idx(self, diff):
        idx = []
        for i, d in enumerate(diff):
            # 根据一个动态窗口, 在动态窗口内寻找差异峰值, 避免使用太多高峰值旁边的一些值
            ub = len(diff) - 1
            lb = 0
            if not ((i-(self.window//2)) < lb):
                lb = i - self.window // 2

            if not ((i+(self.window//2)) > ub):
                ub = i + self.window // 2

            comp_window = diff[lb:ub]
            if d >= max(comp_window):
                idx.append(i)
        return idx

    def exponential_smoothing(self, data):
        # 一阶差分指数平滑法
        s_temp = [data[0]]
        for i in range(1, len(data), 1):
            s_temp.append(self.alpha * data[i - 1] +
                          (1 - self.alpha) * s_temp[i - 1])
        return s_temp

    def get_key_frame(self, video):
        '''
        Save the key frame image
        :return:
        '''
        diff = self.load_diff(video)
        idx = self.pick_idx(diff)
        return video[idx].clone()

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        # [seq, h, w, 3] RGB
        key_frame = video.numpy()
        key_frame = self.get_key_frame(key_frame)

        if key_frame.shape[0] > 5:
            return torch.from_numpy(key_frame)
        else:
            return video

        # return self.get_key_frame(video)


class KeyFrameGetterBasedIntervalSampling(torch.nn.Module):
    r""" 基于帧间差分的视频关键帧提取
    link:
        https://github.com/monkeyDemon/AI-Toolbox/tree/master/computer_vision/image_classification_keras/multi_label_classification
        https://github.com/monkeyDemon/AI-Toolbox/blob/master/preprocess%20ToolBox/keyframes_extract_tool/keyframes_extract_diff.py
        https://github.com/MiaoMiaoKuangFei/VideoFaceDetector/blob/master/KeyFrameGetter.py
    Args:

    """

    def __init__(self, interval=3) -> None:
        super().__init__()
        self.interval = interval

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        # [seq, h, w, 3] RGB
        indx = [0]
        for i in range(video.shape[0]):
            if (i+1) % self.interval == 0:
                indx.append(i)
        return video[indx].clone()

        # return self.get_key_frame(video)


def CAST(v, L, H):
    # CAST(v, L, H) ( (v) > (H) ? 255 : (v) < (L) ? 0 : cvRound(255*((v) - (L))/((H)-(L))))
    # 255 if (v) > (H) else 0 if (v) < (L) else round(255*((v) - (L))/((H)-(L)))
    if (v) > (H):
        return 255
    elif (v) < (L):
        return 0
    else:
        return round(255*((v) - (L))/((H)-(L)))


class calcOpticalFlow(torch.nn.Module):
    r""" 根据提供的参数, 提取 OpticalFlow.
    Args:
        keepRGB (bool, optional): 是否保留原 RGB 通道. Defaults to "right".
    """

    def __init__(self, type: str = "Farneback", keepRGB: bool = True) -> None:
        super(calcOpticalFlow, self).__init__()
        self.type = type
        self.keeprgb = keepRGB

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        # [seq, h, w, 3] RGB
        if isinstance(video, torch.Tensor):
            video = video.numpy()
        flows = []
        prev_gray = cv2.cvtColor(video[0], cv2.COLOR_BGR2GRAY)
        for capture_frame in video[1:]:
            capture_image = capture_frame
            capture_gray = cv2.cvtColor(capture_image, cv2.COLOR_RGB2GRAY)

            if self.type == "Farneback":
                # OPTFLOW_FARNEBACK_GAUSSIAN, OPTFLOW_USE_INITIAL_FLOW
                # 返回一个两通道的光流向量，实际上是每个点的像素位移值
                # https://github.com/chuanenlin/optical-flow/blob/master/dense-solution.py
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, capture_gray, None, 0.702, 5, 10, 2, 7, 1.5, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
                flow_x, flow_y = flow[..., 0], flow[..., 1]

                # img_x = np.zeros_like(flow_x)
                # img_y = np.zeros_like(flow_y)

                # for i in range(flow_x.shape[0]):
                #     for j in range(flow_y.shape[1]):
                #         x = flow_x[i, j]
                #         y = flow_y[i, j]
                #         img_x[i, j] = CAST(x, -20, 20)
                #         img_y[i, j] = CAST(y, -20, 20)
                flows.append(np.stack([flow_x, flow_y], axis=-1))
                prev_gray = capture_gray
            elif self.type == "PyrLK":
                raise ValueError("还未实现的功能")
            else:
                raise ValueError("不是所支持的功能")
        flows.append(np.stack([flow_x, flow_y], axis=-1))
        flows = np.array(flows)
        if self.keeprgb:
            flows = np.concatenate([video, flows], axis=-1)
        return torch.Tensor(flows.copy())


class calcRGBDifference(torch.nn.Module):
    r""" 根据提供的参数, 提取 Difference.
    Args:
        keepRGB (bool, optional): 是否保留原 RGB 通道. Defaults to "right".
    """

    def __init__(self, type: str = "Farneback", keepRGB: bool = True) -> None:
        super(calcRGBDifference, self).__init__()
        self.type = type
        self.keeprgb = keepRGB

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        # [seq, h, w, 3] RGB
        if isinstance(video, torch.Tensor):
            video = video.numpy()
        differences = video.copy()
        for x in reversed(list(range(1, video.shape[0]))):
            # 后一张图像减去前一张图像
            differences[x, :, :, :] = video[x,
                                            :, :, :] - video[x - 1, :, :, :]

        if self.keeprgb:
            differences = np.concatenate([video, differences], axis=-1)
        return torch.Tensor(differences.copy())


class calcOpticalFlowAndRGBDifference(torch.nn.Module):
    r""" 根据提供的参数, 提取 OpticalFlow.
    Args:
        keepRGB (bool, optional): 是否保留原 RGB 通道. Defaults to "right".
    """

    def __init__(self, type: str = "Farneback", keepRGB: bool = True) -> None:
        super(calcOpticalFlowAndRGBDifference, self).__init__()
        self.type = type
        self.keeprgb = keepRGB
        self.opticalflow = calcOpticalFlow(type=type, keepRGB=False)
        self.difference = calcRGBDifference(keepRGB=False)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        # [seq, h, w, 3] RGB
        if isinstance(video, torch.Tensor):
            video = video.numpy()
        flows = self.opticalflow(video)
        differences = self.difference(video)
        if self.keeprgb:
            return torch.Tensor(np.concatenate([video, flows, differences], axis=-1).copy())
        return torch.Tensor(np.concatenate([flows, differences], axis=-1).copy())


if __name__ == '__main__':

    with open("/sdb/visitors2/SCW/data/IEMOCAP/Session3/sentences/video/Ses03M_script01_1/Ses03M_script01_1_M037.avi", "rb") as fh:
        video_file = io.BytesIO(fh.read())

    _av_reader = decord.VideoReader(
        uri=video_file,
        ctx=decord.cpu(0),
        width=-1,
        height=-1,
        num_threads=0,
        fault_tol=-1,
    )
    _fps = _av_reader.get_avg_fps()
    _duration = float(len(_av_reader)) / float(_fps)
    frame_idxs = list(range(0, len(_av_reader)))
    video = _av_reader.get_batch(frame_idxs)
    video = video.asnumpy()  # NxHxWx3 RGB

    width, height = video.shape[2], video.shape[1]
    fourcc = cv2.VideoWriter_fourcc(
        *'XVID')  # 视频的编码
    # 定义视频对象输出
    writer = cv2.VideoWriter(
        "/home/visitors2/SCW/torchTraining/custom_transforms/result.avi", fourcc, _fps, (width, height))

    # framegetter = KeyFrameGetterBasedInterDifference(
    #     window=5, smooth=False, alpha=0.07)
    # framegetter = KeyFrameGetterBasedIntervalSampling(interval=3)
    framegetter = calcOpticalFlowAndRGBDifference()
    video = framegetter(torch.Tensor(video))

    for img in video:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        writer.write(img)  # 视频保存

    writer.release()
