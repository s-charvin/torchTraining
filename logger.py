"""
logger.py

Altered version of Logger.py by hujinsen. Original source can be found at:
    
"""
import os
import smtplib
from email.mime.text import MIMEText
from email.header import Header
from email.utils import formataddr
from torch.utils.tensorboard import SummaryWriter


class Logger(object):
    """Tensorboard logger."""

    def __init__(self, log_dir, model_name, flush_secs=120):
        """Initialize summary writer.
        para log_dir: 日志保存基路径
        para model_name: 模型名称
        para flush_secs: 将记载日志刷新到磁盘的频率，默认120秒
        """

        self.file_dir = os.path.join(log_dir, model_name)
        if not os.path.exists(self.file_dir):
            os.makedirs(self.file_dir, exist_ok=True)
        self.writer = SummaryWriter(
            log_dir=self.file_dir, flush_secs=flush_secs)
        print("构建logger.")

    def scalar_summary(self, tag, value, step):
        """在独立图上画一条线或多条线
        para tag: (可选section/)数据名称，用于区分训练过程不同类型数值的变化
        para value: 单个float型数值或字典包含的float型数值, {'h': h_value, 'w': w_value}
        para step: 训练的 step
        para walltime: 记录发生的时间，默认为 time.time()
        """

        if isinstance(value, float):  # 判断类型，确定画单线还是多线
            self.writer.add_scalar(
                tag=tag, scalar_value=value, global_step=step)
        elif isinstance(value, dict):
            self.writer.add_scalars(
                main_tag=tag, tag_scalar_dict=value, global_step=step)
        self.writer.flush()

    def histo_summary(self, tag, values, step, bins="tensorflow"):
        """记录一组数据的直方图。
        para tag: (可选section/)数据名称，用于区分训练过程不同类型数值的变化
        para values: Tensor, array型的 histogram 数据
        para step: 训练的 step
        para bins: 数字: 等宽的分桶范围 序列: 非均匀分桶宽度 字符串: 选择计算最佳分桶方法
        para walltime: 记录发生的时间，默认为 time.time()
        para maxbins: 最大分桶数
        """
        # 直方图信息 日志
        self.writer.add_histogram(
            tag, values=values, global_step=step, bins=bins, walltime=None, max_bins=None)
        self.writer.flush()

    def image_summary(self, tag, images, step):
        """显示单张或多张图片
        para tag: (可选section/)数据名称，用于区分训练过程不同类型数值的变化
        para images: 单张图片或图片tensor、array
        para step: 训练的 step
        para walltime: 记录发生的时间，默认为 time.time()
        """
        if len(images.shape) == 2:
            self.writer.add_image(
                tag=tag, img_tensor=images, global_step=step, walltime=None, dataformats='CHW')
        elif len(images.shape) == 3:
            self.writer.add_images(
                tag=tag, img_tensor=images, global_step=step, walltime=None, dataformats='NCHW')
        self.writer.flush()

    def embedding_summary(self, mat, metadata=None, label_img=None, global_step=None, tag='default', metadata_header=None):
        """在二维或三维空间可视化 embedding 向量。、
        para tag: 
        para mat: 一个矩阵，每行代表特征空间的一个数据点
        para metadatas: 一维列表，mat 中每行数据的 label，大小应和 mat 行数相同
        para label_img: 一个形如 NxCxHxW 的张量，对应 mat 每一行数据显示出的图像，N 应和 mat 行数相同
        para step: 训练的 step
        para metadata_header: 数据名称，不同名称的数据将分别展示
        """
        self.writer.add_embedding(mat=mat, metadata=metadata, label_img=label_img,
                                  global_step=global_step, tag=tag, metadata_header=metadata_header)
        self.writer.flush()

    def graph_summary(self, model, input_to_model, verbose=False, use_strict_trace=True):
        """可视化神经网络
        para model: 待可视化的网络模型
        para input_to_model: 待输入神经网络的变量或一组变量
        para verbose: 
        para use_strict_trace: 
        """
        self.writer.add_graph(model, input_to_model=input_to_model,
                              verbose=verbose, use_strict_trace=use_strict_trace)
        self.writer.flush()

    # def video_summary(self, tag, images, step):
    #     """在独立图上画一条线或多条线
    #     para tag: (可选section/)数据名称，用于区分训练过程不同类型数值的变化
    #     para images: 单张图片或图片tensor、array
    #     para step: 训练的 step
    #     para walltime: 记录发生的时间，默认为 time.time()
    #     """
    #     self.writer.add_video(tag, vid_tensor, global_step=None, fps=4, walltime=None)
    #

    # def audio_summary(self, tag, images, step):
    #     """在独立图上画一条线或多条线
    #     para tag: (可选section/)数据名称，用于区分训练过程不同类型数值的变化
    #     para images: 单张图片或图片tensor、array
    #     para step: 训练的 step
    #     para walltime: 记录发生的时间，默认为 time.time()
    #     """
    #     self.writer.add_audio(tag, snd_tensor, global_step=None, sample_rate=44100, walltime=None)
    #

    # def text_summary(self, tag, images, step):
    #     """在独立图上画一条线或多条线
    #     para tag: (可选section/)数据名称，用于区分训练过程不同类型数值的变化
    #     para images: 单张图片或图片tensor、array
    #     para step: 训练的 step
    #     para walltime: 记录发生的时间，默认为 time.time()
    #     """
    #     self.writer.add_text(tag, text_string, global_step=None, walltime=None)
    #

    # def pr_curve_summary(self, tag, images, step):
    #     """Add images summary. 在独立图上画一条线或多条线
    #     para tag: (可选section/)数据名称，用于区分训练过程不同类型数值的变化
    #     para images: 单张图片或图片tensor、array
    #     para step: 训练的 step
    #     para walltime: 记录发生的时间，默认为 time.time()
    #     """
    #     self.writer.add_pr_curve(tag, labels, predictions, global_step=None, num_thresholds=127, weights=None, walltime=None)
    #

    # def mesh_summary(self, tag, images, step):
    #     """Add images summary. 在独立图上画一条线或多条线
    #     para tag: (可选section/)数据名称，用于区分训练过程不同类型数值的变化
    #     para images: 单张图片或图片tensor、array
    #     para step: 训练的 step
    #     para walltime: 记录发生的时间，默认为 time.time()
    #     """
    #     self.writer.add_mesh(tag, vertices, colors=None, faces=None, config_dict=None, global_step=None, walltime=None)
    #

    # def hparams_summary(self, tag, images, step):
    #     """Add images summary. 在独立图上画一条线或多条线
    #     para tag: (可选section/)数据名称，用于区分训练过程不同类型数值的变化
    #     para images: 单张图片或图片tensor、array
    #     para step: 训练的 step
    #     para walltime: 记录发生的时间，默认为 time.time()
    #     """
    #     self.writer.add_hparams(hparam_dict, metric_dict, hparam_domain_discrete=None, run_name=None)
    #


def sendmail(Text, config):
    """向指定邮箱发送含Text内容的邮件
    Args:
        Text (str): 右键主体内容
        config (dict): 指定参数（发信人、收信人、第三方SMTP参数(可选)）
    """
    # 第三方 SMTP 服务
    mail_host = config["logs"]["SMTP"]["mail_host"]  # 服务器
    mail_port = int(config["logs"]["SMTP"]["mail_port"])  # 端口
    mail_user = config["logs"]["SMTP"]["mail_user"]  # 用户名
    mail_pass = config["logs"]["SMTP"]["mail_pass"]  # 口令

    # 三个参数：第一个为文本内容，第二个 plain 设置文本格式，第三个 utf-8 设置编码
    message = MIMEText(Text, 'plain', 'utf-8')  # 发送的主体内容
    message["Accept-Language"] = Header("zh-CN", 'utf-8')
    # message["Accept-Charset"] = Header("ISO-8859-1,utf-8", 'utf-8')

    # 标准邮件需要三个头部信息： From, To, 和 Subject
    message['Subject'] = Header('自动化脚本', 'utf-8')  # 标题
    message['To'] = formataddr(
        ["用户", config["logs"]["to_mail"]], 'utf-8')  # 接收者

    try:
        if mail_host:  # 使用第三方SMTP服务
            message['From'] = formataddr(
                ["自动化脚本_健康打卡", mail_user], 'utf-8')  # 发送者

            smtpObj = smtplib.SMTP_SSL(
                mail_host, mail_port)  # 发件人邮箱中的SMTP服务器，端口是
            smtpObj.login(mail_user, mail_pass)  # 括号中对应的是发件人邮箱账号、邮箱密码
            smtpObj.sendmail(mail_user,
                             config["logs"]["to_mail"], message.as_string())
        else:  # 使用本地服务
            message['From'] = formataddr(
                ["自动化脚本_健康打卡", config["from_mail"]], 'utf-8')  # 发送者
            smtpObj = smtplib.SMTP('localhost')
            smtpObj.sendmail(config["logs"]["from_mail"],
                             config["logs"]["to_mail"], message.as_string())
        print("邮件发送成功")
        smtpObj.quit()  # 关闭连接
    except smtplib.SMTPException as Error:
        print(f"无法发送邮件\n Error: {Error}")
