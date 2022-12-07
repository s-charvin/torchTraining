import torch
import sys
import os
import json
from nvitop import select_devices, Device
import time
import datetime
from redis import Redis
import psutil


class RedisClient:
    ''' 创建 Redis 客户端示例
        1. 用 List 列表存储已经加入训练队伍的任务 wait_queue, run_queue, complete_queue, close_queue
        2. 每个任务保存的信息格式:
        task_content = {
            "task_id": None, # 当前任务特有的任务 ID
            "state":None, # 当前任务状态(running, waiting, completed, closed)
            "system_pid": os.getpid(), # 当前任务进程的 PID
            "create_time": None, # 任务实际创建时间
            "update_time": None, # 任务更新(查询/重启触发)时间
            .
            .
            .}
    '''
    #

    def __init__(self, host, port, min_free_memory, password=None, username=None):
        self.min_free_memory = min_free_memory
        self.client = Redis(host=host,
                            port=port,
                            password=password,
                            username=username,
                            decode_responses=True,
                            charset='UTF-8',
                            encoding='UTF-8')

    def join_wait_queue(self, task_name):
        """加入当前进程执行的任务到等待队列
        Returns:
            str: task_id, 返回任务特有的 ID
        """
        curr_time = datetime.datetime.now()
        creat_time = datetime.datetime.strftime(
            curr_time, '%Y-%m-%d %H:%M:%S')  # 获取当前创建时间
        self.task_id = str(os.getpid()) + '*' + str(
            int(time.mktime(time.strptime(creat_time, "%Y-%m-%d %H:%M:%S"))))  # 计算当前创建任务 ID
        content = {
            "task_name": task_name,
            "task_id": self.task_id,
            "system_pid": os.getpid(),
            "use_gpus": "",
            "state": "waiting",
            "create_time": creat_time,
        }
        # 获取等待任务列表的所有任务元素数量
        wait_num = len(self.client.lrange('wait_queue', 0, -1))
        # 将任务的字典类型数据转换为符合 json 格式的字符串, 并将其放入等待任务列表的尾部
        self.client.rpush("wait_queue", json.dumps(content))
        if wait_num == 0:
            print(f"任务加入成功, 目前排第一位哦, 稍后就可以开始训练了！")
        else:
            print(f"任务加入成功, 正在排队中！ 前方还有 {wait_num} 个训练任务！")
        print(f"\n *** tips: 如果想要对正在排队的任务进行调整可以移步 Redis 客户端进行数据修改, 只建议进行修改 state 参数以及删除训练任务操作, 其他操作可能会影响任务排队的稳定性. *** \n ")
        return self.task_id

    def is_my_turn(self):
        """
        判断当前任务是否排到等待队列的第一位.
        """
        # 获取等待任务列表的第一个任务元素
        curr_task = json.loads(self.client.lrange('wait_queue', 0, -1)[0])
        return curr_task['task_id'] == self.task_id

    def pop_wait_queue(self):
        """
        如果当前任务在等待队列第一位, 且经判断 GPU 环境有余量, 则尝试将当前任务弹出, 进入运行队列中.
        如果成功, 返回 True. 否则就继续存放在等待列表, 且返回 False.
        (最好仅在 GPU 空闲, 且能支持此任务运行时调用, 否则运行出错后仍会重新进入等待队列, 且为最末处)
        """
        if self.is_my_turn():
            task = json.loads(self.client.lrange('wait_queue', 0, -1)[0])
            if task['task_id'] == self.task_id:
                gpus = Device.from_cuda_visible_devices()
                available_gpus = select_devices(
                    gpus, min_free_memory=self.min_free_memory)
                if len(gpus) == len(available_gpus):
                    curr_time = datetime.datetime.now()
                    curr_time = datetime.datetime.strftime(
                        curr_time, '%Y-%m-%d %H:%M:%S')
                    next_task = self.client.lpop("wait_queue")
                    task["state"] = "running"
                    task["use_gpus"] = ",".join(
                        [str(i) for i in available_gpus]),
                    task["run_time"] = curr_time
                    self.client.rpush("run_queue", json.dumps(task))
                    print(f"\n 更新任务队列成功, 当前任务已被置于运行队列!")
                    return True
                wait_num = len(self.client.lrange('wait_queue', 0, -1))
                print(f"尝试更新任务队列失败, 当前任务仍需等待. 显存容量不足！")
                return False
        wait_num = len(self.client.lrange('wait_queue', 0, -1))
        print(f"尝试更新任务队列失败, 当前任务仍需等待. 前方还有 {wait_num} 个训练任务！")
        return False

    def is_can_run(self):
        """
        判断当前任务可以运行队列中, 是则返回 True, 否则返回 False.
        """
        # 获取运行任务列表的所有任务元素列表
        run_tasks = self.client.lrange('run_queue', 0, -1)

        for i, task_ in enumerate(run_tasks):
            task = json.loads(task_)
            if task["task_id"] == self.task_id:
                return True
        return False

    def pop_run_queue(self, success=True):
        """
        如果当前任务在运行队列中, 则从运行队列弹出当前任务, 进入等待(运行失败)或成功队列中, 成功后, 返回 True. 否则返回 False.
        (运行出错的任务会重新进入等待队列, 且为最末处)
        """
        run_tasks = self.client.lrange('run_queue', 0, -1)

        for i, task_ in enumerate(run_tasks):
            task = json.loads(task_)
            if task["task_id"] == self.task_id:
                self.client.lrem("run_queue", 0, json.dumps(task))
                if success:
                    curr_time = datetime.datetime.now()
                    curr_time = datetime.datetime.strftime(
                        curr_time, '%Y-%m-%d %H:%M:%S')
                    task["completed_time"] = curr_time
                    task["state"] = "completed"
                    self.client.rpush("complete_queue", json.dumps(task))
                    print(f"当前任务已训练完成!")
                else:
                    task["state"] = "waiting"
                    task.pop("run_time", "")
                    self.client.rpush("wait_queue", json.dumps(task))
                    print(f"显存占用超出导致任务运行出错, 已将当前任务重新置于等待列表, 等候稍后重试!")
        return False

    def check_data(self):

        run_tasks = self.client.lrange('run_queue', 0, -1)
        wait_tasks = self.client.lrange('wait_queue', 0, -1)

        for i, task_ in enumerate(run_tasks):
            task = json.loads(task_)
            pid = int(task['system_pid'])
            if not psutil.pid_exists(pid):
                print(f"发现运行队列有残余数据, 任务进程为{pid}, 本地并无此任务!")
                self.client.lrem("run_queue", 0, json.dumps(task))

        for i, task_ in enumerate(wait_tasks):
            task = json.loads(task_)
            pid = int(task['system_pid'])
            if not psutil.pid_exists(pid):
                self.client.lrem("wait_queue", 0, json.dumps(task))
                print(f"发现等待队列有残余数据, 任务进程为{pid}, 本地并无此任务!")


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def collate_fn_pad(batch):

    # 一个 batch 大小的列表，内部含有数据元组，如：(waveform, label)
    # Gather in lists

    tensors = []
    genders = []
    labels = []
    nfs = None

    for spec, gender, label in batch:
        tensors.append(spec)
        genders.append(gender)
        labels.append(label)
        # texts.append(texts)
    tensors_lengths = [i.shape[0] for i in tensors]
    tensors = torch.nn.utils.rnn.pad_sequence(
        tensors, batch_first=True, padding_value=0.)

    packed_tensors = torch.nn.utils.rnn.pack_padded_sequence(
        tensors, tensors_lengths, batch_first=True, enforce_sorted=False)

    return packed_tensors, genders, torch.LongTensor(labels), nfs


def collate_fn_split(batch):
    fix_len = 260
    hop_len = 130

    # 一个batch大小的列表，内部含有数据元组，如：()
    # Gather in lists
    tensors = []
    genders = []
    labels = []
    # texts = []
    nfs = []
    for spec, gender, label in batch:
        if len(spec.shape) == 2:
            # item, label: [frame, feature] , int
            # frame 分段为固定大小, 125代表采样率为16000, 帧移为256条件下, 2秒时间长度的数据
            # 计算总分段数
            if spec.shape[0] < fix_len - hop_len:
                nf = 1
            else:
                nf = (spec.shape[0] - fix_len + hop_len) // hop_len + 1
            nfs.append(nf)
            # 计算每段的起始位置
            indf = [i*hop_len for i in range(nf)]
            # 正式开始分段数据
            for i in range(nf-1):
                tensors.append(
                    spec[indf[i]:indf[i] + fix_len, :])
                genders.append(gender)
                labels.append(label)
            tensors.append(spec[indf[-1]:, :])
            genders.append(gender)
            labels.append(label)

        elif len(spec.shape) == 1:
            fix_len = 16000 * 4
            hop_len = 16000 * 2
            if spec.shape[0] < fix_len - hop_len:
                nf = 1
            else:
                nf = (spec.shape[0] - fix_len + hop_len) // hop_len + 1
            nfs.append(nf)
            # 计算每段的起始位置
            indf = [i*hop_len for i in range(nf)]
            # 正式开始分段数据
            for i in range(nf-1):
                tensors.append(
                    spec[indf[i]:indf[i] + fix_len])
                genders.append(gender)
                labels.append(label)
            tensors.append(spec[indf[-1]:])
            genders.append(gender)
            labels.append(label)
            # 对不等长的数据进行填充
    tensors_lengths = [i.shape[0] for i in tensors]
    tensors = torch.nn.utils.rnn.pad_sequence(
        tensors, batch_first=True, padding_value=0.)
    if tensors.shape[1] != fix_len:
        print(tensors.shape[1])
        tensors = torch.nn.functional.pad(
            tensors, (0, 0, 0, fix_len-tensors.shape[1], 0, 0), mode='constant', value=0.)
    packed_tensors = torch.nn.utils.rnn.pack_padded_sequence(
        tensors, tensors_lengths, batch_first=True, enforce_sorted=False)

    return packed_tensors, genders, torch.LongTensor(labels), nfs
