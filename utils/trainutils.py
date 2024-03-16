from requests import delete
import torch
import sys
import os
import json
from nvitop import select_devices, Device
import time
import datetime
from redis import Redis
import psutil
import errno
import os
import sys
import logging
import threading
import multiprocessing

import time



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

    def __init__(self, host, port, min_free_memory, gpu_maxutil, password, username):
        self.min_free_memory = min_free_memory
        self.max_gpu_util = gpu_maxutil
        self.client = Redis(host=host,
                            port=port,
                            password=password,
                            username=username,
                            decode_responses=True,
                            charset='UTF-8',
                            encoding='UTF-8')

    def join_wait_queue(self, task_name, task_config):
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
            "task_config": task_config,
            "system_ppid": os.getppid(),
            "system_pid": os.getpid(),
            "state": "waiting",
            "create_time": creat_time,
        }
        # 获取等待任务列表的所有任务元素数量
        wait_num = len(self.client.lrange('wait_queue', 0, -1))
        # 将任务的字典类型数据转换为符合 json 格式的字符串, 并将其放入等待任务列表的尾部
        self.client.rpush("wait_queue", json.dumps(content))
        if wait_num == 0:
            logging.info(
                f"任务加入成功, 任务进程({os.getppid()} | {os.getpid()}), 目前排第一位哦, 稍后就可以开始训练了！")
        else:
            logging.info(
                f"任务加入成功, 任务进程({os.getppid()} | {os.getpid()}), 正在排队中！ 前方还有 {wait_num-1} 个训练任务！")
        logging.info(
            f"\n *** tips: 如果想要对正在排队的任务进行调整可以移步 Redis 客户端进行数据修改, 其他操作可能会影响任务排队的稳定性. *** \n ")
        return self.task_id

    # 通过 "waiting" 状态判断当前程序是否还需要等待执行, 如果状态改变了, 就直接退出等待循环, 终止程序
    def is_need_wait(self):
        wait_tasks = self.client.lrange('wait_queue', 0, -1)
        for i, task_ in enumerate(wait_tasks):
            task = json.loads(task_)
            if task["task_id"] == self.task_id:
                return task["state"] == "waiting"
        return False
    # 删除当前程序的任务配置信息

    def delete_waitting_task(self):
        wait_tasks = self.client.lrange('wait_queue', 0, -1)
        for i, task_ in enumerate(wait_tasks):
            task = json.loads(task_)
            if task["task_id"] == self.task_id:
                self.client.lrem("run_queue", 0, task_)
        return True

    def delete_running_task(self):
        run = self.client.lrange('run_queue', 0, -1)
        for i, task_ in enumerate(run):
            task = json.loads(task_)
            if task["task_id"] == self.task_id:
                self.client.lrem("run_queue", 0, task_)
        return True

    def is_my_turn(self):
        """
        判断当前任务是否排到等待队列的第一位.
        """
        # 获取等待任务列表的第一个任务元素
        curr_task = json.loads(self.client.lrange('wait_queue', 0, -1)[0])
        return curr_task['task_id'] == self.task_id

    def pop_wait_queue(self):
        """
        如果当前任务在等待队列第一位, 且经判断 GPU 环境有余量,且正在运行的任务不多, 则尝试将当前任务弹出, 进入运行队列中.
        如果成功, 返回存储的任务配置. 否则就让出第一名位置, 去队尾继续等待, 返回 False.
        (最好仅在 GPU 空闲, 且能支持此任务运行时调用, 否则运行出错后仍会重新进入等待队列, 且为最末处)
        """
        if self.is_my_turn():
            task = json.loads(self.client.lrange('wait_queue', 0, -1)[0])
            run_tasks = self.client.lrange('run_queue', 0, -1)
            if task['task_id'] == self.task_id:
                gpus = Device.from_cuda_visible_devices()
                available_gpus = select_devices(
                    gpus, min_free_memory=self.min_free_memory, max_gpu_utilization=self.max_gpu_util)

                curgpus_intask = [g.strip() for g in task["task_config"]
                                  ['train']['available_gpus'].split(",")]
                num_gpus_task = int(len(curgpus_intask))

                # 获取当前使用的 GPU 上, 已经运行的任务数量, 如果太多就继续等待
                tasks_ingpu = [0 for i in curgpus_intask]
                for ind, gpu in enumerate(curgpus_intask):
                    for run_task_ in run_tasks:
                        run_task = json.loads(run_task_)
                        if gpu in run_task["task_config"]["train"]["available_gpus"]:
                            tasks_ingpu[ind] += 1

                # # 获取当前 GPU 上, 自己已经运行的任务数量, 如果太多旧继续等待
                # tasks_ingpu = [0 for i in gpus]
                # for ind, gpu in enumerate(available_gpus):
                #     for run_task_ in run_tasks:
                #         run_task = json.loads(run_task_)
                #         if str(gpu) in run_task["task_config"]["train"]["available_gpus"]:
                #             tasks_ingpu[ind] += 1

                if (max(tasks_ingpu) >= task["task_config"]["train"]["gpu_queuer"]["task_maxnum"]):
                    next_task = self.client.lpop("wait_queue")
                    self.client.rpush("wait_queue", next_task)
                    logging.info(f"尝试更新任务队列失败, 当前任务仍需等待. 所使用显卡正在运行的任务数量过多!")
                    return False

                if (num_gpus_task == len(available_gpus)):
                    curr_time = datetime.datetime.now()
                    curr_time = datetime.datetime.strftime(
                        curr_time, '%Y-%m-%d %H:%M:%S')
                    next_task = self.client.lpop("wait_queue")
                    task["state"] = "running"
                    task["run_time"] = curr_time
                    self.client.rpush("run_queue", json.dumps(task))
                    logging.info(f"\n 更新任务队列成功, 当前任务已被置于运行队列!")
                    return task["task_config"]

                next_task = self.client.lpop("wait_queue")
                self.client.rpush("wait_queue", next_task)
                logging.info(f"尝试更新任务队列失败, 当前任务仍需等待. 显卡容量不足!")
                return False

        wait_num = len(self.client.lrange('wait_queue', 0, -1))
        logging.info(f"尝试更新任务队列失败, 当前任务仍需等待. 前方还有 {wait_num} 个训练任务！")
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
                    logging.info(f"当前任务已训练完成!")
                else:
                    task["state"] = "waiting"
                    task.pop("run_time", "")
                    self.client.rpush("wait_queue", json.dumps(task))
                    logging.info(f"显存占用超出导致任务运行出错, 已将当前任务重新置于等待列表, 等候稍后重试!")
        return False

    def check_data(self):

        run_tasks = self.client.lrange('run_queue', 0, -1)
        wait_tasks = self.client.lrange('wait_queue', 0, -1)
        for i, task_ in enumerate(run_tasks):
            task = json.loads(task_)
            pid = int(task['system_pid'])
            if not pid_exists(pid):
                logging.info(f"发现运行队列有残余数据, 任务进程为{pid}, 本地并无此任务!")
                self.client.lrem("run_queue", 0, task_)

        for i, task_ in enumerate(wait_tasks):
            task = json.loads(task_)
            pid = int(task['system_pid'])
            if not pid_exists(pid):
                logging.info(f"发现等待队列有残余数据, 任务进程为{pid}, 本地并无此任务!")
                self.client.lrem("wait_queue", 0, task_)



class TaskManager:
    _instance = None
    _lock = multiprocessing.Lock()

    @classmethod
    def get_instance(cls, min_free_memory=2):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    manager = multiprocessing.Manager()
                    cls._instance = manager.TaskManager(min_free_memory)
        return cls._instance

    def __init__(self, min_free_memory=2):
        self.min_free_memory = min_free_memory
        assert min_free_memory >= 2
        self.memoryEnough = multiprocessing.Manager().Value('i', False)
        self.gpu_occupied = multiprocessing.Manager().Value('i', False)
        self.task_running = multiprocessing.Manager().Value('i', False)
        self.lock = multiprocessing.Manager().Lock()
        self.occupied_tensors = multiprocessing.Manager().list()
        self.monitor_thread = multiprocessing.Process(target=self.monitor_gpu_memory)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def init(self, min_free_memory=2):
        self.min_free_memory = min_free_memory
        assert min_free_memory >= 2
        manager = multiprocessing.Manager()
        self.memoryEnough = multiprocessing.Manager().Value('i', False)
        self.gpu_occupied = multiprocessing.Manager().Value('i', False)
        self.task_running = multiprocessing.Manager().Value('i', False)
        self.lock = manager.Lock()
        self.occupied_tensors = manager.list()
        self.monitor_thread = multiprocessing.Process(target=self.monitor_gpu_memory)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def monitor_gpu_memory(self):
        while True:
            if not self.task_running.value:
                time.sleep(5)
                gpus = Device.from_cuda_visible_devices()
                all_memory_enough = all(gpu.memory_free() / 1024 / 1024 / 1024 >= self.min_free_memory for gpu in gpus)
                with self.lock:
                    self.memoryEnough.value = all_memory_enough
                if self.memoryEnough.value:
                    self.occupy_gpu()

    # def is_need_wait(self):
    #     return not self.memoryEnough
        
    def check_can_run(self):
        with self.lock:
            if self.memoryEnough.value and self.gpu_occupied.value:
                self.release_gpu()
        return self.memoryEnough.value and not self.gpu_occupied.value

    
    def occupy_gpu(self):
        with self.lock:
            if not self.gpu_occupied.value:
                self.gpu_occupied.value = True
                gpus = Device.from_cuda_visible_devices()
                logging.info(f"尝试占据显存，目前显存情况：{[gpu for gpu in gpus]}")
                for gpu in gpus:
                    # 计算需要占用的显存大小，留出1GB左右的空闲
                    occupy_size = max(gpu.memory_free() - 1.9 * 1024 * 1024 * 1024, 0)  # 避免负数
                    tensor_size = torch.zeros((int(occupy_size // 4),)).cuda(gpu.cuda_index)
                    # 将占用的 Tensor 存储起来，以便释放显存时使用
                    self.occupied_tensors.append(tensor_size)
                

    def release_gpu(self):
        with self.lock:
            for tensor_size in self.occupied_tensors:
                # tensor_size = tensor_size.cpu()
                del tensor_size
            self.occupied_tensors.clear()
            torch.cuda.empty_cache()  # 清空PyTorch的CUDA缓存
            self.gpu_occupied.value = False

    def task_runed(self):
        self.task_running.value = True
       
    def init_task(self):
        with self.lock:
            self.memoryEnough.value = False
            self.gpu_occupied.value = False
            self.task_running.value = False
            self.occupied_tensors[:] = []

def pid_exists(pid):
    """Check whether pid exists in the current process table.
    UNIX only.
    """
    if pid < 0:
        return False
    if pid == 0:
        # According to "man 2 kill" PID 0 refers to every process
        # in the process group of the calling process.
        # On certain systems 0 is a valid PID but we have no way
        # to know that in a portable fashion.
        raise ValueError('invalid PID 0')
    try:
        os.kill(pid, 0)
    except OSError as err:
        if err.errno == errno.ESRCH:
            # ESRCH == No such process
            return False
        elif err.errno == errno.EPERM:
            # EPERM clearly means there's a process to deny access to
            return True
        else:
            # According to "man 2 kill" possible error values are
            # (EINVAL, EPERM, ESRCH)
            raise
    else:
        return True


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
        logging.info(tensors.shape[1])
        tensors = torch.nn.functional.pad(
            tensors, (0, 0, 0, fix_len-tensors.shape[1], 0, 0), mode='constant', value=0.)
    packed_tensors = torch.nn.utils.rnn.pack_padded_sequence(
        tensors, tensors_lengths, batch_first=True, enforce_sorted=False)

    return packed_tensors, genders, torch.LongTensor(labels), nfs
