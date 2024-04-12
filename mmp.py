import os
from nvitop import  Device
import torch
import time

# 参数设置
cuda_visible_devices = "0"  # 可见的CUDA设备列表，用逗号分隔
min_gpu_memory_gb = 25  # 最小GPU剩余内存（GB），低于此值将不被考虑用于占用
min_gpu_memory_to_occupy_gb = 6  # GPU剩余内存大于此值（GB）时，计划进行占用
occupy_size_factor = 0.5  # 占用GPU内存大小的因子
occupy_interval_seconds = 0.5  # 占用GPU的时间间隔（秒）

tensor_sizes = []

os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

def cpu_intensive_task():
    a = torch.randn(100, 100)
    b = torch.randn(100, 100)
    result = torch.matmul(a, b)
    return result
    
    
if __name__ == '__main__':
    pid = os.getpid()
    print(f"当前进程PID：{pid}")
    
    while True:
        if all(gpu.memory_free()/ 1024 / 1024 / 1024 > 4 for gpu in Device.from_cuda_visible_devices()):
            gpus = Device.from_cuda_visible_devices()
            for gpu in gpus:
                tensor_sizes.append(torch.zeros((1,)).cuda(gpu.cuda_index))
            while True:
                for gpu in gpus:
                    gpu_memory_free = gpu.memory_free()
                    gpu_memory_free_h = gpu_memory_free / 1024 / 1024 / 1024
                    if gpu_memory_free_h > min_gpu_memory_to_occupy_gb:
                        occupy_size = gpu_memory_free
                        print(f"GPU{gpu.cuda_index}: 剩余({gpu_memory_free_h}), 计划占用({occupy_size * occupy_size_factor / 1024 / 1024 / 1024 })")
                        tensor_sizes.append(torch.ones((int((occupy_size * occupy_size_factor) // 4),)).cuda(gpu.cuda_index))
                for tensor_ in tensor_sizes:
                    tensor_ = tensor_.add_(1)
                    tensor_.grad = None
                cpu_intensive_task()
                time.sleep(occupy_interval_seconds)
        time.sleep(occupy_interval_seconds)