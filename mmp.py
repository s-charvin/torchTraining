import os
from nvitop import select_devices, Device
import torch
import time
import random

tensor_sizes = []
if __name__ == '__main__':
    
    gpus = Device.from_cuda_visible_devices()
    for gpu in gpus:
        tensor_sizes.append(torch.zeros((1,)).cuda(gpu.cuda_index))

    
    while True:
        for gpu in gpus:
            if gpu.memory_free() / 1024 / 1024 / 1024 >= 2.4:
                occupy_size = max(gpu.memory_free() -  random.uniform(0.5, 1) * 1024 * 1024 * 1024, 0)
                print(occupy_size / 1024 / 1024 / 1024 / 2)
                tensor_sizes.append(torch.ones((int((occupy_size / 2) // 4),)).cuda(gpu.cuda_index))
        for tensor_ in tensor_sizes:
            tensor_ = tensor_ * 1
            tensor_.grad = None
        time.sleep(0.5)