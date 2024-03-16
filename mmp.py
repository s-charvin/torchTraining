import os
from nvitop import select_devices, Device
import torch

tensor_sizes = []
while True:
    gpus = Device.from_cuda_visible_devices()
    for gpu in gpus:
        if gpu.memory_free() / 1024 / 1024 / 1024 >= 4:
            occupy_size = max(gpu.memory_free() - 7.3 * 1024 * 1024 * 1024, 0)
            tensor_sizes.append(torch.zeros((int(occupy_size // 4),)).cuda(gpu.cuda_index))
    for tensor_ in tensor_sizes:
        tensor_ = tensor_ + 1