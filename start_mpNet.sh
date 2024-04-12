#!/bin/bash

# 等待一小时
sleep 1h

# 运行程序并将日志重定向到指定文件
nohup python ~/SCW/torchTraining/mpNet.py > ~/SCW/torchTraining/mpNet.log 2>&1 &