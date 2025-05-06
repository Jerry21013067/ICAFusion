# 恢复YOLOv5目录中所有中断的训练，包括分布式数据并行（DDP）训练
# 使用方法：$ python utils/aws/resume.py

# 导入必要的模块
import os
import sys
from pathlib import Path

import torch
import yaml

sys.path.append('./')  # 将当前目录添加到系统路径中，以便运行子目录中的脚本

port = 0  # 定义端口号，用于分布式训练
path = Path('').resolve()  # 获取当前目录的绝对路径
for last in path.rglob('*/**/last.pt'):  # 遍历当前目录及其子目录，查找所有名为"last.pt"的文件
    ckpt = torch.load(last)  # 加载"last.pt"文件，获取检查点信息
    if ckpt['optimizer'] is None:  # 如果检查点中没有优化器信息，则跳过
        continue

    # 加载"opt.yaml"文件，获取训练选项
    with open(last.parent.parent / 'opt.yaml') as f:
        opt = yaml.safe_load(f)

    # 获取设备信息
    d = opt['device'].split(',')  # 设备列表
    nd = len(d)  # 设备数量
    ddp = nd > 1 or (nd == 0 and torch.cuda.device_count() > 1)  # 判断是否使用分布式数据并行（DDP）

    if ddp:  # 多GPU
        port += 1  # 端口号递增
        cmd = f'python -m torch.distributed.launch --nproc_per_node {nd} --master_port {port} train.py --resume {last}'
    else:  # 单GPU
        cmd = f'python train.py --resume {last}'

    cmd += ' > /dev/null 2>&1 &'  # 将命令输出重定向到/dev/null，并在后台线程中运行
    print(cmd)
    os.system(cmd)  # 执行命令
