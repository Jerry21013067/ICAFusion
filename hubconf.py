"""YOLOv5 PyTorch Hub模型 https://pytorch.org/hub/ultralytics_yolov5/

Usage:
    import torch
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
"""

from pathlib import Path

import torch

from models.yolo import Model
from utils.general import check_requirements, set_logging
from utils.google_utils import attempt_download
from utils.torch_utils import select_device

dependencies = ['torch', 'yaml']  # 定义依赖包
check_requirements(Path(__file__).parent / 'requirements.txt', exclude=('pycocotools', 'thop'))  # 检查依赖


def create(name, pretrained, channels, classes, autoshape, verbose):
    """
    创建指定的YOLOv5模型

    参数：
        name (str): 模型名称，例如'yolov5s'
        pretrained (bool): 是否加载预训练权重
        channels (int): 输入通道数
        classes (int): 模型类别数
        autoshape (bool): 是否启用自动形状调整
        verbose (bool): 是否启用详细日志

    返回：
        PyTorch模型
    """
    try:
        set_logging(verbose=verbose)  # 设置日志级别

        # 获取模型配置文件路径
        cfg = list((Path(__file__).parent / 'models').rglob(f'{name}.yaml'))[0]
        model = Model(cfg, channels, classes)  # 创建模型
        if pretrained:
            fname = f'{name}.pt'  # 预训练权重文件名
            attempt_download(fname)  # 如果本地没有找到，尝试下载
            ckpt = torch.load(fname, map_location=torch.device('cpu'))  # 加载权重
            msd = model.state_dict()  # 模型状态字典
            csd = ckpt['model'].float().state_dict()  # 预训练权重状态字典（转换为FP32）
            csd = {k: v for k, v in csd.items() if msd[k].shape == v.shape}  # 过滤形状不匹配的权重
            model.load_state_dict(csd, strict=False)  # 加载权重
            if len(ckpt['model'].names) == classes:
                model.names = ckpt['model'].names  # 设置类别名称
            if autoshape:
                model = model.autoshape()  # 启用自动形状调整
        device = select_device('0' if torch.cuda.is_available() else 'cpu')  # 选择设备
        return model.to(device)  # 将模型移动到指定设备

    except Exception as e:
        help_url = 'https://github.com/ultralytics/yolov5/issues/36'  # 提供帮助链接
        s = '缓存可能过期，请尝试设置force_reload=True。详情请查看%s' % help_url
        raise Exception(s) from e  # 抛出异常


def custom(path_or_model='path/to/model.pt', autoshape=True, verbose=True):
    """
    自定义YOLOv5模型

    参数（3种选择）：
        path_or_model (str): 模型权重文件路径
        path_or_model (dict): torch.load('path/to/model.pt')
        path_or_model (nn.Module): torch.load('path/to/model.pt')['model']

    返回：
        PyTorch模型
    """
    set_logging(verbose=verbose)  # 设置日志级别

    # 加载模型权重
    model = torch.load(path_or_model) if isinstance(path_or_model, str) else path_or_model
    if isinstance(model, dict):
        model = model['ema' if model.get('ema') else 'model']  # 加载模型

    # 创建模型
    hub_model = Model(model.yaml).to(next(model.parameters()).device)
    hub_model.load_state_dict(model.float().state_dict())  # 加载状态字典
    hub_model.names = model.names  # 设置类别名称
    if autoshape:
        hub_model = hub_model.autoshape()  # 启用自动形状调整
    device = select_device('0' if torch.cuda.is_available() else 'cpu')  # 选择设备
    return hub_model.to(device)  # 将模型移动到指定设备


# 定义不同大小的YOLOv5模型
def yolov5s(pretrained=True, channels=3, classes=80, autoshape=True, verbose=True):
    return create('yolov5s', pretrained, channels, classes, autoshape, verbose)


def yolov5m(pretrained=True, channels=3, classes=80, autoshape=True, verbose=True):
    return create('yolov5m', pretrained, channels, classes, autoshape, verbose)


def yolov5l(pretrained=True, channels=3, classes=80, autoshape=True, verbose=True):
    return create('yolov5l', pretrained, channels, classes, autoshape, verbose)


def yolov5x(pretrained=True, channels=3, classes=80, autoshape=True, verbose=True):
    return create('yolov5x', pretrained, channels, classes, autoshape, verbose)


def yolov5s6(pretrained=True, channels=3, classes=80, autoshape=True, verbose=True):
    return create('yolov5s6', pretrained, channels, classes, autoshape, verbose)


def yolov5m6(pretrained=True, channels=3, classes=80, autoshape=True, verbose=True):
    return create('yolov5m6', pretrained, channels, classes, autoshape, verbose)


def yolov5l6(pretrained=True, channels=3, classes=80, autoshape=True, verbose=True):
    return create('yolov5l6', pretrained, channels, classes, autoshape, verbose)


def yolov5x6(pretrained=True, channels=3, classes=80, autoshape=True, verbose=True):
    return create('yolov5x6', pretrained, channels, classes, autoshape, verbose)


if __name__ == '__main__':
    # 创建一个YOLOv5s模型并进行推理
    model = create(name='yolov5s', pretrained=True, channels=3, classes=80, autoshape=True, verbose=True)  # 加载预训练模型
    # model = custom(path_or_model='path/to/model.pt')  # 自定义模型

    # 验证推理
    import cv2
    import numpy as np
    from PIL import Image

    # 定义输入图像
    imgs = ['data/images/zidane.jpg',  # 文件名
            'https://github.com/ultralytics/yolov5/releases/download/v1.0/zidane.jpg',  # URI
            cv2.imread('data/images/bus.jpg')[:, :, ::-1],  # OpenCV图像
            Image.open('data/images/bus.jpg'),  # PIL图像
            np.zeros((320, 640, 3))]  # NumPy数组

    results = model(imgs)  # 批量推理
    results.print()  # 打印结果
    results.save()  # 保存结果
