# YOLOv5 general utils

import glob  # 文件路径匹配
import logging  # 日志记录
import math  # 数学运算
import os  # 操作系统接口
import platform  # 获取平台信息
import random  # 随机数生成
import re  # 正则表达式
import subprocess  # 子进程调用
import time  # 时间相关操作
from itertools import repeat  # 迭代器工具
from multiprocessing.pool import ThreadPool  # 多线程池
from pathlib import Path  # 路径操作

import cv2  # OpenCV库
import numpy as np  # 数值计算库
import pandas as pd  # 数据分析库
import pkg_resources as pkg  # 包资源管理
import torch  # PyTorch深度学习框架
import torchvision  # PyTorch的计算机视觉库
import yaml  # YAML格式解析

from utils.google_utils import gsutil_getsize  # Google云存储工具
from utils.metrics import fitness  # 模型性能评估工具
from utils.torch_utils import init_torch_seeds  # PyTorch种子初始化工具
import torch.backends.cudnn as cudnn  # CUDA深度神经网络加速库

# 设置PyTorch和NumPy的打印选项
torch.set_printoptions(linewidth=320, precision=5, profile='long')
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # 设置NumPy打印宽度和浮点数格式
pd.options.display.max_columns = 10
cv2.setNumThreads(0)  # 禁用OpenCV的多线程（与PyTorch DataLoader不兼容）
os.environ['NUMEXPR_MAX_THREADS'] = str(min(os.cpu_count(), 8))  # 设置NumExpr的最大线程数

logger = logging.getLogger(__name__)


def set_logging(rank=-1, verbose=True):  # 设置日志级别
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO if (verbose and rank in [-1, 0]) else logging.WARN)


def init_seeds(seed=0, deterministic=False):
    # 初始化随机数生成器种子，确保可复现性 https://pytorch.org/docs/stable/notes/randomness.html
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 设置所有GPU的CUDA随机数种子（多GPU安全）
    # torch.backends.cudnn.benchmark = True  # AutoBatch问题 https://github.com/ultralytics/yolov5/issues/9287
    if deterministic and check_version(torch.__version__, '1.12.0'):  # https://github.com/ultralytics/yolov5/pull/8213
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        os.environ['PYTHONHASHSEED'] = str(seed)


def get_latest_run(search_dir='.'):
    # 返回最近的'last.pt'文件路径（用于--resume）
    last_list = glob.glob(f'{search_dir}/**/last*.pt', recursive=True)
    return max(last_list, key=os.path.getctime) if last_list else ''


def isdocker():
    # 检查是否在Docker容器中
    return Path('/workspace').exists()  # or Path('/.dockerenv').exists()


def emojis(str=''):
    # 返回平台兼容的字符串（去除Windows系统中的emoji）
    return str.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else str


def file_size(file):
    # 返回文件大小（单位：MB）
    return Path(file).stat().st_size / 1e6


def check_online():
    # 检查网络连接
    import socket
    try:
        socket.create_connection(("1.1.1.1", 443), 5)  # 尝试连接到主机
        return True
    except OSError:
        return False


def check_git_status():
    # 检查代码是否为最新版本
    print(colorstr('github: '), end='')
    try:
        assert Path('.git').exists(), 'skipping check (not a git repository)'
        assert not isdocker(), 'skipping check (Docker image)'
        assert check_online(), 'skipping check (offline)'

        cmd = 'git fetch && git config --get remote.origin.url'
        url = subprocess.check_output(cmd, shell=True).decode().strip().rstrip('.git')
        branch = subprocess.check_output('git rev-parse --abbrev-ref HEAD', shell=True).decode().strip()
        n = int(subprocess.check_output(f'git rev-list {branch}..origin/master --count', shell=True))  # 获取落后提交数
        if n > 0:
            s = f"⚠️ WARNING: code is out of date by {n} commit{'s' * (n > 1)}. " \
                f"Use 'git pull' to update or 'git clone {url}' to download latest."
        else:
            s = f'up to date with {url} ✅'
        print(emojis(s))  # 打印结果
    except Exception as e:
        print(e)


def check_requirements(requirements='requirements.txt', exclude=()):
    # 检查安装的依赖是否满足要求
    import pkg_resources as pkg
    prefix = colorstr('red', 'bold', 'requirements:')
    if isinstance(requirements, (str, Path)):  # 如果是文件路径
        file = Path(requirements)
        if not file.exists():
            print(f"{prefix} {file.resolve()} not found, check failed.")
            return
        requirements = [f'{x.name}{x.specifier}' for x in pkg.parse_requirements(file.open()) if x.name not in exclude]
    else:  # 如果是包列表
        requirements = [x for x in requirements if x not in exclude]

    n = 0  # 更新的包数量
    for r in requirements:
        try:
            pkg.require(r)
        except Exception as e:  # 如果不满足
            n += 1
            print(f"{prefix} {r} not found and is required by YOLOv5, attempting auto-update...")
            print(subprocess.check_output(f"pip install '{r}'", shell=True).decode())

    if n:  # 如果有更新
        source = file.resolve() if 'file' in locals() else requirements
        s = f"{prefix} {n} package{'s' * (n > 1)} updated per {source}\n" \
            f"{prefix} ⚠️ {colorstr('bold', 'Restart runtime or rerun command for updates to take effect')}\n"
        print(emojis(s))  # 打印结果


def check_img_size(img_size, s=32):
    # 验证图像尺寸是否为步长的倍数
    new_size = make_divisible(img_size, int(s))  # 向上取最接近的步长倍数
    if new_size != img_size:
        print('WARNING: --img-size %g must be multiple of max stride %g, updating to %g' % (img_size, s, new_size))
    return new_size


def check_imshow():
    # 检查环境是否支持图像显示
    try:
        assert not isdocker(), 'cv2.imshow() is disabled in Docker environments'
        cv2.imshow('test', np.zeros((1, 1, 3)))
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        return True
    except Exception as e:
        print(f'WARNING: Environment does not support cv2.imshow() or PIL Image.show() image displays\n{e}')
        return False


def check_file(file):
    # 搜索文件（如果未找到）
    if Path(file).is_file() or file == '':
        return file
    else:
        files = glob.glob('./**/' + file, recursive=True)  # 查找文件
        assert len(files), f'File Not Found: {file}'  # 确保文件存在
        assert len(files) == 1, f"Multiple files match '{file}', specify exact path: {files}"  # 确保唯一
        return files[0]  # 返回文件路径


def check_dataset(dict):
    # 如果数据集未找到，则下载
    val, s = dict.get('val'), dict.get('download')
    if val and len(val):
        val = [Path(x).resolve() for x in (val if isinstance(val, list) else [val])]  # 解析验证集路径
        if not all(x.exists() for x in val):
            print('\nWARNING: Dataset not found, nonexistent paths: %s' % [str(x) for x in val if not x.exists()])
            if s and len(s):  # 如果有下载脚本
                if s.startswith('http') and s.endswith('.zip'):  # URL
                    f = Path(s).name  # 文件名
                    print(f'Downloading {s} ...')
                    torch.hub.download_url_to_file(s, f)
                    r = os.system(f'unzip -q {f} -d ../ && rm {f}')  # 解压
                elif s.startswith('bash '):  # 如果是bash脚本
                    print(f'Running {s} ...')
                    r = os.system(s)
                else:  # 如果是Python脚本
                    r = exec(s)  # 执行脚本
                print('Dataset autodownload %s\n' % ('success' if r in (0, None) else 'failure'))  # 打印结果
            else:
                raise Exception('Dataset not found.')


def check_version(current='0.0.0', minimum='0.0.0', name='version ', pinned=False, hard=False, verbose=False):
    # 检查版本是否满足要求
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = (current == minimum) if pinned else (current >= minimum)  # 比较版本
    s = f'WARNING ⚠️ {name}{minimum} is required by YOLOv5, but {name}{current} is currently installed'
    if hard:
        assert result, emojis(s)  # 如果不满足，抛出异常
    if verbose and not result:
        logger.warning(s)  # 如果不满足且需要详细输出，记录警告
    return result


def download(url, dir='.', multi_thread=False):
    # 多线程下载和解压文件
    def download_one(url, dir):
        # 下载单个文件
        f = dir / Path(url).name  # 文件路径
        if not f.exists():
            print(f'Downloading {url} to {f}...')
            torch.hub.download_url_to_file(url, f, progress=True)  # 下载
        if f.suffix in ('.zip', '.gz'):
            print(f'Unzipping {f}...')
            if f.suffix == '.zip':
                os.system(f'unzip -qo {f} -d {dir} && rm {f}')  # 解压
            elif f.suffix == '.gz':
                os.system(f'tar xfz {f} --directory {f.parent} && rm {f}')

    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)  # 创建目录
    if multi_thread:
        ThreadPool(8).imap(lambda x: download_one(*x), zip(url, repeat(dir)))  # 多线程下载
    else:
        for u in tuple(url) if isinstance(url, str) else url:
            download_one(u, dir)


def make_divisible(x, divisor):
    # 返回能被divisor整除的x
    return math.ceil(x / divisor) * divisor


def clean_str(s):
    # 清理字符串中的特殊字符
    return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨´><+]", repl="_", string=s)


def one_cycle(y1=0.0, y2=1.0, steps=100):
    # 返回一个正弦波函数（用于学习率调度）
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


def colorstr(*input):
    # 返回带有ANSI颜色代码的字符串 https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # 解析颜色参数和字符串
    colors = {'black': '\033[30m',  # 基础颜色
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # 亮色
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # 结束符
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


def labels_to_class_weights(labels, nc=80):
    # 根据训练标签计算类别权重（逆频率）
    if labels[0] is None:  # 如果没有标签
        return torch.Tensor()

    labels = np.concatenate(labels, 0)  # 合并标签
    classes = labels[:, 0].astype(np.int)  # 提取类别
    weights = np.bincount(classes, minlength=nc)  # 计算每个类别的出现次数

    weights[weights == 0] = 1  # 替换空类别为1
    weights = 1 / weights  # 计算逆频率
    weights /= weights.sum()  # 归一化
    return torch.from_numpy(weights)


def labels_to_image_weights(labels, nc=80, class_weights=np.ones(80)):
    # 根据图像内容和类别权重计算图像权重
    class_counts = np.array([np.bincount(x[:, 0].astype(np.int), minlength=nc) for x in labels])
    image_weights = (class_weights.reshape(1, nc) * class_counts).sum(1)
    return image_weights


def coco80_to_coco91_class():
    # 将80类COCO索引转换为91类（论文）
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    return x

def xyxy2xywh2(x):
    # 将边界框从[x1, y1, x2, y2]转换为[x1, y1, w, h]
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0]  # x top-left
    y[:, 1] = x[:, 1]  # y top-left
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xyxy2xywh(x):
    # 将边界框从[x1, y1, x2, y2]转换为[x, y, w, h]
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # 将边界框从[x, y, w, h]转换为[x1, y1, x2, y2]
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # 将归一化的边界框从[x, y, w, h]转换为[x1, y1, x2, y2]
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y


def xyn2xy(x, w=640, h=640, padw=0, padh=0):
    # 将归一化的点从[x, y]转换为[x, y]
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * x[:, 0] + padw  # top left x
    y[:, 1] = h * x[:, 1] + padh  # top left y
    return y


def segment2box(segment, width=640, height=640):
    # 将分割标签转换为边界框
    x, y = segment.T  # 提取点
    inside = (x >= 0) & (y >= 0) & (x <= width) & (y <= height)
    x, y, = x[inside], y[inside]
    return np.array([x.min(), y.min(), x.max(), y.max()]) if any(x) else np.zeros((1, 4))  # 返回边界框


def segments2boxes(segments):
    # 将分割标签转换为边界框
    boxes = []
    for s in segments:
        x, y = s.T
        boxes.append([x.min(), y.min(), x.max(), y.max()])
    return xyxy2xywh(np.array(boxes))


def resample_segments(segments, n=1000):
    # 重新采样分割标签
    for i, s in enumerate(segments):
        x = np.linspace(0, len(s) - 1, n)
        xp = np.arange(len(s))
        segments[i] = np.concatenate([np.interp(x, xp, s[:, i]) for i in range(2)]).reshape(2, -1).T
    return segments


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # 将坐标从img1_shape缩放到img0_shape
    if ratio_pad is None:  # 如果没有缩放比例
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # 计算缩放比例
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # 计算填充
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # 减去x填充
    coords[:, [1, 3]] -= pad[1]  # 减去y填充
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, img_shape):
    # 裁剪边界框
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # 计算IoU
    box2 = box2.T

    if x1y1x2y2:  # 如果是[x1, y1, x2, y2]格式
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # 如果是[x, y, w, h]格式
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # 最小外接矩形宽度
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # 最小外接矩形高度
        if CIoU or DIoU:  # 如果是DIoU或CIoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # 最小外接矩形对角线平方
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # 中心点距离平方
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    # 计算边界框的IoU
    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def wh_iou(wh1, wh2):
    # 计算宽度和高度的IoU
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]
    inter = torch.min(wh1, wh2).prod(2)  # [N,M]
    return inter / (wh1.prod(2) + wh2.prod(2) - inter)  # iou = inter / (area1 + area2 - inter)


def python_nms(dets, scores, iou_thresh):
    # 使用Python实现非极大值抑制
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    keep = []
    index = scores.argsort()[::-1]
    while index.size > 0:
        i = index[0]
        keep.append(i)

        x11 = torch.max(x1[i], x1[index[1:]])
        y11 = torch.max(y1[i], y1[index[1:]])
        x22 = torch.min(x2[i], x2[index[1:]])
        y22 = torch.min(y2[i], y2[index[1:]])

        w = np.maximum(0, x22 - x11 + 1)
        h = np.maximum(0, y22 - y11 + 1)

        overlaps = w * h
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)

        idx = torch.where(ious <= iou_thresh)[0]

        index = index[idx + 1]

    return keep


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=()):
    # 非极大值抑制
    nc = prediction.shape[2] - 5  # 类别数量
    xc = prediction[..., 4] > conf_thres  # 候选框

    min_wh, max_wh = 2, 4096  # 最小和最大边界框宽度和高度
    max_det = 300  # 每张图像的最大检测数量
    max_nms = 30000  # 最大NMS数量
    time_limit = 10.0  # 时间限制
    redundant = True  # 是否需要冗余检测
    multi_label &= nc > 1  # 是否支持多标签
    merge = False  # 是否使用合并NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]  # 筛选置信度大于阈值的框

        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]
            v[:, 4] = 1.0
            v[range(len(l)), l[:, 0].long() + 5] = 1.0
            x = torch.cat((x, v), 0)

        if not x.shape[0]:
            continue

        x[:, 5:] *= x[:, 4:5]  # 计算类别置信度

        box = xywh2xyxy(x[:, :4])

        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        n = x.shape[0]
        if not n:
            continue
        elif n > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        c = x[:, 5:6] * (0 if agnostic else max_wh)
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:
            i = i[:max_det]
        if merge and (1 < n < 3E3):
            iou = box_iou(boxes[i], boxes) > iou_thres
            weights = iou * scores[None]
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)
            if redundant:
                i = i[iou.sum(1) > 1]

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break

    return output


def strip_optimizer(f='best.pt', s=''):
    # 去掉优化器信息
    x = torch.load(f, map_location=torch.device('cpu'))
    if x.get('ema'):
        x['model'] = x['ema']
    for k in 'optimizer', 'training_results', 'wandb_id', 'ema', 'updates':
        x[k] = None
    x['epoch'] = -1
    x['model'].half()
    for p in x['model'].parameters():
        p.requires_grad = False
    torch.save(x, s or f)
    mb = os.path.getsize(s or f) / 1E6
    print(f"Optimizer stripped from {f},{(' saved as %s,' % s) if s else ''} {mb:.1f}MB")


def print_mutation(hyp, results, yaml_file='hyp_evolved.yaml', bucket=''):
    # 打印进化结果
    a = '%10s' * len(hyp) % tuple(hyp.keys())
    b = '%10.3g' * len(hyp) % tuple(hyp.values())
    c = '%10.4g' * len(results) % results
    print('\n%s\n%s\nEvolved fitness: %s\n' % (a, b, c))

    if bucket:
        url = 'gs://%s/evolve.txt' % bucket
        if gsutil_getsize(url) > (os.path.getsize('evolve.txt') if os.path.exists('evolve.txt') else 0):
            os.system('gsutil cp %s .' % url)

    with open('evolve.txt', 'a') as f:  # append result
        f.write(c + b + '\n')
    x = np.unique(np.loadtxt('evolve.txt', ndmin=2), axis=0)  # load unique rows
    x = x[np.argsort(-fitness(x))]  # sort
    np.savetxt('evolve.txt', x, '%10.3g')  # save sort by fitness

    # Save yaml
    for i, k in enumerate(hyp.keys()):
        hyp[k] = float(x[0, i + 7])
    with open(yaml_file, 'w') as f:
        results = tuple(x[0, :7])
        c = '%10.4g' * len(results) % results  # results (P, R, mAP@0.5, mAP@0.5:0.95, val_losses x 3)
        f.write('# Hyperparameter Evolution Results\n# Generations: %g\n# Metrics: ' % len(x) + c + '\n\n')
        yaml.safe_dump(hyp, f, sort_keys=False)

    if bucket:
        os.system('gsutil cp evolve.txt %s gs://%s' % (yaml_file, bucket))  # upload


def apply_classifier(x, model, img, im0):
    # 应用分类器
    im0 = [im0] if isinstance(im0, np.ndarray) else im0
    for i, d in enumerate(x):
        if d is not None and len(d):
            d = d.clone()

            b = xyxy2xywh(d[:, :4])  # boxes
            b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)
            b[:, 2:] = b[:, 2:] * 1.3 + 30  # pad
            d[:, :4] = xywh2xyxy(b).long()

            scale_coords(img.shape[2:], d[:, :4], im0[i].shape)

            pred_cls1 = d[:, 5].long()
            ims = []
            for j, a in enumerate(d):
                cutout = im0[i][int(a[1]):int(a[3]), int(a[0]):int(a[2])]
                im = cv2.resize(cutout, (224, 224))

                im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                im = np.ascontiguousarray(im, dtype=np.float32)  # uint8 to float32
                im /= 255.0  # 0 - 255 to 0.0 - 1.0
                ims.append(im)

            pred_cls2 = model(torch.Tensor(ims).to(d.device)).argmax(1)
            x[i] = x[i][pred_cls1 == pred_cls2]

    return x


def save_one_box(xyxy, im, file='image.jpg', gain=1.02, pad=10, square=False, BGR=False):
    # 保存裁剪后的图像
    xyxy = torch.tensor(xyxy).view(-1, 4)
    b = xyxy2xywh(xyxy)
    if square:
        b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)
    b[:, 2:] = b[:, 2:] * gain + pad  # box wh * gain + pad
    xyxy = xywh2xyxy(b).long()
    clip_coords(xyxy, im.shape)
    crop = im[int(xyxy[0, 1]):int(xyxy[0, 3]), int(xyxy[0, 0]):int(xyxy[0, 2])]
    cv2.imwrite(str(increment_path(file, mkdir=True).with_suffix('.jpg')), crop if BGR else crop[..., ::-1])


def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # 增量路径
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        suffix = path.suffix
        path = path.with_suffix('')
        dirs = glob.glob(f"{path}{sep}*")
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2
        path = Path(f"{path}{sep}{n}{suffix}")  # update path
    dir = path if path.suffix == '' else path.parent  # directory
    if not dir.exists() and mkdir:
        dir.mkdir(parents=True, exist_ok=True)  # make directory
    return path
