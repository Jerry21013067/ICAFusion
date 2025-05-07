# YOLOv5 PyTorch utils

import datetime
import logging
import math
import os
import platform
import subprocess
import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torchvision

try:
    import thop  # 导入thop库，用于计算FLOPS（浮点运算次数）
except ImportError:
    thop = None
logger = logging.getLogger(__name__)


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    装饰器，用于分布式训练中让所有进程等待本地主进程执行某些操作。
    """
    if local_rank not in [-1, 0]:  # 如果当前进程不是本地主进程
        torch.distributed.barrier()  # 等待本地主进程完成
    yield  # 执行主进程的操作
    if local_rank == 0:  # 如果当前进程是本地主进程
        torch.distributed.barrier()  # 等待其他进程完成


def init_torch_seeds(seed=0):
    # 初始化PyTorch随机种子，用于控制随机性 https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(seed)  # 设置PyTorch随机种子
    if seed == 0:  # 如果种子为0，优先保证可复现性
        cudnn.benchmark, cudnn.deterministic = False, True  # 禁用cudnn的benchmark，启用deterministic
    else:  # 如果种子不为0，优先保证性能
        cudnn.benchmark, cudnn.deterministic = True, False  # 启用cudnn的benchmark，禁用deterministic


def date_modified(path=__file__):
    # 返回文件的最后修改日期，格式为'年-月-日', i.e. '2021-3-26'
    t = datetime.datetime.fromtimestamp(Path(path).stat().st_mtime)
    return f'{t.year}-{t.month}-{t.day}'


def git_describe(path=Path(__file__).parent):  # path must be a directory
    # 返回当前目录的git描述信息, i.e. v5.0-5-g3e25f1e https://git-scm.com/docs/git-describe
    s = f'git -C {path} describe --tags --long --always'
    try:
        return subprocess.check_output(s, shell=True, stderr=subprocess.STDOUT).decode()[:-1]
    except subprocess.CalledProcessError as e:
        return ''  # 如果不是git仓库，返回空字符串


def select_device(device='', batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    s = f'YOLOv5 🚀 {git_describe() or date_modified()} torch {torch.__version__} '  # string
    cpu = device.lower() == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # 设置可见的GPU
        assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'  # 检查CUDA是否可用

    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        n = torch.cuda.device_count()
        if n > 1 and batch_size:  # 如果有多个GPU且指定了batch_size
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        space = ' ' * len(s)
        for i, d in enumerate(device.split(',') if device else range(n)):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2}MB)\n"  # 构造日志字符串
    else:
        s += 'CPU\n'

    logger.info(s.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else s)  # 输出日志
    return torch.device('cuda:0' if cuda else 'cpu')


def time_synchronized():
    # 返回同步后的当前时间
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def profile(x, ops, n=100, device=None):
    # 对PyTorch模块或模块列表进行性能分析 Example usage:
    #     x = torch.randn(16, 3, 640, 640)  # input
    #     m1 = lambda x: x * torch.sigmoid(x)
    #     m2 = nn.SiLU()
    #     profile(x, [m1, m2], n=100)  # profile speed over 100 iterations

    device = device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    x = x.to(device)
    x.requires_grad = True
    print(torch.__version__, device.type, torch.cuda.get_device_properties(0) if device.type == 'cuda' else '')
    print(f"\n{'Params':>12s}{'GFLOPS':>12s}{'forward (ms)':>16s}{'backward (ms)':>16s}{'input':>24s}{'output':>24s}")
    for m in ops if isinstance(ops, list) else [ops]:
        m = m.to(device) if hasattr(m, 'to') else m  # 将模块移动到设备
        m = m.half() if hasattr(m, 'half') and isinstance(x, torch.Tensor) and x.dtype is torch.float16 else m  # 如果输入是半精度，将模块转换为半精度
        dtf, dtb, t = 0., 0., [0., 0., 0.]  # 初始化时间变量
        try:
            flops = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2  # GFLOPS
        except:
            flops = 0

        for _ in range(n):
            t[0] = time_synchronized()
            y = m(x)
            t[1] = time_synchronized()
            try:
                _ = y.sum().backward()
                t[2] = time_synchronized()
            except:  # 如果没有反向传播方法
                t[2] = float('nan')
            dtf += (t[1] - t[0]) * 1000 / n  # 计算前向传播时间
            dtb += (t[2] - t[1]) * 1000 / n  # 计算反向传播时间

        s_in = tuple(x.shape) if isinstance(x, torch.Tensor) else 'list'
        s_out = tuple(y.shape) if isinstance(y, torch.Tensor) else 'list'
        p = sum(list(x.numel() for x in m.parameters())) if isinstance(m, nn.Module) else 0  # 计算参数数量
        print(f'{p:12}{flops:12.4g}{dtf:16.4g}{dtb:16.4g}{str(s_in):>24s}{str(s_out):>24s}')


def is_parallel(model):
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def intersect_dicts(da, db, exclude=()):
    # 返回两部字典的交集，排除指定的键
    return {k: v for k, v in da.items() if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape}


def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass
            #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True


def find_modules(model, mclass=nn.Conv2d):
    # 找到模型中指定类型的模块
    return [i for i, m in enumerate(model.module_list) if isinstance(m, mclass)]


def sparsity(model):
    # 计算模型的稀疏度
    a, b = 0., 0.
    for p in model.parameters():
        a += p.numel()
        b += (p == 0).sum()
    return b / a


def prune(model, amount=0.3):
    # 对模型进行剪枝
    import torch.nn.utils.prune as prune
    print('Pruning model... ', end='')
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            prune.l1_unstructured(m, name='weight', amount=amount)  # 使用L1剪枝
            prune.remove(m, 'weight')  # 移除剪枝
    print(' %.3g global sparsity' % sparsity(model))


def fuse_conv_and_bn(conv, bn):
    # 将卷积层和批量归一化层融合 https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    # 准备滤波器
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # 准备偏置
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


def model_info(model, verbose=False, img_size=640):
    # 输出模型信息, i.e. img_size=640 or img_size=[640, 320]
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    if verbose:
        print('%5s %40s %9s %12s %20s %10s %10s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))

    try:  # FLOPS
        from thop import profile
        img_size = img_size if isinstance(img_size, list) else [img_size, img_size]  # 转换为列表
        img = torch.zeros((1, 3, img_size[0], img_size[1]), device=next(model.parameters()).device)  # 创建输入张量
        flops = profile(deepcopy(model), inputs=img, verbose=False)[0] / 1E9  # stride GFLOPS
        fs = '%.1f GFLOPS' % (flops)  # 640x640 GFLOPS
    except (ImportError, Exception):
        fs = ''

    logger.info(f"Model Summary: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients, {fs}")


def load_classifier(name='resnet101', n=2):
    # 加载预训练分类器模型
    model = torchvision.models.__dict__[name](pretrained=True)

    # 获取全连接层输入特征数量
    filters = model.fc.weight.shape[1]
    model.fc.bias = nn.Parameter(torch.zeros(n), requires_grad=True)
    model.fc.weight = nn.Parameter(torch.zeros(n, filters), requires_grad=True)
    model.fc.out_features = n
    return model


def scale_img(img, ratio=1.0, same_shape=False, gs=32):  # img(16,3,256,416)
    # 缩放图像
    if ratio == 1.0:
        return img
    else:
        h, w = img.shape[2:]
        s = (int(h * ratio), int(w * ratio))  # 计算新尺寸
        img = F.interpolate(img, size=s, mode='bilinear', align_corners=False)  # 缩放图像
        if not same_shape:  # 如果不需要保持形状
            h, w = [math.ceil(x * ratio / gs) * gs for x in (h, w)]
        return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)  # 填充图像


def copy_attr(a, b, include=(), exclude=()):
    # 将b的属性复制到a，可以选择包含或排除某些属性
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)


class ModelEMA:
    """
    模型指数移动平均。
    """
    def __init__(self, model, decay=0.9999, updates=0):
        # 初始化EMA
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()  # 创建EMA模型
        self.updates = updates  # 初始化更新次数
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # 定义衰减函数
        for p in self.ema.parameters():  # 遍历EMA模型参数
            p.requires_grad_(False)  # 禁用梯度计算

    def update(self, model):
        # 更新EMA模型
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.module.state_dict() if is_parallel(model) else model.state_dict()  # 获取模型状态字典
            for k, v in self.ema.state_dict().items():  # 遍历EMA模型状态字典
                if v.dtype.is_floating_point:  # 如果是浮点类型
                    v *= d  # 应用衰减
                    v += (1. - d) * msd[k].detach()  # 更新EMA模型参数

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # 更新EMA模型属性
        copy_attr(self.ema, model, include, exclude)
