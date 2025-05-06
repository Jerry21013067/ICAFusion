# YOLOv5实验性模块

import numpy as np
import torch
import torch.nn as nn
from numpy.core.multiarray import _reconstruct

from models.common import Conv, DWConv
from utils.google_utils import attempt_download

# 添加安全的全局变量
torch.serialization.add_safe_globals([_reconstruct])


class CrossConv(nn.Module):
    """
    交叉卷积下采样模块
    参数：
    c1: 输入通道数
    c2: 输出通道数
    k: 卷积核大小，默认为3
    s: 步长，默认为1
    g: 分组数，默认为1
    e: 扩展率，默认为1.0
    shortcut: 是否使用残差连接，默认为False
    """
    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
        super(CrossConv, self).__init__()
        c_ = int(c2 * e)  # 中间通道数
        self.cv1 = Conv(c1, c_, (1, k), (1, s))
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class Sum(nn.Module):
    """
    多层加权求和模块 https://arxiv.org/abs/1911.09070
    参数：
    n: 输入层数
    weight: 是否应用权重，默认为False
    """
    def __init__(self, n, weight=False):
        super(Sum, self).__init__()
        self.weight = weight  # 是否应用权重
        self.iter = range(n - 1)  # 迭代对象
        if weight:
            self.w = nn.Parameter(-torch.arange(1., n) / 2, requires_grad=True)  # 层权重

    def forward(self, x):
        y = x[0]  # 不加权
        if self.weight:
            w = torch.sigmoid(self.w) * 2
            for i in self.iter:
                y = y + x[i + 1] * w[i]
        else:
            for i in self.iter:
                y = y + x[i + 1]
        return y


class GhostConv(nn.Module):
    """
    Ghost卷积模块 https://github.com/huawei-noah/ghostnet
    参数：
    c1: 输入通道数
    c2: 输出通道数
    k: 卷积核大小，默认为1
    s: 步长，默认为1
    g: 分组数，默认为1
    act: 是否使用激活函数，默认为True
    """
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super(GhostConv, self).__init__()
        c_ = c2 // 2  # 中间通道数
        self.cv1 = Conv(c1, c_, k, s, None, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat([y, self.cv2(y)], 1)


class GhostBottleneck(nn.Module):
    """
    Ghost瓶颈模块 https://github.com/huawei-noah/ghostnet
    参数：
    c1: 输入通道数
    c2: 输出通道数
    k: 卷积核大小，默认为3
    s: 步长，默认为1
    """
    def __init__(self, c1, c2, k=3, s=1):
        super(GhostBottleneck, self).__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(GhostConv(c1, c_, 1, 1),  # 点卷积
                                  DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # 深度可分离卷积
                                  GhostConv(c_, c2, 1, 1, act=False))  # 点卷积线性
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False),
                                      Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class MixConv2d(nn.Module):
    """
    混合深度卷积模块 https://arxiv.org/abs/1907.09595
    参数：
    c1: 输入通道数
    c2: 输出通道数
    k: 卷积核大小，默认为(1, 3)
    s: 步长，默认为1
    equal_ch: 是否等通道，默认为True
    """
    def __init__(self, c1, c2, k=(1, 3), s=1, equal_ch=True):
        super(MixConv2d, self).__init__()
        groups = len(k)
        if equal_ch:  # 等通道
            i = torch.linspace(0, groups - 1E-6, c2).floor()  # c2索引
            c_ = [(i == g).sum() for g in range(groups)]  # 中间通道数
        else:  # 等权重
            b = [c2] + [0] * groups
            a = np.eye(groups + 1, groups, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            c_ = np.linalg.lstsq(a, b, rcond=None)[0].round()  # 解等权重索引

        self.m = nn.ModuleList([nn.Conv2d(c1, int(c_[g]), k[g], s, k[g] // 2, bias=False) for g in range(groups)])
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return x + self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))


class Ensemble(nn.ModuleList):
    """
    模型集成类
    """
    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x, augment=False):
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        # y = torch.stack(y).max(0)[0]  # 最大集成
        # y = torch.stack(y).mean(0)  # 平均集成
        y = torch.cat(y, 1)  # NMS集成
        return y, None  # 推理，训练输出


def attempt_load(weights, map_location=None):
    """
    加载模型权重
    参数：
    weights: 权重路径，可以是单个权重文件或权重列表
    map_location: 映射位置，默认为None
    """
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        attempt_download(w)
        ckpt = torch.load(w, map_location=map_location, weights_only=False)  # 加载
        model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval())  # FP32模型

    # 兼容性更新
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True  # pytorch 1.7.0兼容性
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0兼容性

    if len(model) == 1:
        return model[-1]  # 返回模型
    else:
        print('Ensemble created with %s\n' % weights)
        for k in ['names', 'stride']:
            setattr(model, k, getattr(model[-1], k))
        return model  # 返回集成
