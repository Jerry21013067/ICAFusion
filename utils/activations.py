# Activation functions

import torch
import torch.nn as nn
import torch.nn.functional as F


# SiLU激活函数 https://arxiv.org/pdf/1606.08415.pdf ----------------------------------------------------------------------------
class SiLU(nn.Module):  # 适用于导出的版本，兼容nn.SiLU()
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


class Hardswish(nn.Module):  # Hardswish激活函数
    @staticmethod
    def forward(x):
        # return x * F.hardsigmoid(x)  # 适用于TorchScript和CoreML
        return x * F.hardtanh(x + 3, 0., 6.) / 6.  # 适用于TorchScript、CoreML和ONNX


# Mish激活函数 https://github.com/digantamisra98/Mish --------------------------------------------------------------------------
class Mish(nn.Module):
    @staticmethod
    def forward(x):
        return x * F.softplus(x).tanh()

# Mish激活函数（内存高效版本） -----------------------------------------------------------------------------
class MemoryEfficientMish(nn.Module):
    class F(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)  # 保存输入张量，用于反向传播
            return x.mul(torch.tanh(F.softplus(x)))  # x * tanh(ln(1 + exp(x)))

        @staticmethod
        def backward(ctx, grad_output):
            x = ctx.saved_tensors[0]  # 获取保存的输入张量
            sx = torch.sigmoid(x)
            fx = F.softplus(x).tanh()
            return grad_output * (fx + x * sx * (1 - fx * fx))  # 自定义反向传播公式

    def forward(self, x):
        return self.F.apply(x)  # 应用自定义的激活函数


# FReLU激活函数 https://arxiv.org/abs/2007.11824 -------------------------------------------------------------------------------
class FReLU(nn.Module):
    def __init__(self, c1, k=3):  # 输入通道数c1，卷积核大小k
        super().__init__()
        self.conv = nn.Conv2d(c1, c1, k, 1, 1, groups=c1, bias=False)  # 深度可分离卷积
        self.bn = nn.BatchNorm2d(c1)  # 批量归一化层

    def forward(self, x):
        return torch.max(x, self.bn(self.conv(x)))  # FReLU公式：max(x, T(x))


# ACON激活函数 https://arxiv.org/pdf/2009.04759.pdf ----------------------------------------------------------------------------
class AconC(nn.Module):
    r""" ACON激活函数（激活或不激活）。
    AconC: (p1*x - p2*x) * sigmoid(beta*(p1*x - p2*x)) + p2*x
    根据论文 "Activate or Not: Learning Customized Activation" (https://arxiv.org/pdf/2009.04759.pdf)。
    """
    def __init__(self, c1):
        super().__init__()
        self.p1 = nn.Parameter(torch.randn(1, c1, 1, 1))
        self.p2 = nn.Parameter(torch.randn(1, c1, 1, 1))
        self.beta = nn.Parameter(torch.ones(1, c1, 1, 1))

    def forward(self, x):
        dpx = (self.p1 - self.p2) * x
        return dpx * torch.sigmoid(self.beta * dpx) + self.p2 * x


class MetaAconC(nn.Module):
    r""" MetaAcon激活函数（激活或不激活）。
    MetaAconC: (p1*x - p2*x) * sigmoid(beta*(p1*x - p2*x)) + p2*x
    beta由一个小网络生成，根据论文 "Activate or Not: Learning Customized Activation" (https://arxiv.org/pdf/2009.04759.pdf)。
    """
    def __init__(self, c1, k=1, s=1, r=16):  # 输入通道数c1，卷积核大小k，步长s，压缩比r
        super().__init__()
        c2 = max(r, c1 // r)  # 计算中间通道数
        self.p1 = nn.Parameter(torch.randn(1, c1, 1, 1))
        self.p2 = nn.Parameter(torch.randn(1, c1, 1, 1))
        self.fc1 = nn.Conv2d(c1, c2, k, s, bias=True)  # 第一个全连接层
        self.fc2 = nn.Conv2d(c2, c1, k, s, bias=True)  # 第二个全连接层
        # self.bn1 = nn.BatchNorm2d(c2)
        # self.bn2 = nn.BatchNorm2d(c1)

    def forward(self, x):
        y = x.mean(dim=2, keepdims=True).mean(dim=3, keepdims=True)  # 计算全局平均值
        # batch-size 1 bug/instabilities https://github.com/ultralytics/yolov5/issues/2891
        # beta = torch.sigmoid(self.bn2(self.fc2(self.bn1(self.fc1(y)))))  # 原始实现（包含批量归一化）
        beta = torch.sigmoid(self.fc2(self.fc1(y)))  # 修复批量归一化问题
        dpx = (self.p1 - self.p2) * x
        return dpx * torch.sigmoid(beta * dpx) + self.p2 * x
