# YOLOv5 common modules

import math
from copy import copy
from pathlib import Path
import warnings

import cv2
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from torch import einsum
from PIL import Image
from torch.cuda import amp
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.modules.utils import _triple, _pair, _single
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from utils.datasets import letterbox
from utils.general import non_max_suppression, make_divisible, scale_coords, increment_path, xyxy2xywh, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import time_synchronized
from timm.models.layers import DropPath

from torch.nn import init, Sequential
import math
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np

# 自动计算padding，使得卷积操作后特征图尺寸保持不变
def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


# 深度可分离卷积
def DWConv(c1, c2, k=1, s=1, act=True):
    return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class Conv(nn.Module):
    # 标准卷积
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class TransformerLayer(nn.Module):
    # Transformer层 https://arxiv.org/abs/2010.11929
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    # 视觉Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)
        self.tr = nn.Sequential(*[TransformerLayer(c2, num_heads) for _ in range(num_layers)])
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2)
        p = p.unsqueeze(0)
        p = p.transpose(0, 3)
        p = p.squeeze(3)
        e = self.linear(p)
        x = p + e

        x = self.tr(x)
        x = x.unsqueeze(3)
        x = x.transpose(0, 3)
        x = x.reshape(b, self.c2, w, h)
        return x


class VGGblock(nn.Module):
    # VGG模块
    def __init__(self, num_convs, c1, c2):
        super(VGGblock, self).__init__()
        self.blk = []
        for num in range(num_convs):
            if num == 0:
                self.blk.append(nn.Sequential(nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=3, padding=1),
                                              nn.ReLU(),
                                              ))
            else:
                self.blk.append(nn.Sequential(nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=3, padding=1),
                                              nn.ReLU(),
                                              ))
        self.blk.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.vggblock = nn.Sequential(*self.blk)

    def forward(self, x):
        out = self.vggblock(x)

        return out


class ResNetblock(nn.Module):
    # ResNet残差块
    expansion = 4

    def __init__(self, c1, c2, stride=1):
        super(ResNetblock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c2)
        self.conv2 = nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.conv3 = nn.Conv2d(in_channels=c2, out_channels=self.expansion*c2, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*c2)

        self.shortcut = nn.Sequential()
        if stride != 1 or c1 != self.expansion*c2:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels=c1, out_channels=self.expansion*c2, kernel_size=1, stride=stride, bias=False),
                                          nn.BatchNorm2d(self.expansion*c2),
                                          )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)

        return out


class ResNetlayer(nn.Module):
    # ResNet层
    expansion = 4

    def __init__(self, c1, c2, stride=1, is_first=False, num_blocks=1):
        super(ResNetlayer, self).__init__()
        self.blk = []
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=7, stride=2, padding=3, bias=False),
                                        nn.BatchNorm2d(c2),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        else:
            self.blk.append(ResNetblock(c1, c2, stride))
            for i in range(num_blocks - 1):
                self.blk.append(ResNetblock(self.expansion*c2, c2, 1))
            self.layer = nn.Sequential(*self.blk)

    def forward(self, x):
        out = self.layer(x)

        return out


class Bottleneck(nn.Module):
    # 标准瓶颈层
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # CSP瓶颈层 https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class C3(nn.Module):
    # CSP瓶颈层，包含3个卷积
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super(C3, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class C3TR(C3):
    # C3模块，使用TransformerBlock替代部分结构
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class SPP(nn.Module):
    # 空间金字塔池化层
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    # 空间金字塔池化-快速层
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class Focus(nn.Module):
    # 将宽高信息聚焦到通道空间
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

    def forward(self, x):
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))


class Contract(nn.Module):
    # 将宽高压缩到通道中，例如将x(1,64,80,80)转换为x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()
        s = self.gain
        x = x.view(N, C, H // s, s, W // s, s)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        return x.view(N, C * s * s, H // s, W // s)


class Expand(nn.Module):
    # 将通道扩展到宽高，例如将x(1,64,80,80)转换为x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()
        s = self.gain
        x = x.view(N, s, s, C // s ** 2, H, W)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()
        return x.view(N, C // s ** 2, H * s, W * s)


class Concat(nn.Module):
    # 沿指定维度拼接张量列表
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        # print(x.shape)
        return torch.cat(x, self.d)


class Add(nn.Module):
    # 将张量列表相加并平均
    def __init__(self, weight=0.5):
        super().__init__()
        self.w = weight

    def forward(self, x):
        return x[0] * self.w + x[1] * (1 - self.w)


class Add2(nn.Module):
    # 选择性相加，x + transformer[0] 或 x + transformer[1]
    def __init__(self, c1, index):
        super().__init__()
        self.index = index

    def forward(self, x):
        if self.index == 0:
            return torch.add(x[0], x[1][0])
        elif self.index == 1:
            return torch.add(x[0], x[1][1])
        # return torch.add(x[0], x[1])


class NiNfusion(nn.Module):
    # NiN融合模块
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        super(NiNfusion, self).__init__()

        self.concat = Concat(dimension=1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        y = self.concat(x)
        y = self.act(self.conv(y))

        return y


class DMAF(nn.Module):
    # DMAF模块
    def __init__(self, c2):
        super(DMAF, self).__init__()


    def forward(self, x):
        x1 = x[0]
        x2 = x[1]

        subtract_vis = x1 - x2
        avgpool_vis = nn.AvgPool2d(kernel_size=(subtract_vis.size(2), subtract_vis.size(3)))
        weight_vis = torch.tanh(avgpool_vis(subtract_vis))

        subtract_ir = x2 - x1
        avgpool_ir = nn.AvgPool2d(kernel_size=(subtract_ir.size(2), subtract_ir.size(3)))
        weight_ir = torch.tanh(avgpool_ir(subtract_ir))

        x1_weight = subtract_vis * weight_ir
        x2_weight = subtract_ir * weight_vis

        return x1_weight, x2_weight


class NMS(nn.Module):
    # 非极大值抑制模块
    conf = 0.25
    iou = 0.45
    classes = None

    def __init__(self):
        super(NMS, self).__init__()

    def forward(self, x):
        return non_max_suppression(x[0], conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)


class autoShape(nn.Module):
    # 自动适应输入形状的模型封装，支持多种输入格式，并包含预处理、推理和NMS
    conf = 0.25
    iou = 0.45
    classes = None

    def __init__(self, model):
        super(autoShape, self).__init__()
        self.model = model.eval()

    def autoshape(self):
        print('autoShape already enabled, skipping... ')
        return self

    @torch.no_grad()
    def forward(self, imgs, size=640, augment=False, profile=False):
        t = [time_synchronized()]
        p = next(self.model.parameters())
        if isinstance(imgs, torch.Tensor):
            with amp.autocast(enabled=p.device.type != 'cpu'):
                return self.model(imgs.to(p.device).type_as(p), augment, profile)

        # Pre-process
        n, imgs = (len(imgs), imgs) if isinstance(imgs, list) else (1, [imgs])
        shape0, shape1, files = [], [], []
        for i, im in enumerate(imgs):
            f = f'image{i}'
            if isinstance(im, str):
                im, f = np.asarray(Image.open(requests.get(im, stream=True).raw if im.startswith('http') else im)), im
            elif isinstance(im, Image.Image):
                im, f = np.asarray(im), getattr(im, 'filename', f) or f
            files.append(Path(f).with_suffix('.jpg').name)
            if im.shape[0] < 5:
                im = im.transpose((1, 2, 0))
            im = im[:, :, :3] if im.ndim == 3 else np.tile(im[:, :, None], 3)
            s = im.shape[:2]
            shape0.append(s)
            g = (size / max(s))
            shape1.append([y * g for y in s])
            imgs[i] = im if im.data.contiguous else np.ascontiguousarray(im)
        shape1 = [make_divisible(x, int(self.stride.max())) for x in np.stack(shape1, 0).max(0)]
        x = [letterbox(im, new_shape=shape1, auto=False)[0] for im in imgs]
        x = np.stack(x, 0) if n > 1 else x[0][None]
        x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))
        x = torch.from_numpy(x).to(p.device).type_as(p) / 255.
        t.append(time_synchronized())

        with amp.autocast(enabled=p.device.type != 'cpu'):
            y = self.model(x, augment, profile)[0]
            t.append(time_synchronized())

            y = non_max_suppression(y, conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)
            for i in range(n):
                scale_coords(shape1, y[i][:, :4], shape0[i])

            t.append(time_synchronized())
            return Detections(imgs, y, files, t, self.names, x.shape)


class Detections:
    # YOLOv5推理结果的检测类
    def __init__(self, imgs, pred, files, times=None, names=None, shape=None):
        super(Detections, self).__init__()
        d = pred[0].device
        gn = [torch.tensor([*[im.shape[i] for i in [1, 0, 1, 0]], 1., 1.], device=d) for im in imgs]
        self.imgs = imgs
        self.pred = pred
        self.names = names
        self.files = files
        self.xyxy = pred
        self.xywh = [xyxy2xywh(x) for x in pred]
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]
        self.n = len(self.pred)
        self.t = tuple((times[i + 1] - times[i]) * 1000 / self.n for i in range(3))
        self.s = shape

    def display(self, pprint=False, show=False, save=False, crop=False, render=False, save_dir=Path('')):
        for i, (im, pred) in enumerate(zip(self.imgs, self.pred)):
            str = f'image {i + 1}/{len(self.pred)}: {im.shape[0]}x{im.shape[1]} '
            if pred is not None:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()
                    str += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "
                if show or save or render or crop:
                    for *box, conf, cls in pred:
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        if crop:
                            save_one_box(box, im, file=save_dir / 'crops' / self.names[int(cls)] / self.files[i])
                        else:
                            plot_one_box(box, im, label=label, color=colors(cls))

            im = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im
            if pprint:
                print(str.rstrip(', '))
            if show:
                im.show(self.files[i])
            if save:
                f = self.files[i]
                im.save(save_dir / f)
                print(f"{'Saved' * (i == 0)} {f}", end=',' if i < self.n - 1 else f' to {save_dir}\n')
            if render:
                self.imgs[i] = np.asarray(im)

    def print(self):
        self.display(pprint=True)
        print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {tuple(self.s)}' % self.t)

    def show(self):
        self.display(show=True)

    def save(self, save_dir='runs/hub/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/hub/exp', mkdir=True)
        self.display(save=True, save_dir=save_dir)

    def crop(self, save_dir='runs/hub/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/hub/exp', mkdir=True)
        self.display(crop=True, save_dir=save_dir)
        print(f'Saved results to {save_dir}\n')

    def render(self):
        self.display(render=True)
        return self.imgs

    def pandas(self):
        new = copy(self)
        ca = 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'
        cb = 'xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name'
        for k, c in zip(['xyxy', 'xyxyn', 'xywh', 'xywhn'], [ca, ca, cb, cb]):
            a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in getattr(self, k)]
            setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
        return new

    def tolist(self):
        x = [Detections([self.imgs[i]], [self.pred[i]], self.names, self.s) for i in range(self.n)]
        for d in x:
            for k in ['imgs', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
                setattr(d, k, getattr(d, k)[0])
        return x

    def __len__(self):
        return self.n


class Classify(nn.Module):
    # 分类头，将x(b,c1,20,20)转换为x(b,c2)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        super(Classify, self).__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g)
        self.flat = nn.Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)
        return self.flat(self.conv(z))


class LearnableCoefficient(nn.Module):
    def __init__(self):
        super(LearnableCoefficient, self).__init__()
        self.bias = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)

    def forward(self, x):
        out = x * self.bias
        return out


class LearnableWeights(nn.Module):
    def __init__(self):
        super(LearnableWeights, self).__init__()
        self.w1 = nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        self.w2 = nn.Parameter(torch.tensor([0.5]), requires_grad=True)

    def forward(self, x1, x2):
        out = x1 * self.w1 + x2 * self.w2
        return out


class CrossAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, attn_pdrop=.1, resid_pdrop=.1):
        '''
        :param d_model: 模型的输出维度
        :param d_k: 查询和键的维度
        :param d_v: 值的维度
        :param h: 头的数量
        '''
        super(CrossAttention, self).__init__()
        assert d_k % h == 0
        self.d_model = d_model
        self.d_k = d_model // h
        self.d_v = d_model // h
        self.h = h

        self.que_proj_vis = nn.Linear(d_model, h * self.d_k)
        self.key_proj_vis = nn.Linear(d_model, h * self.d_k)
        self.val_proj_vis = nn.Linear(d_model, h * self.d_v)

        self.que_proj_ir = nn.Linear(d_model, h * self.d_k)
        self.key_proj_ir = nn.Linear(d_model, h * self.d_k)
        self.val_proj_ir = nn.Linear(d_model, h * self.d_v)

        self.out_proj_vis = nn.Linear(h * self.d_v, d_model)
        self.out_proj_ir = nn.Linear(h * self.d_v, d_model)

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        # layer norm
        self.LN1 = nn.LayerNorm(d_model)
        self.LN2 = nn.LayerNorm(d_model)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, attention_mask=None, attention_weights=None):
        rgb_fea_flat = x[0]
        ir_fea_flat = x[1]
        b_s, nq = rgb_fea_flat.shape[:2]
        nk = rgb_fea_flat.shape[1]

        # Self-Attention
        rgb_fea_flat = self.LN1(rgb_fea_flat)
        q_vis = self.que_proj_vis(rgb_fea_flat).contiguous().view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)
        k_vis = self.key_proj_vis(rgb_fea_flat).contiguous().view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)
        v_vis = self.val_proj_vis(rgb_fea_flat).contiguous().view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)

        ir_fea_flat = self.LN2(ir_fea_flat)
        q_ir = self.que_proj_ir(ir_fea_flat).contiguous().view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)
        k_ir = self.key_proj_ir(ir_fea_flat).contiguous().view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)
        v_ir = self.val_proj_ir(ir_fea_flat).contiguous().view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)

        att_vis = torch.matmul(q_ir, k_vis) / np.sqrt(self.d_k)
        att_ir = torch.matmul(q_vis, k_ir) / np.sqrt(self.d_k)

        att_vis = torch.softmax(att_vis, -1)
        att_vis = self.attn_drop(att_vis)
        att_ir = torch.softmax(att_ir, -1)
        att_ir = self.attn_drop(att_ir)

        # output
        out_vis = torch.matmul(att_vis, v_vis).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)
        out_vis = self.resid_drop(self.out_proj_vis(out_vis))
        out_ir = torch.matmul(att_ir, v_ir).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)
        out_ir = self.resid_drop(self.out_proj_ir(out_ir))

        return [out_vis, out_ir]


class CrossTransformerBlock(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop, loops_num=1):
        """
        :param d_model: 模型的输出维度
        :param d_k: 查询和键的维度
        :param d_v: 值的维度
        :param h: 头的数量
        :param block_exp: MLP（前馈网络）的扩展因子
        """
        super(CrossTransformerBlock, self).__init__()
        self.loops = loops_num
        self.ln_input = nn.LayerNorm(d_model)
        self.ln_output = nn.LayerNorm(d_model)
        self.crossatt = CrossAttention(d_model, d_k, d_v, h, attn_pdrop, resid_pdrop)
        self.mlp_vis = nn.Sequential(nn.Linear(d_model, block_exp * d_model),
                                     nn.GELU(),
                                     nn.Linear(block_exp * d_model, d_model),
                                     nn.Dropout(resid_pdrop),
                                     )
        self.mlp_ir = nn.Sequential(nn.Linear(d_model, block_exp * d_model),
                                    nn.GELU(),
                                    nn.Linear(block_exp * d_model, d_model),
                                    nn.Dropout(resid_pdrop),
                                    )
        self.mlp = nn.Sequential(nn.Linear(d_model, block_exp * d_model),
                                 nn.GELU(),
                                 nn.Linear(block_exp * d_model, d_model),
                                 nn.Dropout(resid_pdrop),
                                 )

        self.LN1 = nn.LayerNorm(d_model)
        self.LN2 = nn.LayerNorm(d_model)

        self.coefficient1 = LearnableCoefficient()
        self.coefficient2 = LearnableCoefficient()
        self.coefficient3 = LearnableCoefficient()
        self.coefficient4 = LearnableCoefficient()
        self.coefficient5 = LearnableCoefficient()
        self.coefficient6 = LearnableCoefficient()
        self.coefficient7 = LearnableCoefficient()
        self.coefficient8 = LearnableCoefficient()

    def forward(self, x):
        rgb_fea_flat = x[0]
        ir_fea_flat = x[1]
        assert rgb_fea_flat.shape[0] == ir_fea_flat.shape[0]
        bs, nx, c = rgb_fea_flat.size()
        h = w = int(math.sqrt(nx))

        for loop in range(self.loops):
            rgb_fea_out, ir_fea_out = self.crossatt([rgb_fea_flat, ir_fea_flat])
            rgb_att_out = self.coefficient1(rgb_fea_flat) + self.coefficient2(rgb_fea_out)
            ir_att_out = self.coefficient3(ir_fea_flat) + self.coefficient4(ir_fea_out)
            rgb_fea_flat = self.coefficient5(rgb_att_out) + self.coefficient6(self.mlp_vis(self.LN2(rgb_att_out)))
            ir_fea_flat = self.coefficient7(ir_att_out) + self.coefficient8(self.mlp_ir(self.LN2(ir_att_out)))


        return [rgb_fea_flat, ir_fea_flat]


class TransformerFusionBlock(nn.Module):
    def __init__(self, d_model, vert_anchors=16, horz_anchors=16, h=8, block_exp=4, n_layer=1, embd_pdrop=0.1, attn_pdrop=0.1, resid_pdrop=0.1):
        super(TransformerFusionBlock, self).__init__()

        self.n_embd = d_model
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors
        d_k = d_model
        d_v = d_model

        self.pos_emb_vis = nn.Parameter(torch.zeros(1, vert_anchors * horz_anchors, self.n_embd))
        self.pos_emb_ir = nn.Parameter(torch.zeros(1, vert_anchors * horz_anchors, self.n_embd))


        self.avgpool = AdaptivePool2d(self.vert_anchors, self.horz_anchors, 'avg')
        self.maxpool = AdaptivePool2d(self.vert_anchors, self.horz_anchors, 'max')

        self.vis_coefficient = LearnableWeights()
        self.ir_coefficient = LearnableWeights()

        self.apply(self._init_weights)

        self.crosstransformer = nn.Sequential(*[CrossTransformerBlock(d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop) for layer in range(n_layer)])

        self.concat = Concat(dimension=1)

        self.conv1x1_out = Conv(c1=d_model * 2, c2=d_model, k=1, s=1, p=0, g=1, act=True)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        rgb_fea = x[0]
        ir_fea = x[1]
        assert rgb_fea.shape[0] == ir_fea.shape[0]
        bs, c, h, w = rgb_fea.shape

        new_rgb_fea = self.vis_coefficient(self.avgpool(rgb_fea), self.maxpool(rgb_fea))
        new_c, new_h, new_w = new_rgb_fea.shape[1], new_rgb_fea.shape[2], new_rgb_fea.shape[3]
        rgb_fea_flat = new_rgb_fea.contiguous().view(bs, new_c, -1).permute(0, 2, 1) + self.pos_emb_vis

        new_ir_fea = self.ir_coefficient(self.avgpool(ir_fea), self.maxpool(ir_fea))
        ir_fea_flat = new_ir_fea.contiguous().view(bs, new_c, -1).permute(0, 2, 1) + self.pos_emb_ir

        rgb_fea_flat, ir_fea_flat = self.crosstransformer([rgb_fea_flat, ir_fea_flat])

        rgb_fea_CFE = rgb_fea_flat.contiguous().view(bs, new_h, new_w, new_c).permute(0, 3, 1, 2)
        if self.training == True:
            rgb_fea_CFE = F.interpolate(rgb_fea_CFE, size=([h, w]), mode='nearest')
        else:
            rgb_fea_CFE = F.interpolate(rgb_fea_CFE, size=([h, w]), mode='bilinear')
        new_rgb_fea = rgb_fea_CFE + rgb_fea
        ir_fea_CFE = ir_fea_flat.contiguous().view(bs, new_h, new_w, new_c).permute(0, 3, 1, 2)
        if self.training == True:
            ir_fea_CFE = F.interpolate(ir_fea_CFE, size=([h, w]), mode='nearest')
        else:
            ir_fea_CFE = F.interpolate(ir_fea_CFE, size=([h, w]), mode='bilinear')
        new_ir_fea = ir_fea_CFE + ir_fea

        new_fea = self.concat([new_rgb_fea, new_ir_fea])
        new_fea = self.conv1x1_out(new_fea)

        
        return new_fea


class AdaptivePool2d(nn.Module):
    def __init__(self, output_h, output_w, pool_type='avg'):
        super(AdaptivePool2d, self).__init__()

        self.output_h = output_h
        self.output_w = output_w
        self.pool_type = pool_type

    def forward(self, x):
        bs, c, input_h, input_w = x.shape

        if (input_h > self.output_h) or (input_w > self.output_w):
            self.stride_h = input_h // self.output_h
            self.stride_w = input_w // self.output_w
            self.kernel_size = (input_h - (self.output_h - 1) * self.stride_h, input_w - (self.output_w - 1) * self.stride_w)

            if self.pool_type == 'avg':
                y = nn.AvgPool2d(kernel_size=self.kernel_size, stride=(self.stride_h, self.stride_w), padding=0)(x)
            else:
                y = nn.MaxPool2d(kernel_size=self.kernel_size, stride=(self.stride_h, self.stride_w), padding=0)(x)
        else:
            y = x

        return y

class SE_Block(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super(SE_Block, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.gap(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        return x * y.expand_as(x)


class Channel_Attention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        '''
        :param in_channels: 输入通道数
        :param reduction_ratio: 输出通道数量的缩放系数
        :param pool_types: 池化类型
        '''

        super(Channel_Attention, self).__init__()

        self.pool_types = pool_types
        self.in_channels = in_channels
        self.shared_mlp = nn.Sequential(nn.Flatten(),
                                        nn.Linear(in_features=in_channels, out_features=in_channels//reduction_ratio),
                                        nn.ReLU(),
                                        nn.Linear(in_features=in_channels//reduction_ratio, out_features=in_channels)
                                        )

    def forward(self, x):
        channel_attentions = []

        for pool_types in self.pool_types:
            if pool_types == 'avg':
                pool_init = nn.AvgPool2d(kernel_size=(x.size(2), x.size(3)))
                avg_pool = pool_init(x)
                channel_attentions.append(self.shared_mlp(avg_pool))
            elif pool_types == 'max':
                pool_init = nn.MaxPool2d(kernel_size=(x.size(2), x.size(3)))
                max_pool = pool_init(x)
                channel_attentions.append(self.shared_mlp(max_pool))

        pooling_sums = torch.stack(channel_attentions, dim=0).sum(dim=0)
        output = nn.Sigmoid()(pooling_sums).unsqueeze(2).unsqueeze(3).expand_as(x)

        return x * output


# 空间注意力模块
class Spatial_Attention(nn.Module):
    def __init__(self, kernel_size=7):
        super(Spatial_Attention, self).__init__()

        self.spatial_attention = nn.Sequential(nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, stride=1, dilation=1, padding=(kernel_size-1)//2, bias=False),
                                               nn.BatchNorm2d(num_features=1, eps=1e-5, momentum=0.01, affine=True)
                                               )

    def forward(self, x):
        x_compress = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)  # 在通道维度上分别计算平均值和最大值，并在通道维度上进行拼接
        x_output = self.spatial_attention(x_compress)  # 使用7x7卷积核进行卷积
        scaled = nn.Sigmoid()(x_output)

        return x * scaled  # 将输入F'和通道注意力模块的输出Ms相乘，得到F''


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, pool_types=['avg', 'max'], spatial=True):
        super(CBAM, self).__init__()

        self.spatial = spatial
        self.channel_attention = Channel_Attention(in_channels=in_channels, reduction_ratio=reduction_ratio, pool_types=pool_types)

        if self.spatial:
            self.spatial_attention = Spatial_Attention(kernel_size=7)

    def forward(self, x):
        x_out = self.channel_attention(x)
        if self.spatial:
            x_out = self.spatial_attention(x_out)

        return x_out
