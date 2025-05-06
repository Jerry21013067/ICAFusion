# YOLOv5 YOLO特定模块

import argparse
import logging
import sys
from copy import deepcopy
from pathlib import Path

sys.path.append(Path(__file__).parent.parent.absolute().__str__())  # 为了在子目录中运行'$ python *.py'文件
logger = logging.getLogger(__name__)

from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import make_divisible, check_file, set_logging
from utils.torch_utils import time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, \
    select_device, copy_attr
#from mmcv.ops import DeformConv2dPack as DCN

try:
    import thop  # 用于计算FLOPS
except ImportError:
    thop = None


class Detect(nn.Module):
    # 检测层
    stride = None  # 在构建过程中计算的步长
    export = False  # onnx导出

    def __init__(self, nc=80, anchors=(), ch=()):
        """
        初始化检测层
        参数：
            nc: 类别数量，默认为80
            anchors: 锚点，默认为空
            ch: 输入通道数，默认为空
        """
        super(Detect, self).__init__()
        self.nc = nc  # 类别数量
        self.no = nc + 5  # 每个锚点的输出数量
        self.nl = len(anchors)  # 检测层数量
        self.na = len(anchors[0]) // 2  # 每个检测层的锚点数量
        self.grid = [torch.zeros(1)] * self.nl  # 初始化网格
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # 输出卷积
        #self.m = nn.ModuleList(DCN(x, self.no * self.na, kernel_size=(3, 3), stride=1, padding=1, dilation=1, deform_groups=1) for x in ch)  # 输出DCN卷积3x3

    def forward(self, x):
        """
        前向传播
        参数：
            x: 输入张量
        返回：
            z: 推理输出
            logits_: logits输出
        """
        z = []  # 推理输出
        logits_ = []
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # 卷积
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # 推理
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device )

                logits = x[i][..., 5:]

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))
                logits_.append(logits.view(bs, -1, self.no - 5))

        return x if self.training else (torch.cat(z, 1), torch.cat(logits_, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        """
        创建网格
        参数：
            nx: 网格宽度，默认为20
            ny: 网格高度，默认为20
        返回：
            网格张量
        """
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Model(nn.Module):
    """
    YOLOv5模型
    """
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):
        """
        初始化YOLOv5模型
        参数：
            cfg: 模型配置文件路径或字典，默认为'yolov5s.yaml'
            ch: 输入通道数，默认为3
            nc: 类别数量，默认为None
            anchors: 锚点，默认为None
        """
        super(Model, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict

        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # 定义模型
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            logger.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            logger.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names

        # 构建步长和锚点
        m = self.model[-1]  # Detect()
        # print(m)

        if isinstance(m, Detect):
            s = 256  # 2x min stride
            # m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s), torch.zeros(1, ch, s, s))])  # forward
            m.stride = torch.Tensor([8.0, 16.0, 32.0])
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            #self._initialize_biases()  # only run once

        # 初始化权重和偏置
        initialize_weights(self)
        self.info()
        logger.info('')

    def forward(self, x, x2, augment=False, profile=False):
        """
        前向传播
        参数：
            x: 输入张量
            x2: 第二个输入张量
            augment: 是否进行数据增强，默认为False
            profile: 是否进行性能分析，默认为False
        返回：
            输出张量
        """
        if augment:
            img_size = x.shape[-2:]  # 图像高度和宽度
            s = [1, 0.83, 0.67]  # 缩放比例
            f = [None, 3, None]  # 翻转方向 (2-上下翻转, 3-左右翻转)
            y = []  # 输出列表
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
                yi = self.forward_once(xi)[0]  # 前向传播
                yi[..., :4] /= si  # 缩放坐标
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # 上下翻转
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # 左右翻转
                y.append(yi)
            return torch.cat(y, 1), None  # 数据增强推理
        else:
            return self.forward_once(x, x2, profile)  # 单尺度推理


    def forward_once(self, x, x2, profile=False):
        """
        单次前向传播
        参数：
            x: 输入张量
            x2: 第二个输入张量
            profile: 是否进行性能分析，默认为False
        返回：
            输出张量
        """
        y, dt = [], []  # 输出列表和时间列表
        i = 0
        for m in self.model:
            if m.f != -1:  # 如果不是从上一层获取数据
                if m.f != -4:
                    x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # 从前面的层获取数据

            if profile:
                o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPS
                t = time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((time_synchronized() - t) * 100)
                if m == self.model[0]:
                    logger.info(f"{'time (ms)':>10s} {'GFLOPS':>10s} {'params':>10s}  {'module'}")
                logger.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')

            if m.f == -4:
                x = m(x2)
            else:
                x = m(x)  # 前向传播
            y.append(x if m.i in self.save else None)  # 保存输出
            i += 1

        if profile:
            logger.info('%.1fms total' % sum(dt))
        return x

    def _initialize_biases(self, cf=None):  # 初始化偏置
        # https://arxiv.org/abs/1708.02002 section 3.3
        """
        初始化检测层的偏置
        参数：
            cf: 类别频率，默认为None
        """
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # 遍历检测层
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        """
        打印偏置信息
        """
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # 遍历检测层
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            logger.info(
                ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    def fuse(self):  # 融合Conv2d和BatchNorm2d层
        logger.info('Fusing layers... ')
        for m in self.model.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # 更新卷积层
                delattr(m, 'bn')  # 删除BatchNorm2d层
                m.forward = m.fuseforward  # 更新前向传播
        self.info()
        return self

    def nms(self, mode=True):  # 添加或移除NMS模块
        present = type(self.model[-1]) is NMS  # 最后一层是否为NMS
        if mode and not present:
            logger.info('Adding NMS... ')
            m = NMS()  # 创建NMS模块
            m.f = -1  # from
            m.i = self.model[-1].i + 1  # 索引
            self.model.add_module(name='%s' % m.i, module=m)  # add
            self.eval()
        elif not mode and present:
            logger.info('Removing NMS... ')
            self.model = self.model[:-1]  # remove
        return self

    def autoshape(self):  # 添加autoShape模块
        logger.info('Adding autoShape... ')
        m = autoShape(self)  # 包装模型
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # 复制属性
        return m

    def info(self, verbose=False, img_size=640):  # 打印模型信息
        model_info(self, verbose, img_size)


def parse_model(d, ch):  # 解析模型
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # 锚点数量
    no = na * (nc + 5)  # 输出数量 = 锚点数量 * (类别数量 + 5)

    layers, save, c2 = [], [], ch[-1]  # 层列表、保存列表、输出通道数
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # 遍历模型结构
        m = eval(m) if isinstance(m, str) else m  # 解析模块
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # 解析参数
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # 深度增益
        if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv, BottleneckCSP,
                 C3, C3TR]:
            # 配置卷积层
            if m is Focus:
                c1, c2 = 3, args[0]
                if c2 != no:  # 如果不是输出层
                    c2 = make_divisible(c2 * gw, 8)
                args = [c1, c2, *args[1:]]
            elif m is Conv and args[0] == 64:    # new
                c1, c2 = 3, args[0]
                if c2 != no:  # 如果不是输出层
                    c2 = make_divisible(c2 * gw, 8)
                args = [c1, c2, *args[1:]]
            else:
                c1, c2 = ch[f], args[0]
                if c2 != no:  # 如果不是输出层
                    c2 = make_divisible(c2 * gw, 8)

                args = [c1, c2, *args[1:]]
                if m in [BottleneckCSP, C3, C3TR]:
                    args.insert(2, n)  # 插入重复次数
                    n = 1

        # 配置其他模块
        elif m is ResNetlayer:
            if args[3] == True:
                c2 = args[1]
            else:
                c2 = args[1]*4
        elif m is VGGblock:
            c2 = args[2]
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        elif m in [Add, DMAF]:
            c2 = ch[f[0]]
            args = [c2]
        elif m is Add2:
            c2 = ch[f[0]]
            args = [c2, args[1]]
        elif m is Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # 锚点数量
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        elif m is NiNfusion:
            c1 = sum([ch[x] for x in f])
            c2 = c1 // 2
            args = [c1, c2, *args]
        elif m is TransformerFusionBlock:
            c2 = ch[f[0]]
            args = [c2, *args[1:]]
        else:
            c2 = ch[f]

        # 创建模块
        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # 模块
        t = str(m)[8:-2].replace('__main__.', '')  # 模块类型
        np = sum([x.numel() for x in m_.parameters()])  # 参数数量
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # 添加索引、来源、类型和参数数量
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # 添加到保存列表
        layers.append(m_)
        if i == 0:
            ch = []

        ch.append(c2)
        
    return nn.Sequential(*layers), sorted(save)  # 返回模型序列和保存列表


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='/home/fqy/proj/paper/YOLOFusion/models/transformer/yolov5s_fusion_transformer(x3)_vedai.yaml', help='model.yaml')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # 检查文件
    set_logging()
    device = select_device(opt.device)
    print(device)

    # 创建模型
    model = Model(opt.cfg).to(device)
    input_rgb = torch.Tensor(8, 3, 640, 640).to(device)
    input_ir = torch.Tensor(8, 3, 640, 640).to(device)
    # 前向传播
    output = model(input_rgb, input_ir)
    
