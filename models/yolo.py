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

try:
    import thop  # 用于计算FLOPS
except ImportError:
    thop = None


class Detect(nn.Module):
    """
    检测层
    """
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
        self.register_buffer('anchors', a)  # 形状(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # 形状(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # 输出卷积

    def forward(self, x):
        """
        前向传播
        参数：
            x: 输入张量
        返回：
            z: 推理输出
        """
        z = []  # 推理输出
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # 卷积
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # 推理
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

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
            cfg: 模型配置文件，默认为'yolov5s.yaml'
            ch: 输入通道数，默认为3
            nc: 类别数量，默认为None
            anchors: 锚点，默认为None
        """
        super(Model, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # 模型字典

        else:  # is *.yaml
            import yaml  # 用于torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.safe_load(f)  # 模型字典
            print("YAML")
            print(self.yaml)

        # 定义模型
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # 输入通道数
        if nc and nc != self.yaml['nc']:
            logger.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # 覆盖yaml值
        if anchors:
            logger.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # 覆盖yaml值
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # 默认名称
        # logger.info([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

        # 构建步长，锚点
        m = self.model[-1]  # Detect()
        # print("Detect")
        # print(m)
        if isinstance(m, Detect):
            s = 256  # 2x最小步长
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # 前向传播
            # print("m.stride", m.stride)
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # 仅运行一次
            # logger.info('Strides: %s' % m.stride.tolist())

        # 初始化权重，偏置
        initialize_weights(self)
        self.info()
        logger.info('')

    def forward(self, x, augment=False, profile=False):
        """
        前向传播
        参数：
            x: 输入张量
            augment: 是否增强，默认为False
            profile: 是否分析，默认为False
        返回：
            输出张量
        """
        if augment:
            img_size = x.shape[-2:]  # 高度，宽度
            s = [1, 0.83, 0.67]  # 缩放比例
            f = [None, 3, None]  # 翻转 (2-ud, 3-lr)
            y = []  # 输出
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
                yi = self.forward_once(xi)[0]  # 前向传播
                # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # 保存
                yi[..., :4] /= si  # 反缩放
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # 反翻转ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # 反翻转lr
                y.append(yi)
            return torch.cat(y, 1), None  # 增强推理，训练
        else:
            return self.forward_once(x, profile)  # 单尺度推理，训练

    def forward_once(self, x, profile=False):
        """
        单次前向传播
        参数：
            x: 输入张量
            profile: 是否分析，默认为False
        返回：
            输出张量
        """
        y, dt = [], []  # 输出
        for m in self.model:
            if m.f != -1:  # 如果不是来自前一层
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # 来自更早的层

            if profile:
                o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPS
                t = time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((time_synchronized() - t) * 100)
                if m == self.model[0]:
                    logger.info(f"{'time (ms)':>10s} {'GFLOPS':>10s} {'params':>10s}  {'module'}")
                logger.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')

            x = m(x)  # 运行
            y.append(x if m.i in self.save else None)  # 保存输出

        if profile:
            logger.info('%.1fms total' % sum(dt))
        return x

    def _initialize_biases(self, cf=None):
        # https://arxiv.org/abs/1708.02002 section 3.3
        """
        初始化偏置
        参数：
            cf: 类别频率，默认为None
        """
        m = self.model[-1]  # Detect()模块
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        """
        打印偏置
        """
        m = self.model[-1]  # Detect()模块
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            logger.info(
                ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))


    def fuse(self):
        """
        融合模型Conv2d() + BatchNorm2d()层
        """
        logger.info('Fusing layers... ')
        for m in self.model.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # 更新卷积
                delattr(m, 'bn')  # 删除批量归一化
                m.forward = m.fuseforward  # 更新前向传播
        self.info()
        return self

    def nms(self, mode=True):
        """
        添加或移除NMS模块
        参数：
            mode: 是否添加，默认为True
        """
        present = type(self.model[-1]) is NMS  # 最后一层是否为NMS
        if mode and not present:
            logger.info('Adding NMS... ')
            m = NMS()  # 模块
            m.f = -1  # from
            m.i = self.model[-1].i + 1  # 索引
            self.model.add_module(name='%s' % m.i, module=m)  # add
            self.eval()
        elif not mode and present:
            logger.info('Removing NMS... ')
            self.model = self.model[:-1]  # remove
        return self

    def autoshape(self):
        """
        添加autoShape模块
        """
        logger.info('Adding autoShape... ')
        m = autoShape(self)  # 包装模型
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # 复制属性
        return m

    def info(self, verbose=False, img_size=640):
        """
        打印模型信息
        参数：
            verbose: 是否详细，默认为False
            img_size: 图像大小，默认为640
        """
        model_info(self, verbose, img_size)


def parse_model(d, ch):
    """
    解析模型
    参数：
        d: 模型字典
        ch: 输入通道数
    返回：
        模型，savelist
    """
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # 锚点数量
    no = na * (nc + 5)  # 输出数量 = 锚点 * (类别 + 5)

    layers, save, c2 = [], [], ch[-1]  # 层，savelist，输出通道数
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # 解析字符串
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # 解析字符串
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # 深度增益
        if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, DWConv, MixConv2d, Focus, CrossConv, BottleneckCSP,
                 C3, C3TR]:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # 如果不是输出
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3, C3TR]:
                args.insert(2, n)  # 重复次数
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Add:
            c2 = ch[f[0]]
            args = [c2]
        elif m is Add2:
            c2 = ch[f[0]]
            args = [c2, args[1]]
        # elif m is CMAFF:
        #     c2 = ch[f[0]]
        #     args = [c2]
        elif m is GPT:
            c2 = ch[f[0]]
            args = [c2]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        elif m is Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # 锚点数量
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # 模块
        t = str(m)[8:-2].replace('__main__.', '')  # 模块类型
        np = sum([x.numel() for x in m_.parameters()])  # 参数数量
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # 附加索引，'来自'索引，类型，参数数量
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # 打印
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # 添加到savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # 检查文件
    set_logging()
    device = select_device(opt.device)

    # 创建模型
    model = Model(opt.cfg).to(device)
    input_rgb = torch.Tensor(8, 3, 640, 640).to(device)
    output = model(input_rgb)

    # 打印模型
    # model.train()
    # torch.save(model, "yolov5s.pth")

    # 分析
    # img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 320, 320).to(device)
    # y = model(img, profile=True)

    # Tensorboard (not working https://github.com/ultralytics/yolov5/issues/2898)
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter('.')
    # logger.info("Run 'tensorboard --logdir=models' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(torch.jit.trace(model, img, strict=False), [])  # 添加模型图
    # tb_writer.add_image('test', img[0], dataformats='CWH')  # 添加模型到tensorboard
