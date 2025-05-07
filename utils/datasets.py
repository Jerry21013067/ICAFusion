# 数据集工具和数据加载器

import glob  # 用于文件路径匹配
import logging  # 用于日志记录
import math  # 数学函数
import os  # 操作系统接口
import random  # 随机数生成
import shutil  # 文件操作
import time  # 时间相关函数
from itertools import repeat  # 迭代器工具
from multiprocessing.pool import ThreadPool  # 线程池
from pathlib import Path  # 路径操作
from threading import Thread  # 线程

import cv2  # OpenCV库，用于图像处理
import numpy as np  # NumPy库，用于数值计算
import torch  # PyTorch库，用于深度学习
import torch.nn.functional as F  # PyTorch神经网络函数
from PIL import Image, ExifTags  # PIL库，用于图像处理；ExifTags用于读取图片的EXIF信息
from torch.utils.data import Dataset  # PyTorch数据集类
from tqdm import tqdm  # 进度条工具

from utils.general import check_requirements, xyxy2xywh, xywh2xyxy, xywhn2xyxy, xyn2xy, segment2box, segments2boxes, \
    resample_segments, clean_str  # 从utils.general模块导入一些通用工具函数
from utils.torch_utils import torch_distributed_zero_first  # 从utils.torch_utils模块导入分布式训练相关工具

import global_var  # 导入全局变量模块


# 参数定义
help_url = 'https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # 可接受的图片格式
vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # 可接受的视频格式
logger = logging.getLogger(__name__)  # 获取日志记录器

# FQY 构建随机采样的sampler，为了之后双模态输入
class RandomSampler(torch.utils.data.sampler.RandomSampler):
    # 随机采样器类
    def __init__(self, data_source, replacement=False, num_samples=None):
        # 初始化函数
        self.data_source = data_source  # 数据源
        self.replacement = replacement  # 是否有放回采样
        self._num_samples = num_samples  # 采样数量

        if not isinstance(self.replacement, bool):
            # 检查replacement是否为布尔值
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(self.replacement))

        if self._num_samples is not None and not replacement:
            # 如果指定了采样数量且没有放回采样，则抛出错误
            raise ValueError("With replacement=False, num_samples should not be specified, "
                             "since a random permute will be performed.")

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            # 检查num_samples是否为正整数
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))

    @property
    def num_samples(self):
        # 返回采样数量
        # 数据集大小可能在运行时改变
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        # 返回迭代器
        n = len(self.data_source)
        if self.replacement:
            # 如果有放回采样
            return iter(torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64).tolist())
        # print("-------------------------")
        s = global_var.get_value('s')
        return iter(s)

    def __len__(self):
        # 返回采样数量
        return self.num_samples


# 获取图片的EXIF方向标签
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


def get_hash(files):
    # 计算一组文件的哈希值
    # 返回文件大小的总和作为哈希值
    return sum(os.path.getsize(f) for f in files if os.path.isfile(f))


def exif_size(img):
    # 获取图片的EXIF尺寸
    s = img.size  # 图片的原始尺寸 (宽度, 高度)
    try:
        rotation = dict(img._getexif().items())[orientation]
        # 根据EXIF方向标签调整图片尺寸
        if rotation == 6:  # 旋转270度
            s = (s[1], s[0])
        elif rotation == 8:  # 旋转90度
            s = (s[1], s[0])
    except:
        pass

    return s


def create_dataloader_rgb_ir(path1, path2,  imgsz, batch_size, stride, opt, hyp=None, augment=False, cache=False, pad=0.0, rect=False,
                             rank=-1, world_size=1, workers=8, image_weights=False, quad=False, prefix='', sampler=None):
    # 创建双模态（RGB和IR）数据加载器
    # 确保在分布式训练中，只有第一个进程处理数据集，其他进程可以使用缓存
    with torch_distributed_zero_first(rank):
        dataset = LoadMultiModalImagesAndLabels(path1, path2, imgsz, batch_size,
                                                augment=augment,  # 是否增强图片
                                                hyp=hyp,  # 增强参数
                                                rect=rect,  # 是否使用矩形训练
                                                cache_images=cache,
                                                single_cls=opt.single_cls,
                                                stride=int(stride),
                                                pad=pad,
                                                image_weights=image_weights,
                                                prefix=prefix)

    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count() // world_size, batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if rank != -1 else None
    loader = torch.utils.data.DataLoader if image_weights else InfiniteDataLoader

    # global_var.set_value('s', torch.randperm(len(dataset)).tolist())
    # sampler = RandomSampler(dataset)

    # sampler = torch.utils.data.sampler.RandomSampler(dataset)

    # 根据是否使用图片权重选择数据加载器
    dataloader = loader(dataset,
                        batch_size=batch_size,
                        num_workers=nw,
                        sampler=sampler,
                        pin_memory=True,
                        collate_fn=LoadImagesAndLabels.collate_fn4 if quad else LoadImagesAndLabels.collate_fn)
    # 创建数据加载器
    return dataloader, dataset


class InfiniteDataLoader(torch.utils.data.dataloader.DataLoader):
    """ Dataloader that reuses workers
    重用线程的数据加载器
    Uses same syntax as vanilla DataLoader
    使用与普通DataLoader相同的语法
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        # 使用重复采样器
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever
    无限重复的采样器
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)
            # 无限循环采样器


class LoadImages:  # 用于推理的图片加载类
    def __init__(self, path, img_size=640, stride=32):
        p = str(Path(path).absolute())  # 获取绝对路径
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # 如果路径中包含通配符，则递归匹配文件
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # 如果是目录，则匹配目录下的所有文件
        elif os.path.isfile(p):
            files = [p]  # 如果是文件，则直接使用该文件
        else:
            raise Exception(f'ERROR: {p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in img_formats]  # 筛选出图片文件
        videos = [x for x in files if x.split('.')[-1].lower() in vid_formats]  # 筛选出视频文件
        ni, nv = len(images), len(videos)  # 图片数量和视频数量

        self.img_size = img_size  # 图片尺寸
        self.stride = stride  # 步幅
        self.files = images + videos  # 文件列表（图片和视频）
        self.nf = ni + nv  # 文件总数
        self.video_flag = [False] * ni + [True] * nv  # 标记是否为视频文件
        self.mode = 'image'  # 默认模式为图片
        if any(videos):  # 如果有视频文件
            self.new_video(videos[0])  # 打开第一个视频
        else:
            self.cap = None  # 如果没有视频文件，则释放视频捕获对象
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {img_formats}\nvideos: {vid_formats}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration  # 抛出停止迭代异常
        path = self.files[self.count]

        if self.video_flag[self.count]:
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            if not ret_val:
                self.count += 1  # 增加计数器
                self.cap.release()
                if self.count == self.nf:
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            print(f'video {self.count + 1}/{self.nf} ({self.frame}/{self.nframes}) {path}: ', end='')

        else:
            self.count += 1
            img0 = cv2.imread(path)  # BGR
            assert img0 is not None, 'Image Not Found ' + path
            print(f'image {self.count}/{self.nf} {path}: ', end='')

        img = letterbox(img0, self.img_size, stride=self.stride)[0]  # 对图片进行缩放
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)  # 确保图片是连续的数组

        return path, img, img0, self.cap

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf


class LoadWebcam:  # 用于推理的摄像头加载类
    def __init__(self, pipe='0', img_size=640, stride=32):
        self.img_size = img_size
        self.stride = stride  # 步幅

        if pipe.isnumeric():  # 如果输入是数字，则表示本地摄像头的设备号
            pipe = eval(pipe)  # 将字符串转换为整数

        self.pipe = pipe
        self.cap = cv2.VideoCapture(pipe)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # 设置缓冲区大小为3帧

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if cv2.waitKey(1) == ord('q'):  # q to quit
            self.cap.release()
            cv2.destroyAllWindows()  # 关闭所有OpenCV窗口
            raise StopIteration

        if self.pipe == 0:
            ret_val, img0 = self.cap.read()
            img0 = cv2.flip(img0, 1)  # 水平翻转（镜像）
        else:  # IP camera
            n = 0
            while True:
                n += 1
                self.cap.grab()  # 抓取一帧（不返回图像）
                if n % 30 == 0:  # 每30帧读取一次
                    ret_val, img0 = self.cap.retrieve()
                    if ret_val:
                        break

        assert ret_val, f'Camera Error {self.pipe}'
        img_path = 'webcam.jpg'
        print(f'webcam {self.count}: ', end='')

        img = letterbox(img0, self.img_size, stride=self.stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return img_path, img, img0, None

    def __len__(self):
        return 0


class LoadStreams:  # 多个IP或RTSP摄像头加载类
    def __init__(self, sources='streams.txt', img_size=640, stride=32):
        self.mode = 'stream'
        self.img_size = img_size
        self.stride = stride

        if os.path.isfile(sources):
            with open(sources, 'r') as f:
                sources = [x.strip() for x in f.read().strip().splitlines() if len(x.strip())]
        else:
            sources = [sources]

        n = len(sources)
        self.imgs = [None] * n
        self.sources = [clean_str(x) for x in sources]  # 清理摄像头源字符串
        for i, s in enumerate(sources):
            print(f'{i + 1}/{n}: {s}... ', end='')
            if 'youtube.com/' in s or 'youtu.be/' in s:
                check_requirements(('pafy', 'youtube_dl'))  # 检查依赖
                import pafy
                s = pafy.new(s).getbest(preftype="mp4").url
            s = eval(s) if s.isnumeric() else s
            cap = cv2.VideoCapture(s)
            assert cap.isOpened(), f'Failed to open {s}'
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = cap.get(cv2.CAP_PROP_FPS) % 100  # 获取帧率

            _, self.imgs[i] = cap.read()
            thread = Thread(target=self.update, args=([i, cap]), daemon=True)
            print(f' success ({w}x{h} at {self.fps:.2f} FPS).')
            thread.start()
        print('')  # newline

        s = np.stack([letterbox(x, self.img_size, stride=self.stride)[0].shape for x in self.imgs], 0)  # shapes
        self.rect = np.unique(s, axis=0).shape[0] == 1
        if not self.rect:
            print('WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.')

    def update(self, index, cap):
        n = 0
        while cap.isOpened():
            n += 1
            cap.grab()
            if n == 4:  # 每4帧读取一次
                success, im = cap.retrieve()
                self.imgs[index] = im if success else self.imgs[index] * 0
                n = 0
            time.sleep(1 / self.fps)

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        img0 = self.imgs.copy()
        if cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        img = [letterbox(x, self.img_size, auto=self.rect, stride=self.stride)[0] for x in img0]
        img = np.stack(img, 0)
        img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
        img = np.ascontiguousarray(img)

        return self.sources, img, img0, None

    def __len__(self):
        return 0


def img2label_paths(img_paths):  # 根据图片路径定义标签路径
    sb, t = 'labels', []  # /images/, /labels/
    for x in img_paths:
        if 'visible' in x.split('/'):
            sa = 'visible'
        elif 'infrared' in x.split('/'):
            sa = 'infrared'

        t.append('txt'.join(x.replace(sa, sb, 1).rsplit(x.split('.')[-1], 1)))
    return t


class LoadImagesAndLabels(Dataset):  # 用于训练/测试的图片和标签加载类
    def __init__(self, path, img_size=640, batch_size=16, augment=False, hyp=None, rect=False, image_weights=False,
                 cache_images=False, single_cls=False, stride=32, pad=0.0, prefix=''):
        self.img_size = img_size
        self.augment = augment  # 是否进行数据增强
        self.hyp = hyp  # 数据增强的超参数
        self.image_weights = image_weights  # 是否使用图片权重
        self.rect = False if image_weights else rect  # 是否使用矩形训练
        self.mosaic = self.augment and not self.rect  # 是否使用马赛克增强
        self.mosaic_border = [-img_size // 2, -img_size // 2]  # 马赛克边界的坐标
        self.stride = stride  # 步幅
        self.path = path

        try:
            f = []  # 图片文件列表
            for p in path if isinstance(path, list) else [path]:  # 如果路径是列表，则遍历每个路径
                p = Path(p)  # 转换为Path对象
                if p.is_dir():  # dir
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)  # 递归查找所有文件
                elif p.is_file():  # file
                    with open(p, 'r') as t:
                        t = t.read().strip().splitlines()  # 读取文件内容并按行分割
                        parent = str(p.parent) + os.sep
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]
                else:
                    raise Exception(f'{prefix}{p} does not exist')
            self.img_files = sorted([x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in img_formats])
            assert self.img_files, f'{prefix}No images found'
        except Exception as e:
            raise Exception(f'{prefix}Error loading data from {path}: {e}\nSee {help_url}')

        self.label_files = img2label_paths(self.img_files)  # labels
        cache_path = (p if p.is_file() else Path(self.label_files[0]).parent).with_suffix('.cache')  # cached labels
        if cache_path.is_file():
            cache, exists = torch.load(cache_path), True  # load
            if cache['hash'] != get_hash(self.label_files + self.img_files) or 'version' not in cache:  # changed
                cache, exists = self.cache_labels(cache_path, prefix), False  # re-cache
        else:
            cache, exists = self.cache_labels(cache_path, prefix), False  # cache

        # 显示缓存信息
        nf, nm, ne, nc, n = cache.pop('results')
        if exists:
            d = f"Scanning '{cache_path}' images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupted"
            tqdm(None, desc=prefix + d, total=n, initial=n)  # 显示进度条
        assert nf > 0 or not augment, f'{prefix}No labels in {cache_path}. Can not train without labels. See {help_url}'

        # 读取缓存
        cache.pop('hash')  # 删除哈希值
        cache.pop('version')  # 删除版本信息
        labels, shapes, self.segments = zip(*cache.values())
        self.labels = list(labels)  # 转换为列表
        self.shapes = np.array(shapes, dtype=np.float64)  # 转换为numpy数组
        self.img_files = list(cache.keys())  # 更新图片文件列表
        self.label_files = img2label_paths(cache.keys())  # 更新标签文件列表
        if single_cls:  # 如果是单类别
            for x in self.labels:
                x[:, 0] = 0  # 将所有类别设置为0

        n = len(shapes)  # 图片数量
        bi = np.floor(np.arange(n) / batch_size).astype(np.int)  # 计算每个图片的批次索引
        nb = bi[-1] + 1  # 批次数量
        self.batch = bi  # 批次索引
        self.n = n  # 图片总数
        self.indices = range(n)  # 图片索引

        # 矩形训练
        if self.rect:
            # 按照宽高比排序
            s = self.shapes  # 图片尺寸
            ar = s[:, 1] / s[:, 0]  # 宽高比
            irect = ar.argsort()  # 排序索引
            self.img_files = [self.img_files[i] for i in irect]  # 重新排序图片文件
            self.label_files = [self.label_files[i] for i in irect]  # 重新排序标签文件
            self.labels = [self.labels[i] for i in irect]  # 重新排序标签
            self.shapes = s[irect]  # 重新排序图片尺寸
            ar = ar[irect]  # 重新排序宽高比

            # 设置训练图片尺寸
            shapes = [[1, 1]] * nb  # 初始化批次尺寸
            for i in range(nb):  # 遍历每个批次
                ari = ar[bi == i]  # 当前批次的宽高比
                mini, maxi = ari.min(), ari.max()  # 最小和最大宽高比
                if maxi < 1:  # 如果最大宽高比小于1
                    shapes[i] = [maxi, 1]  # 设置批次尺寸
                elif mini > 1:  # 如果最小宽高比大于1
                    shapes[i] = [1, 1 / mini]  # 设置批次尺寸

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(np.int) * stride

        self.imgs = [None] * n  # 初始化图片缓存
        if cache_images:  # 如果缓存图片
            gb = 0  # 已缓存图片的大小（GB）
            self.img_hw0, self.img_hw = [None] * n, [None] * n  # 初始化图片尺寸
            results = ThreadPool(8).imap(lambda x: load_image(*x), zip(repeat(self), range(n)))  # 使用线程池加载图片
            pbar = tqdm(enumerate(results), total=n)  # 显示进度条
            for i, x in pbar:  # 遍历加载结果
                self.imgs[i], self.img_hw0[i], self.img_hw[i] = x  # 缓存图片和尺寸
                gb += self.imgs[i].nbytes  # 累加缓存大小
                pbar.desc = f'{prefix}Caching images ({gb / 1E9:.1f}GB)'  # 更新进度条描述
            pbar.close()  # 关闭进度条

    def cache_labels(self, path=Path('./labels.cache'), prefix=''):
        # 缓存数据集标签，检查图片并读取尺寸
        x = {}  # 初始化缓存字典
        nm, nf, ne, nc = 0, 0, 0, 0  # 初始化统计变量
        pbar = tqdm(zip(self.img_files, self.label_files), desc='Scanning images', total=len(self.img_files))
        for i, (im_file, lb_file) in enumerate(pbar):
            try:
                im = Image.open(im_file)  # 打开图片
                im.verify()  # 验证图片
                shape = exif_size(im)  # 获取图片尺寸
                segments = []  # 初始化分割信息
                assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
                assert im.format.lower() in img_formats, f'invalid image format {im.format}'

                # 验证标签
                if os.path.isfile(lb_file):
                    nf += 1  # 标签文件计数加1
                    with open(lb_file, 'r') as f:
                        l = [x.split() for x in f.read().strip().splitlines()]
                        if any([len(x) > 8 for x in l]):  # 如果是分割标签
                            classes = np.array([x[0] for x in l], dtype=np.float32)
                            segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in l]  # 提取分割信息
                            l = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # 转换为边界框格式
                        l = np.array(l, dtype=np.float32)
                    if len(l):
                        assert l.shape[1] == 5, 'labels require 5 columns each'
                        assert (l >= 0).all(), 'negative labels'
                        assert (l[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels'
                        assert np.unique(l, axis=0).shape[0] == l.shape[0], 'duplicate labels'
                    else:
                        ne += 1  # 空标签文件计数加1
                        l = np.zeros((0, 5), dtype=np.float32)
                else:
                    nm += 1  # 缺失标签文件计数加1
                    l = np.zeros((0, 5), dtype=np.float32)
                x[im_file] = [l, shape, segments]
            except Exception as e:
                nc += 1
                print(f'{prefix}WARNING: Ignoring corrupted image and/or label {im_file}: {e}')

            pbar.desc = f"{prefix}Scanning '{path.parent / path.stem}' images and labels... " \
                        f"{nf} found, {nm} missing, {ne} empty, {nc} corrupted"
        pbar.close()

        if nf == 0:
            print(f'{prefix}WARNING: No labels found in {path}. See {help_url}')

        x['hash'] = get_hash(self.label_files + self.img_files)
        x['results'] = nf, nm, ne, nc, i + 1
        x['version'] = 0.1  # 缓存版本
        torch.save(x, path)  # 保存缓存
        logging.info(f'{prefix}New cache created: {path}')
        return x

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):

        index = self.indices[index]  # 获取索引
        hyp = self.hyp  # 数据增强超参数
        mosaic = self.mosaic and random.random() < hyp['mosaic']  # 是否使用马赛克增强

        if mosaic:
            # 加载马赛克图片
            img, labels = load_mosaic(self, index)
            shapes = None

            if random.random() < hyp['mixup']:
                img2, labels2 = load_mosaic(self, random.randint(0, self.n - 1))
                r = np.random.beta(8.0, 8.0)  # MixUp比例
                img = (img * r + img2 * (1 - r)).astype(np.uint8)
                labels = np.concatenate((labels, labels2), 0)

        else:
            # 加载普通图片
            img, (h0, w0), (h, w) = load_image(self, index)

            # Letterbox
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # 最终的Letterbox尺寸
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # 保存原始尺寸和调整后的尺寸

            labels = self.labels[index].copy()
            if labels.size:  # 如果标签不为空
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

        if self.augment:
            # 数据增强
            if not mosaic:
                img, labels = random_perspective(img, labels,
                                                 degrees=hyp['degrees'],
                                                 translate=hyp['translate'],
                                                 scale=hyp['scale'],
                                                 shear=hyp['shear'],
                                                 perspective=hyp['perspective'])

            # 颜色空间增强
            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

        nL = len(labels)  # 标签数量
        if nL:
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])  # 转换为xywh格式
            labels[:, [2, 4]] /= img.shape[0]  # 归一化高度
            labels[:, [1, 3]] /= img.shape[1]  # 归一化宽度

        random.seed(index)
        if self.augment:
            # 随机翻转
            if random.random() < hyp['flipud']:
                img = np.flipud(img)
                if nL:
                    labels[:, 2] = 1 - labels[:, 2]

            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)
                if nL:
                    labels[:, 1] = 1 - labels[:, 1]

        labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # 转换图片格式
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img), labels_out, self.img_files[index], shapes

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes = zip(*batch)  # 解包数据
        for i, l in enumerate(label):
            l[:, 0] = i  # 添加目标图片索引
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes

    @staticmethod
    def collate_fn4(batch):
        img, label, path, shapes = zip(*batch)  # 解包数据
        n = len(shapes) // 4  # 计算批次数量
        img4, label4, path4, shapes4 = [], [], path[:n], shapes[:n]

        print("n = len(shapes) // 4", n)
        print("Image Shape", img.shape)

        ho = torch.tensor([[0., 0, 0, 1, 0, 0]])
        wo = torch.tensor([[0., 0, 1, 0, 0, 0]])
        s = torch.tensor([[1, 1, .5, .5, .5, .5]])  # 定义缩放因子
        for i in range(n):  # 遍历每个批次
            i *= 4
            if random.random() < 0.5:
                im = F.interpolate(img[i].unsqueeze(0).float(), scale_factor=2., mode='bilinear', align_corners=False)[
                    0].type(img[i].type())
                l = label[i]
            else:
                im = torch.cat((torch.cat((img[i], img[i + 1]), 1), torch.cat((img[i + 2], img[i + 3]), 1)), 2)
                l = torch.cat((label[i], label[i + 1] + ho, label[i + 2] + wo, label[i + 3] + ho + wo), 0) * s
            img4.append(im)
            label4.append(l)

        for i, l in enumerate(label4):
            l[:, 0] = i  # 添加目标图片索引

        return torch.stack(img4, 0), torch.cat(label4, 0), path4, shapes4


class LoadMultiModalImagesAndLabels(Dataset):  # 用于训练/测试的多模态（RGB和IR）图片和标签加载类
    """
    FQY  载入多模态数据 （RGB 和 IR）
    """
    def __init__(self, path_rgb, path_ir, img_size=640, batch_size=16, augment=False, hyp=None, rect=False, image_weights=False,
                 cache_images=False, single_cls=False, stride=32, pad=0.0, prefix=''):
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect  # 是否使用马赛克增强
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        self.path_rgb = path_rgb
        self.path_ir = path_ir

        try:
            f_rgb = []  # RGB图片文件列表
            f_ir = []
            # -----------------------------  rgb   -----------------------------
            for p_rgb in path_rgb if isinstance(path_rgb, list) else [path_rgb]:
                p_rgb = Path(p_rgb)  # 转换为Path对象
                if p_rgb.is_dir():  # dir
                    f_rgb += glob.glob(str(p_rgb / '**' / '*.*'), recursive=True)
                elif p_rgb.is_file():  # file
                    with open(p_rgb, 'r') as t:
                        t = t.read().strip().splitlines()
                        parent = str(p_rgb.parent) + os.sep
                        f_rgb += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # 替换路径中的相对路径
                else:
                    raise Exception(f'{prefix}{path_rgb} does not exist')

            # -----------------------------  ir   -----------------------------
            for p_ir in path_ir if isinstance(path_ir, list) else [path_ir]:
                p_ir = Path(p_ir)  # 转换为Path对象
                if p_ir.is_dir():  # dir
                    f_ir += glob.glob(str(p_ir / '**' / '*.*'), recursive=True)
                elif p_ir.is_file():  # file
                    with open(p_ir, 'r') as t:
                        t = t.read().strip().splitlines()
                        parent = str(p_ir.parent) + os.sep
                        f_ir += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # 替换路径中的相对路径
                else:
                    raise Exception(f'{prefix}{p_ir} does not exist')

            self.img_files_rgb = sorted([x.replace('/', os.sep) for x in f_rgb if x.split('.')[-1].lower() in img_formats])
            self.img_files_ir = sorted([x.replace('/', os.sep) for x in f_ir if x.split('.')[-1].lower() in img_formats])

            assert (self.img_files_rgb, self.img_files_ir), (f'{prefix}No images found', f'{prefix}No images found')
        except Exception as e:
            raise Exception(f'{prefix}Error loading data from {path_rgb,path_ir}: {e}\nSee {help_url}')

        # 检查缓存
        # 检查RGB缓存
        self.label_files_rgb = img2label_paths(self.img_files_rgb)  # 获取RGB标签文件路径
        cache_rgb_path = (p_rgb if p_rgb.is_file() else Path(self.label_files_rgb[0]).parent).with_suffix('.cache')  # 缓存文件路径
        if cache_rgb_path.is_file():
            cache_rgb, exists_rgb = torch.load(cache_rgb_path), True  # 加载缓存
            if cache_rgb['hash'] != get_hash(self.label_files_rgb + self.img_files_rgb) or 'version' not in cache_rgb:  # 如果缓存过期
                cache_rgb, exists_rgb = self.cache_labels(self.img_files_rgb,self.label_files_rgb,
                                                          cache_rgb_path, prefix), False  # 重新缓存
        else:
            cache_rgb, exists_rgb = self.cache_labels(self.img_files_rgb,self.label_files_rgb,
                                                      cache_rgb_path, prefix), False  # 创建缓存

        # 检查IR缓存
        self.label_files_ir = img2label_paths(self.img_files_ir)  # 获取IR标签文件路径
        cache_ir_path = (p_ir if p_ir.is_file() else Path(self.label_files_ir[0]).parent).with_suffix('.cache')  # 缓存文件路径
        if cache_ir_path.is_file():  # 如果缓存文件存在
            cache_ir, exists_ir = torch.load(cache_ir_path), True  # 加载缓存
            if cache_ir['hash'] != get_hash(self.label_files_ir + self.img_files_ir) or 'version' not in cache_ir:  # 如果缓存过期
                cache_ir, exists_ir = self.cache_labels(self.img_files_ir, self.label_files_ir, cache_ir_path,
                                                        prefix), False  # 重新缓存
        else:
            cache_ir, exists_ir = self.cache_labels(self.img_files_ir, self.label_files_ir, cache_ir_path,
                                                    prefix), False  # 如果缓存文件不存在，则创建缓存

        # 显示缓存信息
        nf_rgb, nm_rgb, ne_rgb, nc_rgb, n_rgb = cache_rgb.pop('results')  # 获取RGB缓存结果
        nf_ir, nm_ir, ne_ir, nc_ir, n_ir = cache_ir.pop('results')  # 获取IR缓存结果
        if exists_rgb:
            d = f"Scanning RGB '{cache_rgb_path}' images and labels... {nf_rgb} found, {nm_rgb} missing, {ne_rgb} empty, {nc_rgb} corrupted"
            tqdm(None, desc=prefix + d, total=n_rgb, initial=n_rgb)  # 显示进度条
        if exists_ir:
            d = f"Scanning IR '{cache_rgb_path}' images and labels... {nf_ir} found, {nm_ir} missing, {ne_ir} empty, {nc_ir} corrupted"
            tqdm(None, desc=prefix + d, total=n_ir, initial=n_ir)  # 显示进度条

        assert nf_rgb > 0 or not augment, f'{prefix}No labels in {cache_rgb_path}. Can not train without labels. See {help_url}'  # 确保有RGB标签文件

        # 读取缓存
        # 读取RGB缓存
        cache_rgb.pop('hash')  # 删除哈希值
        cache_rgb.pop('version')  # 删除版本信息
        labels_rgb, shapes_rgb, self.segments_rgb = zip(*cache_rgb.values())
        self.labels_rgb = list(labels_rgb)
        self.shapes_rgb = np.array(shapes_rgb, dtype=np.float64)
        self.img_files_rgb = list(cache_rgb.keys())  # 更新RGB图片文件列表
        self.label_files_rgb = img2label_paths(cache_rgb.keys())  # 更新RGB标签文件列表
        if single_cls:
            for x in self.labels_rgb:
                x[:, 0] = 0

        n_rgb = len(shapes_rgb)  # RGB图片数量
        bi_rgb = np.floor(np.arange(n_rgb) / batch_size).astype(np.int)  # 计算每个RGB图片的批次索引
        nb_rgb = bi_rgb[-1] + 1  # RGB批次数量
        self.batch_rgb = bi_rgb  # RGB批次索引
        self.n_rgb = n_rgb  # RGB图片总数
        self.indices_rgb = range(n_rgb)  # RGB图片索引

        # 读取IR缓存
        cache_ir.pop('hash')  # 删除哈希值
        cache_ir.pop('version')  # 删除版本信息
        labels_ir, shapes_ir, self.segments_ir = zip(*cache_ir.values())
        self.labels_ir = list(labels_ir)
        self.shapes_ir = np.array(shapes_ir, dtype=np.float64)
        self.img_files_ir = list(cache_ir.keys())  # 更新IR图片文件列表
        self.label_files_ir = img2label_paths(cache_ir.keys())  # 更新IR标签文件列表
        if single_cls:
            for x in self.labels_ir:
                x[:, 0] = 0

        n_ir = len(shapes_ir)  # IR图片数量
        bi_ir = np.floor(np.arange(n_ir) / batch_size).astype(np.int)  # 计算每个IR图片的批次索引
        nb_ir = bi_ir[-1] + 1  # IR批次数量
        self.batch_ir = bi_ir  # IR批次索引
        self.n_ir = n_ir
        self.indices_ir = range(n_ir)

        # 矩形训练
        if self.rect:
            # RGB
            # 按照宽高比排序
            s_rgb = self.shapes_rgb  # RGB图片尺寸
            ar_rgb = s_rgb[:, 1] / s_rgb[:, 0]  # RGB宽高比
            irect_rgb = ar_rgb.argsort()
            self.img_files_rgb = [self.img_files_rgb[i] for i in irect_rgb]
            self.label_files_rgb = [self.label_files_rgb[i] for i in irect_rgb]
            self.labels_rgb = [self.labels_rgb[i] for i in irect_rgb]
            self.shapes_rgb = s_rgb[irect_rgb]  # 重新排序RGB图片尺寸
            ar_rgb = ar_rgb[irect_rgb]

            # 设置训练RGB图片尺寸
            shapes_rgb = [[1, 1]] * nb_rgb
            for i in range(nb_rgb):
                ari_rgb = ar_rgb[bi_rgb == i]
                mini, maxi = ari_rgb.min(), ari_rgb.max()
                if maxi < 1:
                    shapes_rgb[i] = [maxi, 1]
                elif mini > 1:
                    shapes_rgb[i] = [1, 1 / mini]

            self.batch_shapes_rgb = np.ceil(np.array(shapes_rgb) * img_size / stride + pad).astype(np.int) * stride

            # IR
            # 按照宽高比排序
            s_ir = self.shapes_ir  # IR图片尺寸
            ar_ir = s_ir[:, 1] / s_ir[:, 0]  # IR宽高比
            irect_ir = ar_ir.argsort()  # 排序索引
            self.img_files_ir = [self.img_files_ir[i] for i in irect_ir]
            self.label_files_ir = [self.label_files_ir[i] for i in irect_ir]
            self.labels_ir = [self.labels_ir[i] for i in irect_ir]
            self.shapes_ir = s_ir[irect_ir]  # 重新排序IR图片尺寸
            ar_ir = ar_ir[irect_ir]

            # 设置训练IR图片尺寸
            shapes_ir = [[1, 1]] * nb_ir
            for i in range(nb_ir):
                ari_ir = ar_ir[bi_ir == i]
                mini, maxi = ari_ir.min(), ari_ir.max()
                if maxi < 1:
                    shapes_ir[i] = [maxi, 1]
                elif mini > 1:
                    shapes_ir[i] = [1, 1 / mini]

            self.batch_shapes_ir = np.ceil(np.array(shapes_ir) * img_size / stride + pad).astype(np.int) * stride

        # 缓存图片到内存以加快训练速度（警告：大数据集可能会超出系统RAM）
        self.imgs_rgb = [None] * n_rgb
        self.imgs_ir = [None] * n_ir

        self.labels = self.labels_rgb
        self.shapes = self.shapes_rgb
        self.indices = self.indices_rgb

    def cache_labels(self, imgfiles, labelfiles, path=Path('./labels.cache'), prefix=''):
        # 缓存数据集标签，检查图片并读取尺寸
        img_files = imgfiles
        label_files = labelfiles
        x = {}  # 初始化缓存字典
        nm, nf, ne, nc = 0, 0, 0, 0  # 初始化统计变量
        pbar = tqdm(zip(img_files, label_files), desc='Scanning images', total=len(img_files))
        for i, (im_file, lb_file) in enumerate(pbar):
            try:
                im = Image.open(im_file)
                im.verify()  # 验证图片
                shape = exif_size(im)  # 获取图片尺寸
                segments = []  # 初始化分割信息
                assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
                assert im.format.lower() in img_formats, f'invalid image format {im.format}'

                # 验证标签
                if os.path.isfile(lb_file):
                    nf += 1  # 标签文件计数加1
                    with open(lb_file, 'r') as f:
                        l = [x.split() for x in f.read().strip().splitlines()]
                        if any([len(x) > 8 for x in l]):  # 如果是分割标签
                            classes = np.array([x[0] for x in l], dtype=np.float32)
                            segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in l]  # 提取分割信息
                            l = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # 转换为边界框格式
                        l = np.array(l, dtype=np.float32)
                    if len(l):
                        assert l.shape[1] == 5, 'labels require 5 columns each'
                        assert (l >= 0).all(), 'negative labels'
                        assert (l[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels'
                        assert np.unique(l, axis=0).shape[0] == l.shape[0], 'duplicate labels'
                    else:
                        ne += 1  # 空标签文件计数加1
                        l = np.zeros((0, 5), dtype=np.float32)
                else:
                    nm += 1  # 缺失标签文件计数加1
                    l = np.zeros((0, 5), dtype=np.float32)
                x[im_file] = [l, shape, segments]
            except Exception as e:
                nc += 1
                print(f'{prefix}WARNING: Ignoring corrupted image and/or label {im_file}: {e}')

            pbar.desc = f"{prefix}Scanning '{path.parent / path.stem}' images and labels... " \
                        f"{nf} found, {nm} missing, {ne} empty, {nc} corrupted"
        pbar.close()

        if nf == 0:
            print(f'{prefix}WARNING: No labels found in {path}. See {help_url}')

        x['hash'] = get_hash(label_files + img_files)
        x['results'] = nf, nm, ne, nc, i + 1
        x['version'] = 0.1  # 缓存版本
        torch.save(x, path)  # 保存缓存
        logging.info(f'{prefix}New cache created: {path}')
        return x

    def __len__(self):
        return len(self.img_files_rgb)

    def __getitem__(self, index):
        # 获取单个数据项
        index_rgb = self.indices_rgb[index]  # 获取RGB索引
        index_ir = self.indices_ir[index]  # 获取IR索引

        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp['mosaic']
        if mosaic:
            # 加载马赛克图片
            img_rgb, labels_rgb, img_ir, labels_ir = load_mosaic_RGB_IR(self, index_rgb, index_ir)

            shapes = None

        else:
            # 加载普通图片
            img_rgb, img_ir, (h0, w0), (h, w) = load_image_rgb_ir(self, index)

            # Letterbox
            shape = self.batch_shapes_rgb[self.batch_rgb[index]] if self.rect else self.img_size  # 最终的Letterbox尺寸
            img_rgb, ratio, pad = letterbox(img_rgb, shape, auto=False, scaleup=self.augment)
            img_ir, ratio, pad = letterbox(img_ir, shape, auto=False, scaleup=self.augment)

            shapes = (h0, w0), ((h / h0, w / w0), pad)  # 保存原始尺寸和调整后的尺寸

            labels = self.labels_rgb[index].copy()
            if labels.size:  # 如果标签不为空
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

            labels_rgb = labels
            labels_ir = labels

        if self.augment:
            # 数据增强
            augment_hsv(img_rgb, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])
            augment_hsv(img_ir, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

        nL = len(labels_rgb)  # RGB标签数量
        if nL:
            labels_rgb[:, 1:5] = xyxy2xywh(labels_rgb[:, 1:5])  # 转换为xywh格式
            labels_rgb[:, [2, 4]] /= img_rgb.shape[0]  # 归一化高度
            labels_rgb[:, [1, 3]] /= img_rgb.shape[1]  # 归一化宽度

        if self.augment:
            # 随机翻转
            if random.random() < hyp['flipud']:
                img_rgb = np.flipud(img_rgb)
                img_ir = np.flipud(img_ir)
                if nL:
                    labels_rgb[:, 2] = 1 - labels_rgb[:, 2]

            if random.random() < hyp['fliplr']:
                img_rgb = np.fliplr(img_rgb)
                img_ir = np.fliplr(img_ir)

                if nL:
                    labels_rgb[:, 1] = 1 - labels_rgb[:, 1]

        labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels_rgb)

        img_rgb = img_rgb[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img_rgb = np.ascontiguousarray(img_rgb)
        img_ir = img_ir[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img_ir = np.ascontiguousarray(img_ir)

        img_all = np.concatenate((img_rgb, img_ir), axis=0)

        return torch.from_numpy(img_all), labels_out, self.img_files_rgb[index], shapes

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes = zip(*batch)  # 解包数据
        for i, l in enumerate(label):
            l[:, 0] = i  # 添加目标图片索引
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes

    @staticmethod
    def collate_fn4(batch):
        img, label, path, shapes = zip(*batch)  # 解包数据
        n = len(shapes) // 4
        img4, label4, path4, shapes4 = [], [], path[:n], shapes[:n]

        ho = torch.tensor([[0., 0, 0, 1, 0, 0]])
        wo = torch.tensor([[0., 0, 1, 0, 0, 0]])
        s = torch.tensor([[1, 1, .5, .5, .5, .5]])  # 定义缩放因子
        for i in range(n):  # 遍历每个批次
            i *= 4
            if random.random() < 0.5:
                im = F.interpolate(img[i].unsqueeze(0).float(), scale_factor=2., mode='bilinear', align_corners=False)[
                    0].type(img[i].type())
                l = label[i]
            else:
                im = torch.cat((torch.cat((img[i], img[i + 1]), 1), torch.cat((img[i + 2], img[i + 3]), 1)), 2)
                l = torch.cat((label[i], label[i + 1] + ho, label[i + 2] + wo, label[i + 3] + ho + wo), 0) * s
            img4.append(im)
            label4.append(l)

        for i, l in enumerate(label4):
            l[:, 0] = i  # 添加目标图片索引

        return torch.stack(img4, 0), torch.cat(label4, 0), path4, shapes4

# 辅助功能 --------------------------------------------------------------------------------------------------

def shift_augment(self, img):  # 随机平移增强
    direction = [[-1, 1], [1, 1], [1, -1], [-1, -1]]  # 平移方向：左上、右上、右下、左下

    shift_X, shift_Y = random.randint(0, 10), random.randint(0, 10)
    shift_direction = direction[random.randint(0, 3)]
    c, h, w = img.shape

    shift_img = np.full((c, h, w), 114, dtype=np.uint8)
    if shift_direction == [-1, 1]:  # 左上平移
        shift_img[:, :h - shift_Y, :w - shift_X] = img[:, shift_Y:, shift_X:]
    elif shift_direction == [1, 1]:  # 右上平移
        shift_img[:, :h - shift_Y, shift_X:] = img[:, shift_Y:, :w - shift_X]
    elif shift_direction == [1, -1]:  # 右下平移
        shift_img[:, shift_Y:, shift_X:] = img[:, :h - shift_Y, :w - shift_X]
    else:
        shift_img[:, :h - shift_Y, shift_X:] = img[:, shift_Y:, :w - shift_X]

    return shift_img

def load_image(self, index):
    # 加载单张图片
    img = self.imgs[index]
    if img is None:  # 如果图片未缓存
        path = self.img_files[index]
        img = cv2.imread(path)  # 使用OpenCV加载图片
        assert img is not None, 'Image Not Found ' + path
        h0, w0 = img.shape[:2]  # 获取原始图片的高度和宽度
        r = self.img_size / max(h0, w0)  # 计算缩放比例
        if r != 1:  # 如果需要缩放
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)),
                             interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR)
        return img, (h0, w0), img.shape[:2]  # 返回图片、原始尺寸和缩放后的尺寸
    else:
        return self.imgs[index], self.img_hw0[index], self.img_hw[index]  # 返回缓存的图片和尺寸


def load_image_rgb_ir(self, index):
    # 加载RGB和IR图片
    img_rgb = self.imgs_rgb[index]
    img_ir = self.imgs_ir[index]

    if (img_rgb is None) and (img_ir is None):  # 如果RGB和IR图片都未缓存

        path_rgb = self.img_files_rgb[index]
        path_ir = self.img_files_ir[index]

        img_rgb = cv2.imread(path_rgb)  # 使用OpenCV加载RGB图片
        img_ir = cv2.imread(path_ir)  # 使用OpenCV加载IR图片

        assert img_rgb is not None, 'Image RGB Not Found ' + path_rgb
        assert img_ir is not None, 'Image IR Not Found ' + path_ir

        h0, w0 = img_rgb.shape[:2]  # 获取原始图片的高度和宽度
        r = self.img_size / max(h0, w0)  # 计算缩放比例
        if r != 1:  # 如果需要缩放
            img_rgb = cv2.resize(img_rgb, (int(w0 * r), int(h0 * r)),
                             interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR)
            img_ir = cv2.resize(img_ir, (int(w0 * r), int(h0 * r)),
                             interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR)
        return img_rgb, img_ir, (h0, w0), img_rgb.shape[:2]  # 返回RGB和IR图片、原始尺寸和缩放后的尺寸
    else:
        return self.imgs_rgb[index], self.imgs_ir[index], self.img_hw0_rgb[index], self.img_hw_rgb[index]  # 返回缓存的RGB和IR图片和尺寸



def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # 随机生成HSV增益
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # 获取图片的数据类型

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # 将图片从HSV转换回BGR


def hist_equalize(img, clahe=True, bgr=False):
    # 直方图均衡化
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV if bgr else cv2.COLOR_RGB2YUV)
    if clahe:
        c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        yuv[:, :, 0] = c.apply(yuv[:, :, 0])
    else:
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])  # 对Y通道应用直方图均衡化
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR if bgr else cv2.COLOR_YUV2RGB)  # 将图片从YUV转换回BGR或RGB


def load_mosaic(self, index):
    # 加载4张图片的马赛克
    labels4, segments4 = [], []
    s = self.img_size
    yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # 马赛克中心坐标
    indices = [index] + random.choices(self.indices, k=3)  # 随机选择3张图片的索引
    for i, index in enumerate(indices):
        # 加载图片
        img, _, (h, w) = load_image(self, index)

        if i == 0:  # 左上
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # 创建马赛克图片
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # 计算放置位置
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # 计算裁剪位置
        elif i == 1:  # 右上
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # 左下
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # 右下
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # 将图片放置在马赛克图片中
        padw = x1a - x1b
        padh = y1a - y1b

        # 处理标签
        labels, segments = self.labels[index].copy(), self.segments[index].copy()
        if labels.size:
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # 将标签从归一化格式转换为像素格式
            segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
        labels4.append(labels)
        segments4.extend(segments)

    # 合并和裁剪标签
    labels4 = np.concatenate(labels4, 0)
    for x in (labels4[:, 1:], *segments4):
        np.clip(x, 0, 2 * s, out=x)  # 裁剪标签

    # 增强
    img4, labels4 = random_perspective(img4, labels4, segments4,
                                       degrees=self.hyp['degrees'],
                                       translate=self.hyp['translate'],
                                       scale=self.hyp['scale'],
                                       shear=self.hyp['shear'],
                                       perspective=self.hyp['perspective'],
                                       border=self.mosaic_border)
    return img4, labels4

def load_mosaic_RGB_IR(self, index1, index2):
    # 加载RGB和IR图片的4张马赛克
    index_rgb = index1
    index_ir = index2

    labels4_rgb, segments4_rgb = [], []
    labels4_ir, segments4_ir = [], []

    s = self.img_size

    yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # 马赛克中心坐标

    assert index_rgb == index_ir, 'INDEX RGB 不等于 INDEX IR'

    indices = [index_rgb] + random.choices(self.indices_rgb, k=3)  # 随机选择3张RGB图片的索引

    for i, index in enumerate(indices):
        # 加载图片
        img_rgb, img_ir, _, (h, w) = load_image_rgb_ir(self, index)

        # 将图片放置在马赛克图片中
        if i == 0:  # 左上
            img4_rgb = np.full((s * 2, s * 2, img_rgb.shape[2]), 114, dtype=np.uint8)  # 创建RGB马赛克图片
            img4_ir = np.full((s * 2, s * 2, img_ir.shape[2]), 114, dtype=np.uint8)  # 创建IR马赛克图片
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # 计算放置位置
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # 计算裁剪位置
        elif i == 1:  # 右上
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # 左下
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # 右下
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4_rgb[y1a:y2a, x1a:x2a] = img_rgb[y1b:y2b, x1b:x2b]  # 将RGB图片放置在马赛克图片中
        img4_ir[y1a:y2a, x1a:x2a] = img_ir[y1b:y2b, x1b:x2b]  # 将IR图片放置在马赛克图片中
        padw = x1a - x1b  # 计算填充宽度
        padh = y1a - y1b  # 计算填充高度

        labels_rgb, segments_rgb = self.labels_rgb[index].copy(), self.segments_rgb[index].copy()
        labels_ir, segments_ir = self.labels_ir[index].copy(), self.segments_ir[index].copy()
        if labels_rgb.size:
            labels_rgb[:, 1:] = xywhn2xyxy(labels_rgb[:, 1:], w, h, padw, padh)  # 将RGB标签从归一化格式转换为像素格式
            labels_ir[:, 1:] = xywhn2xyxy(labels_ir[:, 1:], w, h, padw, padh)  # 将IR标签从归一化格式转换为像素格式
            segments_rgb = [xyn2xy(x, w, h, padw, padh) for x in segments_rgb]
            segments_ir = [xyn2xy(x, w, h, padw, padh) for x in segments_ir]
        labels4_rgb.append(labels_rgb)
        segments4_rgb.extend(segments_rgb)
        labels4_ir.append(labels_ir)
        segments4_ir.extend(segments_ir)

    labels4_rgb = np.concatenate(labels4_rgb, 0)
    labels4_ir = np.concatenate(labels4_ir, 0)
    for x in (labels4_rgb[:, 1:], *segments4_rgb):
        np.clip(x, 0, 2 * s, out=x)  # 裁剪RGB标签
    for x in (labels4_ir[:, 1:], *segments4_ir):
        np.clip(x, 0, 2 * s, out=x)  # 裁剪IR标签

    img4_rgb, img4_ir, labels4_rgb, labels4_ir = random_perspective_rgb_ir(img4_rgb,
                                                                           img4_ir,
                                                                           labels4_rgb,
                                                                           labels4_ir,
                                                                           segments4_rgb,
                                                                           segments4_ir,
                                                                           degrees=self.hyp['degrees'],
                                                                           translate=self.hyp['translate'],
                                                                           scale=self.hyp['scale'],
                                                                           shear=self.hyp['shear'],
                                                                           perspective=self.hyp['perspective'],
                                                                           border=self.mosaic_border
                                                                           )

    labels4_ir = labels4_rgb  # 将IR标签设置为RGB标签

    return img4_rgb, labels4_rgb, img4_ir, labels4_ir



def load_mosaic9(self, index):
    # 加载9张图片的马赛克

    labels9, segments9 = [], []
    s = self.img_size
    indices = [index] + random.choices(self.indices, k=8)  # 随机选择8张额外的图片索引，加上当前索引
    for i, index in enumerate(indices):
        # 加载图片
        img, _, (h, w) = load_image(self, index)

        # 将图片放置到马赛克图片中
        if i == 0:  # 中心图片
            img9 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)  # 创建一个3s x 3s的马赛克图片，填充值为114
            h0, w0 = h, w
            c = s, s, s + w, s + h  # 计算当前图片在马赛克中的位置
        elif i == 1:  # 上部图片
            c = s, s - h, s + w, s
        elif i == 2:  # 右上图片
            c = s + wp, s - h, s + wp + w, s
        elif i == 3:  # 右侧图片
            c = s + w0, s, s + w0 + w, s + h
        elif i == 4:  # 右下图片
            c = s + w0, s + hp, s + w0 + w, s + hp + h
        elif i == 5:  # 下部图片
            c = s + w0 - w, s + h0, s + w0, s + h0 + h
        elif i == 6:  # 左下图片
            c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
        elif i == 7:  # 左侧图片
            c = s - w, s + h0 - h, s, s + h0
        elif i == 8:  # 左上图片
            c = s - w, s + h0 - hp - h, s, s + h0 - hp

        padx, pady = c[:2]
        x1, y1, x2, y2 = [max(x, 0) for x in c]  # 确保坐标不小于0

        # 处理标签
        labels, segments = self.labels[index].copy(), self.segments[index].copy()
        if labels.size:
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padx, pady)  # 将标签从归一化格式转换为像素格式
            segments = [xyn2xy(x, w, h, padx, pady) for x in segments]
        labels9.append(labels)
        segments9.extend(segments)

        img9[y1:y2, x1:x2] = img[y1 - pady:, x1 - padx:]  # 将当前图片放置到马赛克图片的指定位置
        hp, wp = h, w  # 保存当前图片的高和宽，用于后续图片的放置

    # 偏移马赛克图片
    yc, xc = [int(random.uniform(0, s)) for _ in self.mosaic_border]  # 随机生成马赛克中心的x和y偏移量
    img9 = img9[yc:yc + 2 * s, xc:xc + 2 * s]

    # 合并和裁剪标签
    labels9 = np.concatenate(labels9, 0)
    labels9[:, [1, 3]] -= xc
    labels9[:, [2, 4]] -= yc
    c = np.array([xc, yc])  # 马赛克中心点
    segments9 = [x - c for x in segments9]

    for x in (labels9[:, 1:], *segments9):
        np.clip(x, 0, 2 * s, out=x)  # 裁剪标签和分割信息，确保它们在马赛克图片的范围内

    # 数据增强
    img9, labels9 = random_perspective(img9, labels9, segments9,
                                       degrees=self.hyp['degrees'],
                                       translate=self.hyp['translate'],
                                       scale=self.hyp['scale'],
                                       shear=self.hyp['shear'],
                                       perspective=self.hyp['perspective'],
                                       border=self.mosaic_border)  # 边界裁剪

    return img9, labels9


def replicate(img, labels):
    # 复制标签
    h, w = img.shape[:2]
    boxes = labels[:, 1:].astype(int)
    x1, y1, x2, y2 = boxes.T
    s = ((x2 - x1) + (y2 - y1)) / 2  # 边长 (像素)
    for i in s.argsort()[:round(s.size * 0.5)]:  # 最小的索引
        x1b, y1b, x2b, y2b = boxes[i]
        bh, bw = y2b - y1b, x2b - x1b
        yc, xc = int(random.uniform(0, h - bh)), int(random.uniform(0, w - bw))  # 偏移 x, y
        x1a, y1a, x2a, y2a = [xc, yc, xc + bw, yc + bh]
        img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        labels = np.append(labels, [[labels[i, 0], x1a, y1a, x2a, y2a]], axis=0)

    return img, labels


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # 调整图片大小并填充以满足步幅约束
    shape = img.shape[:2]  # 当前形状 [高度, 宽度]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # 缩放比例 (新 / 旧)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # 只缩小，不放大（为了更好的测试 mAP）
        r = min(r, 1.0)

    # 计算填充
    ratio = r, r  # 宽度、高度比例
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh 填充

    dw /= 2  # 将填充分成两边
    dh /= 2

    if shape[::-1] != new_unpad:  # 如果需要调整大小
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # 添加边界

    return img, ratio, (dw, dh)


def random_perspective(img, targets=(), segments=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0,
                       border=(0, 0)):
    # 随机透视变换
    height = img.shape[0] + border[0] * 2  # 图片高度
    width = img.shape[1] + border[1] * 2

    # 中心点
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x 偏移 (像素)
    C[1, 2] = -img.shape[0] / 2  # y 偏移 (像素)

    # 透视
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x 透视 (关于 y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y 透视 (关于 x)

    # 旋转和缩放
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    s = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # 剪切
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # 随机生成x方向的剪切角度并转换为弧度
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # 随机生成y方向的剪切角度并转换为弧度

    # 平移
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # 随机生成x方向的平移量
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # 随机生成y方向的平移量

    # 组合变换矩阵
    M = T @ S @ R @ P @ C  # 按照从右到左的顺序组合所有变换矩阵
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # 如果有边界或者图像发生了变换
        if perspective:  # 如果应用透视变换
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))  # 对图像应用透视变换
        else:  # 如果只应用仿射变换
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))  # 对图像应用仿射变换

    # 可视化变换前后的图像（调试用）
    # import matplotlib.pyplot as plt
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(img[:, :, ::-1])  # 显示原始图像
    # ax[1].imshow(img2[:, :, ::-1])  # 显示变换后的图像

    # 转换标签坐标
    n = len(targets)
    if n:
        use_segments = any(x.any() for x in segments)
        new = np.zeros((n, 4))
        if use_segments:  # 如果使用分割信息
            segments = resample_segments(segments)  # 对分割信息进行重采样
            for i, segment in enumerate(segments):
                xy = np.ones((len(segment), 3))
                xy[:, :2] = segment
                xy = xy @ M.T  # 对分割信息应用变换矩阵
                xy = xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]  # 如果应用透视变换则进行归一化

                # 裁剪分割信息
                new[i] = segment2box(xy, width, height)

        else:  # 如果不使用分割信息
            xy = np.ones((n * 4, 3))  # 创建一个全1的数组，用于添加齐次坐标
            xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # 将目标框坐标赋值给xy
            xy = xy @ M.T  # 对目标框坐标应用变换矩阵
            xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # 如果应用透视变换则进行归一化

            # 创建新的目标框
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # 裁剪目标框
            new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
            new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        # 过滤候选目标框
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=new.T, area_thr=0.01 if use_segments else 0.10)
        targets = targets[i]
        targets[:, 1:5] = new[i]

    return img, targets

def random_perspective_rgb_ir(img_rgb, img_ir, targets_rgb=(),targets_ir=(), segments_rgb=(), segments_ir=(),
                              degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0, border=(0, 0)):
    # 随机透视变换函数，用于对RGB和IR图像进行随机的旋转、缩放、剪切、平移和透视变换
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]

    img = img_rgb
    targets = targets_rgb
    segments = segments_rgb

    height = img.shape[0] + border[0] * 2  # 计算变换后的图像高度，考虑边界
    width = img.shape[1] + border[1] * 2

    # 中心点平移矩阵
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # 计算x方向的平移量，使图像中心移动到原点
    C[1, 2] = -img.shape[0] / 2  # 计算y方向的平移量，使图像中心移动到原点

    # 透视变换矩阵
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # 随机生成x方向的透视变换参数
    P[2, 1] = random.uniform(-perspective, perspective)  # 随机生成y方向的透视变换参数

    # 旋转和缩放矩阵
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    s = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # 剪切矩阵
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # 随机生成x方向的剪切角度并转换为弧度
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # 随机生成y方向的剪切角度并转换为弧度

    # 平移矩阵
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # 随机生成x方向的平移量
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # 随机生成y方向的平移量

    # 组合变换矩阵
    M = T @ S @ R @ P @ C  # 按照从右到左的顺序组合所有变换矩阵
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # 如果有边界或者图像发生了变换
        if perspective:  # 如果应用透视变换
            # 对RGB和IR图像分别应用透视变换
            img_rgb = cv2.warpPerspective(img_rgb, M, dsize=(width, height), borderValue=(114, 114, 114))
            img_ir = cv2.warpPerspective(img_ir, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # 如果只应用仿射变换
            # 对RGB和IR图像分别应用仿射变换
            img_rgb = cv2.warpAffine(img_rgb, M[:2], dsize=(width, height), borderValue=(114, 114, 114))
            img_ir = cv2.warpAffine(img_ir, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # 可视化变换前后的图像（调试用）
    # import matplotlib.pyplot as plt
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(img_rgb[:, :, ::-1])  # 显示原始RGB图像
    # ax[1].imshow(img_ir[:, :, ::-1])  # 显示变换后的IR图像


    # 转换标签坐标
    n = len(targets)
    if n:
        use_segments = any(x.any() for x in segments)
        new = np.zeros((n, 4))
        if use_segments:  # 如果使用分割信息
            segments = resample_segments(segments)  # 对分割信息进行重采样
            for i, segment in enumerate(segments):
                xy = np.ones((len(segment), 3))
                xy[:, :2] = segment
                xy = xy @ M.T  # 对分割信息应用变换矩阵
                xy = xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]  # 如果应用透视变换则进行归一化

                # 裁剪分割信息
                new[i] = segment2box(xy, width, height)

        else:  # 如果不使用分割信息
            xy = np.ones((n * 4, 3))
            xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # 将目标框坐标赋值给xy
            xy = xy @ M.T  # 对目标框坐标应用变换矩阵
            xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # 如果应用透视变换则进行归一化

            # 创建新的目标框
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # 裁剪目标框
            new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
            new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        # 过滤候选目标框
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=new.T, area_thr=0.01 if use_segments else 0.10)
        targets = targets[i]
        targets[:, 1:5] = new[i]

    return img_rgb, img_ir, targets, targets


def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1, eps=1e-16):  # box1(4,n), box2(4,n)
    # 计算候选框：box1为增强前的框，box2为增强后的框，wh_thr为宽高阈值（像素），ar_thr为宽高比阈值，area_thr为面积比阈值
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # 计算宽高比，避免除以0
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # 候选框


def cutout(image, labels):
    # 应用图像cutout增强（随机遮挡图像的一部分） https://arxiv.org/abs/1708.04552
    h, w = image.shape[:2]

    def bbox_ioa(box1, box2):
        # 计算box1和box2的交并比（Intersection over Area），box1为4个坐标，box2为nx4的坐标，坐标格式为x1y1x2y2
        box2 = box2.transpose()

        # 获取边界框的坐标
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

        # 计算交集面积
        inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                     (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

        # 计算box2的面积
        box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + 1e-16

        # 计算交并比
        return inter_area / box2_area

    # 创建随机遮挡区域
    scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16  # 图像尺寸的比例
    for s in scales:
        mask_h = random.randint(1, int(h * s))
        mask_w = random.randint(1, int(w * s))

        # 定义遮挡区域
        xmin = max(0, random.randint(0, w) - mask_w // 2)
        ymin = max(0, random.randint(0, h) - mask_h // 2)
        xmax = min(w, xmin + mask_w)
        ymax = min(h, ymin + mask_h)

        # 应用随机颜色遮挡
        image[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(3)]

        # 返回未被遮挡的标签
        if len(labels) and s > 0.03:
            box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
            ioa = bbox_ioa(box, labels[:, 1:5])  # 计算遮挡区域与标签的交并比
            labels = labels[ioa < 0.60]  # 移除被遮挡超过60%的标签

    return labels


def create_folder(path='./new'):
    # 创建文件夹
    if os.path.exists(path):
        shutil.rmtree(path)  # 删除输出文件夹
    os.makedirs(path)  # 创建新的输出文件夹


def flatten_recursive(path='../coco128'):
    # 将递归目录中的所有文件平铺到顶层目录
    new_path = Path(path + '_flat')
    create_folder(new_path)
    for file in tqdm(glob.glob(str(Path(path)) + '/**/*.*', recursive=True)):
        shutil.copyfile(file, new_path / Path(file).name)


def extract_boxes(path='../coco128/'):  # from utils.datasets import *; extract_boxes('../coco128')
    # 将检测数据集转换为分类数据集，每个类别一个目录

    path = Path(path)  # 图片目录路径
    shutil.rmtree(path / 'classifier') if (path / 'classifier').is_dir() else None  # 如果存在，则删除已有的classifier目录
    files = list(path.rglob('*.*'))
    n = len(files)  # 文件数量
    for im_file in tqdm(files, total=n):
        if im_file.suffix[1:] in img_formats:
            # 图片
            im = cv2.imread(str(im_file))[..., ::-1]  # BGR to RGB
            h, w = im.shape[:2]

            # 标签
            lb_file = Path(img2label_paths([str(im_file)])[0])
            if Path(lb_file).exists():
                with open(lb_file, 'r') as f:
                    lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # 读取标签数据

                for j, x in enumerate(lb):
                    c = int(x[0])  # class
                    f = (path / 'classifier') / f'{c}' / f'{path.stem}_{im_file.stem}_{j}.jpg'  # 新的文件名
                    if not f.parent.is_dir():
                        f.parent.mkdir(parents=True)

                    b = x[1:] * [w, h, w, h]  # 将标签从归一化坐标转换为像素坐标
                    b[2:] = b[2:] * 1.2 + 3  # 增加边界框的大小
                    b = xywh2xyxy(b.reshape(-1, 4)).ravel().astype(np.int)

                    b[[0, 2]] = np.clip(b[[0, 2]], 0, w)  # 裁剪边界框的x坐标，防止超出图片范围
                    b[[1, 3]] = np.clip(b[[1, 3]], 0, h)
                    assert cv2.imwrite(str(f), im[b[1]:b[3], b[0]:b[2]]), f'box failure in {f}'


def autosplit(path='../coco128', weights=(0.9, 0.1, 0.0), annotated_only=False):
    """ 自动将数据集划分为训练集、验证集和测试集，并保存为path/autosplit_*.txt文件
    使用方法：from utils.datasets import *; autosplit('../coco128')
    参数：
        path:           图片目录路径
        weights:        训练集、验证集、测试集的权重（列表）
        annotated_only: 是否只使用有标注的图片
    """
    path = Path(path)  # 图片目录路径
    files = sum([list(path.rglob(f"*.{img_ext}")) for img_ext in img_formats], [])  # 获取目录下所有图片文件
    n = len(files)  # 图片数量
    indices = random.choices([0, 1, 2], weights=weights, k=n)  # 根据权重随机分配每个图片到训练集、验证集或测试集

    txt = ['autosplit_train.txt', 'autosplit_val.txt', 'autosplit_test.txt']  # 3个txt文件名
    [(path / x).unlink() for x in txt if (path / x).exists()]  # 如果文件存在，则删除

    print(f'Autosplitting images from {path}' + ', using *.txt labeled images only' * annotated_only)
    for i, img in tqdm(zip(indices, files), total=n):  # 遍历所有图片
        if not annotated_only or Path(img2label_paths([str(img)])[0]).exists():  # 如果不只使用有标注的图片，或者图片有对应的标注文件
            with open(path / txt[i], 'a') as f:  # 打开对应的txt文件
                f.write(str(img) + '\n')  # 将图片路径写入txt文件
