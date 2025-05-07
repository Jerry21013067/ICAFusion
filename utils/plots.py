# Plotting utils

import glob  # 文件路径匹配
import math  # 数学运算
import os  # 操作系统接口
import random  # 随机数生成
from copy import copy  # 深拷贝
from pathlib import Path  # 路径操作

import cv2  # OpenCV库
import matplotlib
import matplotlib.pyplot as plt  # 绘图库
import numpy as np  # 数值计算库
import pandas as pd  # 数据分析库
import seaborn as sns  # 数据可视化库
import torch  # PyTorch深度学习框架
import yaml  # YAML格式解析
from PIL import Image, ImageDraw, ImageFont  # 图像处理库
from scipy.signal import butter, filtfilt  # 信号处理库

from utils.general import xywh2xyxy, xyxy2xywh, xywhn2xyxy  # 坐标转换工具
from utils.metrics import fitness  # 模型性能评估工具

# 设置Matplotlib字体大小和后端
matplotlib.rc('font', **{'size': 11})
matplotlib.use('Agg')  # 仅用于写入文件


class Colors:
    # Ultralytics颜色调色板 https://ultralytics.com/
    def __init__(self):
        self.palette = [self.hex2rgb(c) for c in matplotlib.colors.TABLEAU_COLORS.values()]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # 将十六进制颜色转换为RGB格式
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # 创建实例


def hist2d(x, y, n=100):
    # 生成二维直方图，用于绘制标签分布图
    xedges, yedges = np.linspace(x.min(), x.max(), n), np.linspace(y.min(), y.max(), n)
    hist, xedges, yedges = np.histogram2d(x, y, (xedges, yedges))
    xidx = np.clip(np.digitize(x, xedges) - 1, 0, hist.shape[0] - 1)
    yidx = np.clip(np.digitize(y, yedges) - 1, 0, hist.shape[1] - 1)
    return np.log(hist[xidx, yidx])


def butter_lowpass_filtfilt(data, cutoff=1500, fs=50000, order=5):
    # 使用Butterworth低通滤波器对数据进行滤波 https://stackoverflow.com/questions/28536191/how-to-filter-smooth-with-scipy-numpy
    def butter_lowpass(cutoff, fs, order):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        return butter(order, normal_cutoff, btype='low', analog=False)

    b, a = butter_lowpass(cutoff, fs, order=order)
    return filtfilt(b, a, data)  # 双向滤波


def plot_one_box(x, im, color=None, label=None, line_thickness=3):
    # 在图像上绘制一个边界框
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # 线宽
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # 字体厚度
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # 填充
        cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def plot_one_box_PIL(box, im, color=None, label=None, line_thickness=None):
    # 使用PIL在图像上绘制一个边界框
    im = Image.fromarray(im)
    draw = ImageDraw.Draw(im)
    line_thickness = line_thickness or max(int(min(im.size) / 200), 2)
    draw.rectangle(box, width=line_thickness, outline=tuple(color))
    if label:
        fontsize = max(round(max(im.size) / 40), 12)
        font = ImageFont.truetype("Arial.ttf", fontsize)
        txt_width, txt_height = font.getsize(label)
        draw.rectangle([box[0], box[1] - txt_height + 4, box[0] + txt_width, box[1]], fill=tuple(color))
        draw.text((box[0], box[1] - txt_height + 1), label, fill=(255, 255, 255), font=font)
    return np.asarray(im)


def plot_wh_methods():  # 比较YOLOv3和YOLOv5的宽高乘法方法
    x = np.arange(-4.0, 4.0, .1)
    ya = np.exp(x)
    yb = torch.sigmoid(torch.from_numpy(x)).numpy() * 2

    fig = plt.figure(figsize=(6, 3), tight_layout=True)
    plt.plot(x, ya, '.-', label='YOLOv3')
    plt.plot(x, yb ** 2, '.-', label='YOLOv5 ^2')
    plt.plot(x, yb ** 1.6, '.-', label='YOLOv5 ^1.6')
    plt.xlim(left=-4, right=4)
    plt.ylim(bottom=0, top=6)
    plt.xlabel('input')
    plt.ylabel('output')
    plt.grid()
    plt.legend()
    fig.savefig('comparison.png', dpi=200)


def output_to_target(output):
    # 将模型输出转换为目标格式
    targets = []
    for i, o in enumerate(output):
        for *box, conf, cls in o.cpu().numpy():
            targets.append([i, cls, *list(*xyxy2xywh(np.array(box)[None])), conf])
    return np.array(targets)


def plot_samples(batch_index, images, path, tcls, tbox, indices, anchors, offsets, targets):
    # 绘制样本图像
    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()
    if isinstance(tbox, torch.Tensor):
        tbox = tbox.cpu().numpy()

    bs, _, h, w = images.shape
    slide = [8, 16, 32]

    if np.max(images[0]) <= 1:
        images *= 255

    file_path = str(path) + '/samples_visual'
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    for j in range(bs):  # 遍历每张图像
        img = images[j]
        img = img.transpose(1, 2, 0)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        for target in targets:
            if target[0] == j:
                t = target[2:6].unsqueeze(0)
                t = xywhn2xyxy(t, w=640, h=640)
                cv2.rectangle(img, (int(t[0][0]), int(t[0][1])), (int(t[0][2]), int(t[0][3])), (255, 255, 255), thickness=2)
            else:
                continue

        for i in range(len(indices)):  # 遍历每一层
            indice = indices[i]
            image_indice = (indice[0]==j).nonzero()
            anchor = anchors[i][image_indice].squeeze(1)
            offset = offsets[i][image_indice].squeeze(1)
            gridy, gridx = indice[2][image_indice], indice[3][image_indice]
            grid = torch.cat([gridx, gridy], dim=1) + offset
            anchor_boxs = torch.cat([grid, anchor], dim=1) * slide[i]
            for anchor_box in anchor_boxs:
                bbox = xywh2xyxy(anchor_box.unsqueeze(0))
                cv2.rectangle(img, (int(bbox[0][0]), int(bbox[0][1])), (int(bbox[0][2]), int(bbox[0][3])), (255, 0, 0), thickness=1)

        save_path = file_path+'/'+'image'+str(batch_index)+'_'+str(j)+'.jpg'
        cv2.imwrite(save_path, img)


def plot_images(images, targets, paths=None, fname='images.jpg', names=None, max_size=640, max_subplots=16):
    # 绘制图像网格并标注边界框

    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    if np.max(images[0]) <= 1:
        images *= 255

    tl = 3  # 线宽
    tf = max(tl - 1, 1)  # 字体厚度
    bs, _, h, w = images.shape
    bs = min(bs, max_subplots)
    ns = np.ceil(bs ** 0.5)

    scale_factor = max_size / max(h, w)
    if scale_factor < 1:
        h = math.ceil(scale_factor * h)
        w = math.ceil(scale_factor * w)

    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)
    for i, img in enumerate(images):
        if i == max_subplots:
            break

        block_x = int(w * (i // ns))
        block_y = int(h * (i % ns))

        img = img.transpose(1, 2, 0)
        if scale_factor < 1:
            img = cv2.resize(img, (w, h))

        mosaic[block_y:block_y + h, block_x:block_x + w, :] = img
        if len(targets) > 0:
            image_targets = targets[targets[:, 0] == i]
            boxes = xywh2xyxy(image_targets[:, 2:6]).T
            classes = image_targets[:, 1].astype('int')
            labels = image_targets.shape[1] == 6
            conf = None if labels else image_targets[:, 6]

            if boxes.shape[1]:
                if boxes.max() <= 1.01:
                    boxes[[0, 2]] *= w
                    boxes[[1, 3]] *= h
                elif scale_factor < 1:
                    boxes *= scale_factor
            boxes[[0, 2]] += block_x
            boxes[[1, 3]] += block_y
            for j, box in enumerate(boxes.T):
                cls = int(classes[j])
                color = colors(cls)
                cls = names[cls] if names else cls
                if labels or conf[j] > 0.25:
                    label = '%s' % cls if labels else '%s %.1f' % (cls, conf[j])
                    plot_one_box(box, mosaic, label=label, color=color, line_thickness=tl)

        if paths:
            label = Path(paths[i]).name[:40]
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            cv2.putText(mosaic, label, (block_x + 5, block_y + t_size[1] + 5), 0, tl / 3, [220, 220, 220], thickness=tf, lineType=cv2.LINE_AA)

        cv2.rectangle(mosaic, (block_x, block_y), (block_x + w, block_y + h), (255, 255, 255), thickness=3)

    if fname:
        r = min(1280. / max(h, w) / ns, 1.0)
        mosaic = cv2.resize(mosaic, (int(ns * w * r), int(ns * h * r)), interpolation=cv2.INTER_AREA)
        Image.fromarray(mosaic).save(fname)

    return mosaic



def plot_lr_scheduler(optimizer, scheduler, epochs=300, save_dir=''):
    # 绘制学习率调度器的曲线
    optimizer, scheduler = copy(optimizer), copy(scheduler)
    y = []
    for _ in range(epochs):
        scheduler.step()
        y.append(optimizer.param_groups[0]['lr'])
    plt.plot(y, '.-', label='LR')
    plt.xlabel('epoch')
    plt.ylabel('LR')
    plt.grid()
    plt.xlim(0, epochs)
    plt.ylim(0)
    plt.savefig(Path(save_dir) / 'LR.png', dpi=200)
    plt.close()


def plot_test_txt():  # 绘制test.txt的直方图
    x = np.loadtxt('test.txt', dtype=np.float32)
    box = xyxy2xywh(x[:, :4])
    cx, cy = box[:, 0], box[:, 1]

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), tight_layout=True)
    ax.hist2d(cx, cy, bins=600, cmax=10, cmin=0)
    ax.set_aspect('equal')
    plt.savefig('hist2d.png', dpi=300)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True)
    ax[0].hist(cx, bins=600)
    ax[1].hist(cy, bins=600)
    plt.savefig('hist1d.png', dpi=200)


def plot_targets_txt():  # 绘制targets.txt的直方图
    x = np.loadtxt('targets.txt', dtype=np.float32).T
    s = ['x targets', 'y targets', 'width targets', 'height targets']
    fig, ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)
    ax = ax.ravel()
    for i in range(4):
        ax[i].hist(x[i], bins=100, label='%.3g +/- %.3g' % (x[i].mean(), x[i].std()))
        ax[i].legend()
        ax[i].set_title(s[i])
    plt.savefig('targets.jpg', dpi=200)


def plot_study_txt(path='', x=None):  # 绘制study.txt的结果
    fig, ax = plt.subplots(2, 4, figsize=(10, 6), tight_layout=True)
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 4), tight_layout=True)
    for f in sorted(Path(path).glob('study*.txt')):
        y = np.loadtxt(f, dtype=np.float32, usecols=[0, 1, 2, 3, 7, 8, 9], ndmin=2).T
        x = np.arange(y.shape[1]) if x is None else np.array(x)
        j = y[3].argmax() + 1
        ax2.plot(y[6, 1:j], y[3, 1:j] * 1E2, '.-', linewidth=2, markersize=8, label=f.stem.replace('study_coco_', '').replace('yolo', 'YOLO'))

    ax2.plot(1E3 / np.array([209, 140, 97, 58, 35, 18]), [34.6, 40.5, 43.0, 47.5, 49.7, 51.5], 'k.-', linewidth=2, markersize=8, alpha=.25, label='EfficientDet')
    ax2.grid(alpha=0.2)
    ax2.set_yticks(np.arange(20, 60, 5))
    ax2.set_xlim(0, 57)
    ax2.set_ylim(30, 55)
    ax2.set_xlabel('GPU Speed (ms/img)')
    ax2.set_ylabel('COCO AP val')
    ax2.legend(loc='lower right')
    plt.savefig(str(Path(path).name) + '.png', dpi=300)


def plot_labels(labels, names=(), save_dir=Path(''), loggers=None):
    # 绘制数据集标签分布
    print('Plotting labels... ')
    c, b = labels[:, 0], labels[:, 1:].transpose()
    nc = int(c.max() + 1)
    x = pd.DataFrame(b.transpose(), columns=['x', 'y', 'width', 'height'])

    sns.pairplot(x, corner=True, diag_kind='auto', kind='hist', diag_kws=dict(bins=50), plot_kws=dict(pmax=0.9))
    plt.savefig(save_dir / 'labels_correlogram.jpg', dpi=200)
    plt.close()

    matplotlib.use('svg')
    ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)[1].ravel()
    ax[0].hist(c, bins=np.linspace(0, nc, nc + 1) - 0.5, rwidth=0.8)
    ax[0].set_ylabel('instances')
    if 0 < len(names) < 30:
        ax[0].set_xticks(range(len(names)))
        ax[0].set_xticklabels(names, rotation=90, fontsize=10)
    else:
        ax[0].set_xlabel('classes')
    sns.histplot(x, x='x', y='y', ax=ax[2], bins=50, pmax=0.9)
    sns.histplot(x, x='width', y='height', ax=ax[3], bins=50, pmax=0.9)

    labels[:, 1:3] = 0.5
    labels[:, 1:] = xywh2xyxy(labels[:, 1:]) * 2000
    img = Image.fromarray(np.ones((2000, 2000, 3), dtype=np.uint8) * 255)
    for cls, *box in labels[:1000]:
        ImageDraw.Draw(img).rectangle(box, width=1, outline=colors(cls))
    ax[1].imshow(img)
    ax[1].axis('off')

    for a in [0, 1, 2, 3]:
        for s in ['top', 'right', 'left', 'bottom']:
            ax[a].spines[s].set_visible(False)

    plt.savefig(save_dir / 'labels.jpg', dpi=200)
    matplotlib.use('Agg')
    plt.close()

    for k, v in loggers.items() or {}:
        if k == 'wandb' and v:
            v.log({"Labels": [v.Image(str(x), caption=x.name) for x in save_dir.glob('*labels*.jpg')]}, commit=False)


def plot_evolution(yaml_file='data/hyp.finetune.yaml'):
    # 绘制超参数进化结果
    with open(yaml_file) as f:
        hyp = yaml.safe_load(f)
    x = np.loadtxt('evolve.txt', ndmin=2)
    f = fitness(x)
    plt.figure(figsize=(10, 12), tight_layout=True)
    matplotlib.rc('font', **{'size': 8})
    for i, (k, v) in enumerate(hyp.items()):
        y = x[:, i + 7]
        mu = y[f.argmax()]
        plt.subplot(6, 5, i + 1)
        plt.scatter(y, f, c=hist2d(y, f, 20), cmap='viridis', alpha=.8, edgecolors='none')
        plt.plot(mu, f.max(), 'k+', markersize=15)
        plt.title('%s = %.3g' % (k, mu), fontdict={'size': 9})
        if i % 5 != 0:
            plt.yticks([])
        print('%15s: %.3g' % (k, mu))
    plt.savefig('evolve.png', dpi=200)
    print('\nPlot saved as evolve.png')


def profile_idetection(start=0, stop=0, labels=(), save_dir=''):
    # 绘制iDetection的日志文件
    ax = plt.subplots(2, 4, figsize=(12, 6), tight_layout=True)[1].ravel()
    s = ['Images', 'Free Storage (GB)', 'RAM Usage (GB)', 'Battery', 'dt_raw (ms)', 'dt_smooth (ms)', 'real-world FPS']
    files = list(Path(save_dir).glob('frames*.txt'))
    for fi, f in enumerate(files):
        try:
            results = np.loadtxt(f, ndmin=2).T[:, 90:-30]
            n = results.shape[1]
            x = np.arange(start, min(stop, n) if stop else n)
            results = results[:, x]
            t = (results[0] - results[0].min())
            results[0] = x
            for i, a in enumerate(ax):
                if i < len(results):
                    label = labels[fi] if len(labels) else f.stem.replace('frames_', '')
                    a.plot(t, results[i], marker='.', label=label, linewidth=1, markersize=5)
                    for side in ['top', 'right']:
                        a.spines[side].set_visible(False)
                else:
                    a.remove()
        except Exception as e:
            print('Warning: Plotting error for %s; %s' % (f, e))
    ax[1].legend()
    plt.savefig(Path(save_dir) / 'idetection_profile.png', dpi=200)


def plot_results_overlay(start=0, stop=0):
    # 绘制训练结果，叠加训练和验证损失
    s = ['train', 'train', 'train', 'Precision', 'mAP@0.5', 'val', 'val', 'val', 'Recall', 'mAP@0.5:0.95']
    t = ['Box', 'Objectness', 'Classification', 'P-R', 'mAP-F1']
    for f in sorted(glob.glob('results*.txt') + glob.glob('../../Downloads/results*.txt')):
        results = np.loadtxt(f, usecols=[2, 3, 4, 8, 9, 12, 13, 14, 10, 11], ndmin=2).T
        n = results.shape[1]
        x = range(start, min(stop, n) if stop else n)
        fig, ax = plt.subplots(1, 5, figsize=(14, 3.5), tight_layout=True)
        ax = ax.ravel()
        for i in range(5):
            for j in [i, i + 5]:
                y = results[j, x]
                ax[i].plot(x, y, marker='.', label=s[j])
            ax[i].set_title(t[i])
            ax[i].legend()
            ax[i].set_ylabel(f) if i == 0 else None
        fig.savefig(f.replace('.txt', '.png'), dpi=200)


def plot_results(file='path/to/results.csv', dir=''):
    # 绘制训练结果
    save_dir = Path(file).parent if file else Path(dir)
    fig, ax = plt.subplots(4, 5, figsize=(12, 6), tight_layout=True)
    ax = ax.ravel()
    files = list(save_dir.glob('results*.csv'))
    assert len(files), f'No results.csv files found in {save_dir.resolve()}, nothing to plot.'
    for fi, f in enumerate(files):
        try:
            data = pd.read_csv(f)
            s = [x.strip() for x in data.columns]
            x = data.values[:, 0]
            for i, j in enumerate([1, 2, 3, 4, 5, 8, 9, 10, 6, 7, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]):
                y = data.values[:, j]
                ax[i].plot(x, y, marker='.', label=f.stem, linewidth=2, markersize=8)
                ax[i].set_title(s[j], fontsize=12)
        except Exception as e:
            print(f'Warning: Plotting error for {f}: {e}')
    ax[1].legend()
    fig.savefig(save_dir / 'results.png', dpi=200)
    plt.close()
