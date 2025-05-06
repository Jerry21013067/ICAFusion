# Auto-anchor utils

import numpy as np
import torch
import yaml
from scipy.cluster.vq import kmeans
from tqdm import tqdm

from utils.general import colorstr  # 导入颜色字符串工具，用于打印带颜色的日志信息

# 检查锚点顺序是否与步长顺序一致
def check_anchor_order(m):
    # 检查YOLOv5 Detect()模块m的锚点顺序是否与步长顺序一致，必要时进行调整
    a = m.anchor_grid.prod(-1).view(-1)  # 计算锚点面积
    da = a[-1] - a[0]  # 锚点面积的差值
    ds = m.stride[-1] - m.stride[0]  # 步长的差值
    if da.sign() != ds.sign():  # 如果顺序不一致
        print('Reversing anchor order')  # 打印提示信息
        m.anchors[:] = m.anchors.flip(0)  # 反转锚点
        m.anchor_grid[:] = m.anchor_grid.flip(0)  # 反转锚点网格


def check_anchors(dataset, model, thr=4.0, imgsz=640):
    # 检查锚点是否适合数据集，如果需要，重新计算锚点
    prefix = colorstr('autoanchor: ')  # 带颜色的前缀
    print(f'\n{prefix}Analyzing anchors... ', end='')  # 打印分析锚点的提示信息
    m = model.module.model[-1] if hasattr(model, 'module') else model.model[-1]  # 获取Detect()模块
    shapes = imgsz * dataset.shapes / dataset.shapes.max(1, keepdims=True)  # 计算图像形状
    scale = np.random.uniform(0.9, 1.1, size=(shapes.shape[0], 1))  # 随机缩放因子
    wh = torch.tensor(np.concatenate([l[:, 3:5] * s for s, l in zip(shapes * scale, dataset.labels)])).float()  # 宽高

    # 定义计算指标的函数
    def metric(k):
        r = wh[:, None] / k[None]  # 计算宽高比
        x = torch.min(r, 1. / r).min(2)[0]  # 比例指标
        best = x.max(1)[0]  # 最佳比例
        aat = (x > 1. / thr).float().sum(1).mean()  # 超过阈值的锚点比例
        bpr = (best > 1. / thr).float().mean()  # 最佳可能召回率
        return bpr, aat

    anchors = m.anchor_grid.clone().cpu().view(-1, 2)  # 当前锚点
    bpr, aat = metric(anchors)  # 计算指标
    print(f'anchors/target = {aat:.2f}, Best Possible Recall (BPR) = {bpr:.4f}', end='')  # 打印指标
    if bpr < 0.98:  # 如果召回率低于阈值
        print('. Attempting to improve anchors, please wait...')  # 提示尝试改进锚点
        na = m.anchor_grid.numel() // 2  # 锚点数量
        try:
            anchors = kmean_anchors(dataset, n=na, img_size=imgsz, thr=thr, gen=1000, verbose=False)  # 使用k-means优化锚点
        except Exception as e:
            print(f'{prefix}ERROR: {e}')  # 打印错误信息
        new_bpr = metric(anchors)[0]  # 计算新锚点的指标
        if new_bpr > bpr:  # 如果新锚点更好
            anchors = torch.tensor(anchors, device=m.anchors.device).type_as(m.anchors) # 转换为张量
            m.anchor_grid[:] = anchors.clone().view_as(m.anchor_grid)  # 更新锚点网格
            m.anchors[:] = anchors.clone().view_as(m.anchors) / m.stride.to(m.anchors.device).view(-1, 1, 1)  # 更新锚点
            check_anchor_order(m)  # 检查锚点顺序
            print(f'{prefix}New anchors saved to model. Update model *.yaml to use these anchors in the future.')  # 提示保存新锚点
        else:
            print(f'{prefix}Original anchors better than new anchors. Proceeding with original anchors.')  # 提示使用原锚点
    print('')  # 换行

# 检查RGB和IR数据集的锚点
def check_anchors_rgb_ir(dataset, model, thr=4.0, imgsz=640):
    # 检查RGB和IR数据集的锚点是否适合数据，必要时重新计算
    prefix = colorstr('autoanchor: ')  # 带颜色的前缀
    print(f'\n{prefix}Analyzing anchors... ', end='')  # 打印分析锚点的提示信息
    m = list(model.model.children())[-1]  # 获取Detect()模块
    print(m)  # 打印模块信息
    shapes = imgsz * dataset.shapes / dataset.shapes.max(1, keepdims=True)  # 计算图像形状
    scale = np.random.uniform(0.9, 1.1, size=(shapes.shape[0], 1))  # 随机缩放因子
    wh = torch.tensor(np.concatenate([l[:, 3:5] * s for s, l in zip(shapes * scale, dataset.labels)])).float()  # 宽高

    # 定义计算指标的函数
    def metric(k):
        r = wh[:, None] / k[None]  # 计算宽高比
        x = torch.min(r, 1. / r).min(2)[0]  # 比例指标
        best = x.max(1)[0]  # 最佳比例
        aat = (x > 1. / thr).float().sum(1).mean()  # 超过阈值的锚点比例
        bpr = (best > 1. / thr).float().mean()  # 最佳可能召回率
        return bpr, aat

    anchors = m.anchor_grid.clone().cpu().view(-1, 2)  # 当前锚点
    bpr, aat = metric(anchors)  # 计算指标
    print(f'anchors/target = {aat:.2f}, Best Possible Recall (BPR) = {bpr:.4f}', end='')  # 打印指标
    if bpr < 0.98:  # 如果召回率低于阈值
        print('. Attempting to improve anchors, please wait...')  # 提示尝试改进锚点
        na = m.anchor_grid.numel() // 2  # 锚点数量
        try:
            anchors = kmean_anchors(dataset, n=na, img_size=imgsz, thr=thr, gen=1000, verbose=False)  # 使用k-means优化锚点
        except Exception as e:
            print(f'{prefix}ERROR: {e}')  # 打印错误信息
        new_bpr = metric(anchors)[0]  # 计算新锚点的指标
        if new_bpr > bpr:  # 如果新锚点更好
            anchors = torch.tensor(anchors, device=m.anchors.device).type_as(m.anchors)  # 转换为张量
            m.anchor_grid[:] = anchors.clone().view_as(m.anchor_grid)  # 更新锚点网格
            m.anchors[:] = anchors.clone().view_as(m.anchors) / m.stride.to(m.anchors.device).view(-1, 1, 1)  # 更新锚点
            check_anchor_order(m)  # 检查锚点顺序
            print(f'{prefix}New anchors saved to model. Update model *.yaml to use these anchors in the future.')  # 提示保存新锚点
        else:
            print(f'{prefix}Original anchors better than new anchors. Proceeding with original anchors.')  # 提示使用原锚点
    print('')  # 换行

# 使用k-means优化锚点
def kmean_anchors(path='./data/coco128.yaml', n=9, img_size=640, thr=4.0, gen=1000, verbose=True):
    """ 使用k-means算法从训练数据集中演化锚点

        参数：
            path: 数据集的*.yaml文件路径，或已加载的数据集
            n: 锚点数量
            img_size: 训练时使用的图像大小
            thr: 训练时使用的锚点-标签宽高比阈值
            gen: 使用遗传算法演化锚点的代数
            verbose: 是否打印所有结果

        返回：
            k: k-means演化的锚点

        使用方法：
            from utils.autoanchor import *; _ = kmean_anchors()
    """
    thr = 1. / thr  # 转换阈值
    prefix = colorstr('autoanchor: ')  # 带颜色的前缀

    # 定义计算指标的函数
    def metric(k, wh):
        r = wh[:, None] / k[None]  # 计算宽高比
        x = torch.min(r, 1. / r).min(2)[0]  # 比例指标
        return x, x.max(1)[0]  # 返回指标和最佳比例

    # 定义锚点适应度函数
    def anchor_fitness(k):
        _, best = metric(torch.tensor(k, dtype=torch.float32), wh)
        return (best * (best > thr).float()).mean()  # 计算适应度

    # 打印结果的函数
    def print_results(k):
        k = k[np.argsort(k.prod(1))]  # 按面积排序
        x, best = metric(k, wh0)
        bpr, aat = (best > thr).float().mean(), (x > thr).float().mean() * n  # 最佳可能召回率和超过阈值的锚点比例
        print(f'{prefix}thr={thr:.2f}: {bpr:.4f} best possible recall, {aat:.2f} anchors past thr')
        print(f'{prefix}n={n}, img_size={img_size}, metric_all={x.mean():.3f}/{best.mean():.3f}-mean/best, '
              f'past_thr={x[x > thr].mean():.3f}-mean: ', end='')
        for i, x in enumerate(k):
            print('%i,%i' % (round(x[0]), round(x[1])), end=',  ' if i < len(k) - 1 else '\n')  # 打印锚点
        return k

    # 加载数据集
    if isinstance(path, str):  # 如果是文件路径
        with open(path) as f:
            data_dict = yaml.safe_load(f)  # 加载数据集配置
        from utils.datasets import LoadImagesAndLabels
        dataset = LoadImagesAndLabels(data_dict['train'], augment=True, rect=True)  # 加载数据集
    else:
        dataset = path  # 如果已经加载了数据集

    # 获取标签宽高
    shapes = img_size * dataset.shapes / dataset.shapes.max(1, keepdims=True)  # 计算图像形状
    wh0 = np.concatenate([l[:, 3:5] * s for s, l in zip(shapes, dataset.labels)])  # 宽高

    # 过滤极小的物体
    i = (wh0 < 3.0).any(1).sum()
    if i:
        print(f'{prefix}WARNING: Extremely small objects found. {i} of {len(wh0)} labels are < 3 pixels in size.')
    wh = wh0[(wh0 >= 2.0).any(1)]  # 过滤小于2像素的物体

    # 执行k-means计算
    print(f'{prefix}Running kmeans for {n} anchors on {len(wh)} points...')
    s = wh.std(0)  # 计算标准差
    k, dist = kmeans(wh / s, n, iter=30)  # 执行k-means
    assert len(k) == n, print(f'{prefix}ERROR: scipy.cluster.vq.kmeans requested {n} points but returned only {len(k)}')
    k *= s  # 恢复原始尺度
    wh = torch.tensor(wh, dtype=torch.float32)  # filtered
    wh0 = torch.tensor(wh0, dtype=torch.float32)  # unfiltered
    k = print_results(k)

    # 遗传算法演化锚点
    npr = np.random
    f, sh, mp, s = anchor_fitness(k), k.shape, 0.9, 0.1  # 适应度、形状、变异概率、标准差
    pbar = tqdm(range(gen), desc=f'{prefix}Evolving anchors with Genetic Algorithm:')  # 进度条
    for _ in pbar:
        v = np.ones(sh)
        while (v == 1).all():  # 确保发生变异
            v = ((npr.random(sh) < mp) * npr.random() * npr.randn(*sh) * s + 1).clip(0.3, 3.0)
        kg = (k.copy() * v).clip(min=2.0)  # 变异后的锚点
        fg = anchor_fitness(kg)  # 计算变异后的适应度
        if fg > f:  # 如果变异后的锚点更好
            f, k = fg, kg.copy()
            pbar.desc = f'{prefix}Evolving anchors with Genetic Algorithm: fitness = {f:.4f}'
            if verbose:
                print_results(k)

    return print_results(k)  # 返回最终的锚点
