# Model validation metrics

from pathlib import Path  # 用于路径操作
import matplotlib.pyplot as plt  # 用于绘图
import numpy as np  # 用于数值计算
import torch  # PyTorch深度学习框架
from . import general  # 导入general模块


def fitness(x):
    # 计算模型的适应度，作为指标的加权组合
    w = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]  # 权重，对应[tp[0], fp[0], fn[0], f1[0], mp, mr, map50, map]
    return (x[:, :8] * w).sum(1)  # 计算加权和


def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir='.', names=()):
    """
    计算每个类别的平均精度（AP），给定召回率和精确度曲线。
    来源：<url id="d0djh07hvltmntaj1lq0" type="url" status="parsed" title="GitHub - rafaelpadilla/Object-Detection-Metrics: Most popular metrics used to evaluate object detection algorithms." wc="21583">https://github.com/rafaelpadilla/Object-Detection-Metrics</url> 。
    # 参数
        tp:  真正例（nparray, nx1 或 nx10）。
        conf:  目标性值，范围从0到1（nparray）。
        pred_cls:  预测的类别（nparray）。
        target_cls:  真实类别（nparray）。
        plot:  是否绘制精确率-召回率曲线（mAP@0.5）。
        save_dir:  图片保存路径。
    # 返回
        按照py-faster-rcnn计算的平均精度。
    """

    # 按目标性排序
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # 找到唯一类别
    unique_classes = np.unique(target_cls)
    nc = unique_classes.shape[0]  # 类别数量，检测数量

    # 创建精确率-召回率曲线并计算每个类别的AP
    px, py = np.linspace(0, 1, 1000), []  # 用于绘图
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = (target_cls == c).sum()  # 标签数量
        n_p = i.sum()  # 预测数量

        if n_p == 0 or n_l == 0:
            continue
        else:
            # 累积FP和TP
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # 召回率
            recall = tpc / (n_l + 1e-16)  # 召回率曲线
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # 负x，xp因为xp递减

            # 精确率
            precision = tpc / (tpc + fpc)  # 精确率曲线
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p在pr_score处

            # 从召回率-精确率曲线计算AP
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
                if plot and j == 0:
                    py.append(np.interp(px, mrec, mpre))  # mAP@0.5处的精确率

    # 计算F1（精确率和召回率的调和平均）
    f1 = 2 * p * r / (p + r + 1e-16)
    if plot:
        plot_pr_curve(px, py, ap, Path(save_dir) / 'PR_curve.png', names)
        plot_mc_curve(px, f1, Path(save_dir) / 'F1_curve.png', names, ylabel='F1')
        plot_mc_curve(px, p, Path(save_dir) / 'P_curve.png', names, ylabel='Precision')
        plot_mc_curve(px, r, Path(save_dir) / 'R_curve.png', names, ylabel='Recall')

    i = f1.mean(0).argmax()  # 最大F1索引
    tp = (r * n_l).round()  # 真正例
    fn = n_l - tp
    fp = (tp / (p + 1e-16) - tp).round()  # 假正例

    return tp[:, i], fp[:, i], fn[:, i], p[:, i], r[:, i], ap, f1[:, i], unique_classes.astype('int32')


def compute_ap(recall, precision):
    """
    给定召回率和精确率曲线，计算平均精度。
    # 参数
        recall:    召回率曲线（列表）。
        precision: 精确率曲线（列表）。
    # 返回
        平均精度，精确率曲线，召回率曲线。
    """

    # 在开头和结尾追加哨兵值
    mrec = np.concatenate(([0.], recall, [recall[-1] + 0.01]))
    mpre = np.concatenate(([1.], precision, [0.]))

    # 计算精确率包络
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # 积分曲线下面积
    method = 'interp'  # 方法：'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101点插值（COCO）
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # 积分
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # x轴（召回率）变化的点
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # 曲线下面积

    return ap, mpre, mrec


class ConfusionMatrix:
    # 更新版本的混淆矩阵类，用于目标检测任务
    def __init__(self, nc, conf=0.25, iou_thres=0.45):
        self.matrix = np.zeros((nc + 1, nc + 1))  # 初始化混淆矩阵
        self.nc = nc  # 类别数量
        self.conf = conf  # 置信度阈值
        self.iou_thres = iou_thres  # IOU阈值

    def process_batch(self, detections, labels):
        """
        返回边界框的交并比（Jaccard指数）。
        预期两组边界框均为(x1, y1, x2, y2)格式。
        参数：
            detections (Array[N, 6])，x1, y1, x2, y2, conf, class
            labels (Array[M, 5])，class, x1, y1, x2, y2
        返回：
            无，相应更新混淆矩阵
        """
        detections = detections[detections[:, 4] > self.conf]  # 过滤低置信度检测
        gt_classes = labels[:, 0].int()  # 真实类别
        detection_classes = detections[:, 5].int()  # 预测类别
        iou = general.box_iou(labels[:, 1:], detections[:, :4])  # 计算IOU

        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(np.int16)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                self.matrix[detection_classes[m1[j]], gc] += 1  # 正确检测
            else:
                self.matrix[self.nc, gc] += 1  # 背景假正例

        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    self.matrix[dc, self.nc] += 1  # 背景假负例

    def matrix(self):
        return self.matrix

    def plot(self, save_dir='', names=()):
        try:
            import seaborn as sn  # 导入seaborn库

            array = self.matrix / (self.matrix.sum(0).reshape(1, self.nc + 1) + 1E-6)  # 归一化
            array[array < 0.005] = np.nan  # 不注释（会显示为0.00）

            fig = plt.figure(figsize=(12, 9), tight_layout=True)
            sn.set(font_scale=1.0 if self.nc < 50 else 0.8)  # 设置字体大小
            labels = (0 < len(names) < 99) and len(names) == self.nc  # 应用类别名称到刻度标签
            sn.heatmap(array, annot=self.nc < 30, annot_kws={"size": 8}, cmap='Blues', fmt='.2f', square=True,
                       xticklabels=names + ['background FP'] if labels else "auto",
                       yticklabels=names + ['background FN'] if labels else "auto").set_facecolor((1, 1, 1))
            fig.axes[0].set_xlabel('True')  # 真实
            fig.axes[0].set_ylabel('Predicted')  # 预测
            fig.savefig(Path(save_dir) / 'confusion_matrix.png', dpi=250)
        except Exception as e:
            pass

    def print(self):
        for i in range(self.nc + 1):
            print(' '.join(map(str, self.matrix[i])))


# Plots ----------------------------------------------------------------------------------------------------------------

def plot_pr_curve(px, py, ap, save_dir='pr_curve.png', names=()):
    # 精确率-召回率曲线
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # 如果类别数量少于21，显示每个类别的图例
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f'{names[i]} {ap[i, 0]:.3f}')  # 绘制召回率-精确率曲线
    else:
        ax.plot(px, py, linewidth=1, color='grey')  # 绘制召回率-精确率曲线

    ax.plot(px, py.mean(1), linewidth=3, color='blue', label='all classes %.3f mAP@0.5' % ap[:, 0].mean())
    ax.set_xlabel('Recall')  # 召回率
    ax.set_ylabel('Precision')  # 精确率
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)


def plot_mc_curve(px, py, save_dir='mc_curve.png', names=(), xlabel='Confidence', ylabel='Metric'):
    # 指标-置信度曲线
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # 如果类别数量少于21，显示每个类别的图例
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f'{names[i]}')  # 绘制置信度-指标曲线
    else:
        ax.plot(px, py.T, linewidth=1, color='grey')  # 绘制置信度-指标曲线

    y = py.mean(0)
    ax.plot(px, y, linewidth=3, color='blue', label=f'all classes {y.max():.2f} at {px[y.argmax()]:.3f}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)
