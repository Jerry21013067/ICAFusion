import argparse
import json
import os
from pathlib import Path
from threading import Thread

import numpy as np
import torch
import yaml
from tqdm import tqdm  # 导入tqdm模块，用于显示进度条

from models.experimental import attempt_load
from utils.datasets import create_dataloader_rgb_ir  # 导入create_dataloader_rgb_ir函数，用于创建数据加载器
from utils.general import logger, coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xyxy2xywh2, xywh2xyxy, set_logging, increment_path, colorstr
from utils.metrics import ap_per_class, ConfusionMatrix  # 导入评估指标计算函数
from utils.plots import plot_images, output_to_target, plot_study_txt
from utils.torch_utils import select_device, time_synchronized
from evaluation_script.evaluation_script import evaluate
from utils.confluence import confluence_process


def test(data,
         weights=None,
         batch_size=32,
         imgsz=640,
         conf_thres=0.001,
         iou_thres=0.5,  # NMS的IoU阈值
         save_json=False,
         single_cls=False,
         augment=False,
         verbose=False,
         model=None,
         dataloader=None,
         save_dir=Path(''),  # 保存图像的目录
         save_txt=False,  # 是否保存为txt格式的标签
         save_hybrid=False,  # 是否保存混合标签
         save_conf=True,  # 是否保存置信度
         plots=False,
         wandb_logger=None,
         compute_loss=None,
         half_precision=True,
         is_coco=False,
         opt=None,
         labels_list=None):
    """
    测试YOLOv5模型的性能。

    参数：
        data (str or dict): 数据集配置文件路径或字典。
        weights (str or list): 模型权重路径或列表。
        batch_size (int): 每个批次的图像数量。
        imgsz (int): 图像大小。
        conf_thres (float): 置信度阈值。
        iou_thres (float): NMS的IoU阈值。
        save_json (bool): 是否保存为JSON格式的结果。
        single_cls (bool): 是否为单类别数据集。
        augment (bool): 是否使用数据增强。
        verbose (bool): 是否打印详细信息。
        model (nn.Module): 模型对象。
        dataloader (DataLoader): 数据加载器。
        save_dir (Path): 保存目录。
        save_txt (bool): 是否保存为txt格式的标签。
        save_hybrid (bool): 是否保存混合标签。
        save_conf (bool): 是否保存置信度。
        plots (bool): 是否绘制图像。
        wandb_logger (object): W&B日志记录器。
        compute_loss (function): 损失计算函数。
        half_precision (bool): 是否使用半精度。
        is_coco (bool): 是否为COCO数据集。
        opt (argparse.Namespace): 命令行参数。
        labels_list (list): 标签列表。

    返回：
        tuple: 包含测试结果的元组。
    """

    # 初始化/加载模型并设置设备
    training = model is not None  # 是否为训练模式
    if training:  # 如果是训练模式
        device = next(model.parameters()).device  # 获取模型设备
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # 创建保存目录
        if save_txt:
            labels_dir = increment_path(Path(save_dir) / 'labels' / 'pred', exist_ok=False, mkdir=True)  # 创建标签目录
    else:  # 如果是测试模式
        set_logging()  # 设置日志
        device = select_device(opt.device, batch_size=batch_size)  # 选择设备
        # 创建保存目录
        save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # 创建保存目录
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # 创建保存目录
        labels_dir = save_dir / 'labels'  # 设置标签目录
        # 加载模型
        model = attempt_load(weights, map_location=device)  # 加载FP32模型
        gs = max(int(model.stride.max()), 32)  # 网格大小（最大步长）
        imgsz = check_img_size(imgsz, s=gs)  # 检查图像大小
        # 多GPU禁用，与.half()不兼容
        # if device.type != 'cpu' and torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)

    # 半精度
    half = device.type != 'cpu' and half_precision  # 半精度仅支持CUDA
    if half:
        model.half()  # 转换为半精度

    # 配置
    model.eval()  # 设置为评估模式
    if isinstance(data, str):  # 如果数据是字符串
        is_coco = data.endswith('coco.yaml')  # 是否为COCO数据集
        with open(data) as f:  # 打开数据文件
            data = yaml.safe_load(f)  # 加载数据
    check_dataset(data)  # 检查数据集
    nc = 1 if single_cls else int(data['nc'])  # 类别数量
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # IoU向量，用于计算mAP@0.5:0.95
    niou = iouv.numel()  # IoU数量

    # 日志
    log_imgs = 0  # 日志图像数量
    if wandb_logger and wandb_logger.wandb:  # 如果使用W&B日志记录器
        log_imgs = min(wandb_logger.log_imgs, 100)  # 设置日志图像数量
    # 数据加载器
    if not training:  # 如果不是训练模式
        task = opt.task if opt.task in ('train', 'val', 'test') else 'val'  # 设置任务类型
        val_path_rgb = data['val_rgb']  # RGB验证路径
        val_path_ir = data['val_ir']  # 红外验证路径
        dataloader = create_dataloader_rgb_ir(val_path_rgb, val_path_ir, imgsz, batch_size, gs, opt, pad=0.5, rect=True,
                                              prefix=colorstr(f'{task}: '))[0]  # 创建数据加载器

    seen = 0  # 已处理的图像数量
    confusion_matrix = ConfusionMatrix(nc=nc)  # 混淆矩阵
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}  # 类别名称
    coco91class = coco80_to_coco91_class()  # COCO类别映射
    if nc == 1:  # 如果是单类别
        s = ('%20s' + '%12s' * 10) % (
        'Class', 'Images', 'Labels', 'TP', 'FP', 'FN', 'F1', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')  # 设置进度条显示信息
    else:
        s = ('%20s' + '%12s' * 7) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.75', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map75, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0, 0., 0.  # 初始化指标
    tp, fp, fn = 0, 0, 0  # 初始化TP、FP、FN
    loss = torch.zeros(4, device=device)  # 初始化损失
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []  # 初始化列表

    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):  # 遍历数据加载器
        img = img.to(device, non_blocking=True)  # 将图像移动到设备
        img = img.half() if half else img.float()  # 转换为半精度或浮点
        img /= 255.0  # 归一化
        targets = targets.to(device)  # 将目标移动到设备
        nb, _, height, width = img.shape  # 获取批次大小、通道数、高度、宽度
        img_rgb = img[:, :3, :, :]  # RGB图像
        img_ir = img[:, 3:, :, :]  # 红外图像

        with torch.no_grad():  # 禁用梯度计算
            t = time_synchronized()  # 记录时间
            out, _, train_out = model(img_rgb, img_ir, augment=augment)  # 模型推理
            t0 += time_synchronized() - t  # 更新推理时间
            # 计算损失
            if compute_loss:
                loss += compute_loss([x.float() for x in train_out], targets)[1][:4]  # 计算损失
            # NMS
            targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # 转换为目标像素坐标
            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # 用于自动标记的标签
            t = time_synchronized()  # 记录时间
            out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True,
                                      agnostic=single_cls)  # NMS
            t1 += time_synchronized() - t  # 更新NMS时间

        # 每张图像的统计信息
        for si, pred in enumerate(out):  # 遍历预测结果
            labels = targets[targets[:, 0] == si, 1:]  # 获取目标标签
            nl = len(labels)  # 目标数量
            tcls = labels[:, 0].tolist() if nl else []  # 目标类别
            path = Path(paths[si])  # 获取图像路径
            seen += 1  # 更新已处理图像数量

            if len(pred) == 0:  # 如果没有预测结果
                if nl:  # 如果有目标
                    stats.append(
                        (torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))  # 添加统计信息
                continue

            # 预测结果
            if single_cls:  # 如果是单类别
                pred[:, 5] = 0  # 设置类别为0
            predn = pred.clone()  # 复制预测结果
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # 转换为原图坐标

            # 保存为txt格式的标签
            if save_txt:  # 如果保存为txt格式
                i = labels_list.index(str(path.stem) + '.txt')  # 获取索引
                gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # 归一化增益
                for *xyxy, conf, cls in predn.tolist():  # 遍历预测结果
                    xywh = (xyxy2xywh2(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()  # 转换为xywh格式
                    line = (i + 1, *xywh, conf) if save_conf else (i + 1, *xywh)  # 标签格式
                    with open(labels_dir / (path.stem + '.txt'), 'a') as f:  # 打开文件
                        f.write(('%g,' * len(line)).rstrip(",") % line + '\n')  # 写入文件

            # W&B日志记录
            if len(wandb_images) < log_imgs and wandb_logger.current_epoch > 0:  # 如果记录日志
                if wandb_logger.current_epoch % wandb_logger.bbox_interval == 0:  # 如果满足间隔条件
                    box_data = [{"position": {"minX": xyxy[0], "minY": xyxy[1], "maxX": xyxy[2], "maxY": xyxy[3]},
                                 "class_id": int(cls),
                                 "box_caption": "%s %.3f" % (names[cls], conf),
                                 "scores": {"class_score": conf},
                                 "domain": "pixel"} for *xyxy, conf, cls in pred.tolist()]  # 创建日志数据
                    boxes = {"predictions": {"box_data": box_data, "class_labels": names}}  # 创建日志字典
                    # wandb_images.append(wandb_logger.wandb.Image(img[si], boxes=boxes, caption=path.name))  # 添加日志图像
            # wandb_logger.log_training_progress(predn, path, names) if wandb_logger and wandb_logger.wandb_run else None  # 记录训练进度

            # 保存为JSON格式的结果
            if save_json:  # 如果保存为JSON格式
                image_id = int(path.stem) if path.stem.isnumeric() else path.stem  # 获取图像ID
                box = xyxy2xywh(predn[:, :4])  # 转换为xywh格式
                box[:, :2] -= box[:, 2:] / 2  # 转换为左上角坐标
                for p, b in zip(pred.tolist(), box.tolist()):  # 遍历预测结果
                    jdict.append({'image_id': image_id,  # 添加到JSON字典
                                  'category_id': coco91class[int(p[5])] if is_coco else int(p[5]),
                                  'bbox': [round(x, 3) for x in b],
                                  'score': round(p[4], 5)})

            # 将所有预测结果标记为不正确
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)  # 初始化正确标记
            if nl:  # 如果有目标
                detected = []  # 检测到的目标索引
                tcls_tensor = labels[:, 0]  # 目标类别张量

                # 目标边界框
                tbox = xywh2xyxy(labels[:, 1:5])  # 转换为xyxy格式
                scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # 转换为原图坐标
                if plots:  # 如果绘制图像
                    confusion_matrix.process_batch(predn, torch.cat((labels[:, 0:1], tbox), 1))  # 更新混淆矩阵

                # 每个目标类别
                for cls in torch.unique(tcls_tensor):  # 遍历目标类别
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # 预测索引
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # 目标索引

                    # 搜索检测结果
                    if pi.shape[0]:  # 如果有预测结果
                        ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # 计算IoU

                        # 添加检测结果
                        detected_set = set()  # 检测到的目标集合
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):  # 遍历IoU大于阈值的预测结果
                            d = ti[i[j]]  # 检测到的目标
                            if d.item() not in detected_set:  # 如果目标未被检测到
                                detected_set.add(d.item())  # 添加到集合
                                detected.append(d)  # 添加到检测列表
                                correct[pi[j]] = ious[j] > iouv  # 更新正确标记
                                if len(detected) == nl:  # 如果所有目标都被检测到
                                    break

            # 添加统计信息
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

        # 绘制图像
        if plots and batch_i < 3:  # 如果绘制图像
            f1 = save_dir / f'test_batch{batch_i}_labels.jpg'  # 标签图像路径
            Thread(target=plot_images, args=(img_rgb, targets, paths, f1, names), daemon=True).start()  # 绘制标签图像
            f2 = save_dir / f'test_batch{batch_i}_pred.jpg'  # 预测图像路径
            Thread(target=plot_images, args=(img_rgb, output_to_target(out), paths, f2, names),
                   daemon=True).start()  # 绘制预测图像

    # 保存所有预测框结果，用于MR计算
    if save_txt:  # 如果保存为txt格式
        temp = []  # 初始化列表
        files = os.listdir(labels_dir)  # 获取文件列表
        files.sort()  # 排序
        for index, file in enumerate(files):  # 遍历文件
            with open(labels_dir / file, 'r') as f:  # 打开文件
                for line in f:  # 遍历文件内容
                    temp.append(line)  # 添加到列表
        with open(labels_dir / 'result.txt', 'a') as ff:  # 打开结果文件
            for ii in temp:  # 遍历列表
                ff.write(ii)  # 写入文件

    # 计算MR指标
    MR_all = 0.0  # 初始化MR指标
    MR_day = 0.0
    MR_night = 0.0
    MR_near = 0.0
    MR_medium = 0.0
    MR_far = 0.0
    MR_none = 0.0
    MR_partial = 0.0
    MR_heavy = 0.0
    recall_all = 0.0
    MRresult = [MR_all, MR_day, MR_night, MR_near, MR_medium, MR_far, MR_none, MR_partial, MR_heavy,
                recall_all]  # MR结果列表

    # 计算统计信息
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # 转换为numpy数组
    if len(stats) and stats[0].any():  # 如果有统计信息
        tp, fp, fn, p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)  # 计算AP
        ap50, ap75, ap = ap[:, 0], ap[:, 5], ap.mean(1)  # AP@0.5, AP@0.75, AP
        mp, mr, map50, map75, map = p.mean(), r.mean(), ap50.mean(), ap75.mean(), ap.mean()  # 平均指标
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # 每个类别的目标数量
    else:
        nt = torch.zeros(1)  # 如果没有统计信息，初始化为0

    # 打印结果
    if nc > 1:  # 如果有多类别
        pf = '%20s' + '%12i' * 2 + '%12.3g' * 5  # 打印格式
        logger.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map75, map))  # 打印总结果
    else:
        pf = '%20s' + '%12i' * 2 + '%12.4g' * 8  # 打印格式
        logger.info(pf % ('all', seen, nt.sum(), tp, fp, fn, f1, mp, mr, map50, map))  # 打印总结果
    logger.info(('%20s' + '%11s' * 9) % (
    'MR-all', 'MR-day', 'MR-night', 'MR-near', 'MR-medium', 'MR-far', 'MR-none', 'MR-partial', 'MR-heavy',
    'Recall-all'))  # 打印MR指标
    logger.info(('%20.2f' + '%11.2f' * 9) % (
    MR_all * 100, MR_day * 100, MR_night * 100, MR_near * 100, MR_medium * 100, MR_far * 100, MR_none * 100,
    MR_partial * 100, MR_heavy * 100, recall_all * 100))  # 打印MR指标

    # 打印每个类别的结果
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):  # 如果打印详细信息
        for i, c in enumerate(ap_class):  # 遍历类别
            logger.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap75[i], ap[i]))  # 打印类别结果

    # 打印速度信息
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # 转换为毫秒
    if not training:  # 如果不是训练模式
        print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)  # 打印速度信息

    # 绘制图像
    if plots:  # 如果绘制图像
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))  # 绘制混淆矩阵
        if wandb_logger and wandb_logger.wandb:  # 如果使用W&B日志记录器
            val_batches = [wandb_logger.wandb.Image(str(f), caption=f.name) for f in
                           sorted(save_dir.glob('test*.jpg'))]  # 获取验证图像
            wandb_logger.log({"Validation": val_batches})  # 记录验证图像
    if wandb_images:  # 如果有W&B图像
        wandb_logger.log({"Bounding Box Debugger/Images": wandb_images})  # 记录图像

    # 保存为JSON格式的结果
    if save_json and len(jdict):  # 如果保存为JSON格式
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # 权重文件名
        anno_json = '../coco/annotations/instances_val2017.json'  # 注释文件路径
        pred_json = str(save_dir / f"{w}_predictions.json")  # 预测文件路径
        print('\nEvaluating pycocotools mAP... saving %s...' % pred_json)  # 打印信息
        with open(pred_json, 'w') as f:  # 打开文件
            json.dump(jdict, f)  # 保存为JSON格式

        try:  # 尝试使用pycocotools评估mAP
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # 初始化注释API
            pred = anno.loadRes(pred_json)  # 初始化预测API
            eval = COCOeval(anno, pred, 'bbox')  # 初始化评估器
            if is_coco:  # 如果是COCO数据集
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # 设置图像ID
            eval.evaluate()  # 评估
            eval.accumulate()  # 累积结果
            eval.summarize()  # 打印结果
            map, map50 = eval.stats[:2]  # 获取mAP值
        except Exception as e:  # 如果发生异常
            print(f'pycocotools unable to run: {e}')  # 打印错误信息

    # 返回结果
    model.float()  # 转换为浮点
    if not training:  # 如果不是训练模式
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''  # 打印保存的标签数量
        print(f"Results saved to {save_dir}{s}")  # 打印保存路径
    maps = np.zeros(nc) + map  # 初始化mAP数组
    for i, c in enumerate(ap_class):  # 遍历类别
        maps[c] = ap[i]  # 更新mAP值

    if not isinstance(tp, int):  # 如果tp不是整数
        return (tp[0], fp[0], fn[0], f1[0], mp, mr, map50, map,
                *(loss.cpu() / len(dataloader)).tolist()), maps, MRresult, t  # 返回结果
    else:
        return (tp, fp, fn, f1, mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, MRresult, t  # 返回结果


if __name__ == '__main__':  # 如果是主程序
    parser = argparse.ArgumentParser(prog='test.py')  # 创建解析器
    parser.add_argument('--weights', nargs='+', type=str,
                        default='/home/shen/Chenyf/exp_save/multispectral-object-detection/5l_FLIR_3class_transformerx2_avgpool+maxpool/weights/best.pt',
                        help='模型权重路径')  # 添加权重参数
    parser.add_argument('--data', type=str, default='./data/multispectral/FLIR-align-3class.yaml',
                        help='数据集配置文件路径')  # 添加数据集参数
    parser.add_argument('--batch-size', type=int, default=1, help='每个批次的图像数量')  # 添加批次大小参数
    parser.add_argument('--img-size', type=int, default=640, help='图像大小')  # 添加图像大小参数
    parser.add_argument('--conf-thres', type=float, default=0.001, help='置信度阈值')  # 添加置信度阈值参数
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS的IoU阈值')  # 添加IoU阈值参数
    parser.add_argument('--task', default='val', help='任务类型')  # 添加任务类型参数
    parser.add_argument('--device', default='', help='设备')  # 添加设备参数
    parser.add_argument('--single-cls', action='store_true', help='是否为单类别数据集')  # 添加单类别参数
    parser.add_argument('--augment', default=False, action='store_true', help='是否使用数据增强')  # 添加数据增强参数
    parser.add_argument('--verbose', action='store_true', help='是否打印详细信息')  # 添加详细信息参数
    parser.add_argument('--save-txt', default=True, action='store_true', help='是否保存为txt格式的标签')  # 添加保存txt标签参数
    parser.add_argument('--save-hybrid', action='store_true', help='是否保存混合标签')  # 添加保存混合标签参数
    parser.add_argument('--save-conf', default=True, action='store_true', help='是否保存置信度')  # 添加保存置信度参数
    parser.add_argument('--save-json', action='store_true', help='是否保存为JSON格式的结果')  # 添加保存JSON结果参数
    parser.add_argument('--project', default='runs/test', help='保存项目路径')  # 添加保存项目路径参数
    parser.add_argument('--name', default='exp', help='保存实验名称')  # 添加保存实验名称参数
    parser.add_argument('--exist-ok', action='store_true', help='是否允许已存在的项目')  # 添加允许已存在项目参数
    opt = parser.parse_args()  # 解析参数
    opt.save_json |= opt.data.endswith('coco.yaml')  # 如果是COCO数据集，保存为JSON格式
    opt.data = check_file(opt.data)  # 检查数据集文件
    print(opt)  # 打印参数
    print(opt.data)  # 打印数据集路径
    check_requirements()  # 检查依赖

    p = "/home/shen/Chenyf/FLIR-align-3class/labels/test"  # 标签路径
    labels_list = os.listdir(p)  # 获取标签列表
    labels_list.sort()  # 排序
    if opt.data in ['./data/multispectral/FLIR-align-3class.yaml', './data/multispectral/FLIR-ADAS.yaml',
                    './data/multispectral/VEDAI.yaml']:  # 如果是特定数据集
        opt.verbose = True  # 打印详细信息

    if opt.task in ('train', 'val', 'test'):  # 如果是训练、验证或测试任务
        test(opt.data,  # 数据集路径
             opt.weights,  # 权重路径
             opt.batch_size,  # 批次大小
             opt.img_size,  # 图像大小
             opt.conf_thres,  # 置信度阈值
             opt.iou_thres,  # IoU阈值
             opt.save_json,  # 是否保存为JSON格式
             opt.single_cls,  # 是否为单类别数据集
             opt.augment,  # 是否使用数据增强
             opt.verbose,  # 是否打印详细信息
             save_txt=opt.save_txt | opt.save_hybrid,  # 是否保存为txt格式的标签
             save_hybrid=opt.save_hybrid,  # 是否保存混合标签
             save_conf=opt.save_conf,  # 是否保存置信度
             opt=opt,  # 参数
             labels_list=labels_list  # 标签列表
             )

    elif opt.task == 'speed':  # 如果是速度测试任务
        for w in opt.weights:  # 遍历权重路径
            test(opt.data, w, opt.batch_size, opt.img_size, 0.25, 0.45, save_json=False, plots=False, opt=opt)  # 测试速度

    elif opt.task == 'study':  # 如果是研究任务
        x = list(range(256, 1536 + 128, 128))  # 图像大小范围
        for w in opt.weights:  # 遍历权重路径
            f = f'study_{Path(opt.data).stem}_{Path(w).stem}.txt'  # 结果文件路径
            y = []  # 初始化结果列表
            for i in x:  # 遍历图像大小
                print(f'\nRunning {f} point {i}...')  # 打印信息
                r, _, t = test(opt.data, w, opt.batch_size, i, opt.conf_thres, opt.iou_thres, opt.save_json,
                               plots=False, opt=opt)  # 测试
                y.append(r + t)  # 添加结果
            np.savetxt(f, y, fmt='%10.4g')  # 保存结果
        os.system('zip -r study.zip study_*.txt')  # 压缩结果文件
        plot_study_txt(x=x)  # 绘制研究图像
