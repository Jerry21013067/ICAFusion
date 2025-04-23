import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from numpy import random

from models.experimental import attempt_load  # 加载模型
from utils.datasets import LoadStreams, LoadImages  # 加载数据集
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box, xywh2xyxy  # 通用工具函数
from utils.plots import colors, plot_one_box  # 绘图工具
from utils.torch_utils import select_device, load_classifier, time_synchronized  # PyTorch工具函数


def detect(opt):
    source1, source2, weights, view_img, save_txt, imgsz = opt.source1, opt.source2, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    # 提取命令行参数
    save_img = not opt.nosave and not source1.endswith('.txt')  # 判断是否保存图像
    webcam = source1.isnumeric() or source1.endswith('.txt') or source1.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))  # 判断是否为摄像头或网络流

    # 创建保存目录
    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # 递增路径
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # 创建标签目录

    # 初始化
    set_logging()  # 设置日志
    device = select_device(opt.device)  # 选择设备
    half = device.type != 'cpu'  # 是否使用半精度

    # 加载模型
    model = attempt_load(weights, map_location=device)  # 加载模型权重
    stride = int(model.stride.max())  # 获取模型步幅
    # imgsz = check_img_size(imgsz, s=stride)
    names = model.module.names if hasattr(model, 'module') else model.names  # 获取类别名称
    if half:
        model.half()  # 转换为半精度

    # 加载分类器（可选）
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # 初始化分类器
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # 设置数据加载器
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()  # 检查是否可以显示图像
        cudnn.benchmark = True  # 加速固定图像尺寸的推理
        dataset = LoadStreams(source1, img_size=imgsz, stride=stride)  # 加载视频流
    else:
        dataset = LoadImages(source1, img_size=imgsz, stride=stride)  # 加载图像数据
        dataset2 = LoadImages(source2, img_size=imgsz, stride=stride)  # 加载第二个数据源

    # 开始推理
    t0 = time.time()
    img_num = 0
    fps_sum = 0
    for (path, img, im0s, vid_cap), (path_, img2, im0s_, vid_cap_) in zip(dataset, dataset2):
        # 遍历两个数据源
        img = torch.from_numpy(img).to(device)  # 转换为张量
        img2 = torch.from_numpy(img2).to(device)

        img = img.half() if half else img.float()  # 转换为半精度或浮点
        img /= 255.0  # 归一化
        if img.ndimension() == 3:
            img = img.unsqueeze(0)  # 添加批次维度
        img2 = img2.half() if half else img2.float()
        img2 /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img2.ndimension() == 3:
            img2 = img2.unsqueeze(0)

        # 推理
        t1 = time_synchronized()
        pred = model(img, img2, augment=opt.augment)[0]  # 模型预测

        # NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # 应用分类器（可选）
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # 处理检测结果
        for i, det in enumerate(pred):  # 遍历每张图像的检测框

            if webcam:  # 处理视频流
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:  # 处理图像
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)
                p, s, im0_, frame = path, '', im0s_.copy(), getattr(dataset2, 'frame', 0)

            p = Path(p)  # 转换为Path对象
            save_path = str(save_dir / p.name)  # 构造保存路径
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # 构造标签路径
            s += '%gx%g ' % img.shape[2:]  # 打印图像尺寸
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # 归一化增益

            # 打印GPU内存占用
            mem = '%.4gM' % (torch.cuda.memory_reserved() / 1E6 if torch.cuda.is_available() else 0)
            print('GPU Memory:', mem)

            # 处理检测框
            if len(det):
                # 缩放检测框坐标
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # 打印检测结果
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # 每个类别的检测数量
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # 添加到打印字符串

                # 保存检测结果
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # 保存为文本文件
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or opt.save_crop or view_img:  # 绘制检测框
                        c = int(cls)
                        label = None if opt.hide_labels else (names[c] if opt.hide_conf else f'{names[c]} {conf:.2f}')

                        plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=opt.line_thickness)
                        plot_one_box(xyxy, im0_, label=label, color=colors(c, True), line_thickness=opt.line_thickness)
                        if opt.save_crop:  # 保存裁剪后的检测框
                            save_one_box(xyxy, im0s, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # 打印推理时间
            print(f'{s}Done. ({t2 - t1:.6f}s, {1/(t2 - t1):.6f}Hz)')
            img_num += 1
            fps_sum += 1/(t2 - t1)

            # 显示结果
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)

            # 保存结果
            if save_img:
                if dataset.mode == 'image':  # 保存图像
                    save_path_rgb = save_path.split('.')[0] + '_rgb.' + save_path.split('.')[1]
                    save_path_ir = save_path.split('.')[0] + '_ir.' + save_path.split('.')[1]
                    print(save_path_rgb)
                    cv2.imwrite(save_path_rgb, im0)
                    cv2.imwrite(save_path_ir, im0_)
                else:  # 保存视频
                    if vid_path != save_path:
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()
                        if vid_cap:
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    # 打印保存路径和总耗时
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')
    print(f'Average Speed: {fps_sum/img_num:.6f}Hz')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/ICAFusion_FLIR.pt', help='模型权重路径')
    parser.add_argument('--source1', type=str, default='FLIR-align-3class/visible/test/', help='第一个数据源路径')
    parser.add_argument('--source2', type=str, default='FLIR-align-3class/infrared/test/', help='第二个数据源路径')
    parser.add_argument('--img-size', type=int, default=640, help='推理图像尺寸')
    parser.add_argument('--conf-thres', type=float, default=0.1, help='置信度阈值')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS的IOU阈值')
    parser.add_argument('--device', default='0', help='设备选择，如0或0,1,2,3或cpu')
    parser.add_argument('--view-img', default=False, action='store_true', help='是否显示结果')
    parser.add_argument('--save-txt', action='store_true', help='是否保存结果到txt文件')
    parser.add_argument('--save-conf', action='store_true', help='是否保存置信度')
    parser.add_argument('--save-crop', action='store_true', help='是否保存裁剪后的检测框')
    parser.add_argument('--nosave', action='store_true', help='是否不保存图像/视频')
    parser.add_argument('--classes', nargs='+', type=int, help='类别过滤')
    parser.add_argument('--agnostic-nms', action='store_true', help='类别无关的NMS')
    parser.add_argument('--augment', action='store_true', help='是否使用增强推理')
    parser.add_argument('--update', action='store_true', help='是否更新模型')
    parser.add_argument('--project', default='runs/detect', help='保存结果的项目路径')
    parser.add_argument('--name', default='exp', help='保存结果的实验名称')
    parser.add_argument('--exist-ok', action='store_true', help='是否覆盖已存在的项目')
    parser.add_argument('--line-thickness', default=2, type=int, help='边界框的线条厚度')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='是否隐藏标签')
    parser.add_argument('--hide-conf', default=True, action='store_true', help='是否隐藏置信度')
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # 更新所有模型
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect(opt=opt)
                strip_optimizer(opt.weights)
        else:
            detect(opt=opt)
