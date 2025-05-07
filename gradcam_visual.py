import os
import time
import argparse
import numpy as np
from models.gradcam import YOLOV5GradCAM
from models.yolo_v5_object_detector import YOLOV5TorchObjectDetector
import cv2
from deep_utils import Box, split_extension

# 定义目标层名称列表
target = ['model_30_cv1_act', 'model_30_cv2_act', 'model_30_cv3_act', \
          'model_33_cv1_act', 'model_33_cv2_act', 'model_33_cv3_act', \
          'model_36_cv1_act', 'model_36_cv2_act', 'model_36_cv3_act']

# 定义命令行参数解析器
parser = argparse.ArgumentParser()
parser.add_argument('--model-path', type=str, default="/home/shen/Chenyf/exp_save/multispectral-object-detection/5m_NiNfusion/weights/best.pt", help='模型路径')
parser.add_argument('--source1', type=str, default='/home/shen/Chenyf/kaist/visible/test', help='可见光图像源路径')  # 文件夹或摄像头
parser.add_argument('--source2', type=str, default='/home/shen/Chenyf/kaist/infrared/test', help='红外图像源路径')  # 文件夹或摄像头
parser.add_argument('--output-dir', type=str, default='/home/shen/Chenyf/kaist/Grad_CAM_visual/outputs_nin_head', help='输出目录')
parser.add_argument('--img-size', type=int, default=640, help="输入图像大小")
parser.add_argument('--target-layer', type=str, default=target,
                    help='GradCAM应用的目标层，层名称用下划线分隔')
parser.add_argument('--method', type=str, default='gradcam', help='使用的方法：gradcam或gradcampp')
parser.add_argument('--device', type=str, default='cpu', help='使用的设备：cuda或cpu')
parser.add_argument('--names', type=str, default='person',
                    help='类别名称，用逗号分隔，例如：person,car,bicycle')
args = parser.parse_args()  # 解析命令行参数


def get_res_img2(heat, mask, res_img):
    """
    生成热力图并将其添加到结果图像中。
    """
    # 将掩码转换为0-255范围的uint8格式
    mask = mask.squeeze(0).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
    # 使用OpenCV生成热力图
    heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    # 将热力图归一化到0-1范围
    n_heatmat = (heatmap / 255).astype(np.float32)
    # 将热力图添加到列表中
    heat.append(n_heatmat)
    return res_img, heat


def get_res_img(bbox, mask, res_img):
    """
    将热力图与边界框结合，并将其添加到结果图像中。
    """
    # 将掩码转换为0-255范围的uint8格式
    mask = mask.squeeze(0).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
    # 使用OpenCV生成热力图
    heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    # 将热力图归一化到0-1范围
    n_heatmat = (heatmap / 255).astype(np.float32)
    # 将热力图与结果图像叠加
    res_img = cv2.addWeighted(res_img, 0.7, n_heatmat, 0.3, 0)
    # 将结果图像归一化
    res_img = (res_img / res_img.max())
    return res_img, n_heatmat


def put_text_box(bbox, cls_name, res_img):
    """
    在结果图像上绘制边界框和类别名称。
    """
    x1, y1, x2, y2 = bbox  # 获取边界框坐标
    # 在结果图像上绘制边界框
    res_img = Box.put_box(res_img, bbox)
    return res_img


def concat_images(images):
    """
    将多个图像水平拼接。
    """
    w, h = images[0].shape[:2]  # 获取图像的宽度和高度
    width = w  # 设置拼接图像的宽度
    height = h * len(images)  # 设置拼接图像的高度
    base_img = np.zeros((width, height, 3), dtype=np.uint8)  # 创建一个空白图像
    for i, img in enumerate(images):  # 遍历图像列表
        base_img[:, h * i:h * (i + 1), ...] = img  # 将每个图像拼接到空白图像上
    return base_img  # 返回拼接后的图像


def main(img_vis_path, img_ir_path):
    """
    主函数，用于处理单对图像。
    """
    device = args.device  # 获取设备
    input_size = (args.img_size, args.img_size)  # 设置输入图像大小
    img_vis, img_ir = cv2.imread(img_vis_path), cv2.imread(img_ir_path)  # 读取可见光和红外图像
    print('[INFO] Loading the model')  # 输出加载模型的信息
    # 加载YOLOv5模型
    model = YOLOV5TorchObjectDetector(args.model_path, device, img_size=input_size,
                                      names=None if args.names is None else args.names.strip().split(","),
                                      confidence=0.3)
    # 对图像进行预处理
    torch_img_vis, torch_img_ir = model.preprocessing(img_vis[..., ::-1], img_ir[..., ::-1])
    # 将预处理后的图像转换为NumPy格式
    result = torch_img_vis.squeeze(0).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).detach().cpu().numpy()
    result = result[..., ::-1]  # 将图像从RGB转换为BGR
    images = []  # 初始化图像列表
    if args.method == 'gradcam':  # 如果使用GradCAM方法
        for layer in args.target_layer:  # 遍历目标层
            # 初始化GradCAM对象
            saliency_method = YOLOV5GradCAM(model=model, layer_name=layer, img_size=input_size)
            tic = time.time()  # 记录开始时间
            # 计算GradCAM掩码和检测结果
            masks, logits, [boxes, _, class_names, confs] = saliency_method(torch_img_vis, torch_img_ir)
            print("total time:", round(time.time() - tic, 4))  # 输出总时间
            res_img = result.copy()  # 复制结果图像
            res_img = res_img / 255  # 将图像归一化到0-1范围
            heat = []  # 初始化热力图列表
            for i, mask in enumerate(masks):  # 遍历掩码
                bbox = boxes[0][i]  # 获取边界框
                # 将掩码转换为0-255范围的uint8格式
                mask = mask.squeeze(0).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
                # 使用OpenCV生成热力图
                heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
                # 将热力图归一化到0-1范围
                n_heatmat = (heatmap / 255).astype(np.float32)
                heat.append(n_heatmat)  # 将热力图添加到列表中
                #res_img, heat_map = get_res_img(bbox, mask, res_img)
                #res_img = put_text_box(bbox, cls_name, res_img)  # 绘制边界框和类别名称
                #images.append(res_img)

            if len(heat) != 0:  # 如果热力图列表不为空
                heat_all = heat[0]  # 初始化热力图总和
                for h in heat[1:]:  # 遍历热力图列表
                    heat_all += h  # 累加热力图
                heat_avg = heat_all / len(heat)  # 计算平均热力图
                res_img = cv2.addWeighted(res_img, 0.3, heat_avg, 0.7, 0)  # 将平均热力图与结果图像叠加
            res_img = (res_img / res_img.max())  # 将结果图像归一化
            cv2.imwrite('temp.jpg', (res_img * 255).astype(np.uint8))  # 保存临时图像
            heat_map = cv2.imread('temp.jpg')  # 读取临时图像
            final_image = heat_map  # 设置最终图像
            images.append(final_image)  # 将最终图像添加到列表中
            # 保存图像
            suffix = '-res-' + layer  # 设置后缀
            img_name = split_extension(os.path.split(img_vis_path)[-1], suffix=suffix)  # 获取图像名称
            output_path = f'{args.output_dir}/{img_name}'  # 设置输出路径
            os.makedirs(args.output_dir, exist_ok=True)  # 创建输出目录
            print(f'[INFO] Saving the final image at {output_path}')  # 输出保存信息
            cv2.imwrite(output_path, final_image)  # 保存最终图像

        # 计算平均热力图
        img_name = split_extension(os.path.split(img_vis_path)[-1], suffix='_avg')  # 获取图像名称
        output_path = f'{args.output_dir}/{img_name}'  # 设置输出路径
        img_all = images[0].astype(np.uint16)  # 初始化图像总和
        for img in images[1:]:  # 遍历图像列表
            img_all += img  # 累加图像
        img_avg = img_all / len(images)  # 计算平均图像
        cv2.imwrite(output_path, img_avg.astype(np.uint8))  # 保存平均图像


if __name__ == '__main__':
    # 如果输入路径是文件夹
    if os.path.isdir(args.source1):
        img_vis_list = os.listdir(args.source1)  # 获取可见光图像列表
        img_vis_list.sort()  # 对图像列表排序
        for item in img_vis_list[1127:]:  # 遍历图像列表
            img_vis_path = os.path.join(args.source1, item)  # 获取可见光图像路径
            if args.source1 == '/home/shen/Chenyf/FLIR-align-3class/visible/test':  # 如果路径匹配
                new_item = item[:-4] + '.jpeg'  # 修改文件扩展名
                img_ir_path = os.path.join(args.source2, new_item)  # 获取红外图像路径
            else:
                img_ir_path = os.path.join(args.source2, item)  # 获取红外图像路径
            main(img_vis_path, img_ir_path)  # 处理图像对
            print(item)  # 输出当前处理的图像名称
    else:
        main(img_vis_path, img_ir_path)  # 如果输入路径是文件，直接处理