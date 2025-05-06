"""
将YOLOv5 *.pt模型导出为ONNX和TorchScript格式

用法：
    $ export PYTHONPATH="$PWD" && python models/export.py --weights yolov5s.pt --img 640 --batch 1
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.append(Path(__file__).parent.parent.absolute().__str__())  # 为了在子目录中运行'$ python *.py'文件

import torch
import torch.nn as nn
from torch.utils.mobile_optimizer import optimize_for_mobile

import models
from models.experimental import attempt_load
from utils.activations import Hardswish, SiLU
from utils.general import colorstr, check_img_size, check_requirements, file_size, set_logging
from utils.torch_utils import select_device

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./yolov5s.pt', help='weights path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size')  # 高度，宽度
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--grid', action='store_true', help='export Detect() layer grid')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--dynamic', action='store_true', help='dynamic ONNX axes')  # 仅ONNX
    parser.add_argument('--simplify', action='store_true', help='simplify ONNX model')  # 仅ONNX
    opt = parser.parse_args()
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # 扩展
    print(opt)
    set_logging()
    t = time.time()

    # 加载PyTorch模型
    device = select_device(opt.device)
    model = attempt_load(opt.weights, map_location=device)  # 加载FP32模型
    labels = model.names

    # 检查
    gs = int(max(model.stride))  # 网格大小（最大步长）
    opt.img_size = [check_img_size(x, gs) for x in opt.img_size]  # 验证img_size是否为gs的倍数

    # 输入
    img = torch.zeros(opt.batch_size, 3, *opt.img_size).to(device)  # 图像大小(1,3,320,192) iDetection

    # 更新模型
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0兼容性
        if isinstance(m, models.common.Conv):  # 分配导出友好的激活函数
            if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()
            elif isinstance(m.act, nn.SiLU):
                m.act = SiLU()
        # elif isinstance(m, models.yolo.Detect):
        #     m.forward = m.forward_export  # 分配forward（可选）
    model.model[-1].export = not opt.grid  # 设置Detect()层网格导出
    for _ in range(2):
        y = model(img)  # 干运行
    print(f"\n{colorstr('PyTorch:')} starting from {opt.weights} ({file_size(opt.weights):.1f} MB)")

    # TorchScript导出 -----------------------------------------------------------------------------------------------
    prefix = colorstr('TorchScript:')
    try:
        print(f'\n{prefix} starting export with torch {torch.__version__}...')
        f = opt.weights.replace('.pt', '.torchscript.pt')  # 文件名
        ts = torch.jit.trace(model, img, strict=False)
        ts = optimize_for_mobile(ts)  # https://pytorch.org/tutorials/recipes/script_optimized.html
        ts.save(f)
        print(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
    except Exception as e:
        print(f'{prefix} export failure: {e}')

    # ONNX导出 ------------------------------------------------------------------------------------------------------
    prefix = colorstr('ONNX:')
    try:
        import onnx

        print(f'{prefix} starting export with onnx {onnx.__version__}...')
        f = opt.weights.replace('.pt', '.onnx')  # 文件名
        torch.onnx.export(model, img, f, verbose=False, opset_version=12, input_names=['images'],
                          dynamic_axes={'images': {0: 'batch', 2: 'height', 3: 'width'},  # size(1,3,640,640)
                                        'output': {0: 'batch', 2: 'y', 3: 'x'}} if opt.dynamic else None)

        # 检查
        model_onnx = onnx.load(f)  # 加载onnx模型
        onnx.checker.check_model(model_onnx)  # 检查onnx模型
        # print(onnx.helper.printable_graph(model_onnx.graph))  # 打印

        # 简化
        if opt.simplify:
            try:
                check_requirements(['onnx-simplifier'])
                import onnxsim

                print(f'{prefix} simplifying with onnx-simplifier {onnxsim.__version__}...')
                model_onnx, check = onnxsim.simplify(model_onnx,
                                                     dynamic_input_shape=opt.dynamic,
                                                     input_shapes={'images': list(img.shape)} if opt.dynamic else None)
                assert check, 'assert check failed'
                onnx.save(model_onnx, f)
            except Exception as e:
                print(f'{prefix} simplifier failure: {e}')
        print(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
    except Exception as e:
        print(f'{prefix} export failure: {e}')

    # CoreML导出 ----------------------------------------------------------------------------------------------------
    prefix = colorstr('CoreML:')
    try:
        import coremltools as ct

        print(f'{prefix} starting export with coremltools {ct.__version__}...')
        # 将模型从torchscript转换并应用像素缩放，如detect.py
        model = ct.convert(ts, inputs=[ct.ImageType(name='image', shape=img.shape, scale=1 / 255.0, bias=[0, 0, 0])])
        f = opt.weights.replace('.pt', '.mlmodel')  # 文件名
        model.save(f)
        print(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
    except Exception as e:
        print(f'{prefix} export failure: {e}')

    # 完成
    print(f'\nExport complete ({time.time() - t:.2f}s). Visualize with https://github.com/lutzroeder/netron.')
