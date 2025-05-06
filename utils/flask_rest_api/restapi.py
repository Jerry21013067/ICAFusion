"""
运行一个REST API，暴露YOLOv5s目标检测模型
"""
import argparse  # 导入argparse模块，用于解析命令行参数
import io  # 导入io模块，用于处理字节流

import torch
from PIL import Image  # 导入Pillow库，用于图像处理
from flask import Flask, request  # 导入Flask框架和请求处理模块

# 创建Flask应用实例
app = Flask(__name__)

# 定义目标检测的API路由
DETECTION_URL = "/v1/object-detection/yolov5s"

# 定义处理POST请求的函数
@app.route(DETECTION_URL, methods=["POST"])
def predict():
    if not request.method == "POST":  # 检查请求方法是否为POST
        return

    if request.files.get("image"):  # 检查是否有上传的图像文件
        image_file = request.files["image"]  # 获取上传的图像文件
        image_bytes = image_file.read()  # 读取图像文件的字节数据

        img = Image.open(io.BytesIO(image_bytes))  # 使用Pillow库从字节数据中打开图像

        results = model(img, size=640)  # size=640表示将图像缩放到640x640像素（可调整为320以加快推理速度）
        return results.pandas().xyxy[0].to_json(orient="records")  # 将检测结果转换为JSON格式并返回


if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="Flask API exposing YOLOv5 model")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()  # 解析命令行参数
    # 加载YOLOv5s模型，force_reload=True表示强制重新加载模型
    model = torch.hub.load("ultralytics/yolov5", "yolov5s", force_reload=True)
    # 启动Flask应用，监听所有IP地址，端口号由命令行参数指定
    app.run(host="0.0.0.0", port=args.port)  # debug=True会导致应用在修改代码后自动重启
