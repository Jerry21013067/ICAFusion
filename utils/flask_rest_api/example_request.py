"""执行测试请求"""
import pprint  # 导入pprint模块，用于美化打印输出

import requests  # 导入requests模块，用于发送HTTP请求

# 定义目标检测的URL地址
DETECTION_URL = "http://localhost:5000/v1/object-detection/yolov5s"
# 定义测试图像的文件名
TEST_IMAGE = "zidane.jpg"

# 打开测试图像文件，并读取其二进制数据
image_data = open(TEST_IMAGE, "rb").read()

# 使用requests.post方法发送POST请求
# 将图像数据作为文件上传，并将返回的JSON数据解析为Python对象
response = requests.post(DETECTION_URL, files={"image": image_data}).json()

# 使用pprint.pprint方法美化打印返回的响应数据
pprint.pprint(response)
