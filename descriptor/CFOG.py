import torch
import numpy as np
#import matlab.engine
import torch.nn.functional as F
from PIL import Image
import scipy.io as sio
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def denseCFOG(image):
    """
    计算输入图像的密集CFOG（Centered Fast Oriented Gradient）描述符。

    参数:
        image (torch.Tensor): 输入图像张量，可以是2D、3D或4D张量。
                              2D: (H, W)
                              3D: (1, H, W)
                              4D: (batch_size, 1, H, W)

    返回:
        des_tensor (torch.Tensor): 计算得到的CFOG描述符张量，形状为(batch_size, 9, H, W)。

    说明:
        该函数根据输入图像的维度，调用相应的处理函数denseCFOG2D。
        对于批量图像，逐个计算每个图像的CFOG描述符，并将结果组合成一个张量。
    """
    s_im = image.shape
    if len(s_im) == 2:
        # 输入为2D图像，直接调用denseCFOG2D计算CFOG描述符
        des_tensor = denseCFOG2D(image)
    elif len(s_im) == 3:
        # 输入为3D图像，去掉通道维度后调用denseCFOG2D
        assert s_im[1] == 1  # 确保通道数为1
        image = image.squeeze(0)  # 去掉通道维度
        des_tensor = denseCFOG2D(image)
    elif len(s_im) == 4:
        # 输入为4D图像（批量图像），逐个计算每个图像的CFOG描述符
        batchSize = s_im[0]  # 获取批量大小
        assert s_im[1] == 1  # 确保通道数为1
        des_tensor = torch.zeros(batchSize, 9, s_im[2], s_im[3]).to(device)  # 初始化描述符张量
        for b in range(batchSize):
            des_tensor[b] = denseCFOG2D(image[b].squeeze(0))  # 逐个计算CFOG描述符
    else:
        # 输入维度不支持，返回0
        des_tensor = 0
    return des_tensor

def denseCFOG2D(image):
    """
    计算单个2D图像的CFOG描述符，调用MATLAB实现的CFOG算法。

    参数:
        image (torch.Tensor): 输入的2D图像张量，形状为(H, W)。

    返回:
        des_tensor (torch.Tensor): 计算得到的CFOG描述符张量，形状为(9, H, W)。

    说明:
        该函数先将输入图像转换为MATLAB支持的格式，然后调用MATLAB引擎运行CFOG算法，
        最后将结果转换回PyTorch张量。
    """
    # 启动MATLAB引擎
    eng = matlab.engine.start_matlab()
    # 切换到CFOG算法所在的目录
    eng.cd('./descriptor', nargout=0)
    # 将输入图像张量转换为numpy数组，并缩放到0-255范围，转换为uint8类型
    im_np = np.array(image.detach().cpu())
    im_np = (im_np * 255).astype(np.uint8)
    # 将numpy数组转换为MATLAB支持的uint8格式
    im_matlab = matlab.uint8(im_np.tolist())
    # 调用MATLAB实现的CFOG算法
    des_matlab = eng.CFOG_matlab(im_matlab)
    # 将MATLAB返回的结果转换为numpy数组
    des_np = np.array(des_matlab)
    # 将numpy数组转换为PyTorch张量，并调整维度顺序
    des_tensor = torch.tensor(des_np, dtype=torch.float32).to(device).permute(2, 0, 1)
    # 关闭MATLAB引擎
    eng.exit()
    return des_tensor
