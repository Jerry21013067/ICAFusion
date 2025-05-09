import cv2
import numpy as np
import torch
import random

# 将边界框坐标从 [x, y, w, h] 转换为 [x1, y1, x2, y2]（其中xy1为左上角，xy2为右下角）
def xywh2xyxy(x):
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # 左上角 x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # 左上角 y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # 右下角 x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # 右下角 y
    return y

# 缩放坐标
def scale_coords_x(img1_shape, coords, img0_shape):
    # 根据图像尺寸变化，调整坐标
    scale_x = img1_shape[1] / img0_shape[1]  # x方向缩放比例
    scale_y = img1_shape[0] / img0_shape[0]  # y方向缩放比例

    coords[:, [0, 2]] /= scale_x  # 缩放x坐标
    coords[:, [1, 3]] /= scale_y  # 缩放y坐标

    clip_coords(coords, img0_shape)  # 裁剪坐标
    return coords


def clip_coords(boxes, img_shape):
    # 将边界框裁剪到图像范围内
    boxes[:, 0] = np.clip(boxes[:, 0], 0, img_shape[1])  # x1
    boxes[:, 1] = np.clip(boxes[:, 1], 0, img_shape[0])  # y1
    boxes[:, 2] = np.clip(boxes[:, 2], 0, img_shape[1])  # x2
    boxes[:, 3] = np.clip(boxes[:, 3], 0, img_shape[0])  # y2
    
def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # 在图像上绘制一个边界框
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # 线/字体粗细
    color = color or [random.randint(0, 255) for _ in range(3)]  # 随机颜色
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))  # 左上角和右下角坐标
    cv2.rectangle(img, c1, c2, color, thickness=tl)  # 绘制矩形框
    if label:
        tf = max(tl - 1, 1)  # 字体粗细
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]  # 获取文本大小
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3  # 文本框右下角坐标
        cv2.rectangle(img, c1, c2, color, -1)  # 填充文本框
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)  # 绘制文本



# Confluence处理
def confluence_process(prediction, conf_thres=0.1, p_thres=0.6):
    """对推理结果执行Confluence操作
         prediction: (bs, anchors*grid*grid, xywh + confidence + classes) , 类型: torch.tensor

    返回:
         检测结果，形状为 nx6 (x1, y1, x2, y2, conf, cls)
    """
    if prediction.dtype is torch.float16:
        prediction = prediction.float()  # 转换为FP32

    nc = prediction[0].shape[1] - 5  # 类别数量
    xc = prediction[..., 4] > conf_thres  # 候选框

    # 设置
    max_det = 300  # 每张图像的最大检测数量
    time_limit = 10.0  # 超时时间
    redundant = True  # 是否需要冗余检测
    multi_label = nc > 1  # 是否允许多标签

    #t = time.time()
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # 遍历每张图像
        x = x[xc[xi]]  # 筛选置信度大于阈值的框

        if not x.shape[0]:
            continue

        x[:, 5:] *= x[:, 4:5]  # 计算置信度
        box = xywh2xyxy(x[:, :4])  # 转换为xyxy格式

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero().t()
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
            
        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # confluence
        dets = x.cpu().numpy()
        i = confluence(dets, nc, p_thres)
        
        output[xi] = x[i]
        #if (time.time() - t) > time_limit:
        #    break  # time limit exceeded

    return output
    
# Confluence算法
def confluence(prediction, class_num, p_thres=0.6):
    """对推理结果执行Confluence操作
         prediction: (n, xyxy + confidence + classID), 类型: numpy.array

    返回:
         检测结果的索引
    """
    index = np.arange(0, len(prediction), 1).reshape(-1,1)
    infos = np.concatenate((prediction, index), 1)
     
    keep = []
        
    for c in range(class_num):       
        pcs = infos[infos[:, 5] == c]      
           
        while (len(pcs)):   
                       
            n = len(pcs)       
            xs = pcs[:, [0, 2]]
            ys = pcs[:, [1, 3]]             
            ps = []
            
            # 遍历 pcs，计算每一个box 和其余 box 的 p 值，然后聚类成簇，再根据 wp 挑出 best
            confluence_min = 10000
            best = None
            for i, pc in enumerate(pcs):
                if (n == 1): 
                    best = 0
                    break

                index_other = [j for j in range(n) if j!= i]
                x_t = xs[i]
                x_t = np.tile(x_t, (n-1, 1))
                x_other = xs[index_other]
                x_all = np.concatenate((x_t, x_other), 1)

                y_t = ys[i]
                y_t = np.tile(y_t, (n-1, 1))
                y_other = ys[index_other]
                y_all = np.concatenate((y_t, y_other), 1)                  

                xmin = x_all.min(1).reshape(-1, 1)
                xmax = x_all.max(1).reshape(-1, 1)
                ymin = y_all.min(1).reshape(-1, 1)
                ymax = y_all.max(1).reshape(-1, 1)            
                               
                x_all = (x_all - xmin)/(xmax - xmin)
                y_all = (y_all - ymin)/(ymax - ymin)

                p = abs(x_all[:,0] - x_all[:,2]) + abs(x_all[:,1] - x_all[:,3]) + \
                    abs(y_all[:,0] - y_all[:,2]) + abs(y_all[:,1] - y_all[:,3])
              
                ps.append(p)
                wp = p / pc[4]
                wp = wp[p < 2]
                
                if (len(wp) == 0):
                    value = 0
                else:
                    value = wp.min()
                if (value < confluence_min):
                    confluence_min = value
                    best = i        

            keep.append(int(pcs[best][6])) 
            if (len(ps) > 0):               
                p = ps[best]
                index_ = np.where(p < p_thres)[0]
                index_ = [i if i < best else i +1 for i in index_]
            else:
                index_ = []
            index_eff = [j for j in range(n) if (j != best and j not in index_)]            
            pcs = pcs[index_eff]
            
    keep = np.unique(keep)
    return keep
     
    
  
# 测试Confluence
def test():
    
    colors = [[0, 0, 255], [0, 255, 0], [255, 0, 0]]
    
    img = cv2.imread('./data/test.png')   
    img = cv2.resize(img, (1080, 720))
    prediction = np.loadtxt('./data/prediction.txt')
    
    nc = 3
    wp_thres = 0.6
    i = confluence(prediction, nc, wp_thres)    
    print(i)    
    print(np.shape((prediction)))
    output = prediction[i]
    
    if output is not None and len(output):
        # Rescale boxes from img_size to im0 size
        output[:, :4] = scale_coords_x([416, 416], output[:, :4], img.shape).round()

        # Write results
        for *xyxy, conf, cls in output:
            label = '%s %.2f' % (str(cls), conf)
            plot_one_box(xyxy, img, label=label, color=colors[int(cls)])
            
    cv2.imshow("xx", img)
    cv2.waitKey(-1) 
    
    
if __name__ == '__main__':
    
    test()
