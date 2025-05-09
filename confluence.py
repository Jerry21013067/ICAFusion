"""
Source: Confluence
Methods
a) assign_boxes_to_classes
b) normalise_coordinates
c) confluence_nms - returns maxima scoring box, removes false positives using confluence - efficient
d) confluence - returns most confluent box, removes false positives using confluence - less efficient but better box
"""

from collections import defaultdict  # 用于创建默认字典
import numpy as np

def assign_boxes_to_classes(bounding_boxes, classes, scores):
    """
    将边界框分配到对应的类别。
    参数：
       bounding_boxes: 边界框列表，格式为 (x1, y1, x2, y2)
       classes: 类别标识符列表（整数值，例如1表示人）
       scores: 类别置信度分数列表（范围为0.0到1.0）
    返回：
       boxes_to_classes: defaultdict(list)，包含类别到边界框和置信度分数的映射
    """
    boxes_to_classes = defaultdict(list)  # 创建默认字典
    for each_box, each_class, each_score in zip(bounding_boxes, classes, scores):  # 遍历边界框、类别和分数
        if each_score >= 0.05:  # 如果置信度分数大于或等于0.05
            boxes_to_classes[each_class].append(np.array([each_box[0], each_box[1], each_box[2], each_box[3], each_score]))  # 将边界框和分数添加到对应类别的列表中
    return boxes_to_classes  # 返回类别到边界框的映射

def normalise_coordinates(x1, y1, x2, y2,min_x,max_x,min_y,max_y):
    """
    归一化边界框坐标。
    参数：
       x1, y1, x2, y2: 需要归一化的边界框坐标
       min_x, max_x, min_y, max_y: 边界框坐标的最小值和最大值（范围为0到1）
    返回：
       归一化后的边界框坐标（范围为0到1）
    """
    x1, y1, x2, y2 = (x1-min_x)/(max_x-min_x), (y1-min_y)/(max_y-min_y), (x2-min_x)/(max_x-min_x), (y2-min_y)/(max_y-min_y)
    return x1, y1, x2, y2

def confluence_nms(bounding_boxes,scores,classes,confluence_thr,gaussian,score_thr=0.05,sigma=0.5):  
    """
    Confluence NMS：通过Confluence算法去除假正例，返回得分最高的边界框。
    参数：
       bounding_boxes: 边界框列表，格式为 (x1, y1, x2, y2)
       classes: 类别标识符列表（整数值，例如1表示人）
       scores: 类别置信度分数列表（范围为0.0到1.0）
       confluence_thr: Confluence阈值，范围为0到2，最佳值为0.5到0.8
       gaussian: 布尔值，是否使用高斯衰减来降低次优边界框的置信度分数（设置为False时，将抑制次优边界框）
       score_thr: 类别置信度分数阈值
       sigma: 高斯衰减中使用的参数，较小的值会导致更严格的衰减
    返回：
       output: 字典，将类别标识符映射到最终保留的边界框（及其对应的置信度分数）
    """
    class_mapping = assign_boxes_to_classes(bounding_boxes, classes, scores)  # 将边界框分配到类别
    output = {}  # 初始化输出字典
    for each_class in class_mapping:  # 遍历每个类别
        dets = np.array(class_mapping[each_class])  # 获取当前类别的边界框和分数
        retain = []  # 初始化保留的边界框列表
        while dets.size > 0:  # 当还有边界框时
            max_idx = np.argmax(dets[:, 4], axis=0)  # 找到置信度最高的边界框索引
            dets[[0, max_idx], :] = dets[[max_idx, 0], :]  # 将置信度最高的边界框移到第一位
            retain.append(dets[0, :])  # 将置信度最高的边界框添加到保留列表中
            x1, y1, x2, y2 = dets[0, 0], dets[0, 1], dets[0, 2], dets[0, 3]  # 获取置信度最高的边界框坐标

            # 计算边界框的最小和最大坐标
            min_x = np.minimum(x1, dets[1:, 0])
            min_y = np.minimum(y1, dets[1:, 1])
            max_x = np.maximum(x2, dets[1:, 2])
            max_y = np.maximum(y2, dets[1:, 3])

            # 归一化边界框坐标
            x1, y1, x2, y2 = normalise_coordinates(x1, y1, x2, y2, min_x, max_x, min_y, max_y)
            xx1, yy1, xx2, yy2 = normalise_coordinates(dets[1:, 0], dets[1:, 1], dets[1:, 2], dets[1:, 3], min_x, max_x, min_y, max_y)

            # 计算曼哈顿距离
            md_x1, md_x2, md_y1, md_y2 = abs(x1 - xx1), abs(x2 - xx2), abs(y1 - yy1), abs(y2 - yy2)
            manhattan_distance = (md_x1 + md_x2 + md_y1 + md_y2)

            # 初始化权重
            weights = np.ones_like(manhattan_distance)

            # 如果启用高斯衰减
            if (gaussian == True):
                gaussian_weights = np.exp(-((1 - manhattan_distance) * (1 - manhattan_distance)) / sigma)
                weights[manhattan_distance <= confluence_thr] = gaussian_weights[manhattan_distance <= confluence_thr]
            else:
                weights[manhattan_distance <= confluence_thr] = manhattan_distance[manhattan_distance <= confluence_thr]

            # 更新边界框的置信度分数
            dets[1:, 4] *= weights
            to_reprocess = np.where(dets[1:, 4] >= score_thr)[0]  # 找到置信度分数高于阈值的边界框
            dets = dets[to_reprocess + 1, :]  # 保留这些边界框
        output[each_class] = retain  # 将保留的边界框添加到输出字典中

    return output  # 返回最终结果

def confluence(bounding_boxes, scores, classes, confluence_thr, gaussian, score_thr=0.05, sigma=0.5):
    """
    Confluence：通过Confluence算法去除假正例，返回最一致的边界框。
    参数：
       bounding_boxes: 边界框列表，格式为 (x1, y1, x2, y2)
       classes: 类别标识符列表（整数值，例如1表示人）
       scores: 类别置信度分数列表（范围为0.0到1.0）
       confluence_thr: Confluence阈值，范围为0到2，最佳值为0.5到0.8
       gaussian: 布尔值，是否使用高斯衰减来降低次优边界框的置信度分数（设置为False时，将抑制次优边界框）
       score_thr: 类别置信度分数阈值
       sigma: 高斯衰减中使用的参数，较小的值会导致更严格的衰减
    返回：
       output: 字典，将类别标识符映射到最终保留的边界框（及其对应的置信度分数）
    """
    class_mapping = assign_boxes_to_classes(bounding_boxes, classes, scores)  # 将边界框分配到类别
    output = {}  # 初始化输出字典
    for each_class in class_mapping:  # 遍历每个类别
        dets = np.array(class_mapping[each_class])  # 获取当前类别的边界框和分数
        retain = []  # 初始化保留的边界框列表
        while dets.size > 0:  # 当还有边界框时
            confluence_scores, proximities = [], []  # 初始化一致性分数和接近度列表
            while len(confluence_scores) < np.size(dets, 0):  # 遍历所有边界框
                current_box = len(confluence_scores)  # 当前边界框索引

                # 获取当前边界框的坐标和置信度分数
                x1, y1, x2, y2 = dets[current_box, 0], dets[current_box, 1], dets[current_box, 2], dets[current_box, 3]
                confidence_score = dets[current_box, 4]

                # 获取其他边界框的坐标和置信度分数
                xx1, yy1, xx2, yy2, cconf = dets[np.arange(len(dets)) != current_box, 0], dets[np.arange(len(dets)) != current_box, 1], dets[np.arange(len(dets)) != current_box, 2], dets[np.arange(len(dets)) != current_box, 3], dets[np.arange(len(dets)) != current_box, 4]

                # 计算边界框的最小和最大坐标
                min_x, min_y, max_x, max_y = np.minimum(x1, xx1), np.minimum(y1, yy1), np.maximum(x2, xx2), np.maximum(y2, yy2)

                # 归一化边界框坐标
                x1, y1, x2, y2 = normalise_coordinates(x1, y1, x2, y2, min_x, max_x, min_y, max_y)
                xx1, yy1, xx2, yy2 = normalise_coordinates(xx1, yy1, xx2, yy2, min_x, max_x, min_y, max_y)

                # 计算水平和垂直距离
                hd_x1, hd_x2, vd_y1, vd_y2 = abs(x1 - xx1), abs(x2 - xx2), abs(y1 - yy1), abs(y2 - yy2)
                proximity = (hd_x1 + hd_x2 + vd_y1 + vd_y2)  # 计算接近度

                # 初始化接近度和置信度分数
                all_proximities = np.ones_like(proximity)
                cconf_scores = np.zeros_like(cconf)

                # 更新接近度和置信度分数
                all_proximities[proximity <= confluence_thr] = proximity[proximity <= confluence_thr]
                cconf_scores[proximity <= confluence_thr] = cconf[proximity <= confluence_thr]

                # 计算一致性分数
                if(cconf_scores.size>0):
                    confluence_score = np.amax(cconf_scores)
                else:
                    confluence_score = confidence_score

                # 计算平均接近度
                if(all_proximities.size>0):
                    proximity = (sum(all_proximities) / all_proximities.size) * (1 - confidence_score)
                else:
                    proximity = sum(all_proximities) * (1 - confidence_score)

                # 将一致性分数和接近度添加到列表中
                confluence_scores.append(confluence_score)
                proximities.append(proximity)

            # 将一致性分数和接近度添加到边界框数组中
            conf = np.array(confluence_scores)
            prox = np.array(proximities)
            dets_temp = np.concatenate((dets, prox[:, None]), axis=1)
            dets_temp = np.concatenate((dets_temp, conf[:, None]), axis=1)

            # 找到接近度最小的边界框索引
            min_idx = np.argmin(dets_temp[:, 5], axis=0)
            dets[[0, min_idx], :] = dets[[min_idx, 0], :]  # 将最一致的边界框移到第一位
            dets_temp[[0, min_idx], :] = dets_temp[[min_idx, 0], :]  # 更新临时数组
            dets[0, 4] = dets_temp[0, 6]  # 更新置信度分数
            retain.append(dets[0, :])  # 将最一致的边界框添加到保留列表中

            # 获取最一致的边界框坐标
            x1, y1, x2, y2 = dets[0, 0], dets[0, 1], dets[0, 2], dets[0, 3]

            # 计算边界框的最小和最大坐标
            min_x = np.minimum(x1, dets[1:, 0])
            min_y = np.minimum(y1, dets[1:, 1])
            max_x = np.maximum(x2, dets[1:, 2])
            max_y = np.maximum(y2, dets[1:, 3])

            # 归一化边界框坐标
            x1, y1, x2, y2 = normalise_coordinates(x1, y1, x2, y2, min_x, max_x, min_y, max_y)
            xx1, yy1, xx2, yy2 = normalise_coordinates(dets[1:, 0], dets[1:, 1], dets[1:, 2], dets[1:, 3], min_x, max_x, min_y, max_y)

            # 计算曼哈顿距离
            md_x1, md_x2, md_y1, md_y2 = abs(x1 - xx1), abs(x2 - xx2), abs(y1 - yy1), abs(y2 - yy2)
            manhattan_distance = (md_x1 + md_x2 + md_y1 + md_y2)

            # 初始化权重
            weights = np.ones_like(manhattan_distance)

            # 如果启用高斯衰减
            if (gaussian == True):
                gaussian_weights = np.exp(-((1 - manhattan_distance) * (1 - manhattan_distance)) / sigma)
                weights[manhattan_distance <= confluence_thr] = gaussian_weights[manhattan_distance <= confluence_thr]
            else:
                weights[manhattan_distance <= confluence_thr] = manhattan_distance[manhattan_distance <= confluence_thr]

            # 更新边界框的置信度分数
            dets[1:, 4] *= weights
            to_reprocess = np.where(dets[1:, 4] >= score_thr)[0]  # 找到置信度分数高于阈值的边界框
            dets = dets[to_reprocess + 1, :]  # 保留这些边界框
        output[each_class] = retain  # 将保留的边界框添加到输出字典中
    return output  # 返回最终结果
