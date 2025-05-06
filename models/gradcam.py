import time
import torch
import torch.nn.functional as F


def find_yolo_layer(model, layer_name):
    """Find yolov5 layer to calculate GradCAM and GradCAM++

    Args:
        model: yolov5 model.
        layer_name (str): the name of layer with its hierarchical information.

    Return:
        target_layer: found layer
    """
    hierarchy = layer_name.split('_')
    target_layer = model.model._modules[hierarchy[0]]

    for h in hierarchy[1:]:
        target_layer = target_layer._modules[h]
    return target_layer


class YOLOV5GradCAM:
    """
    YOLOv5模型的GradCAM和GradCAM++实现类
    """
    def __init__(self, model, layer_name, img_size=(640, 640)):
        """
        初始化YOLOV5GradCAM类

        参数：
            model: YOLOv5模型。
            layer_name: 要计算GradCAM的目标层名称。
            img_size: 输入图像的大小，默认为(640, 640)。
        """
        self.model = model
        self.gradients = dict()
        self.activations = dict()

        def backward_hook(module, grad_input, grad_output):
            """
            反向传播钩子函数，用于捕获梯度值
            """
            self.gradients['value'] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            """
            前向传播钩子函数，用于捕获激活值
            """
            self.activations['value'] = output
            return None

        target_layer = find_yolo_layer(self.model, layer_name)
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

        device = 'cuda' if next(self.model.model.parameters()).is_cuda else 'cpu'
        self.model(torch.zeros(1, 3, *img_size, device=device), torch.zeros(1, 3, *img_size, device=device))
        print('[INFO] saliency_map size :', self.activations['value'].shape[2:])

    def forward(self, img_vis, img_ir, class_idx=True):
        """
        前向传播函数，计算GradCAM和GradCAM++

        参数：
            img_vis: 可见光图像，形状为(1, 3, H, W)。
            img_ir: 红外图像，形状为(1, 3, H, W)。
            class_idx: 是否使用类别索引，默认为True。

        返回：
            saliency_maps: 与输入图像相同空间维度的显著性图列表。
            logits: 模型输出。
            preds: 目标预测结果。
        """
        saliency_maps = []
        b, c, h, w = img_vis.size()
        tic = time.time()
        preds, logits = self.model(img_vis, img_ir)
        print("[INFO] model-forward took: ", round(time.time() - tic, 4), 'seconds')
        for logit, cls, cls_name, conf in zip(logits[0], preds[1][0], preds[2][0], preds[3][0]):
            if class_idx:
                score = logit[cls]
            else:
                score = logit.max()
            self.model.zero_grad()
            tic = time.time()
            score.backward(retain_graph=True)
            print(f"[INFO] {cls_name}, model-backward took: ", round(time.time() - tic, 4), 'seconds')
            gradients = self.gradients['value']
            activations = self.activations['value']
            b, k, u, v = gradients.size()
            alpha = gradients.view(b, k, -1).mean(2)
            weights = alpha.view(b, k, 1, 1)
            saliency_map = (weights * activations).sum(1, keepdim=True)
            saliency_map = F.relu(saliency_map)
            saliency_map = F.interpolate(input=saliency_map, size=(h, w), mode='bilinear', align_corners=False)
            saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
            saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data
            saliency_maps.append(saliency_map)
        return saliency_maps, logits, preds

    def __call__(self, img_vis, img_ir):
        """
        调用YOLOV5GradCAM类的前向传播函数

        参数：
            img_vis: 可见光图像，形状为(1, 3, H, W)。
            img_ir: 红外图像，形状为(1, 3, H, W)。

        返回：
            saliency_maps: 与输入图像相同空间维度的显著性图列表。
            logits: 模型输出。
            preds: 目标预测结果。
        """
        return self.forward(img_vis, img_ir)
