import argparse

import yaml

from wandb_utils import WandbLogger

WANDB_ARTIFACT_PREFIX = 'wandb-artifact://'  # 定义Wandb数据集工件的前缀


def create_dataset_artifact(opt):  # 定义创建数据集工件的函数
    with open(opt.data) as f:  # 打开并加载数据配置文件
        data = yaml.safe_load(f)  # 数据字典
    logger = WandbLogger(opt, '', None, data, job_type='Dataset Creation')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()  # 创建命令行参数解析器
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='data.yaml path')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--project', type=str, default='YOLOv5', help='name of W&B Project')
    opt = parser.parse_args()  # 解析命令行参数
    opt.resume = False  # 明确禁止数据集上传任务的恢复检查
    # 调用函数创建数据集工件
    create_dataset_artifact(opt)
