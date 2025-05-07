import argparse  # 导入argparse模块，用于解析命令行参数
import logging
import math
import os
import random
import time
from copy import deepcopy  # 导入deepcopy，用于深度复制对象
from pathlib import Path
from threading import Thread

import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F  # 导入PyTorch函数式API
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler  # 导入学习率调度器模块
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP  # 导入分布式数据并行模块
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm  # 导入tqdm模块，用于显示进度条

import test  # 导入test模块，用于计算mAP
from models.experimental import attempt_load
from models.yolo_test import Model  # 导入YOLOv5模型类
from utils.autoanchor import check_anchors
from utils.datasets import create_dataloader_rgb_ir
from utils.general import logger, labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    fitness, strip_optimizer, get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    check_requirements, print_mutation, set_logging, one_cycle, colorstr
from utils.google_utils import attempt_download
from utils.loss import ComputeLoss  # 导入ComputeLoss类，用于计算损失
from utils.plots import plot_images, plot_labels, plot_results, plot_evolution
from utils.torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first, is_parallel
from utils.wandb_logging.wandb_utils import WandbLogger, check_wandb_resume

from utils.datasets import RandomSampler
import global_var  # 导入global_var模块，用于全局变量管理


def train_rgb_ir(hyp, opt, device, tb_writer=None):
    """
    训练RGB和红外图像的YOLOv5模型。

    参数：
        hyp (dict): 超参数字典。
        opt (argparse.Namespace): 命令行参数。
        device (torch.device): 训练设备。
        tb_writer (SummaryWriter): TensorBoard日志记录器。
    """
    os.environ["WANDB_MODE"] = "offline"  # 设置WandB模式为离线
    logger.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))  # 打印超参数
    save_dir, epochs, batch_size, total_batch_size, weights, rank = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.global_rank  # 解析参数

    # 创建保存目录
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)  # 创建权重目录
    last = wdir / 'last.pt'  # 最新模型路径
    best = wdir / 'best.pt'  # 最佳模型路径
    results_file = save_dir / 'results.txt'  # 结果文件路径

    # 保存运行设置
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.safe_dump(hyp, f, sort_keys=False)  # 保存超参数
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=False)  # 保存命令行参数

    # 配置
    plots = not opt.evolve  # 是否绘制图像
    cuda = device.type != 'cpu'  # 是否使用CUDA
    init_seeds(seed=1 + rank, deterministic=True)  # 初始化随机种子
    with open(opt.data) as f:
        data_dict = yaml.safe_load(f)  # 加载数据集配置
    is_coco = opt.data.endswith('coco.yaml')  # 是否为COCO数据集

    # 日志记录
    loggers = {'wandb': None}  # 日志记录器字典
    if rank in [-1, 0]:  # 如果是主进程
        opt.hyp = hyp  # 添加超参数
        run_id = torch.load(weights).get('wandb_id') if weights.endswith('.pt') and os.path.isfile(weights) else None  # 获取WandB运行ID
        wandb_logger = WandbLogger(opt, save_dir.stem, run_id, data_dict)  # 初始化WandB日志记录器
        loggers['wandb'] = wandb_logger.wandb  # 添加WandB日志记录器
        data_dict = wandb_logger.data_dict  # 更新数据集配置
        if wandb_logger.wandb:
            weights, epochs, hyp = opt.weights, opt.epochs, opt.hyp  # WandB日志记录器可能更新权重、轮数和超参数

    nc = 1 if opt.single_cls else int(data_dict['nc'])  # 类别数量
    names = ['item'] if opt.single_cls and len(data_dict['names']) != 1 else data_dict['names']  # 类别名称
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # 检查类别数量

    # 模型
    pretrained = weights.endswith('.pt')  # 是否为预训练模型
    if pretrained:
        with torch_distributed_zero_first(rank):
            attempt_download(weights)  # 下载权重文件（如果本地不存在）
        ckpt = torch.load(weights, map_location=device)  # 加载权重
        model = Model(opt.cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # 创建模型
        exclude = ['anchor'] if (opt.cfg or hyp.get('anchors')) and not opt.resume else []  # 排除键
        state_dict = ckpt['model'].float().state_dict()  # 转换为FP32
        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # 交集
        new_state_dict = state_dict
        for key in list(state_dict.keys()):
            new_state_dict[key[:6] + str(int(key[6])+10) + key[7:]] = state_dict[key]  # 修改键名
        model.load_state_dict(new_state_dict, strict=False)  # 加载权重
        logger.info('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weights))  # 打印信息
    else:
        model = Model(opt.cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # 创建模型

    with torch_distributed_zero_first(rank):
        check_dataset(data_dict)  # 检查数据集
    train_path_rgb = data_dict['train_rgb']  # RGB训练路径
    test_path_rgb = data_dict['val_rgb']  # RGB测试路径
    train_path_ir = data_dict['train_ir']  # 红外训练路径
    test_path_ir = data_dict['val_ir']  # 红外测试路径
    labels_path = data_dict['path'] + '/labels/test'  # 标签路径
    labels_list = os.listdir(labels_path)  # 获取标签列表
    labels_list.sort()  # 排序

    # 冻结参数
    freeze = []  # 要冻结的参数名称
    for k, v in model.named_parameters():
        v.requires_grad = True  # 训练所有层
        if any(x in k for x in freeze):
            print('freezing %s' % k)
            v.requires_grad = False  # 冻结参数

    # 优化器
    nbs = 64  # 名义批次大小
    accumulate = max(round(nbs / total_batch_size), 1)  # 累积损失
    hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # 缩放权重衰减
    logger.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    pg0, pg1, pg2 = [], [], []  # 优化器参数组
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # 偏差
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  # 不使用衰减
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # 使用衰减

    if opt.adam:
        optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # 使用Adam优化器
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)  # 使用SGD优化器

    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # 添加pg1
    optimizer.add_param_group({'params': pg2})  # 添加pg2
    logger.info(f"{colorstr('optimizer:')} {type(optimizer).__name__} with parameter groups "
                f"{len(pg0)} weight, {len(pg1)} weight (no decay), {len(pg2)} bias")
    del pg0, pg1, pg2  # 删除临时变量

    if opt.linear_lr:
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']  # 线性学习率调度
    else:
        lf = one_cycle(1, hyp['lrf'], epochs)  # 余弦学习率调度
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # 创建学习率调度器

    # EMA
    ema = ModelEMA(model) if rank in [-1, 0] else None

    # 恢复训练
    start_epoch, best_fitness = 0, 0.0  # 起始轮数和最佳拟合度
    if pretrained:
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])  # 加载优化器状态
            best_fitness = ckpt['best_fitness']

        if ema and ckpt.get('ema'):
            ema.ema.load_state_dict(ckpt['ema'].float().state_dict())  # 加载EMA状态
            ema.updates = ckpt['updates']

        if ckpt.get('training_results') is not None:
            results_file.write_text(ckpt['training_results'])  # 写入结果文件

        start_epoch = ckpt['epoch'] + 1
        if opt.resume:
            assert start_epoch > 0, '%s training to %g epochs is finished, nothing to resume.' % (weights, epochs)
        if epochs < start_epoch:
            logger.info('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                        (weights, ckpt['epoch'], epochs))
            epochs += ckpt['epoch']  # 细调额外的轮数

        del ckpt, state_dict  # 删除临时变量

    # 图像大小
    gs = max(int(model.stride.max()), 32)  # 网格大小（最大步长）
    nl = model.model[-1].nl  # 检测层数量
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # 验证图像大小

    # DP模式
    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)  # 数据并行

    # SyncBatchNorm
    if opt.sync_bn and cuda and rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)  # 同步批量归一化
        logger.info('Using SyncBatchNorm()')

    # 训练加载器
    dataloader, dataset = create_dataloader_rgb_ir(train_path_rgb, train_path_ir, imgsz, batch_size, gs, opt,
                                                   hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect, rank=rank,
                                                   world_size=opt.world_size, workers=opt.workers,
                                                   image_weights=opt.image_weights, quad=opt.quad, prefix=colorstr('train: '))
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # 最大标签类别
    nb = len(dataloader)  # 批次数量
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, opt.data, nc - 1)

    # 主进程
    if rank in [-1, 0]:
        testloader, testdata = create_dataloader_rgb_ir(test_path_rgb, test_path_ir, imgsz_test, 1, gs, opt,
                                                        hyp=hyp, cache=opt.cache_images and not opt.notest, rect=True,
                                                        rank=-1, world_size=opt.world_size, workers=opt.workers,
                                                        pad=0.5, prefix=colorstr('val: '))

        if not opt.resume:
            labels = np.concatenate(dataset.labels, 0)
            c = torch.tensor(labels[:, 0])  # 类别
            if plots:
                plot_labels(labels, names, save_dir, loggers)
                if tb_writer:
                    tb_writer.add_histogram('classes', c, 0)

            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
            model.half().float()  # 预减小锚点精度

    # DDP模式
    if cuda and rank != -1:
        model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank,
                    find_unused_parameters=any(isinstance(layer, nn.MultiheadAttention) for layer in model.modules()))

    # 模型参数
    hyp['box'] *= 3. / nl  # 缩放至层数
    hyp['cls'] *= nc / 80. * 3. / nl  # 缩放至类别和层数
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # 缩放至图像大小和层数
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc  # 添加类别数量至模型
    model.hyp = hyp  # 添加超参数至模型
    model.gr = 1.0  # iou损失比例
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # 添加类别权重
    model.names = names

    # 开始训练
    t0 = time.time()
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)  # 预热迭代次数
    maps = np.zeros(nc)  # 每类别的mAP
    MRresult = 0.0
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1
    scaler = amp.GradScaler(enabled=cuda)
    compute_loss = ComputeLoss(model)  # 初始化损失类
    logger.info(f'Image sizes {imgsz} train, {imgsz_test} test\n'
                f'Using {dataloader.num_workers} dataloader workers\n'
                f'Logging results to {save_dir}\n'
                f'Starting training for {epochs} epochs...')

    for epoch in range(start_epoch, epochs):  # 遍历轮数
        model.train()

        # 更新图像权重（可选）
        if opt.image_weights:
            if rank in [-1, 0]:
                cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # 类别权重
                iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # 图像权重
                dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # 随机加权索引
            if rank != -1:
                indices = (torch.tensor(dataset.indices) if rank == 0 else torch.zeros(dataset.n)).int()
                dist.broadcast(indices, 0)
                if rank != 0:
                    dataset.indices = indices.cpu().numpy()

        # 更新马赛克边界
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        mloss = torch.zeros(4, device=device)  # 平均损失
        if rank != -1:
            dataloader.sampler.set_epoch(epoch)
        pbar = enumerate(dataloader)
        logger.info(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'rank', 'labels', 'img_size'))
        if rank in [-1, 0]:
            pbar = tqdm(pbar, total=nb)  # 进度条
        optimizer.zero_grad()

        for i, (imgs, targets, paths, _) in pbar:  # 遍历批次
            ni = i + nb * epoch  # 总批次数
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # 转换为浮点数
            imgs_rgb = imgs[:, :3, :, :]
            imgs_ir = imgs[:, 3:, :, :]

            # FQY my code 训练数据可视化
            flage_visual = global_var.get_value('flag_visual_training_dataset')
            if flage_visual:
                from torchvision import transforms
                unloader = transforms.ToPILImage()
                for num in range(batch_size):
                    image = imgs[num, :3, :, :].cpu().clone()  # clone the tensor
                    image = image.squeeze(0)  # remove the fake batch dimension
                    image = unloader(image)
                    image.save('example_%s_%s_%s_color.jpg'%(str(epoch), str(i), str(num)))
                    image = imgs[num, 3:, :, :].cpu().clone()  # clone the tensor
                    image = image.squeeze(0)  # remove the fake batch dimension
                    image = unloader(image)
                    image.save('example_%s_%s_%s_ir.jpg'%(str(epoch), str(i), str(num)))

            # 预热
            if ni <= nw:
                xi = [0, nw]  # x interp
                accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # 多尺度
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs
                sf = sz / max(imgs.shape[2:])
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # 前向传播
            with amp.autocast(enabled=cuda):
                pred = model(imgs_rgb, imgs_ir)  # 前向传播
                loss, loss_items = compute_loss(pred, targets.to(device))  # 计算损失
                if rank != -1:
                    loss *= opt.world_size  # 在DDP模式下平均梯度
                if opt.quad:
                    loss *= 4.

            # 反向传播
            scaler.scale(loss).backward()

            # 优化
            if ni % accumulate == 0:
                scaler.step(optimizer)  # 优化器.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)

            # 打印
            if rank in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)  # 更新平均损失
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s = ('%10s' * 2 + '%10.4g' * 6) % ('%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
                pbar.set_description(s)

                if ni < 3:
                    f1 = save_dir / f'train_batch{ni}_vis.jpg'
                    f2 = save_dir / f'train_batch{ni}_inf.jpg'
                    Thread(target=plot_images, args=(imgs_rgb, targets, paths, f1), daemon=True).start()
                    Thread(target=plot_images, args=(imgs_ir, targets, paths, f2), daemon=True).start()

        # 调度器
        lr = [x['lr'] for x in optimizer.param_groups]  # 获取学习率
        scheduler.step()

        # DDP进程0或单GPU
        if rank in [-1, 0]:
            # mAP
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])
            final_epoch = epoch + 1 == epochs
            if not opt.notest or final_epoch:  # 计算mAP
                wandb_logger.current_epoch = epoch + 1
                results, maps, MRresult, times = test.test(data_dict,
                                                           batch_size=1,
                                                           imgsz=imgsz_test,
                                                           model=ema.ema,
                                                           single_cls=opt.single_cls,
                                                           dataloader=testloader,
                                                           save_dir=save_dir,
                                                           save_txt=True,
                                                           save_conf=True,
                                                           verbose=nc < 50 and final_epoch,
                                                           plots=plots and final_epoch,
                                                           wandb_logger=wandb_logger,
                                                           compute_loss=compute_loss,
                                                           is_coco=is_coco,
                                                           labels_list=labels_list,
                                                           )

            # 日志记录
            keys = ['train/box_loss', 'train/obj_loss', 'train/cls_loss', 'train/rank_loss',  # 训练损失
                    'TP', 'FP', 'FN', 'F1', 'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',  # 指标
                    'val/box_loss', 'val/obj_loss', 'val/cls_loss', 'val/rank_loss',  # 验证损失
                    'x/lr0', 'x/lr1', 'x/lr2',  # 学习率
                    'MR_all', 'MR_day', 'MR_night', 'MR_near', 'MR_medium', 'MR_far', 'MR_none', 'MR_partial', 'MR_heavy', 'Recall_all'  # MR
                    ]
            vals = list(mloss) + list(results) + lr + MRresult
            dicts = {k: v for k, v in zip(keys, vals)}  # 字典
            file = save_dir / 'results.csv'
            n = len(dicts) + 1  # 列数
            s = '' if file.exists() else (('%s,' * n % tuple(['epoch'] + keys)).rstrip(',') + '\n')  # 添加标题
            with open(file, 'a') as f:
                f.write(s + ('%g,' * n % tuple([epoch] + vals)).rstrip(',') + '\n')

            # 更新最佳mAP
            fi = fitness(np.array(results).reshape(1, -1))  # 加权组合[P, R, mAP@.5, mAP@.5-.95]
            if fi > best_fitness:
                best_fitness = fi
            #wandb_logger.end_epoch(best_result=best_fitness == fi)

            # 保存模型
            if (not opt.nosave) or (final_epoch and not opt.evolve):
                ckpt = {'epoch': epoch,
                        'best_fitness': best_fitness,
                        'model': deepcopy(model.module if is_parallel(model) else model).half(),
                        'ema': deepcopy(ema.ema).half(),
                        'updates': ema.updates,
                        'optimizer': optimizer.state_dict(),
                        'wandb_id': wandb_logger.wandb_run.id if wandb_logger.wandb else None}

                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if wandb_logger.wandb:
                    if ((epoch + 1) % opt.save_period == 0 and not final_epoch) and opt.save_period != -1:
                        wandb_logger.log_model(
                            last.parent, opt, epoch, fi, best_model=best_fitness == fi)
                del ckpt

        # 结束轮次 ----------------------------------------------------------------------------------------------------
    # 结束训练
    t1 = time.time()
    t = t1 - t0
    if rank in [-1, 0]:
        # 绘图
        if plots:
            plot_results(file=save_dir / 'results.csv')  # 保存为results.png
            if wandb_logger.wandb:
                files = ['results.png', 'confusion_matrix.png', *[f'{x}_curve.png' for x in ('F1', 'PR', 'P', 'R')]]
                wandb_logger.log({"Results": [wandb_logger.wandb.Image(str(save_dir / f), caption=f) for f in files
                                              if (save_dir / f).exists()]})
        # 测试最佳模型 best.pt
        logger.info('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
        for m in (last, best) if best.exists() else (last):  # 测试速度和mAP
            results, _, MRresult, _ = test.test(opt.data,
                                                batch_size=1,
                                                imgsz=imgsz_test,
                                                conf_thres=0.001,
                                                iou_thres=0.5,
                                                model=attempt_load(m, device).half(),
                                                single_cls=opt.single_cls,
                                                dataloader=testloader,
                                                save_dir=save_dir,
                                                save_txt=True,
                                                save_conf=True,
                                                save_json=False,
                                                plots=False,
                                                is_coco=is_coco,
                                                labels_list=labels_list,
                                                verbose=nc > 1,
                                                )

        # 剥离优化器
        final = best if best.exists() else last  # 最终模型
        for f in last, best:
            if f.exists():
                strip_optimizer(f)  # 剥离优化器
        if opt.bucket:
            os.system(f'gsutil cp {final} gs://{opt.bucket}/weights')  # 上传
        if wandb_logger.wandb and not opt.evolve:  # 记录剥离后的模型
            wandb_logger.wandb.log_artifact(str(final), type='model',
                                            name='run_' + wandb_logger.wandb_run.id + '_model',
                                            aliases=['last', 'best', 'stripped'])
        wandb_logger.finish_run()
    else:
        dist.destroy_process_group()
    torch.cuda.empty_cache()
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov5l.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='./models/transformer/yolov5l_Transfusion_FLIR.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default='./data/multispectral/FLIR-align-3class.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=8, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--workers', type=int, default=0, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--upload_dataset', action='store_true', help='Upload dataset as W&B artifact table')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval for W&B')
    parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
    parser.add_argument('--artifact_alias', type=str, default="latest", help='version of dataset artifact to be used')
    opt = parser.parse_args()

    #opt.rect = False

    # FQY  Flag for visualizing the paired training imgs
    global_var._init()
    global_var.set_value('flag_visual_training_dataset', False)

    # 设置DDP变量
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    set_logging(opt.global_rank)
    if opt.global_rank in [-1, 0]:
        # check_git_status()
        check_requirements()

    # 恢复训练
    wandb_run = check_wandb_resume(opt)
    if opt.resume and not wandb_run:  # 恢复中断的训练
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # 指定或最近的路径
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        apriori = opt.global_rank, opt.local_rank
        with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
            opt = argparse.Namespace(**yaml.safe_load(f))  # 替换
        opt.cfg, opt.weights, opt.resume, opt.batch_size, opt.global_rank, opt.local_rank = \
            '', ckpt, True, opt.total_batch_size, *apriori  # 重新设置
        logger.info('Resuming training from %s' % ckpt)
    else:
        opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # 检查文件
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # 扩展到2个大小（训练，测试）
        opt.name = 'evolve' if opt.evolve else opt.name
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve))

    # DDP模式
    opt.total_batch_size = opt.batch_size
    device = select_device(opt.device, batch_size=opt.batch_size)
    if opt.local_rank != -1:
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        device = torch.device('cuda', opt.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')  # 分布式后端
        assert opt.batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
        opt.batch_size = opt.total_batch_size // opt.world_size

    # 超参数
    with open(opt.hyp) as f:
        hyp = yaml.safe_load(f)  # 加载超参数

    # 训练
    logger.info(opt)
    if not opt.evolve:
        tb_writer = None  # 初始化日志记录器
        if opt.global_rank in [-1, 0]:
            prefix = colorstr('tensorboard: ')
            logger.info(f"{prefix}Start with 'tensorboard --logdir {opt.project}', view at http://localhost:6006/")
            tb_writer = SummaryWriter(opt.save_dir)  # Tensorboard

            train_rgb_ir(hyp, opt, device, tb_writer)

    # 超参数进化（可选）
    else:
        # 超参数进化元数据（变异比例0-1，下限，上限）
        meta = {'lr0': (1, 1e-5, 1e-1),  # 初始学习率（SGD=1E-2, Adam=1E-3）
                'lrf': (1, 0.01, 1.0),  # 最终OneCycleLR学习率（lr0 * lrf）
                'momentum': (0.3, 0.6, 0.98),  # SGD动量/Adam beta1
                'weight_decay': (1, 0.0, 0.001),  # 优化器权重衰减
                'warmup_epochs': (1, 0.0, 5.0),  # 预热轮次（可以是分数）
                'warmup_momentum': (1, 0.0, 0.95),  # 预热初始动量
                'warmup_bias_lr': (1, 0.0, 0.2),  # 预热初始偏差学习率
                'box': (1, 0.02, 0.2),  # box损失增益
                'cls': (1, 0.2, 4.0),  # cls损失增益
                'cls_pw': (1, 0.5, 2.0),  # cls BCELoss正权重
                'obj': (1, 0.2, 4.0),  # obj损失增益（按像素缩放）
                'obj_pw': (1, 0.5, 2.0),  # obj BCELoss正权重
                'iou_t': (0, 0.1, 0.7),  # IoU训练阈值
                'anchor_t': (1, 2.0, 8.0),  # 锚点倍数阈值
                'anchors': (2, 2.0, 10.0),  # 每个输出网格的锚点数量（0表示忽略）
                'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet默认gamma=1.5)
                'hsv_h': (1, 0.0, 0.1),  # 图像HSV-Hue增强（比例）
                'hsv_s': (1, 0.0, 0.9),  # 图像HSV-Saturation增强（比例）
                'hsv_v': (1, 0.0, 0.9),  # 图像HSV-Value增强（比例）
                'degrees': (1, 0.0, 45.0),  # 图像旋转（+/-度）
                'translate': (1, 0.0, 0.9),  # 图像平移（+/-比例）
                'scale': (1, 0.0, 0.9),  # 图像缩放（+/-增益）
                'shear': (1, 0.0, 10.0),  # 图像剪切（+/-度）
                'perspective': (0, 0.0, 0.001),  # 图像透视（+/-比例），范围0-0.001
                'flipud': (1, 0.0, 1.0),  # 图像上下翻转（概率）
                'fliplr': (0, 0.0, 1.0),  # 图像左右翻转（概率）
                'mosaic': (1, 0.0, 1.0),  # 图像马赛克（概率）
                'mixup': (1, 0.0, 1.0)}  # 图像混合（概率）

        assert opt.local_rank == -1, 'DDP模式未实现用于 --evolve'
        opt.notest, opt.nosave = True, True  # 仅测试/保存最终轮次
        yaml_file = Path(opt.save_dir) / 'hyp_evolved.yaml'  # 保存最佳结果
        if opt.bucket:
            os.system('gsutil cp gs://%s/evolve.txt .' % opt.bucket)  # 下载evolve.txt（如果存在）

        for _ in range(300):  # 生成代数以进化
            if Path('evolve.txt').exists():  # 如果evolve.txt存在：选择最佳超参数并变异
                # 选择父代
                parent = 'single'  # 父代选择方法：'single'或'weighted'
                x = np.loadtxt('evolve.txt', ndmin=2)
                n = min(5, len(x))  # 考虑的前n个结果数量
                x = x[np.argsort(-fitness(x))][:n]  # 前n个突变
                w = fitness(x) - fitness(x).min()  # 权重
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # 随机选择
                    x = x[random.choices(range(n), weights=w)[0]]  # 加权选择
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # 加权组合

                # 变异
                mp, s = 0.8, 0.2  # 变异概率，sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([x[0] for x in meta.values()])  # 增益0-1
                ng = len(meta)
                v = np.ones(ng)
                while all(v == 1):  # 变异直到发生改变（防止重复）
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = float(x[i + 7] * v[i])  # 变异

            # 约束在限制范围内
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  # 下限
                hyp[k] = min(hyp[k], v[2])  # 上限
                hyp[k] = round(hyp[k], 5)  # 有效数字

            # 训练变异
            results = train_rgb_ir(hyp.copy(), opt, device)

            # 写入变异结果
            print_mutation(hyp.copy(), results, yaml_file, opt.bucket)

        # 绘制结果
        plot_evolution(yaml_file)
        print(f'Hyperparameter evolution complete. Best results saved as: {yaml_file}\n'
              f'Command to train a new model with these hyperparameters: $ python train.py --hyp {yaml_file}')