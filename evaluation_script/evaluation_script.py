"""
评估多光谱行人检测基准上的性能

此脚本评估多光谱检测性能。
我们采用了 [cocoapi](https://github.com/cocodataset/cocoapi) 并对其进行了少量修改以适应 KAISTPed 基准。
"""
from collections import defaultdict
import argparse
import copy
import datetime
import json
import matplotlib
import numpy as np
import os
import pdb
import sys
import tempfile
import traceback

# matplotlib.use('Agg')
# from matplotlib.patches import Polygon
import matplotlib.pyplot as plt

from evaluation_script.coco import COCO
from evaluation_script.cocoeval import COCOeval, Params

font = {'size': 22}
matplotlib.rc('font', **font)


class KAISTPedEval(COCOeval):
    """
    KAISTPed 评估类，继承自 COCOeval。
    """
    def __init__(self, kaistGt=None, kaistDt=None, iouType='segm', method='unknown'):
        """
        使用 KAISTPed API 初始化 KAISTPedEval 对象。
        :param kaistGt: 包含真实标注的 KAISTPed API 对象。
        :param kaistDt: 包含检测结果的 KAISTPed API 对象。
        :param iouType: 评估类型，可以是 'segm'（分割）或 'bbox'（边界框）。
        :param method: 检测方法名称。
        """
        super().__init__(kaistGt, kaistDt, iouType)

        self.params = KAISTParams(iouType=iouType)   # 参数
        self.method = method

    def _prepare(self, id_setup):
        """
        根据参数准备用于评估的真实标注和检测结果。
        :param id_setup: 设置 ID。
        :return: None
        """
        p = self.params
        if p.useCats:
            gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
            dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
        else:
            gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds))
            dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds))

        # 设置忽略标志
        for gt in gts:
            gt['ignore'] = gt['ignore'] if 'ignore' in gt else 0
            gbox = gt['bbox']
            gt['ignore'] = 1 \
                if gt['height'] < self.params.HtRng[id_setup][0] \
                or gt['height'] > self.params.HtRng[id_setup][1] \
                or gt['occlusion'] not in self.params.OccRng[id_setup] \
                or gbox[0] < self.params.bndRng[0] \
                or gbox[1] < self.params.bndRng[1] \
                or gbox[0] + gbox[2] > self.params.bndRng[2] \
                or gbox[1] + gbox[3] > self.params.bndRng[3] \
                else gt['ignore']

        self._gts = defaultdict(list)       # 用于评估的真实标注
        self._dts = defaultdict(list)       # 用于评估的检测结果
        for gt in gts:
            self._gts[gt['image_id'], gt['category_id']].append(gt)
        for dt in dts:
            self._dts[dt['image_id'], dt['category_id']].append(dt)

        self.evalImgs = defaultdict(list)   # 每张图片每个类别的评估结果
        self.eval = {}                      # 累积的评估结果

    def evaluate(self, id_setup):
        """
        对给定的图片运行单张图片评估，并将结果存储在 self.evalImgs 中。
        :param id_setup: 设置 ID。
        :return: None
        """
        p = self.params
        # 添加向后兼容性，如果在参数中指定了 useSegm
        if p.useSegm is not None:
            p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
            #print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
        # print('Evaluate annotation type *{}*'.format(p.iouType))
        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params = p

        self._prepare(id_setup)
        # 遍历图片、面积范围、最大检测数量
        catIds = p.catIds if p.useCats else [-1]

        computeIoU = self.computeIoU

        self.ious = {(imgId, catId): computeIoU(imgId, catId)
                     for imgId in p.imgIds for catId in catIds}

        evaluateImg = self.evaluateImg
        maxDet = p.maxDets[-1]
        HtRng = self.params.HtRng[id_setup]
        OccRng = self.params.OccRng[id_setup]
        self.evalImgs = [evaluateImg(imgId, catId, HtRng, OccRng, maxDet)
                         for catId in catIds
                         for imgId in p.imgIds]

        self._paramsEval = copy.deepcopy(self.params)

    def computeIoU(self, imgId, catId):
        """
        计算给定图片和类别的真实标注和检测结果之间的 IoU。
        :param imgId: 图片 ID
        :param catId: 类别 ID
        :return: IoU 矩阵
        """
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return []
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt = dt[0:p.maxDets[-1]]

        if p.iouType == 'segm':
            g = [g['segmentation'] for g in gt]
            d = [d['segmentation'] for d in dt]
        elif p.iouType == 'bbox':
            g = [g['bbox'] for g in gt]
            d = [d['bbox'] for d in dt]
        else:
            raise Exception('unknown iouType for iou computation')

        # 计算每个检测结果和真实标注之间的 IoU
        iscrowd = [int(o['ignore']) for o in gt]
        ious = self.iou(d, g, iscrowd)
        return ious

    def iou(self, dts, gts, pyiscrowd):
        """
        计算检测结果和真实标注之间的 IoU。
        :param dts: 检测结果的边界框。
        :param gts: 真实标注的边界框。
        :param pyiscrowd: 是否为拥挤标注。
        :return: IoU 矩阵
        """
        dts = np.asarray(dts)
        gts = np.asarray(gts)
        pyiscrowd = np.asarray(pyiscrowd)
        ious = np.zeros((len(dts), len(gts)))
        for j, gt in enumerate(gts):
            gx1 = gt[0]
            gy1 = gt[1]
            gx2 = gt[0] + gt[2]
            gy2 = gt[1] + gt[3]
            garea = gt[2] * gt[3]
            for i, dt in enumerate(dts):
                dx1 = dt[0]
                dy1 = dt[1]
                dx2 = dt[0] + dt[2]
                dy2 = dt[1] + dt[3]
                darea = dt[2] * dt[3]

                unionw = min(dx2, gx2) - max(dx1, gx1)
                if unionw <= 0:
                    continue
                unionh = min(dy2, gy2) - max(dy1, gy1)
                if unionh <= 0:
                    continue
                t = unionw * unionh
                if pyiscrowd[j]:
                    unionarea = darea
                else:
                    unionarea = darea + garea - t

                ious[i, j] = float(t) / unionarea
        return ious

    def evaluateImg(self, imgId, catId, hRng, oRng, maxDet):
        """
        对单张图片和类别进行评估。
        :param imgId: 图片 ID
        :param catId: 类别 ID
        :param hRng: 高度范围
        :param oRng: 遮挡范围
        :param maxDet: 最大检测数量
        :return: 评估结果字典
        """
        try:
            p = self.params
            if p.useCats:
                gt = self._gts[imgId, catId]
                dt = self._dts[imgId, catId]
            else:
                gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
                dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
            
            if len(gt) == 0 and len(dt) == 0:
                return None

            for g in gt:
                if g['ignore']:
                    g['_ignore'] = 1
                else:
                    g['_ignore'] = 0

            # 按分数排序检测结果，按忽略标志排序真实标注
            gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
            gt = [gt[i] for i in gtind]
            dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
            dt = [dt[i] for i in dtind[0:maxDet]]

            if len(dt) == 0:
                return None

            # 加载计算好的 IoU
            ious = self.ious[imgId, catId][dtind, :] if len(self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]
            ious = ious[:, gtind]

            T = len(p.iouThrs)
            G = len(gt)
            D = len(dt)
            gtm = np.zeros((T, G))
            dtm = np.zeros((T, D))
            gtIg = np.array([g['_ignore'] for g in gt])
            dtIg = np.zeros((T, D))

            if not len(ious) == 0:
                for tind, t in enumerate(p.iouThrs):
                    for dind, d in enumerate(dt):
                        # 记录最佳匹配（m=-1 表示未匹配）
                        iou = min([t, 1 - 1e-10])
                        bstOa = iou
                        bstg = -2
                        bstm = -2
                        for gind, g in enumerate(gt):
                            m = gtm[tind, gind]
                            # 如果真实标注已匹配且不是拥挤标注，跳过
                            if m > 0:
                                continue
                            # 如果检测结果匹配到常规真实标注，且当前真实标注为忽略标注，停止
                            if bstm != -2 and gtIg[gind] == 1:
                                break
                            # 如果 IoU 小于最佳匹配 IoU，跳过
                            if ious[dind, gind] < bstOa:
                                continue
                            # 如果匹配成功且是最佳匹配，记录匹配信息
                            bstOa = ious[dind, gind]
                            bstg = gind
                            if gtIg[gind] == 0:
                                bstm = 1
                            else:
                                bstm = -1

                        # 如果匹配成功，记录匹配信息
                        if bstg == -2:
                            continue
                        dtIg[tind, dind] = gtIg[bstg]
                        dtm[tind, dind] = gt[bstg]['id']
                        if bstm == 1:
                            gtm[tind, bstg] = d['id']

        except Exception:

            ex_type, ex_value, ex_traceback = sys.exc_info()            

            # 提取未格式化的堆栈跟踪作为元组
            trace_back = traceback.extract_tb(ex_traceback)

            # 格式化堆栈跟踪
            stack_trace = list()

            for trace in trace_back:
                stack_trace.append("File : %s , Line : %d, Func.Name : %s, Message : %s" % (trace[0], trace[1], trace[2], trace[3]))

            sys.stderr.write("[Error] Exception type : %s \n" % ex_type.__name__)
            sys.stderr.write("[Error] Exception message : %s \n" % ex_value)
            for trace in stack_trace:
                sys.stderr.write("[Error] (Stack trace) %s\n" % trace)

            pdb.set_trace()

        # 存储当前图片和类别的评估结果
        return {
            'image_id': imgId,
            'category_id': catId,
            'hRng': hRng,
            'oRng': oRng,
            'maxDet': maxDet,
            'dtIds': [d['id'] for d in dt],
            'gtIds': [g['id'] for g in gt],
            'dtMatches': dtm,
            'gtMatches': gtm,
            'dtScores': [d['score'] for d in dt],
            'gtIgnore': gtIg,
            'dtIgnore': dtIg,
        }

    def accumulate(self, p=None):
        """
        累积每张图片的评估结果，并存储在 self.eval 中。
        :param p: 评估参数
        :return: None
        """
        if not self.evalImgs:
            pass
            #print('Please run evaluate() first')
        # 允许输入自定义参数
        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T = len(p.iouThrs)
        R = len(p.fppiThrs)
        K = len(p.catIds) if p.useCats else 1
        M = len(p.maxDets)
        ys = -np.ones((T, R, K, M))     # -1 表示缺失类别的精度

        xx_graph = []
        yy_graph = []

        # 创建索引字典
        _pe = self._paramsEval
        catIds = [1]                    # _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # 获取需要评估的索引
        k_list = [n for n, k in enumerate(p.catIds) if k in setK]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        i_list = [n for n, i in enumerate(p.imgIds) if i in setI]
        I0 = len(_pe.imgIds)
        
        # 按类别、面积范围和最大检测数量获取评估结果
        for k, k0 in enumerate(k_list):
            Nk = k0 * I0
            for m, maxDet in enumerate(m_list):
                E = [self.evalImgs[Nk + i] for i in i_list]
                E = [e for e in E if e is not None]
                if len(E) == 0:
                    continue

                dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])

                # 不同的排序方法会产生略有不同的结果。
                # 使用 mergesort 以保持与 Matlab 实现的一致性。

                inds = np.argsort(-dtScores, kind='mergesort')

                dtm = np.concatenate([e['dtMatches'][:, 0:maxDet] for e in E], axis=1)[:, inds]
                dtIg = np.concatenate([e['dtIgnore'][:, 0:maxDet] for e in E], axis=1)[:, inds]
                gtIg = np.concatenate([e['gtIgnore'] for e in E])
                npig = np.count_nonzero(gtIg == 0)
                if npig == 0:
                    continue
                tps = np.logical_and(dtm, np.logical_not(dtIg))
                fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg))
                inds = np.where(dtIg == 0)[1]
                tps = tps[:, inds]
                fps = fps[:, inds]

                tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float64)
                fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float64)
            
                for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                    tp = np.array(tp)
                    fppi = np.array(fp) / I0
                    nd = len(tp)
                    recall = tp / npig
                    q = np.zeros((R,))

                    xx_graph.append(fppi)
                    yy_graph.append(1 - recall)

                    # numpy 在没有 cython 优化时访问元素会很慢。
                    # 使用 Python 列表可以获得显著的速度提升
                    recall = recall.tolist()
                    q = q.tolist()

                    for i in range(nd - 1, 0, -1):
                        if recall[i] < recall[i - 1]:
                            recall[i - 1] = recall[i]

                    inds = np.searchsorted(fppi, p.fppiThrs, side='right') - 1
                    try:
                        for ri, pi in enumerate(inds):
                            q[ri] = recall[pi]
                    except Exception:
                        pass
                    ys[t, :, k, m] = np.array(q)
        
        self.eval = {
            'params': p,
            'counts': [T, R, K, M],
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'TP': ys,
            'xx': xx_graph,
            'yy': yy_graph
        }

    @staticmethod
    def draw_figure(ax, eval_results, methods, colors):
        """
        绘制图形。
        :param ax: Matplotlib 轴对象。
        :param eval_results: 评估结果列表。
        :param methods: 方法名称列表。
        :param colors: 颜色列表。
        """
        assert len(eval_results) == len(methods) == len(colors)

        for eval_result, method, color in zip(eval_results, methods, colors):
            mrs = 1 - eval_result['TP']
            mean_s = np.log(mrs[mrs < 2])
            mean_s = np.mean(mean_s)
            mean_s = float(np.exp(mean_s) * 100)

            xx = eval_result['xx']
            yy = eval_result['yy']

            ax.plot(xx[0], yy[0], color=color, linewidth=3, label=f'{mean_s:.2f}%, {method}')

        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.legend()

        yt = [1, 5] + list(range(10, 60, 10)) + [64, 80]
        yticklabels = ['.{:02d}'.format(num) for num in yt]

        yt += [100]
        yt = [yy / 100.0 for yy in yt]
        yticklabels += [1]
        
        ax.set_yticks(yt)
        ax.set_yticklabels(yticklabels)
        ax.grid(which='major', axis='both')
        ax.set_ylim(0.01, 1)
        ax.set_xlim(2e-4, 50)
        ax.set_ylabel('miss rate')
        ax.set_xlabel('false positives per image')

    def summarize(self, id_setup, res_file=None):
        """
        计算并显示评估结果的摘要指标。
        :param id_setup: 设置 ID。
        :param res_file: 结果文件路径。
        :return: 平均误检率。
        """
        def _summarize(iouThr=None, maxDets=100):
            OCC_TO_TEXT = ['none', 'partial_occ', 'heavy_occ']

            p = self.params
            iStr = ' {:<18} {} @ {:<18} [ IoU={:<9} | height={:>6s} | visibility={:>6s} ] = {:0.2f}%'
            titleStr = 'Average Miss Rate'
            typeStr = '(MR)'
            setupStr = p.SetupLbl[id_setup]
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)
            heightStr = '[{:0.0f}:{:0.0f}]'.format(p.HtRng[id_setup][0], p.HtRng[id_setup][1])
            occlStr = '[' + '+'.join(['{:s}'.format(OCC_TO_TEXT[occ]) for occ in p.OccRng[id_setup]]) + ']'

            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

            # dimension of precision: [TxRxKxAxM]
            s = self.eval['TP']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            mrs = 1 - s[:, :, :, mind]

            if len(mrs[mrs < 2]) == 0:
                mean_s = -1
            else:
                mean_s = np.log(mrs[mrs < 2] + 1e-5)
                mean_s = np.mean(mean_s)
                mean_s = np.exp(mean_s)

            if res_file:
                res_file.write(iStr.format(titleStr, typeStr, setupStr, iouStr, heightStr, occlStr, mean_s * 100))
                res_file.write('\n')
            return mean_s

        if not self.eval:
            raise Exception('Please run accumulate() first')
        
        return _summarize(iouThr=.5, maxDets=1000)


class KAISTParams(Params):
    """
    KAISTPed 评估参数类。
    """
    def setDetParams(self):
        super().setDetParams()

        # 覆盖 KAISTPed 基准的变量
        self.iouThrs = np.array([0.5])
        self.maxDets = [1000]

        # KAISTPed 特定设置
        self.fppiThrs = np.array([0.0100, 0.0178, 0.0316, 0.0562, 0.1000, 0.1778, 0.3162, 0.5623, 1.0000])
        #self.HtRng = [[55, 1e5 ** 2], [50, 75], [50, 1e5 ** 2], [20, 1e5 ** 2]]
        self.HtRng = [[55, 1e5 ** 2], [115, 1e5 ** 2], [45, 115], [1, 45], [1, 1e5 ** 2], [1, 1e5 ** 2], [1, 1e5 ** 2]] # jifengshen
        #self.OccRng = [[0, 1], [0, 1], [2], [0, 1, 2]]
        self.OccRng = [[0, 1], [0], [0], [0], [0], [1], [2]]
        #self.SetupLbl = ['Reasonable', 'Reasonable_small', 'Reasonable_occ=heavy', 'All']
        self.SetupLbl = ['Reasonable', 'scale=near', 'scale=medium', 'scale=far', 'occ=none', 'occ=partial', 'occ=heavy', 'All']
        
        self.bndRng = [5, 5, 635, 507]  # 抛弃超出此像素范围的边界框


class KAIST(COCO):
    """
    KAIST 数据集类，继承自 COCO。
    """
    def txt2json(self, txt):
        """
        将 txt 文件转换为 COCO JSON 格式。
        :param txt: 注释文件路径。
        :return: JSON 格式的注释。
        """
        predict_result = []
        f = open(txt, 'r')
        #print(f)
        lines = f.readlines()
        for line in lines:
            json_format = {}
            pred_info = [float(ll) for ll in line.split(',')]
            json_format["image_id"] = pred_info[0] - 1                                      # 图片 ID
            json_format["category_id"] = 1                                                  # 行人类别
            json_format["bbox"] = [pred_info[1], pred_info[2], pred_info[3], pred_info[4]]  # 边界框
            json_format["score"] = pred_info[5]

            predict_result.append(json_format)
        return predict_result

    def loadRes(self, resFile):
        """
        加载结果文件并返回结果 API 对象。
        :param resFile: 结果文件路径。
        :return: 结果 API 对象。
        """
        # 如果结果文件是 txt 文件，转换为 JSON
        if type(resFile) == str and resFile.endswith('.txt'):
            anns = self.txt2json(resFile)
            _resFile = next(tempfile._get_candidate_names())
            with open(_resFile, 'w') as f:
                json.dump(anns, f, indent=4)
            res = super().loadRes(_resFile)
            os.remove(_resFile)
        elif type(resFile) == str and resFile.endswith('.json'):
            res = super().loadRes(resFile)
        else:
            raise Exception('[Error] Exception extension : %s \n' % resFile.split('.')[-1]) 

        return res


def evaluate(test_annotation_file: str, user_submission_file: str, phase_codename: str = 'Multispectral', plot=False):
    """
    评估特定挑战阶段的提交并返回分数。

    :param test_annotation_file: 测试注释文件路径。
    :param user_submission_file: 用户提交文件路径。
    :param phase_codename: 提交所在的阶段。
    :param plot: 是否绘制图形。
    :return: 所有/白天/夜晚的 KAISTPedEval 对象。
    """
    kaistGt = KAIST(test_annotation_file)
    kaistDt = kaistGt.loadRes(user_submission_file)

    imgIds = sorted(kaistGt.getImgIds())
    method = os.path.basename(user_submission_file).split('_')[0]
    kaistEval = KAISTPedEval(kaistGt, kaistDt, 'bbox', method)

    kaistEval.params.catIds = [1]

    eval_result = {
        'all': copy.deepcopy(kaistEval),
        'day': copy.deepcopy(kaistEval),
        'night': copy.deepcopy(kaistEval),
        'near': copy.deepcopy(kaistEval),
        'medium': copy.deepcopy(kaistEval),
        'far': copy.deepcopy(kaistEval),
        'none': copy.deepcopy(kaistEval),
        'partial': copy.deepcopy(kaistEval),
        'heavy': copy.deepcopy(kaistEval),
    }

    eval_result['all'].params.imgIds = imgIds
    eval_result['all'].evaluate(0)
    eval_result['all'].accumulate()
    MR_all = eval_result['all'].summarize(0)

    eval_result['day'].params.imgIds = imgIds[:1455]
    eval_result['day'].evaluate(0)
    eval_result['day'].accumulate()
    MR_day = eval_result['day'].summarize(0)

    eval_result['night'].params.imgIds = imgIds[1455:]
    eval_result['night'].evaluate(0)
    eval_result['night'].accumulate()
    MR_night = eval_result['night'].summarize(0)
    
    eval_result['near'].params.imgIds = imgIds
    eval_result['near'].evaluate(1)
    eval_result['near'].accumulate()
    MR_near = eval_result['near'].summarize(1)
    
    eval_result['medium'].params.imgIds = imgIds
    eval_result['medium'].evaluate(2)
    eval_result['medium'].accumulate()
    MR_medium = eval_result['medium'].summarize(2)
    
    eval_result['far'].params.imgIds = imgIds
    eval_result['far'].evaluate(3)
    eval_result['far'].accumulate()
    MR_far = eval_result['far'].summarize(3)
    
    eval_result['none'].params.imgIds = imgIds
    eval_result['none'].evaluate(4)
    eval_result['none'].accumulate()
    MR_none = eval_result['none'].summarize(4)
    
    eval_result['partial'].params.imgIds = imgIds
    eval_result['partial'].evaluate(5)
    eval_result['partial'].accumulate()
    MR_partial = eval_result['partial'].summarize(5)
    
    eval_result['heavy'].params.imgIds = imgIds
    eval_result['heavy'].evaluate(6)
    eval_result['heavy'].accumulate()
    MR_heavy = eval_result['heavy'].summarize(6)

    recall_all = 1 - eval_result['all'].eval['yy'][0][-1]

    if plot:
        msg = f'\n########## Method: {method} ##########\n' \
            + f'MR_all: {MR_all * 100:.2f}\n' \
            + f'MR_day: {MR_day * 100:.2f}\n' \
            + f'MR_night: {MR_night * 100:.2f}\n' \
            + f'MR_near: {MR_near * 100:.2f}\n' \
            + f'MR_medium: {MR_medium * 100:.2f}\n' \
            + f'MR_far: {MR_far * 100:.2f}\n' \
            + f'MR_none: {MR_none * 100:.2f}\n' \
            + f'MR_partial: {MR_partial * 100:.2f}\n' \
            + f'MR_heavy: {MR_heavy * 100:.2f}\n' \
            + f'recall_all: {recall_all * 100:.2f}\n' \
            + '######################################\n\n'
        print(msg)

    return eval_result


def draw_all(eval_results, filename='figure.jpg'):
    """
    在单个图形中绘制所有结果，作为误检率与每张图片的假阳性数量（FPPI）曲线。

    :param eval_results: 评估结果列表。
    :param filename: 图形文件名。
    """
    fig, axes = plt.subplots(1, 3, figsize=(45, 10))

    methods = [res['all'].method for res in eval_results]
    colors = [plt.cm.get_cmap('Paired')(ii)[:3] for ii in range(len(eval_results))]

    eval_results_all = [res['all'].eval for res in eval_results]
    KAISTPedEval.draw_figure(axes[0], eval_results_all, methods, colors)
    axes[0].set_title('All')

    eval_results_day = [res['day'].eval for res in eval_results]
    KAISTPedEval.draw_figure(axes[1], eval_results_day, methods, colors)
    axes[1].set_title('Day')

    eval_results_night = [res['night'].eval for res in eval_results]
    KAISTPedEval.draw_figure(axes[2], eval_results_night, methods, colors)
    axes[2].set_title('Night')

    filename += '' if filename.endswith('.jpg') or filename.endswith('.png') else '.jpg'
    plt.savefig(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='eval models')
    parser.add_argument('--annFile', type=str, default='D:/Project/MLPD-Multi-Label-Pedestrian-Detection-main/evaluation_script/KAIST_annotation.json',
                        help='Please put the path of the annotation file. Only support json format.')
    parser.add_argument('--rstFiles', type=str, nargs='+', default=['evaluation_script/MLPD_result.json'],
                        help='Please put the path of the result file. Only support json, txt format.')
    parser.add_argument('--evalFig', type=str, default='KASIT_BENCHMARK.jpg',
                        help='Please put the output path of the Miss rate versus false positive per-image (FPPI) curve')
    args = parser.parse_args()

    args.rstFiles = 'D:/Project/MLPD-Multi-Label-Pedestrian-Detection-main/evaluation_script/yolov5_CMAFF_result.txt'
    phase = "Multispectral"
    results = [evaluate(args.annFile, args.rstFiles, phase)]

    # 按照 MR_all 排序结果
    results = sorted(results, key=lambda x: x['all'].summarize(0), reverse=True)
    draw_all(results, filename=args.evalFig)
