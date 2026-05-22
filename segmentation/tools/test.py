# # Copyright (c) OpenMMLab. All rights reserved.
# import argparse
# import os
# import os.path as osp
# import shutil
# import time
# import warnings

# import mmcv
# import torch
# from mmcv.cnn.utils import revert_sync_batchnorm
# from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
#                          wrap_fp16_model)
# from mmcv.utils import DictAction

# from mmseg import digit_version
# from mmseg.apis import multi_gpu_test, single_gpu_test
# from mmseg.datasets import build_dataloader, build_dataset
# from mmseg.models import build_segmentor
# from mmseg.utils import build_ddp, build_dp, get_device, setup_multi_processes


# def parse_args():
#     parser = argparse.ArgumentParser(
#         description='mmseg test (and eval) a model')
#     parser.add_argument('config', help='test config file path')
#     parser.add_argument('--checkpoint', help='checkpoint file', default=None, type=str)
#     parser.add_argument(
#         '--work-dir',
#         help=('if specified, the evaluation metric results will be dumped'
#               'into the directory as json'))
#     parser.add_argument(
#         '--aug-test', action='store_true', help='Use Flip and Multi scale aug')
#     parser.add_argument('--out', help='output result file in pickle format')
#     parser.add_argument(
#         '--format-only',
#         action='store_true',
#         help='Format the output results without perform evaluation. It is'
#         'useful when you want to format the result to a specific format and '
#         'submit it to the test server')
#     parser.add_argument(
#         '--eval',
#         type=str,
#         nargs='+',
#         help='evaluation metrics, which depends on the dataset, e.g., "mIoU"'
#         ' for generic datasets, and "cityscapes" for Cityscapes')
#     parser.add_argument('--show', action='store_true', help='show results')
#     parser.add_argument(
#         '--show-dir', help='directory where painted images will be saved')
#     parser.add_argument(
#         '--gpu-collect',
#         action='store_true',
#         help='whether to use gpu to collect results.')
#     parser.add_argument(
#         '--gpu-id',
#         type=int,
#         default=0,
#         help='id of gpu to use '
#         '(only applicable to non-distributed testing)')
#     parser.add_argument(
#         '--tmpdir',
#         help='tmp directory used for collecting results from multiple '
#         'workers, available when gpu_collect is not specified')
#     parser.add_argument(
#         '--options',
#         nargs='+',
#         action=DictAction,
#         help="--options is deprecated in favor of --cfg_options' and it will "
#         'not be supported in version v0.22.0. Override some settings in the '
#         'used config, the key-value pair in xxx=yyy format will be merged '
#         'into config file. If the value to be overwritten is a list, it '
#         'should be like key="[a,b]" or key=a,b It also allows nested '
#         'list/tuple values, e.g. key="[(a,b),(c,d)]" Note that the quotation '
#         'marks are necessary and that no white space is allowed.')
#     parser.add_argument(
#         '--cfg-options',
#         nargs='+',
#         action=DictAction,
#         help='override some settings in the used config, the key-value pair '
#         'in xxx=yyy format will be merged into config file. If the value to '
#         'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
#         'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
#         'Note that the quotation marks are necessary and that no white space '
#         'is allowed.')
#     parser.add_argument(
#         '--eval-options',
#         nargs='+',
#         action=DictAction,
#         help='custom options for evaluation')
#     parser.add_argument(
#         '--launcher',
#         choices=['none', 'pytorch', 'slurm', 'mpi'],
#         default='none',
#         help='job launcher')
#     parser.add_argument(
#         '--opacity',
#         type=float,
#         default=0.5,
#         help='Opacity of painted segmentation map. In (0, 1] range.')
#     parser.add_argument('--local_rank', type=int, default=0)
#     args = parser.parse_args()
#     if 'LOCAL_RANK' not in os.environ:
#         os.environ['LOCAL_RANK'] = str(args.local_rank)

#     if args.options and args.cfg_options:
#         raise ValueError(
#             '--options and --cfg-options cannot be both '
#             'specified, --options is deprecated in favor of --cfg-options. '
#             '--options will not be supported in version v0.22.0.')
#     if args.options:
#         warnings.warn('--options is deprecated in favor of --cfg-options. '
#                       '--options will not be supported in version v0.22.0.')
#         args.cfg_options = args.options

#     return args


# def main():
#     args = parse_args()
#     assert args.out or args.eval or args.format_only or args.show \
#         or args.show_dir, \
#         ('Please specify at least one operation (save/eval/format/show the '
#          'results / save the results) with the argument "--out", "--eval"'
#          ', "--format-only", "--show" or "--show-dir"')

#     if args.eval and args.format_only:
#         raise ValueError('--eval and --format_only cannot be both specified')

#     if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
#         raise ValueError('The output file must be a pkl file.')

#     cfg = mmcv.Config.fromfile(args.config)
#     if args.cfg_options is not None:
#         cfg.merge_from_dict(args.cfg_options)

#     # set multi-process settings
#     setup_multi_processes(cfg)

#     # set cudnn_benchmark
#     if cfg.get('cudnn_benchmark', False):
#         torch.backends.cudnn.benchmark = True
#     if args.aug_test:
#         # hard code index
#         cfg.data.test.pipeline[1].img_ratios = [
#             0.5, 0.75, 1.0, 1.25, 1.5, 1.75
#         ]
#         cfg.data.test.pipeline[1].flip = True
#     cfg.model.pretrained = None
#     cfg.data.test.test_mode = True

#     if args.gpu_id is not None:
#         cfg.gpu_ids = [args.gpu_id]

#     # init distributed env first, since logger depends on the dist info.
#     if args.launcher == 'none':
#         cfg.gpu_ids = [args.gpu_id]
#         distributed = False
#         if len(cfg.gpu_ids) > 1:
#             warnings.warn(f'The gpu-ids is reset from {cfg.gpu_ids} to '
#                           f'{cfg.gpu_ids[0:1]} to avoid potential error in '
#                           'non-distribute testing time.')
#             cfg.gpu_ids = cfg.gpu_ids[0:1]
#     else:
#         distributed = True
#         init_dist(args.launcher, **cfg.dist_params)

#     rank, _ = get_dist_info()
#     # allows not to create
#     if args.work_dir is not None and rank == 0:
#         mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
#         timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
#         if args.aug_test:
#             json_file = osp.join(args.work_dir,
#                                  f'eval_multi_scale_{timestamp}.json')
#         else:
#             json_file = osp.join(args.work_dir,
#                                  f'eval_single_scale_{timestamp}.json')
#     elif rank == 0:
#         work_dir = osp.join('./work_dirs',
#                             osp.splitext(osp.basename(args.config))[0])
#         mmcv.mkdir_or_exist(osp.abspath(work_dir))
#         timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
#         if args.aug_test:
#             json_file = osp.join(work_dir,
#                                  f'eval_multi_scale_{timestamp}.json')
#         else:
#             json_file = osp.join(work_dir,
#                                  f'eval_single_scale_{timestamp}.json')

#     # build the dataloader
#     # TODO: support multiple images per gpu (only minor changes are needed)
#     dataset = build_dataset(cfg.data.test)
#     # The default loader config
#     loader_cfg = dict(
#         # cfg.gpus will be ignored if distributed
#         num_gpus=len(cfg.gpu_ids),
#         dist=distributed,
#         shuffle=False)
#     # The overall dataloader settings
#     loader_cfg.update({
#         k: v
#         for k, v in cfg.data.items() if k not in [
#             'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
#             'test_dataloader'
#         ]
#     })
#     test_loader_cfg = {
#         **loader_cfg,
#         'samples_per_gpu': 1,
#         'shuffle': False,  # Not shuffle by default
#         **cfg.data.get('test_dataloader', {})
#     }
#     # build the dataloader
#     data_loader = build_dataloader(dataset, **test_loader_cfg)

#     # build the model and load checkpoint
#     cfg.model.train_cfg = None
#     model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
#     fp16_cfg = cfg.get('fp16', None)
#     if fp16_cfg is not None:
#         wrap_fp16_model(model)
#     if args.checkpoint:
#         checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
#         if 'CLASSES' in checkpoint.get('meta', {}):
#             model.CLASSES = checkpoint['meta']['CLASSES']
#         else:
#             print('"CLASSES" not found in meta, use dataset.CLASSES instead')
#             model.CLASSES = dataset.CLASSES
#         if 'PALETTE' in checkpoint.get('meta', {}):
#             model.PALETTE = checkpoint['meta']['PALETTE']
#         else:
#             print('"PALETTE" not found in meta, use dataset.PALETTE instead')
#             model.PALETTE = dataset.PALETTE

#     print(model)
    

#     # clean gpu memory when starting a new evaluation.
#     torch.cuda.empty_cache()
#     eval_kwargs = {} if args.eval_options is None else args.eval_options

#     # Deprecated
#     efficient_test = eval_kwargs.get('efficient_test', False)
#     if efficient_test:
#         warnings.warn(
#             '``efficient_test=True`` does not have effect in tools/test.py, '
#             'the evaluation and format results are CPU memory efficient by '
#             'default')

#     eval_on_format_results = (
#         args.eval is not None and 'cityscapes' in args.eval)
#     if eval_on_format_results:
#         assert len(args.eval) == 1, 'eval on format results is not ' \
#                                     'applicable for metrics other than ' \
#                                     'cityscapes'
#     if args.format_only or eval_on_format_results:
#         if 'imgfile_prefix' in eval_kwargs:
#             tmpdir = eval_kwargs['imgfile_prefix']
#         else:
#             tmpdir = '.format_cityscapes'
#             eval_kwargs.setdefault('imgfile_prefix', tmpdir)
#         mmcv.mkdir_or_exist(tmpdir)
#     else:
#         tmpdir = None

#     cfg.device = get_device()
#     if not distributed:
#         warnings.warn(
#             'SyncBN is only supported with DDP. To be compatible with DP, '
#             'we convert SyncBN to BN. Please use dist_train.sh which can '
#             'avoid this error.')
#         if not torch.cuda.is_available():
#             assert digit_version(mmcv.__version__) >= digit_version('1.4.4'), \
#                 'Please use MMCV >= 1.4.4 for CPU training!'
#         model = revert_sync_batchnorm(model)
#         model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)
#         results = single_gpu_test(
#             model,
#             data_loader,
#             args.show,
#             args.show_dir,
#             False,
#             args.opacity,
#             pre_eval=args.eval is not None and not eval_on_format_results,
#             format_only=args.format_only or eval_on_format_results,
#             format_args=eval_kwargs)
#     else:
#         model = build_ddp(
#             model,
#             cfg.device,
#             device_ids=[int(os.environ['LOCAL_RANK'])],
#             broadcast_buffers=False)
#         results = multi_gpu_test(
#             model,
#             data_loader,
#             args.tmpdir,
#             args.gpu_collect,
#             False,
#             pre_eval=args.eval is not None and not eval_on_format_results,
#             format_only=args.format_only or eval_on_format_results,
#             format_args=eval_kwargs)

#     rank, _ = get_dist_info()
#     if rank == 0:
#         if args.out:
#             warnings.warn(
#                 'The behavior of ``args.out`` has been changed since MMSeg '
#                 'v0.16, the pickled outputs could be seg map as type of '
#                 'np.array, pre-eval results or file paths for '
#                 '``dataset.format_results()``.')
#             print(f'\nwriting results to {args.out}')
#             mmcv.dump(results, args.out)
#         if args.eval:
#             eval_kwargs.update(metric=args.eval)
#             metric = dataset.evaluate(results, **eval_kwargs)
#             metric_dict = dict(config=args.config, metric=metric)
#             mmcv.dump(metric_dict, json_file, indent=4)
#             if tmpdir is not None and eval_on_format_results:
#                 # remove tmp dir when cityscapes evaluation
#                 shutil.rmtree(tmpdir)


# if __name__ == '__main__':
#     main()

# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import shutil
import time
import warnings

import mmcv
import torch
import numpy as np
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmcv.utils import DictAction

from mmseg import digit_version
from mmseg.apis import multi_gpu_test, single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import build_ddp, build_dp, get_device, setup_multi_processes


def compute_confusion_matrix(results, dataset, num_classes, debug_show=20):
    """
    更健壮的混淆矩阵计算函数，尝试处理常见的 results 格式：
      - (H, W) 或 (1, H, W) -> squeeze -> (H, W)
      - (C, H, W) -> argmax(0) -> (H, W)
      - (H*W,) 或 (H*W,1) -> reshape(gt.shape)
      - 如果是文件路径(str) -> 尝试 np.load / mmcv.imread
      - 其余情况会打印警告并跳过

    debug_show: 前几个样本打印类型/shape 以便调试
    """
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    print('\n开始计算混淆矩阵...')
    for idx, result in enumerate(results):
        if idx % 100 == 0:
            print(f'处理进度: {idx}/{len(results)}')

        # 获取 ground truth
        gt_seg = dataset.get_gt_seg_map_by_idx(idx)
        gt_shape = gt_seg.shape

        # 调试输出前几个样本的result类型/shape
        if idx < debug_show:
            print(f'[DEBUG] idx={idx}, result type={type(result)}')

        pred_seg = None

        # 可能 result 是 tuple/list，取第一个元素（常见于 mmseg 的输出）
        if isinstance(result, (tuple, list)):
            if len(result) == 0:
                if idx < debug_show: print(f'[WARN] idx={idx} empty tuple/list result, skip')
                continue
            # 有时 result = (pred, others...), 有时是多个尺度的数组列表
            # 先尝试直接取第0个
            cand = result[0]
        else:
            cand = result

        # 如果 cand 是文件路径（字符串），尝试加载
        if isinstance(cand, str):
            if idx < debug_show: print(f'[DEBUG] idx={idx} result is string path, try load: {cand}')
            try:
                if cand.endswith('.npy'):
                    cand = np.load(cand)
                else:
                    import mmcv
                    cand = mmcv.imread(cand, flag='unchanged')  # 可能返回 HxW or HxWxC
            except Exception as e:
                print(f'[WARN] idx={idx} 无法加载路径 {cand}: {e}')
                continue

        # 转为 numpy（如果是 torch Tensor）
        if 'torch' in str(type(cand)):
            try:
                cand = cand.detach().cpu().numpy()
            except Exception:
                try:
                    cand = np.array(cand)
                except Exception:
                    if idx < debug_show: print(f'[WARN] idx={idx} 无法将 torch 对象转换为 numpy，跳过')
                    continue

        # 如果还是不是 numpy，尝试转换
        if not isinstance(cand, np.ndarray):
            try:
                cand = np.array(cand)
            except Exception:
                if idx < debug_show: print(f'[WARN] idx={idx} 无法转换为 numpy，类型: {type(cand)}，跳过')
                continue

        if idx < debug_show:
            print(f'[DEBUG] idx={idx} cand.shape={cand.shape} dtype={cand.dtype}')

        # 处理常见维度
        pred = None
        # case: (C, H, W) -> argmax over channel
        if cand.ndim == 3 and cand.shape[0] == num_classes:
            pred = cand.argmax(axis=0)
        # case: (H, W) 或 (1, H, W) 或 (H, W, 1)
        elif cand.ndim == 2:
            pred = cand
        elif cand.ndim == 3 and cand.shape[0] == 1:
            pred = cand.squeeze(0)
        elif cand.ndim == 3 and cand.shape[2] == 1:
            pred = cand.squeeze(2)
        # case: (H*W,) -> 尝试 reshape 为 gt_shape
        elif cand.ndim == 1 and cand.size == gt_seg.size:
            try:
                pred = cand.reshape(gt_shape)
            except Exception as e:
                if idx < debug_show: print(f'[WARN] idx={idx} reshape failed: {e}')
                pred = None
        # case: (C, H, W) where channels last (H, W, C)
        elif cand.ndim == 3 and cand.shape[2] == num_classes:
            pred = cand.argmax(axis=2)
        # case: (num_classes,) -> image-level score（无法做像素级比较），记录并跳过或填充全图（不推荐）
        elif cand.ndim == 1 and cand.size == num_classes:
            if idx < debug_show: print(f'[WARN] idx={idx} result 是 per-image class prob 向量，无法做逐像素比较，跳过')
            pred = None
        else:
            # 其他异常情况，尝试广播/扩展：如果 cand.size == gt.size we already handled；否则跳过并打印
            if idx < debug_show:
                print(f'[WARN] idx={idx} 未知 result 形状: {cand.shape}, 跳过该样本')
            pred = None

        if pred is None:
            continue

        # 确保 pred 与 gt 一致
        if pred.shape != gt_shape:
            # 如果可以 reshape 成 gt_shape 则 reshape
            if pred.size == gt_seg.size:
                try:
                    pred = pred.reshape(gt_shape)
                except Exception:
                    print(f'警告: 第{idx}个样本的预测和GT shape不一致 且无法 reshape')
                    print(f'  pred shape: {pred.shape}, gt shape: {gt_shape}')
                    continue
            else:
                print(f'警告: 第{idx}个样本的预测和GT shape不一致')
                print(f'  pred shape: {pred.shape}, gt shape: {gt_shape}')
                continue

        # 展平并忽略 gt 的 ignore_index（255）
        gt_flat = gt_seg.flatten()
        pred_flat = pred.flatten()
        valid_mask = gt_flat != 255
        gt_flat = gt_flat[valid_mask]
        pred_flat = pred_flat[valid_mask]

        # 更新混淆矩阵（用向量化更快）
        # 过滤越界标签
        valid = (gt_flat >= 0) & (gt_flat < num_classes) & (pred_flat >= 0) & (pred_flat < num_classes)
        if not np.any(valid):
            continue
        gt_valid = gt_flat[valid].astype(np.int64)
        pred_valid = pred_flat[valid].astype(np.int64)
        # 向量化更新混淆矩阵
        for t, p in zip(np.bincount(num_classes * gt_valid + pred_valid), []):
            pass  # 占位（下面采用更直接方法）
        # 更直接的方法：
        for g, p in zip(gt_valid, pred_valid):
            confusion_matrix[g, p] += 1

    print('混淆矩阵计算完成!')
    return confusion_matrix



def save_confusion_matrix(confusion_matrix, class_names, save_path):
    """
    保存混淆矩阵到文件
    
    Args:
        confusion_matrix: 混淆矩阵
        class_names: 类别名称列表
        save_path: 保存路径
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # 保存numpy数组
    np.save(save_path.replace('.png', '.npy'), confusion_matrix)
    print(f'混淆矩阵已保存到: {save_path.replace(".png", ".npy")}')
    
    # 计算归一化的混淆矩阵（按行归一化）
    confusion_matrix_norm = confusion_matrix.astype('float') / (confusion_matrix.sum(axis=1)[:, np.newaxis] + 1e-10)
    
    # 绘制混淆矩阵
    plt.figure(figsize=(12, 10))
    sns.heatmap(confusion_matrix_norm, annot=False, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Normalized Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f'混淆矩阵图已保存到: {save_path}')
    plt.close()
    
    # 保存为CSV格式（方便查看具体数值）
    csv_path = save_path.replace('.png', '.csv')
    with open(csv_path, 'w') as f:
        # 写入表头
        f.write('True\\Pred,' + ','.join(class_names) + '\n')
        # 写入数据
        for i, row in enumerate(confusion_matrix):
            f.write(class_names[i] + ',' + ','.join(map(str, row)) + '\n')
    print(f'混淆矩阵CSV已保存到: {csv_path}')
    
    # 计算并保存各类别的精确率、召回率、F1分数
    metrics_path = save_path.replace('.png', '_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write('Class-wise Metrics:\n')
        f.write('=' * 80 + '\n')
        f.write(f'{"Class":<20} {"Precision":<12} {"Recall":<12} {"F1-Score":<12} {"Support"}\n')
        f.write('=' * 80 + '\n')
        
        for i, class_name in enumerate(class_names):
            tp = confusion_matrix[i, i]
            fp = confusion_matrix[:, i].sum() - tp
            fn = confusion_matrix[i, :].sum() - tp
            
            precision = tp / (tp + fp + 1e-10)
            recall = tp / (tp + fn + 1e-10)
            f1 = 2 * precision * recall / (precision + recall + 1e-10)
            support = confusion_matrix[i, :].sum()
            
            f.write(f'{class_name:<20} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {support}\n')
        
        f.write('=' * 80 + '\n')
        
        # 计算总体准确率
        accuracy = np.trace(confusion_matrix) / (confusion_matrix.sum() + 1e-10)
        f.write(f'\nOverall Accuracy: {accuracy:.4f}\n')
    
    print(f'类别指标已保存到: {metrics_path}')


def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file', default=None, type=str)
    parser.add_argument(
        '--work-dir',
        help=('if specified, the evaluation metric results will be dumped'
              'into the directory as json'))
    parser.add_argument(
        '--aug-test', action='store_true', help='Use Flip and Multi scale aug')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "mIoU"'
        ' for generic datasets, and "cityscapes" for Cityscapes')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu_collect is not specified')
    parser.add_argument(
        '--save-confusion-matrix',
        action='store_true',
        help='whether to save confusion matrix')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help="--options is deprecated in favor of --cfg_options' and it will "
        'not be supported in version v0.22.0. Override some settings in the '
        'used config, the key-value pair in xxx=yyy format will be merged '
        'into config file. If the value to be overwritten is a list, it '
        'should be like key="[a,b]" or key=a,b It also allows nested '
        'list/tuple values, e.g. key="[(a,b),(c,d)]" Note that the quotation '
        'marks are necessary and that no white space is allowed.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options. '
            '--options will not be supported in version v0.22.0.')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options. '
                      '--options will not be supported in version v0.22.0.')
        args.cfg_options = args.options

    return args


def main():
    args = parse_args()
    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir or args.save_confusion_matrix, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show", "--show-dir" or "--save-confusion-matrix"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if args.aug_test:
        # hard code index
        cfg.data.test.pipeline[1].img_ratios = [
            0.5, 0.75, 1.0, 1.25, 1.5, 1.75
        ]
        cfg.data.test.pipeline[1].flip = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    if args.gpu_id is not None:
        cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        cfg.gpu_ids = [args.gpu_id]
        distributed = False
        if len(cfg.gpu_ids) > 1:
            warnings.warn(f'The gpu-ids is reset from {cfg.gpu_ids} to '
                          f'{cfg.gpu_ids[0:1]} to avoid potential error in '
                          'non-distribute testing time.')
            cfg.gpu_ids = cfg.gpu_ids[0:1]
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    rank, _ = get_dist_info()
    # allows not to create
    if args.work_dir is not None and rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        if args.aug_test:
            json_file = osp.join(args.work_dir,
                                 f'eval_multi_scale_{timestamp}.json')
        else:
            json_file = osp.join(args.work_dir,
                                 f'eval_single_scale_{timestamp}.json')
    elif rank == 0:
        work_dir = osp.join('./work_dirs',
                            osp.splitext(osp.basename(args.config))[0])
        mmcv.mkdir_or_exist(osp.abspath(work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        if args.aug_test:
            json_file = osp.join(work_dir,
                                 f'eval_multi_scale_{timestamp}.json')
        else:
            json_file = osp.join(work_dir,
                                 f'eval_single_scale_{timestamp}.json')

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    # The default loader config
    loader_cfg = dict(
        # cfg.gpus will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        shuffle=False)
    # The overall dataloader settings
    loader_cfg.update({
        k: v
        for k, v in cfg.data.items() if k not in [
            'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
            'test_dataloader'
        ]
    })
    test_loader_cfg = {
        **loader_cfg,
        'samples_per_gpu': 1,
        'shuffle': False,  # Not shuffle by default
        **cfg.data.get('test_dataloader', {})
    }
    # build the dataloader
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    if args.checkpoint:
        checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
        if 'CLASSES' in checkpoint.get('meta', {}):
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            print('"CLASSES" not found in meta, use dataset.CLASSES instead')
            model.CLASSES = dataset.CLASSES
        if 'PALETTE' in checkpoint.get('meta', {}):
            model.PALETTE = checkpoint['meta']['PALETTE']
        else:
            print('"PALETTE" not found in meta, use dataset.PALETTE instead')
            model.PALETTE = dataset.PALETTE

    print(model)

    # clean gpu memory when starting a new evaluation.
    torch.cuda.empty_cache()
    eval_kwargs = {} if args.eval_options is None else args.eval_options

    # Deprecated
    efficient_test = eval_kwargs.get('efficient_test', False)
    if efficient_test:
        warnings.warn(
            '``efficient_test=True`` does not have effect in tools/test.py, '
            'the evaluation and format results are CPU memory efficient by '
            'default')

    eval_on_format_results = (
        args.eval is not None and 'cityscapes' in args.eval)
    if eval_on_format_results:
        assert len(args.eval) == 1, 'eval on format results is not ' \
                                    'applicable for metrics other than ' \
                                    'cityscapes'
    if args.format_only or eval_on_format_results:
        if 'imgfile_prefix' in eval_kwargs:
            tmpdir = eval_kwargs['imgfile_prefix']
        else:
            tmpdir = '.format_cityscapes'
            eval_kwargs.setdefault('imgfile_prefix', tmpdir)
        mmcv.mkdir_or_exist(tmpdir)
    else:
        tmpdir = None

    cfg.device = get_device()
    if not distributed:
        warnings.warn(
            'SyncBN is only supported with DDP. To be compatible with DP, '
            'we convert SyncBN to BN. Please use dist_train.sh which can '
            'avoid this error.')
        if not torch.cuda.is_available():
            assert digit_version(mmcv.__version__) >= digit_version('1.4.4'), \
                'Please use MMCV >= 1.4.4 for CPU training!'
        model = revert_sync_batchnorm(model)
        model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)
        results = single_gpu_test(
            model,
            data_loader,
            args.show,
            args.show_dir,
            False,
            args.opacity,
            pre_eval=args.eval is not None and not eval_on_format_results,
            format_only=args.format_only or eval_on_format_results,
            format_args=eval_kwargs)
    else:
        model = build_ddp(
            model,
            cfg.device,
            device_ids=[int(os.environ['LOCAL_RANK'])],
            broadcast_buffers=False)
        results = multi_gpu_test(
            model,
            data_loader,
            args.tmpdir,
            args.gpu_collect,
            False,
            pre_eval=args.eval is not None and not eval_on_format_results,
            format_only=args.format_only or eval_on_format_results,
            format_args=eval_kwargs)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            warnings.warn(
                'The behavior of ``args.out`` has been changed since MMSeg '
                'v0.16, the pickled outputs could be seg map as type of '
                'np.array, pre-eval results or file paths for '
                '``dataset.format_results()``.')
            print(f'\nwriting results to {args.out}')
            mmcv.dump(results, args.out)
        
        # 计算并保存混淆矩阵
        if args.save_confusion_matrix:
            num_classes = len(dataset.CLASSES)
            confusion_matrix = compute_confusion_matrix(results, dataset, num_classes)
            
            # 确定保存路径
            if args.work_dir is not None:
                save_dir = args.work_dir
            else:
                save_dir = osp.join('./work_dirs',
                                   osp.splitext(osp.basename(args.config))[0])
            
            confusion_matrix_path = osp.join(save_dir, f'confusion_matrix_{timestamp}.png')
            save_confusion_matrix(confusion_matrix, dataset.CLASSES, confusion_matrix_path)
        
        if args.eval:
            eval_kwargs.update(metric=args.eval)
            metric = dataset.evaluate(results, **eval_kwargs)
            metric_dict = dict(config=args.config, metric=metric)
            mmcv.dump(metric_dict, json_file, indent=4)
            if tmpdir is not None and eval_on_format_results:
                # remove tmp dir when cityscapes evaluation
                shutil.rmtree(tmpdir)


if __name__ == '__main__':
    main()