# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp
import sys
import time
import warnings

import mmcv
import torch
import torch.distributed as dist
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import Config, DictAction, get_git_hash

# Ensure local project mmseg has highest priority in sys.path.
PROJECT_ROOT = osp.abspath(osp.join(osp.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(line_buffering=True)

# Guard against accidentally importing mmseg from sibling repositories.
sys.path = [p for p in sys.path if '/home/xyz/Code/SegMAN-main/segmentation' not in p]
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mmseg import __version__
from mmseg.apis import init_random_seed, set_random_seed, train_segmentor
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import (collect_env, get_device, get_root_logger,
                         setup_multi_processes)


# 设置CUDA内存分配器
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# 清空缓存
torch.cuda.empty_cache()

# 打印初始内存状态
print(f"Initial GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB", flush=True)



# ========== 新增：实时绘图相关 ==========
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，适合服务器
import matplotlib.pyplot as plt
import json
from mmcv.runner import HOOKS, Hook


class LossPlotter:
    """实时绘制训练和验证损失曲线"""
    def __init__(self, work_dir, plot_interval=1):
        self.work_dir = work_dir
        self.plot_interval = plot_interval
        self.loss_file = osp.join(work_dir, 'loss_log.json')
        self.plot_file = osp.join(work_dir, 'loss_curve.png')
        
        # 初始化数据存储
        self.data = {
            'epochs': [],
            'train_loss': [],
            'val_loss': [],
            'val_mIoU': []  # 也记录mIoU
        }
        
        # 如果存在历史记录，加载它（用于resume训练）
        if osp.exists(self.loss_file):
            try:
                with open(self.loss_file, 'r') as f:
                    self.data = json.load(f)
                print(f'Loaded existing loss log from {self.loss_file}')
            except Exception as e:
                print(f'Failed to load loss log: {e}')
    
    def update(self, epoch, train_loss=None, val_loss=None, val_miou=None):
        """更新损失值"""
        # 检查是否是新的epoch
        if epoch not in self.data['epochs']:
            self.data['epochs'].append(epoch)
            self.data['train_loss'].append(None)
            self.data['val_loss'].append(None)
            self.data['val_mIoU'].append(None)
        
        idx = self.data['epochs'].index(epoch)
        
        if train_loss is not None:
            self.data['train_loss'][idx] = float(train_loss)
        if val_loss is not None:
            self.data['val_loss'][idx] = float(val_loss)
        if val_miou is not None:
            self.data['val_mIoU'][idx] = float(val_miou)
        
        # 保存数据到JSON
        try:
            with open(self.loss_file, 'w') as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            print(f'Failed to save loss log: {e}')
        
        # 每隔plot_interval个epoch绘制一次
        if epoch % self.plot_interval == 0:
            self.plot()
    
    def plot(self):
        """绘制损失曲线"""
        if not self.data['epochs']:
            # 先输出一个占位图，避免用户误以为没有生效
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.set_title('Training and Validation Metrics')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.text(
                0.5, 0.5, 'Waiting for first logging point...',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
            plt.tight_layout()
            plt.savefig(self.plot_file, dpi=150, bbox_inches='tight')
            plt.close()
            return
        
        try:
            # 创建双y轴图
            fig, ax1 = plt.subplots(figsize=(14, 6))
            
            epochs = self.data['epochs']
            
            # 绘制训练和验证损失（左y轴）
            train_losses = [l for l in self.data['train_loss'] if l is not None]
            train_epochs = [e for e, l in zip(epochs, self.data['train_loss']) if l is not None]
            lines = []
            if train_losses:
                line1 = ax1.plot(train_epochs, train_losses, 'b-o', label='Train Loss',
                                 linewidth=2, markersize=5, alpha=0.8)
                lines += line1

            val_losses = [l for l in self.data['val_loss'] if l is not None]
            val_epochs = [e for e, l in zip(epochs, self.data['val_loss']) if l is not None]
            if val_losses:
                line2 = ax1.plot(val_epochs, val_losses, 'r-s', label='Val Loss',
                                 linewidth=2, markersize=5, alpha=0.8)
                lines += line2
            
            ax1.set_xlabel('Epoch', fontsize=13, fontweight='bold')
            ax1.set_ylabel('Loss', fontsize=13, fontweight='bold', color='black')
            ax1.tick_params(axis='y', labelcolor='black')
            ax1.grid(True, alpha=0.3, linestyle='--')
            
            # 绘制mIoU（右y轴）
            val_mious = [m for m in self.data['val_mIoU'] if m is not None]
            miou_epochs = [e for e, m in zip(epochs, self.data['val_mIoU']) if m is not None]
            
            if val_mious:
                ax2 = ax1.twinx()
                line3 = ax2.plot(miou_epochs, val_mious, 'g-^', label='Val mIoU', 
                                linewidth=2, markersize=5, alpha=0.8)
                ax2.set_ylabel('mIoU (%)', fontsize=13, fontweight='bold', color='green')
                ax2.tick_params(axis='y', labelcolor='green')
                lines += line3

            labels = [l.get_label() for l in lines]
            
            if lines:
                ax1.legend(lines, labels, fontsize=11, loc='upper right')
            
            plt.title('Training and Validation Metrics', fontsize=15, fontweight='bold')
            plt.tight_layout()
            
            # 保存图片
            plt.savefig(self.plot_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f'✓ Loss curve saved to: {self.plot_file}')
            
        except Exception as e:
            print(f'Failed to plot loss curve: {e}')


@HOOKS.register_module()
class LossPlotHook(Hook):
    """自定义Hook用于捕获每个epoch的损失并绘图"""
    
    def __init__(self, interval=1, iter_interval=50):
        self.interval = interval
        self.iter_interval = iter_interval
        self.loss_plotter = None
    
    def before_run(self, runner):
        """训练开始前获取loss_plotter"""
        if hasattr(runner, 'cfg') and hasattr(runner.cfg, 'loss_plotter'):
            self.loss_plotter = runner.cfg.loss_plotter
            runner.logger.info('LossPlotHook initialized from cfg.loss_plotter')
        else:
            self.loss_plotter = LossPlotter(
                runner.work_dir, plot_interval=self.interval)
            runner.logger.info(
                'LossPlotHook initialized with fallback LossPlotter')

        # 训练开始先写一个空的 json 和占位图
        try:
            with open(self.loss_plotter.loss_file, 'w') as f:
                json.dump(self.loss_plotter.data, f, indent=2)
            self.loss_plotter.plot()
        except Exception as e:
            runner.logger.warning(f'Failed to initialize loss plot files: {e}')

    def after_train_iter(self, runner):
        """训练过程中按迭代间隔更新一次曲线，避免必须等到epoch结束。"""
        if self.loss_plotter is None:
            return
        if not self.every_n_iters(runner, self.iter_interval):
            return

        epoch = runner.epoch + 1
        train_loss = None
        if hasattr(runner, 'log_buffer') and runner.log_buffer is not None:
            output = runner.log_buffer.output
            if 'loss' in output:
                train_loss = output['loss']
            elif 'decode.loss_seg' in output:
                train_loss = output['decode.loss_seg']
        if hasattr(train_loss, 'item'):
            train_loss = train_loss.item()
        if train_loss is not None:
            self.loss_plotter.update(epoch, train_loss=float(train_loss))
    
    def after_train_epoch(self, runner):
        """训练epoch结束后"""
        if self.loss_plotter is None:
            return
        
        epoch = runner.epoch + 1  # 转换为1-based
        
        # 从log_buffer获取训练损失
        if hasattr(runner, 'log_buffer') and 'loss' in runner.log_buffer.output:
            train_loss = runner.log_buffer.output['loss']
            self.loss_plotter.update(epoch, train_loss=train_loss)
            runner.logger.info(f'✓ Epoch {epoch} - Train Loss: {train_loss:.4f}')
    
    def after_val_epoch(self, runner):
        """验证epoch结束后"""
        if self.loss_plotter is None:
            return
        
        epoch = runner.epoch + 1
        
        val_loss = None
        val_miou = None
        
        # 尝试获取验证损失（兼容 EpochBasedRunner 的 loss_val 命名）
        if hasattr(runner, 'log_buffer'):
            output = runner.log_buffer.output
            if 'loss_val' in output:
                val_loss = output['loss_val']
            elif 'val_loss' in output:
                val_loss = output['val_loss']
            elif 'loss' in output:
                val_loss = output['loss']
        
        # 尝试获取mIoU
        if hasattr(runner, 'log_buffer'):
            output = runner.log_buffer.output
            # MMSeg通常使用mIoU作为主要指标
            if 'mIoU' in output:
                val_miou = output['mIoU'] * 100  # 转换为百分比
            elif 'val_mIoU' in output:
                val_miou = output['val_mIoU'] * 100
            elif 'aAcc' in output:
                val_miou = output['aAcc'] * 100
        
        # 更新绘图
        self.loss_plotter.update(epoch, val_loss=val_loss, val_miou=val_miou)
        
        log_str = f'✓ Epoch {epoch}'
        if val_loss is not None:
            log_str += f' - Val Loss: {val_loss:.4f}'
        if val_miou is not None:
            log_str += f' - Val mIoU: {val_miou:.2f}%'
        runner.logger.info(log_str)


@HOOKS.register_module()
class EarlyStoppingHook(Hook):
    """Early stop training when monitored metric stops improving."""

    def __init__(self,
                 monitor='mIoU',
                 mode='max',
                 patience=12,
                 min_delta=1e-4,
                 warmup_epochs=10):
        self.monitor = monitor
        self.mode = mode
        self.patience = max(1, int(patience))
        self.min_delta = float(min_delta)
        self.warmup_epochs = max(0, int(warmup_epochs))
        self.best = None
        self.bad_epochs = 0
        self.stopped = False

    def _is_improved(self, current):
        if self.best is None:
            return True
        if self.mode == 'max':
            return current > (self.best + self.min_delta)
        return current < (self.best - self.min_delta)

    def _extract_metric(self, runner):
        if not hasattr(runner, 'log_buffer') or runner.log_buffer is None:
            return None
        output = runner.log_buffer.output
        if self.monitor in output:
            return output[self.monitor]
        if self.monitor.lower() == 'miou' and 'mIoU' in output:
            return output['mIoU']
        if self.monitor.lower() == 'loss' and 'loss_val' in output:
            return output['loss_val']
        return None

    def after_val_epoch(self, runner):
        if self.stopped:
            return

        epoch = runner.epoch + 1
        if epoch <= self.warmup_epochs:
            return

        metric = self._extract_metric(runner)
        if metric is None:
            runner.logger.warning(
                f'EarlyStoppingHook: monitor "{self.monitor}" not found in val logs.')
            return
        if hasattr(metric, 'item'):
            metric = metric.item()
        metric = float(metric)

        if self._is_improved(metric):
            self.best = metric
            self.bad_epochs = 0
            runner.logger.info(
                f'EarlyStoppingHook: {self.monitor} improved to {metric:.6f}')
            return

        self.bad_epochs += 1
        runner.logger.info(
            f'EarlyStoppingHook: no improvement on {self.monitor} '
            f'({self.bad_epochs}/{self.patience}), current={metric:.6f}, best={self.best:.6f}')

        if self.bad_epochs >= self.patience:
            self.stopped = True
            runner.logger.warning(
                f'EarlyStoppingHook: trigger stop at epoch {epoch}. '
                f'best_{self.monitor}={self.best:.6f}')
            # Stop after current epoch by shrinking total epochs.
            runner._max_epochs = epoch


def _extract_log_metric(output, monitor):
    if output is None:
        return None
    if monitor in output:
        return output[monitor]

    monitor_lower = monitor.lower()
    aliases = {
        'miou': 'mIoU',
        'loss': 'loss',
        'loss_val': 'loss_val',
    }
    alias = aliases.get(monitor_lower)
    if alias is not None and alias in output:
        return output[alias]
    return None


@HOOKS.register_module()
class MetricThresholdHook(Hook):
    """Stop training once the monitored metric reaches a target threshold."""

    def __init__(self,
                 monitor='IoU.garbage',
                 threshold=0.83,
                 mode='max',
                 warmup_epochs=1):
        self.monitor = monitor
        self.threshold = float(threshold)
        self.mode = mode
        self.warmup_epochs = max(0, int(warmup_epochs))
        self.stopped = False

    def _is_reached(self, metric):
        if self.mode == 'max':
            return metric >= self.threshold
        return metric <= self.threshold

    def after_val_epoch(self, runner):
        if self.stopped:
            return

        epoch = runner.epoch + 1
        if epoch <= self.warmup_epochs:
            return

        if not hasattr(runner, 'log_buffer') or runner.log_buffer is None:
            return
        metric = _extract_log_metric(runner.log_buffer.output, self.monitor)
        if metric is None:
            runner.logger.warning(
                f'MetricThresholdHook: monitor "{self.monitor}" not found in val logs.')
            return
        if hasattr(metric, 'item'):
            metric = metric.item()
        metric = float(metric)

        if not self._is_reached(metric):
            return

        self.stopped = True
        runner.logger.warning(
            f'MetricThresholdHook: trigger stop at epoch {epoch} because '
            f'{self.monitor}={metric:.6f} reached threshold={self.threshold:.6f}')
        # Stop after current epoch by shrinking total epochs.
        runner._max_epochs = epoch


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--load-from', help='the checkpoint file to load weights from')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='(Deprecated, please use --gpu-id) number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed training)')
    
    parser.add_argument(
        '--drop-path',
        default=-1,
        type=float,
        help='drop-path-rate of the backbone network')
    
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--diff_seed',
        action='store_true',
        help='Whether or not set different seeds for different ranks')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
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
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument(
        '--auto-resume',
        action='store_true',
        help='resume from the latest checkpoint automatically.')
    
    # ========== 新增：绘图参数 ==========
    parser.add_argument(
        '--plot-interval',
        type=int,
        default=1,
        help='interval (in epochs) to update loss curve plot')
    parser.add_argument(
        '--disable-early-stop',
        action='store_true',
        help='disable early stopping')
    parser.add_argument(
        '--early-stop-monitor',
        type=str,
        default='mIoU',
        help='metric name monitored by early stopping')
    parser.add_argument(
        '--early-stop-mode',
        type=str,
        default='max',
        choices=['max', 'min'],
        help='whether monitored metric should be maximized or minimized')
    parser.add_argument(
        '--early-stop-patience',
        type=int,
        default=12,
        help='early stopping patience in epochs')
    parser.add_argument(
        '--early-stop-min-delta',
        type=float,
        default=1e-4,
        help='minimum metric improvement to reset patience')
    parser.add_argument(
        '--early-stop-warmup',
        type=int,
        default=10,
        help='epochs to skip before enabling early stopping')
    parser.add_argument(
        '--stop-threshold-monitor',
        type=str,
        default=None,
        help='metric name for threshold stop, e.g. IoU.garbage')
    parser.add_argument(
        '--stop-threshold-value',
        type=float,
        default=None,
        help='target threshold for stop-threshold-monitor')
    parser.add_argument(
        '--stop-threshold-mode',
        type=str,
        default='max',
        choices=['max', 'min'],
        help='whether threshold stop triggers when metric >= threshold or <= threshold')
    parser.add_argument(
        '--stop-threshold-warmup',
        type=int,
        default=1,
        help='epochs to skip before enabling threshold stop')
    
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

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg.find_unused_parameters = False
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.load_from is not None:
        cfg.load_from = args.load_from
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpus is not None:
        cfg.gpu_ids = range(1)
        warnings.warn('`--gpus` is deprecated because we only support '
                      'single GPU mode in non-distributed training. '
                      'Use `gpus=1` now.')
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed training. Use the first GPU '
                      'in `gpu_ids` now.')
    if args.gpus is None and args.gpu_ids is None:
        cfg.gpu_ids = [args.gpu_id]

    cfg.auto_resume = args.auto_resume

    if args.drop_path >= 0:
        try:
            cfg.model.backbone.drop_path_rate = args.drop_path
        except:
            pass
    
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # set multi-process settings
    setup_multi_processes(cfg)

    # init the meta dict to record some important information
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    cfg.device = get_device()
    seed = init_random_seed(args.seed, device=cfg.device)
    seed = seed + dist.get_rank() if args.diff_seed else seed
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta['seed'] = seed
    meta['exp_name'] = osp.basename(args.config)

    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()

    # SyncBN is not support for DP
    if not distributed:
        warnings.warn(
            'SyncBN is only supported with DDP. To be compatible with DP, '
            'we convert SyncBN to BN. Please use dist_train.sh which can '
            'avoid this error.')
        model = revert_sync_batchnorm(model)

    logger.info(model)

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        cfg.checkpoint_config.meta = dict(
            mmseg_version=f'{__version__}+{get_git_hash()[:7]}',
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE)
    
    model.CLASSES = datasets[0].CLASSES
    meta.update(cfg.checkpoint_config.meta)
    
    # ========== 初始化损失曲线绘制器 ==========
    loss_plotter = LossPlotter(cfg.work_dir, plot_interval=args.plot_interval)
    cfg.loss_plotter = loss_plotter
    logger.info(f'Loss plotter initialized. Files will be saved to:')
    logger.info(f'  - Loss log: {loss_plotter.loss_file}')
    logger.info(f'  - Loss curve: {loss_plotter.plot_file}')
    
    # ========== 注册自定义Hook ==========
    if not hasattr(cfg, 'custom_hooks') or cfg.custom_hooks is None:
        cfg.custom_hooks = []
    
    # 添加LossPlotHook
    cfg.custom_hooks.append(
        dict(type='LossPlotHook', interval=args.plot_interval)
    )
    logger.info(f'LossPlotHook registered with interval={args.plot_interval}')
    if not args.disable_early_stop:
        cfg.custom_hooks.append(
            dict(
                type='EarlyStoppingHook',
                monitor=args.early_stop_monitor,
                mode=args.early_stop_mode,
                patience=args.early_stop_patience,
                min_delta=args.early_stop_min_delta,
                warmup_epochs=args.early_stop_warmup))
        logger.info(
            'EarlyStoppingHook registered: '
            f'monitor={args.early_stop_monitor}, mode={args.early_stop_mode}, '
            f'patience={args.early_stop_patience}, min_delta={args.early_stop_min_delta}, '
            f'warmup_epochs={args.early_stop_warmup}')
    if args.stop_threshold_monitor is not None and args.stop_threshold_value is not None:
        cfg.custom_hooks.append(
            dict(
                type='MetricThresholdHook',
                monitor=args.stop_threshold_monitor,
                threshold=args.stop_threshold_value,
                mode=args.stop_threshold_mode,
                warmup_epochs=args.stop_threshold_warmup))
        logger.info(
            'MetricThresholdHook registered: '
            f'monitor={args.stop_threshold_monitor}, threshold={args.stop_threshold_value}, '
            f'mode={args.stop_threshold_mode}, warmup_epochs={args.stop_threshold_warmup}')
    # =========================================
    
    train_segmentor(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
