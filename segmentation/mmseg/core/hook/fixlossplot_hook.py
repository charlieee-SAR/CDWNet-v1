import os
import json
from mmcv.runner import HOOKS, Hook
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


@HOOKS.register_module()
class FixedLossPlotHook(Hook):
    """稳定的损失绘图Hook - 实时保存和绘制曲线"""
    
    def __init__(self, interval=1, save_dir='./work_dirs/metrics'):
        self.interval = interval
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        self.json_file = os.path.join(save_dir, 'metrics.json')
        self.plot_file = os.path.join(save_dir, 'loss_curve.png')
        
        # 初始化数据结构
        self.metrics = {}
        
        # 加载已有数据（支持resume训练）
        if os.path.exists(self.json_file):
            try:
                with open(self.json_file, 'r') as f:
                    self.metrics = json.load(f)
                print(f'✓ Loaded existing metrics from {self.json_file}')
            except Exception as e:
                print(f'✗ Failed to load metrics: {e}')
    
    def after_train_epoch(self, runner):
        """每个训练epoch后被调用"""
        if not self.every_n_epochs(runner, self.interval):
            return
        
        epoch = runner.epoch + 1  # 转换为1-based epoch
        epoch_str = str(epoch)
        
        # 确保该epoch的记录存在
        if epoch_str not in self.metrics:
            self.metrics[epoch_str] = {'epoch': epoch}
        
        # 获取训练loss
        try:
            if hasattr(runner, 'log_buffer') and runner.log_buffer is not None:
                loss_dict = runner.log_buffer.output
                
                # 尝试获取loss
                loss = None
                if 'loss' in loss_dict:
                    loss = loss_dict['loss']
                elif 'decode.loss_seg' in loss_dict:
                    loss = loss_dict['decode.loss_seg']
                
                # 处理tensor类型
                if hasattr(loss, 'item'):
                    loss = loss.item()
                
                if loss is not None:
                    self.metrics[epoch_str]['train_loss'] = float(loss)
                    runner.logger.info(f'✓ Epoch {epoch}: Train Loss = {float(loss):.6f}')
        
        except Exception as e:
            runner.logger.warning(f'Failed to get train loss: {e}')
        
        self._save_and_plot(runner)
    
    def after_val_epoch(self, runner):
        """每个验证epoch后被调用"""
        epoch = runner.epoch + 1
        epoch_str = str(epoch)
        
        # 确保该epoch的记录存在
        if epoch_str not in self.metrics:
            self.metrics[epoch_str] = {'epoch': epoch}
        
        # 获取验证指标
        try:
            # MMSeg通常在eval_results中存储验证结果
            if hasattr(runner, 'eval_results') and runner.eval_results:
                eval_res = runner.eval_results
                
                # 获取mIoU
                if 'mIoU' in eval_res:
                    miou = float(eval_res['mIoU']) * 100  # 转换为百分比
                    self.metrics[epoch_str]['val_miou'] = miou
                    runner.logger.info(f'✓ Epoch {epoch}: Val mIoU = {miou:.2f}%')
                
                # 获取mAcc
                if 'mAcc' in eval_res:
                    macc = float(eval_res['mAcc']) * 100
                    self.metrics[epoch_str]['val_macc'] = macc
                    runner.logger.info(f'✓ Epoch {epoch}: Val mAcc = {macc:.2f}%')
                
                # 获取aAcc
                if 'aAcc' in eval_res:
                    aacc = float(eval_res['aAcc']) * 100
                    self.metrics[epoch_str]['val_aacc'] = aacc
        
        except Exception as e:
            runner.logger.warning(f'Failed to get val metrics: {e}')
        
        self._save_and_plot(runner)
    
    def _save_and_plot(self, runner):
        """保存JSON并绘制曲线"""
        # 保存JSON
        try:
            with open(self.json_file, 'w') as f:
                json.dump(self.metrics, f, indent=2)
        except Exception as e:
            runner.logger.warning(f'Failed to save metrics JSON: {e}')
        
        # 绘制曲线
        try:
            self._plot_metrics(runner)
        except Exception as e:
            runner.logger.warning(f'Failed to plot metrics: {e}')
            import traceback
            traceback.print_exc()
    
    def _plot_metrics(self, runner):
        """绘制loss和mIoU曲线"""
        if not self.metrics:
            return
        
        # 提取数据
        epochs = []
        train_losses = []
        val_mious = []
        
        for epoch_str in sorted(self.metrics.keys(), key=lambda x: int(x)):
            data = self.metrics[epoch_str]
            epoch = data.get('epoch', int(epoch_str))
            
            epochs.append(epoch)
            train_losses.append(data.get('train_loss', None))
            val_mious.append(data.get('val_miou', None))
        
        # 过滤有效数据
        valid_train_epochs = [e for e, l in zip(epochs, train_losses) if l is not None]
        valid_train_losses = [l for l in train_losses if l is not None]
        
        valid_val_epochs = [e for e, m in zip(epochs, val_mious) if m is not None]
        valid_val_mious = [m for m in val_mious if m is not None]
        
        # 如果没有任何数据，不绘制
        if not valid_train_losses and not valid_val_mious:
            return
        
        # 创建图表
        fig, ax1 = plt.subplots(figsize=(14, 6))
        
        # 绘制训练loss（左y轴）
        if valid_train_losses:
            ax1.plot(valid_train_epochs, valid_train_losses, 'b-o', 
                    label='Train Loss', linewidth=2, markersize=6, alpha=0.8)
            ax1.set_xlabel('Epoch', fontsize=13, fontweight='bold')
            ax1.set_ylabel('Loss', fontsize=13, fontweight='bold', color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            ax1.grid(True, alpha=0.3, linestyle='--')
        
        # 绘制验证mIoU（右y轴）
        if valid_val_mious:
            ax2 = ax1.twinx()
            ax2.plot(valid_val_epochs, valid_val_mious, 'g-^', 
                    label='Val mIoU', linewidth=2, markersize=6, alpha=0.8)
            ax2.set_ylabel('mIoU (%)', fontsize=13, fontweight='bold', color='green')
            ax2.tick_params(axis='y', labelcolor='green')
        
        plt.title('Training Metrics', fontsize=15, fontweight='bold')
        
        # 合并图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        if valid_val_mious:
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=11, loc='upper left')
        else:
            ax1.legend(lines1, labels1, fontsize=11, loc='upper left')
        
        plt.tight_layout()
        plt.savefig(self.plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        runner.logger.info(f'✓ Loss curve saved to: {self.plot_file}')