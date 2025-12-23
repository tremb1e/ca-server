import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from transformer import VQGANTransformer
from utils import load_data, plot_images
from data_loaders import CIFAR10DataLoader_pro
from construct_dataset import * 
from torchsummary import summary
from evaluation import *
from mutils import Bar, Logger, AverageMeter, mkdir_p, savefig
from sklearn.metrics import roc_auc_score, roc_curve, f1_score
import time
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path

# 设置矢量PDF输出和Times New Roman字体
matplotlib.use('pdf')  # 使用PDF后端确保矢量输出

# 设置matplotlib参数确保完全矢量化的PDF输出和Times New Roman字体
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif', 'serif']
plt.rcParams['font.sans-serif'] = ['Times New Roman', 'Arial', 'DejaVu Sans', 'sans-serif']
plt.rcParams['font.family'] = 'serif'  # 使用serif字体族，优先Times New Roman
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42  # TrueType字体嵌入，确保文字为矢量
plt.rcParams['ps.fonttype'] = 42   # PostScript字体为矢量
plt.rcParams['text.usetex'] = False  # 禁用LaTeX以避免光栅化
plt.rcParams['savefig.dpi'] = 300  # 高DPI确保质量
plt.rcParams['savefig.format'] = 'pdf'
plt.rcParams['figure.dpi'] = 100  # 显示DPI

# 验证Times New Roman字体是否可用
available_fonts = [f.name for f in fm.fontManager.ttflist]
if 'Times New Roman' in available_fonts:
    print("✅ Times New Roman font is available")
else:
    print("⚠️  Times New Roman font not found, using fallback fonts")

# 添加全局变量定义
best_acc = 0.0

def save_vectorized_pdf(filepath, **kwargs):
    """
    保存完全矢量化的PDF文件，使用Times New Roman字体
    
    Args:
        filepath: 保存路径
        **kwargs: 额外的savefig参数
    """
    default_params = {
        'format': 'pdf',
        'bbox_inches': 'tight',
        'dpi': 'figure',
        'facecolor': 'white',
        'edgecolor': 'none',
        'transparent': False,
        'pad_inches': 0.1
    }
    # 合并用户参数和默认参数
    save_params = {**default_params, **kwargs}
    plt.savefig(filepath, **save_params)

def set_times_new_roman_style(ax, title=None, xlabel=None, ylabel=None):
    """
    为指定的轴设置Times New Roman字体样式
    
    Args:
        ax: matplotlib轴对象
        title: 图表标题
        xlabel: X轴标签
        ylabel: Y轴标签
    """
    if title:
        ax.set_title(title, fontfamily='serif', fontsize=14, pad=15)
    if xlabel:
        ax.set_xlabel(xlabel, fontfamily='serif', fontsize=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontfamily='serif', fontsize=12)
    
    # 设置刻度标签字体
    ax.tick_params(axis='both', which='major', labelsize=10)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily('serif')
    
    # 设置图例字体
    legend = ax.get_legend()
    if legend:
        for text in legend.get_texts():
            text.set_fontfamily('serif')

class TrainTransformer:
    def __init__(self, args):
        # 创建必要的目录
        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs("results", exist_ok=True)
        os.makedirs("charts", exist_ok=True)  # 新增图表目录
        
        self.model = VQGANTransformer(args).to(device=args.device)
        self.optim = self.configure_optimizers()
        
        # 初始化训练历史记录
        self.training_history = {
            'epochs': [],
            'losses': [],
            'aurocs': [],
            'fars': [],
            'frrs': [],
            'eers': [],
            'f1_scores': []
        }

        self.train(args)

    def configure_optimizers(self):
        decay, no_decay = set(), set()
        whitelist_weight_modules = (nn.Linear, )
        blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)

        for mn, m in self.model.transformer.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn

                if pn.endswith("bias"):
                    no_decay.add(fpn)

                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)

                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        no_decay.add("pos_emb")

        param_dict = {pn: p for pn, p in self.model.transformer.named_parameters()}

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=4.5e-06, betas=(0.9, 0.95))
        return optimizer

    def save_training_loss_chart(self, epoch, loss):
        """保存训练损失曲线图"""
        try:
            self.training_history['epochs'].append(epoch + 1)
            self.training_history['losses'].append(loss.cpu().detach().numpy().item())
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(self.training_history['epochs'], self.training_history['losses'], 'b-', linewidth=2, marker='o')
            ax.grid(True, alpha=0.3)
            set_times_new_roman_style(ax, 
                'VQGAN-Transformer Training Loss Curve', 
                'Epoch', 
                'Cross-Entropy Loss')
            
            plt.tight_layout()
            save_vectorized_pdf(f"charts/training_loss_epoch_{epoch+1}.pdf")
            plt.close()
        except Exception as e:
            print(f"Warning: Could not save loss chart: {e}")

    def save_evaluation_metrics_chart(self, epoch, results):
        """保存评估指标图表"""
        try:
            latest_results, avg_results, has_valid = results
            if not has_valid:
                return
                
            auroc, far, frr, eer, f1 = latest_results
            
            # 记录历史数据
            self.training_history['aurocs'].append(auroc)
            self.training_history['fars'].append(far)
            self.training_history['frrs'].append(frr)
            self.training_history['eers'].append(eer)
            self.training_history['f1_scores'].append(f1)
            
            # 1. 当前epoch的评估指标条形图
            fig, ax = plt.subplots(figsize=(12, 8))
            metrics = ['AUROC', 'FAR', 'FRR', 'EER', 'F1 Score']
            values = [auroc, far, frr, eer, f1]
            colors = ['blue', 'red', 'orange', 'green', 'purple']
            
            bars = ax.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
            
            # 在柱子上添加数值标签
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.4f}', ha='center', va='bottom', fontfamily='serif')
            
            ax.set_ylim(0, max(max(values) * 1.2, 1.0))
            ax.grid(True, alpha=0.3, axis='y')
            set_times_new_roman_style(ax, 
                f'Evaluation Metrics - Epoch {epoch+1}', 
                'Metrics', 
                'Values')
            
            plt.tight_layout()
            save_vectorized_pdf(f"charts/evaluation_metrics_epoch_{epoch+1}.pdf")
            plt.close()
            
            # 2. 如果有多个epoch的数据，绘制指标变化趋势图
            if len(self.training_history['epochs']) > 1:
                self._save_metrics_trend_chart(epoch)
                
        except Exception as e:
            print(f"Warning: Could not save evaluation chart: {e}")

    def _save_metrics_trend_chart(self, current_epoch):
        """保存评估指标趋势图"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            epochs = self.training_history['epochs']
            
            # AUROC趋势
            ax1.plot(epochs, self.training_history['aurocs'], 'b-', linewidth=2, marker='o')
            set_times_new_roman_style(ax1, 'AUROC Trend', 'Epoch', 'AUROC')
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 1)
            
            # EER趋势
            ax2.plot(epochs, self.training_history['eers'], 'g-', linewidth=2, marker='s')
            set_times_new_roman_style(ax2, 'EER Trend', 'Epoch', 'EER')
            ax2.grid(True, alpha=0.3)
            
            # FAR和FRR趋势
            ax3.plot(epochs, self.training_history['fars'], 'r-', linewidth=2, marker='^', label='FAR')
            ax3.plot(epochs, self.training_history['frrs'], 'orange', linewidth=2, marker='v', label='FRR')
            set_times_new_roman_style(ax3, 'FAR & FRR Trend', 'Epoch', 'Rate')
            ax3.grid(True, alpha=0.3)
            ax3.legend(prop={'family': 'serif'})
            
            # F1 Score趋势
            ax4.plot(epochs, self.training_history['f1_scores'], 'purple', linewidth=2, marker='d')
            set_times_new_roman_style(ax4, 'F1 Score Trend', 'Epoch', 'F1 Score')
            ax4.grid(True, alpha=0.3)
            ax4.set_ylim(0, 1)
            
            plt.tight_layout()
            save_vectorized_pdf(f"charts/metrics_trend_up_to_epoch_{current_epoch+1}.pdf")
            plt.close()
        except Exception as e:
            print(f"Warning: Could not save trend chart: {e}")

    def save_roc_curve_chart(self, y_true, y_scores, epoch):
        """保存ROC曲线图"""
        try:
            if len(np.unique(y_true)) < 2:
                print("Warning: Cannot plot ROC curve - only one class present")
                return
                
            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            auroc = roc_auc_score(y_true, y_scores)
            
            # 计算EER点
            fnr = 1 - tpr
            eer_diff = np.abs(fpr - fnr)
            eer_index = np.argmin(eer_diff)
            eer_point = (fpr[eer_index] + fnr[eer_index]) / 2.0
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # 绘制ROC曲线
            ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {auroc:.4f})')
            
            # 绘制EER点
            ax.plot(fpr[eer_index], tpr[eer_index], 'ro', markersize=10, 
                   label=f'EER Point ({eer_point:.4f})')
            
            # 绘制对角线
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Random Classifier')
            
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.grid(True, alpha=0.3)
            set_times_new_roman_style(ax, 
                f'ROC Curve - Epoch {epoch+1}', 
                'False Positive Rate (FAR)', 
                'True Positive Rate (1-FRR)')
            ax.legend(prop={'family': 'serif'}, loc='lower right')
            
            plt.tight_layout()
            save_vectorized_pdf(f"charts/roc_curve_epoch_{epoch+1}.pdf")
            plt.close()
        except Exception as e:
            print(f"Warning: Could not save ROC curve: {e}")

    def save_comprehensive_summary_chart(self):
        """保存训练的综合总结图表"""
        try:
            if len(self.training_history['epochs']) == 0:
                return
                
            fig = plt.figure(figsize=(16, 12))
            
            # 创建网格布局
            gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
            
            epochs = self.training_history['epochs']
            
            # 1. 训练损失
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.plot(epochs, self.training_history['losses'], 'b-', linewidth=2, marker='o')
            set_times_new_roman_style(ax1, 'Training Loss', 'Epoch', 'Loss')
            ax1.grid(True, alpha=0.3)
            
            # 2. AUROC
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.plot(epochs, self.training_history['aurocs'], 'g-', linewidth=2, marker='s')
            set_times_new_roman_style(ax2, 'AUROC Performance', 'Epoch', 'AUROC')
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 1)
            
            # 3. FAR和FRR
            ax3 = fig.add_subplot(gs[1, 0])
            ax3.plot(epochs, self.training_history['fars'], 'r-', linewidth=2, marker='^', label='FAR')
            ax3.plot(epochs, self.training_history['frrs'], 'orange', linewidth=2, marker='v', label='FRR')
            set_times_new_roman_style(ax3, 'Error Rates', 'Epoch', 'Rate')
            ax3.grid(True, alpha=0.3)
            ax3.legend(prop={'family': 'serif'})
            
            # 4. EER
            ax4 = fig.add_subplot(gs[1, 1])
            ax4.plot(epochs, self.training_history['eers'], 'purple', linewidth=2, marker='d')
            set_times_new_roman_style(ax4, 'Equal Error Rate', 'Epoch', 'EER')
            ax4.grid(True, alpha=0.3)
            
            # 5. F1 Score
            ax5 = fig.add_subplot(gs[2, 0])
            ax5.plot(epochs, self.training_history['f1_scores'], 'brown', linewidth=2, marker='h')
            set_times_new_roman_style(ax5, 'F1 Score', 'Epoch', 'F1 Score')
            ax5.grid(True, alpha=0.3)
            ax5.set_ylim(0, 1)
            
            # 6. 最终指标总结
            ax6 = fig.add_subplot(gs[2, 1])
            if len(self.training_history['aurocs']) > 0:
                final_metrics = ['AUROC', 'FAR', 'FRR', 'EER', 'F1']
                final_values = [
                    self.training_history['aurocs'][-1],
                    self.training_history['fars'][-1],
                    self.training_history['frrs'][-1],
                    self.training_history['eers'][-1],
                    self.training_history['f1_scores'][-1]
                ]
                colors = ['blue', 'red', 'orange', 'green', 'purple']
                bars = ax6.bar(final_metrics, final_values, color=colors, alpha=0.7)
                
                # 添加数值标签
                for bar, value in zip(bars, final_values):
                    height = bar.get_height()
                    ax6.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontfamily='serif', fontsize=9)
                
                set_times_new_roman_style(ax6, 'Final Metrics', 'Metric', 'Value')
                ax6.set_ylim(0, max(max(final_values) * 1.2, 1.0))
            
            fig.suptitle('VQGAN-Transformer Training Comprehensive Summary', 
                        fontsize=16, fontweight='bold', fontfamily='serif')
            
            save_vectorized_pdf("charts/training_comprehensive_summary.pdf")
            plt.close()
        except Exception as e:
            print(f"Warning: Could not save comprehensive summary: {e}")

    def train(self, args):
        user_num = 1
        x_train, y_train, x_valid, y_valid, x_plot, y_plot, subjects = get_mnist_con_another(cls = user_num)
        
        train_dataset = CIFAR10DataLoader_pro(x_train, y_train, split="train", batch_size = args.batch_size, user_num=user_num)
        test_dataloader = CIFAR10DataLoader_pro(x_valid, y_valid, split="test", batch_size = args.batch_size, user_num=user_num)
        
        for epoch in range(args.epochs):
            with tqdm(range(len(train_dataset))) as pbar:
                for i, (imgs,_) in zip(pbar, train_dataset):
                    self.optim.zero_grad()
                    imgs = imgs.type(torch.FloatTensor)
                    imgs = imgs.to(device=args.device)
                    logits, targets = self.model(imgs)
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                    loss.backward()
                    self.optim.step()
                    pbar.set_postfix(Transformer_Loss=np.round(loss.cpu().detach().numpy().item(), 4))
                    pbar.update(0)
            
            # 保存训练损失图表
            self.save_training_loss_chart(epoch, loss)
            
            log, rec, _, sampled_imgs = self.model.log_images(imgs)
            print(f"Epoch {epoch+1}/{args.epochs} - Testing...")
            latest_results, avg_results, has_valid = self.test(args, test_dataloader, self.model, epoch)
            
            # 保存评估指标图表
            self.save_evaluation_metrics_chart(epoch, (latest_results, avg_results, has_valid))
            
            if has_valid:
                print(f"✅ Epoch {epoch+1} 完成 - 最新AUROC: {latest_results[0]:.4f}, 最新EER: {latest_results[3]:.4f}")
            else:
                print(f"⚠️  Epoch {epoch+1} 完成 - 无有效计算结果，使用默认值")
        
        # 保存综合总结图表
        self.save_comprehensive_summary_chart()
        
        # 保存模型
        checkpoint_path = os.path.join("checkpoints", f"transformer_{user_num}.pt")
        print(f"Saving model to {checkpoint_path}")
        try:
            torch.save(self.model.state_dict(), checkpoint_path)
            print("Model saved successfully!")
        except Exception as e:
            print(f"Error saving model: {e}")
            print("Please check if the checkpoints directory exists and has write permissions.")

    def test(self, args, testloader, transformer, epoch):
        global best_acc
        
        print(f"Starting test with {len(testloader)} batches...")
    
        batch_time = AverageMeter()
        data_time = AverageMeter()
        top1 = AverageMeter()
        far = AverageMeter()
        frr = AverageMeter()
        eer = AverageMeter()
        f1_score_meter = AverageMeter()
        
        # 收集所有批次的预测结果和标签
        all_scores = []
        all_targets = []
        all_half_scores = []
        
        # 添加变量保存最新的结果（初始化为合理的默认值）
        latest_auroc = 0.5  # AUC的中性值
        latest_far = 50.0   # FAR的中性值
        latest_frr = 50.0   # FRR的中性值
        latest_eer = 50.0   # EER的中性值
        latest_f1 = 0.5     # F1的中性值
        
        # 标记是否有成功的计算
        has_valid_computation = False
    
        # switch to evaluate mode
        transformer.eval()
        end = time.time()
        bar = Bar('Processing', max=len(testloader))
        
        print("Running full test with real computations...")
        
        with torch.no_grad():  # 添加no_grad来节省显存和加快推理
            for batch_idx, (inputs, targets) in enumerate(testloader):
                try:
                    if batch_idx == 0:
                        print(f"Processing first batch with {inputs.shape}")
                        
                    # measure data loading time
                    data_time.update(time.time() - end)
                    inputs = inputs.type(torch.FloatTensor)
                    inputs, targets = inputs.cuda(), targets.cuda()
                    
                    if batch_idx == 0:
                        print("Computing transformer log_images...")
                    
                    # 真正的计算：使用transformer的log_images方法
                    log, rec, half, sampled_imgs = transformer.log_images(inputs)
                    
                    if batch_idx == 0:
                        print("Computing reconstruction errors...")
                    
                    # 计算重建误差（注意：误差越小说明越可能是同一用户）
                    # 所以分数应该是负的误差，这样分数越高越可能是同一用户
                    reconstruction_errors = torch.mean(torch.pow((inputs - rec), 2), dim=[1,2,3])
                    half_reconstruction_errors = torch.mean(torch.pow((inputs - half), 2), dim=[1,2,3])
                    
                    # 转换为分数：使用负的重建误差，这样分数越高表示越可能是同一用户
                    scores = -reconstruction_errors
                    half_scores = -half_reconstruction_errors
                    
                    if batch_idx == 0:
                        print("Converting to numpy...")
                    
                    # 将数据转换为numpy
                    targets_numpy = targets.cpu().detach().numpy()
                    scores_numpy = scores.cpu().detach().numpy()
                    half_scores_numpy = half_scores.cpu().detach().numpy()
                    
                    # 收集所有批次的数据
                    all_scores.extend(scores_numpy)
                    all_targets.extend(targets_numpy)
                    all_half_scores.extend(half_scores_numpy)
                    
                    # 数据有效性检查
                    if np.any(np.isnan(scores_numpy)) or np.any(np.isnan(half_scores_numpy)):
                        if batch_idx == 0:
                            print("⚠️  警告：检测到NaN值在分数中")
                        continue
                    
                    if len(targets_numpy) == 0:
                        if batch_idx == 0:
                            print("⚠️  警告：targets为空")
                        continue
                    
                    # 检查标签的唯一值
                    unique_targets = np.unique(targets_numpy)
                    
                    if batch_idx == 0:
                        print(f"🔍 数据分析:")
                        print(f"   Unique targets: {unique_targets}")
                        print(f"   Targets shape: {targets_numpy.shape}, Scores shape: {scores_numpy.shape}")
                        print(f"   Targets distribution: {np.bincount(targets_numpy)}")
                        print(f"   Reconstruction errors range: [{reconstruction_errors.min().item():.4f}, {reconstruction_errors.max().item():.4f}]")
                        print(f"   Scores range: [{scores_numpy.min():.4f}, {scores_numpy.max():.4f}]")
                        print(f"   Half_scores range: [{half_scores_numpy.min():.4f}, {half_scores_numpy.max():.4f}]")
                    
                    # measure elapsed time
                    batch_time.update(time.time() - end)
                    end = time.time()
            
                    # plot progress
                    bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:}'.format(
                            batch=batch_idx + 1,
                            size=len(testloader),
                            data=data_time.avg,
                            bt=batch_time.avg,
                            total=bar.elapsed_td,
                            eta=bar.eta_td,
                            )
                    bar.next()
                    
                    if batch_idx == 0:
                        print("First batch completed successfully!")
                        
                except Exception as e:
                    print(f"Error in batch {batch_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    # 不要break，继续处理下一个batch
                    
                    batch_time.update(time.time() - end)
                    end = time.time()
                    bar.next()
            
        bar.finish()
        
        # 在所有批次处理完后，统一计算评估指标
        if len(all_scores) > 0:
            all_scores = np.array(all_scores)
            all_targets = np.array(all_targets)
            all_half_scores = np.array(all_half_scores)
            
            print("\n📊 整体数据统计:")
            print(f"   总样本数: {len(all_targets)}")
            print(f"   标签分布: {np.bincount(all_targets)}")
            print(f"   唯一标签: {np.unique(all_targets)}")
            
            # 将多分类转换为二分类：用户1 vs 其他用户
            # 假设用户1是目标用户（正样本），其他用户是负样本
            binary_targets = (all_targets == 1).astype(int)
            unique_binary = np.unique(binary_targets)
            
            print(f"   二分类分布: 正样本(用户1): {np.sum(binary_targets==1)}, 负样本(其他): {np.sum(binary_targets==0)}")
            
            if len(unique_binary) == 2:
                try:
                    # 计算ROC AUC
                    auroc = roc_auc_score(binary_targets, all_scores)
                    auroc_half = roc_auc_score(binary_targets, all_half_scores)
                    
                    print(f"\n✅ ROC AUC (使用rec): {auroc:.4f}")
                    print(f"✅ ROC AUC (使用half): {auroc_half:.4f}")
                    
                    # 计算ROC曲线和EER
                    fpr, tpr, thresholds = roc_curve(binary_targets, all_scores)
                    fnr = 1 - tpr
                    
                    # 计算EER：FAR和FRR相等的点
                    eer_diff = np.abs(fpr - fnr)
                    eer_index = np.argmin(eer_diff)
                    
                    eer_value = (fpr[eer_index] + fnr[eer_index]) / 2.0
                    far_value = fpr[eer_index]
                    frr_value = fnr[eer_index]
                    
                    # 使用EER阈值计算F1分数
                    threshold_eer = thresholds[eer_index]
                    predictions_eer = (all_scores >= threshold_eer).astype(int)
                    f1_value = f1_score(binary_targets, predictions_eer)
                    
                    print(f"\n📈 使用rec分数的评估指标:")
                    print(f"   EER阈值: {threshold_eer:.4f}")
                    print(f"   FAR: {far_value:.4f}")
                    print(f"   FRR: {frr_value:.4f}")
                    print(f"   EER: {eer_value:.4f}")
                    print(f"   F1 Score: {f1_value:.4f}")
                    
                    # 保存ROC曲线图表
                    self.save_roc_curve_chart(binary_targets, all_scores, epoch)
                    
                    # 同样计算half分数的指标
                    fpr_half, tpr_half, thresholds_half = roc_curve(binary_targets, all_half_scores)
                    fnr_half = 1 - tpr_half
                    eer_diff_half = np.abs(fpr_half - fnr_half)
                    eer_index_half = np.argmin(eer_diff_half)
                    
                    eer_value_half = (fpr_half[eer_index_half] + fnr_half[eer_index_half]) / 2.0
                    far_value_half = fpr_half[eer_index_half]
                    frr_value_half = fnr_half[eer_index_half]
                    threshold_eer_half = thresholds_half[eer_index_half]
                    predictions_eer_half = (all_half_scores >= threshold_eer_half).astype(int)
                    f1_value_half = f1_score(binary_targets, predictions_eer_half)
                    
                    print(f"\n📈 使用half分数的评估指标:")
                    print(f"   EER阈值: {threshold_eer_half:.4f}")
                    print(f"   FAR: {far_value_half:.4f}")
                    print(f"   FRR: {frr_value_half:.4f}")
                    print(f"   EER: {eer_value_half:.4f}")
                    print(f"   F1 Score: {f1_value_half:.4f}")
                    
                    # 更新结果（使用rec分数的结果作为主要结果）
                    latest_auroc = auroc
                    # FAR/FRR/EER 以小数记录（0.80 而非 80%），便于与日志统一
                    latest_far = far_value
                    latest_frr = frr_value
                    latest_eer = eer_value
                    latest_f1 = f1_value
                    has_valid_computation = True

                    # 更新平均计量器
                    top1.update(auroc, len(all_targets))
                    far.update(far_value, 1)
                    frr.update(frr_value, 1)
                    eer.update(eer_value, 1)
                    f1_score_meter.update(f1_value, 1)
                    
                except Exception as e:
                    print(f"计算评估指标时出错: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"⚠️  警告：只有一个类别，无法计算二分类指标")
                print(f"   二分类唯一值: {unique_binary}")
        else:
            print("⚠️  警告：没有收集到有效数据")
        
        # 修复：test结束后将模型切换回训练模式
        transformer.train()
        
        # 输出详细的评价指标
        print("="*60)
        print("TEST RESULTS:")
        print(f"总batch数: {len(testloader)}")
        print(f"总样本数: {len(all_targets) if 'all_targets' in locals() else 0}")
        print(f"是否有有效计算: {'✅ 是' if has_valid_computation else '❌ 否'}")
        print(f"AUROC (Area Under ROC Curve): {latest_auroc:.4f}")
        print(f"FAR (False Accept Rate):      {latest_far:.4f}")
        print(f"FRR (False Reject Rate):      {latest_frr:.4f}")
        print(f"EER (Equal Error Rate):       {latest_eer:.4f}")
        print(f"F1 Score:                     {latest_f1:.4f}")
        print("="*60)
        if has_valid_computation:
            print("✅ 评估指标基于所有测试数据计算得出")
        else:
            print("⚠️  警告：没有成功的有效计算，所有值都是默认值")
        print("="*60)
        
        # 返回最新值、平均值和有效计算标记
        latest_results = (latest_auroc, latest_far, latest_frr, latest_eer, latest_f1)
        avg_results = (top1.avg, far.avg, frr.avg, eer.avg, f1_score_meter.avg)
        return latest_results, avg_results, has_valid_computation


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument('--latent-dim', type=int, default=256, help='Latent dimension n_z.')
    parser.add_argument('--image-size', type=int, default=32, help='Image height and width.)')
    parser.add_argument('--num-codebook-vectors', type=int, default=1024, help='Number of codebook vectors.')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar.')
    parser.add_argument('--image-channels', type=int, default=1, help='Number of channels of images.')
    parser.add_argument('--dataset-path', type=str, default='./data', help='Path to data.')
    parser.add_argument('--checkpoint-path', type=str, default='./checkpoints/vqgan_epoch_1.pt', help='Path to checkpoint.')
    parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on')
    parser.add_argument('--batch-size', type=int, default=32, help='Input batch size for training.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--learning-rate', type=float, default=2.25e-05, help='Learning rate.')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta param.')
    parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta param.')
    parser.add_argument('--disc-start', type=int, default=10000, help='When to start the discriminator.')
    parser.add_argument('--disc-factor', type=float, default=1., help='Weighting factor for the Discriminator.')
    parser.add_argument('--l2-loss-factor', type=float, default=1., help='Weighting factor for reconstruction loss.')
    parser.add_argument('--perceptual-loss-factor', type=float, default=1., help='Weighting factor for perceptual loss.')

    parser.add_argument('--pkeep', type=float, default=0.5, help='Percentage for how much latent codes to keep.')
    parser.add_argument('--sos-token', type=int, default=0, help='Start of Sentence token.')

    args = parser.parse_args()
    # 修复：删除Windows特定路径，使用默认路径
    # args.dataset_path = r"C:\Users\dome\datasets\flowers"
    #args.checkpoint_path = r".\checkpoints\vqgan_last_ckpt.pt"

    train_transformer = TrainTransformer(args)
