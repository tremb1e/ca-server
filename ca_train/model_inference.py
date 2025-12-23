#!/usr/bin/env python3
"""
VQGAN Transformer 模型推理示例

此脚本展示如何：
1. 加载已训练的用户模型
2. 对新数据进行身份认证
3. 批量处理多个用户的认证任务
"""

import os
import numpy as np
import torch
import argparse
from transformer import VQGANTransformer
from construct_dataset import get_mnist_con_another
from data_loaders import CIFAR10DataLoader_pro
from evaluation import utils_eer
from sklearn.metrics import roc_auc_score
import glob

class VQGANInference:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.models = {}
        
    def load_single_model(self, user_id):
        """加载单个用户的模型"""
        model_path = os.path.join(self.args.checkpoint_path, f"transformer_{user_id}.pt")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"用户 {user_id} 的模型文件不存在: {model_path}")
        
        model = VQGANTransformer(self.args).to(device=self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        
        self.models[user_id] = model
        print(f"成功加载用户 {user_id} 的模型")
        return model
    
    def load_all_models(self):
        """加载所有可用的用户模型"""
        model_pattern = os.path.join(self.args.checkpoint_path, "transformer_*.pt")
        model_files = glob.glob(model_pattern)
        
        if not model_files:
            print(f"在 {self.args.checkpoint_path} 中未找到任何模型文件")
            return {}
        
        print(f"找到 {len(model_files)} 个模型文件，开始加载...")
        
        for model_file in sorted(model_files):
            try:
                filename = os.path.basename(model_file)
                user_id = int(filename.split('_')[1].split('.')[0])
                self.load_single_model(user_id)
            except Exception as e:
                print(f"加载 {model_file} 失败: {str(e)}")
        
        print(f"成功加载 {len(self.models)} 个用户模型: {list(self.models.keys())}")
        return self.models
    
    def authenticate_user(self, user_id, test_data=None):
        """对特定用户进行身份认证"""
        if user_id not in self.models:
            print(f"用户 {user_id} 的模型未加载，正在加载...")
            try:
                self.load_single_model(user_id)
            except FileNotFoundError:
                print(f"用户 {user_id} 的模型不存在")
                return None
        
        model = self.models[user_id]
        
        # 如果没有提供测试数据，使用默认数据
        if test_data is None:
            print(f"为用户 {user_id} 加载测试数据...")
            x_train, y_train, x_valid, y_valid, x_plot, y_plot, subjects = get_mnist_con_another(cls=user_id)
            test_dataloader = CIFAR10DataLoader_pro(x_plot, y_plot, split="test", 
                                                   batch_size=self.args.batch_size, user_num=user_id)
        else:
            test_dataloader = test_data
        
        print(f"开始用户 {user_id} 的身份认证测试...")
        results = self.evaluate_model(model, test_dataloader)
        
        return results
    
    def evaluate_model(self, model, test_dataloader):
        """评估模型性能"""
        model.eval()
        
        all_scores = []
        all_targets = []
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_dataloader):
                inputs = inputs.type(torch.FloatTensor).to(self.device)
                targets = targets.to(self.device)
                
                # 获取重构结果
                log, rec, half, sampled_imgs = model.log_images(inputs)
                
                # 计算重构误差
                scores = torch.mean(torch.pow((inputs - rec), 2), dim=[1,2,3])
                
                all_scores.extend(scores.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        all_scores = np.array(all_scores)
        all_targets = np.array(all_targets)
        
        # 确保是二分类问题
        unique_labels = np.unique(all_targets)
        if len(unique_labels) > 2:
            all_targets = (all_targets == 1).astype(float)
        
        # 计算性能指标
        try:
            auc = roc_auc_score(all_targets, -all_scores)
        except ValueError:
            auc = 0.5
        
        far, frr, eer, f1 = utils_eer(all_targets, -all_scores)
        
        results = {
            'auc': auc,
            'far': far,
            'frr': frr,
            'eer': eer,
            'f1': f1,
            'num_samples': len(all_targets),
            'num_genuine': np.sum(all_targets == 1),
            'num_imposter': np.sum(all_targets == 0)
        }
        
        return results

def main():
    parser = argparse.ArgumentParser(description="VQGAN Transformer 模型推理")
    
    # 模型参数 (与训练时保持一致)
    parser.add_argument('--latent-dim', type=int, default=256, help='Latent dimension n_z.')
    parser.add_argument('--image-size', type=int, default=32, help='Image height and width.')
    parser.add_argument('--num-codebook-vectors', type=int, default=1024, help='Number of codebook vectors.')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar.')
    parser.add_argument('--image-channels', type=int, default=1, help='Number of channels of images.')
    parser.add_argument('--pkeep', type=float, default=0.5, help='Percentage for how much latent codes to keep.')
    parser.add_argument('--sos-token', type=int, default=0, help='Start of Sentence token.')
    
    # 推理参数
    parser.add_argument('--checkpoint-path', type=str, default='./checkpoints_trans',
                       help='模型加载路径')
    parser.add_argument('--device', type=str, default='cuda',
                       help='推理设备')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='批次大小')
    
    # 任务参数
    parser.add_argument('--mode', type=str, default='load_only', 
                       choices=['single', 'load_only'],
                       help='推理模式: single(单用户), load_only(仅加载)')
    parser.add_argument('--user-id', type=int, default=1,
                       help='单用户推理的用户ID')
    
    args = parser.parse_args()
    
    print("=== VQGAN Transformer 模型推理 ===")
    print(f"模型路径: {args.checkpoint_path}")
    print(f"推理设备: {args.device}")
    print(f"模式: {args.mode}")
    
    # 初始化推理器
    inference = VQGANInference(args)
    
    if args.mode == 'load_only':
        # 仅加载模型
        models = inference.load_all_models()
        print(f"已加载 {len(models)} 个用户模型")
        
    elif args.mode == 'single':
        # 单用户认证
        print(f"对用户 {args.user_id} 进行身份认证...")
        results = inference.authenticate_user(args.user_id)
        if results:
            print(f"\n用户 {args.user_id} 认证结果:")
            print(f"AUC: {results['auc']:.4f}")
            print(f"EER: {results['eer']:.4f}%")
            print(f"FAR: {results['far']:.4f}%")
            print(f"FRR: {results['frr']:.4f}%")
            print(f"F1: {results['f1']:.4f}")

if __name__ == "__main__":
    main()
