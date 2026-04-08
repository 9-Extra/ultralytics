#!/usr/bin/env python3
"""
绘制域适应实验的曲线图。

从 ./runs/detect 中读取所有实验的 results.csv，绘制以下曲线：
1. 域分类准确率在源域和目标域的变化
2. 训练集中 domain_loss 的变化
3. 验证集上源域和目标域 mAP50 的变化

布局：同一个实验的不同曲线位于同一列，不同实验的相同曲线位于同一行。
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


def load_experiment_data(exp_path: Path) -> tuple[str, pd.DataFrame | None]:
    """加载单个实验的结果数据。
    
    Args:
        exp_path: 实验文件夹路径
        
    Returns:
        (实验名称, DataFrame或None)
    """
    results_file = exp_path / "results.csv"
    if not results_file.exists():
        return exp_path.name, None
    
    try:
        df = pd.read_csv(results_file)
        # 清理列名（去除空格）
        df.columns = [col.strip() for col in df.columns]
        return exp_path.name, df
    except Exception as e:
        print(f"警告: 无法读取 {results_file}: {e}")
        return exp_path.name, None


def is_domain_adaptation_exp(df: pd.DataFrame) -> bool:
    """检查是否为域适应实验（包含域适应特有的列）。"""
    required_cols = ['train/dom_loss', 'metrics/domain_acc', 'target_metrics/domain_acc']
    return all(col in df.columns for col in required_cols)


def plot_domain_curves(experiments: dict[str, pd.DataFrame], save_dir: Path):
    """绘制所有实验的曲线图。
    
    布局：
    - 行：不同类型的曲线（域分类准确率、domain_loss、val mAP50）
    - 列：不同实验
    
    Args:
        experiments: 实验名称到DataFrame的映射
        save_dir: 保存图片的目录
    """
    # 过滤出域适应实验
    da_experiments = {
        name: df for name, df in experiments.items() 
        if is_domain_adaptation_exp(df)
    }
    
    if not da_experiments:
        print("警告: 未找到域适应实验（需要包含 train/dom_loss, metrics/domain_acc 等列）")
        return
    
    print(f"找到 {len(da_experiments)} 个域适应实验: {list(da_experiments.keys())}")
    
    # 创建图形
    # 3行（3种曲线类型）x N列（N个实验）
    n_exps = len(da_experiments)
    fig, axes = plt.subplots(3, n_exps, figsize=(5 * n_exps, 12), squeeze=False)
    
    # 设置行标题
    row_titles = [
        'Domain Accuracy (Source vs Target)',
        'Domain Loss (Train)',
        'mAP50 (Validation - Source vs Target)'
    ]
    
    for row_idx, title in enumerate(row_titles):
        axes[row_idx, 0].set_ylabel(title, fontsize=11, fontweight='bold')
    
    # 为每个实验绘制曲线
    for col_idx, (exp_name, df) in enumerate(sorted(da_experiments.items())):
        epochs = df['epoch'].values
        
        # 设置列标题
        axes[0, col_idx].set_title(exp_name, fontsize=12, fontweight='bold')
        
        # Row 0: 域分类准确率
        ax = axes[0, col_idx]
        if 'metrics/domain_acc' in df.columns:
            ax.plot(epochs, df['metrics/domain_acc'].values, 
                   label='Source', color='blue', linewidth=1.5)
        if 'target_metrics/domain_acc' in df.columns:
            ax.plot(epochs, df['target_metrics/domain_acc'].values, 
                   label='Target', color='red', linewidth=1.5)
        ax.set_xlabel('Epoch')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        # Row 1: Domain Loss
        ax = axes[1, col_idx]
        if 'train/dom_loss' in df.columns:
            ax.plot(epochs, df['train/dom_loss'].values, 
                   label='Train dom_loss', color='green', linewidth=1.5)
            ax.set_xlabel('Epoch')
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # Row 2: Validation mAP50
        ax = axes[2, col_idx]
        if 'metrics/mAP50(B)' in df.columns:
            ax.plot(epochs, df['metrics/mAP50(B)'].values, 
                   label='Source (Val)', color='blue', linewidth=1.5)
        if 'target_metrics/mAP50(B)' in df.columns:
            ax.plot(epochs, df['target_metrics/mAP50(B)'].values, 
                   label='Target (Val)', color='red', linewidth=1.5)
        ax.set_xlabel('Epoch')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        

    
    plt.tight_layout()
    
    # 保存图片
    save_path = save_dir / 'domain_adaptation_curves.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"曲线图已保存到: {save_path}")
    
    # 同时保存为PDF
    save_path_pdf = save_dir / 'domain_adaptation_curves.pdf'
    plt.savefig(save_path_pdf, bbox_inches='tight')
    print(f"曲线图已保存到: {save_path_pdf}")
    
    plt.close()


def plot_individual_curves(experiments: dict[str, pd.DataFrame], save_dir: Path):
    """为每个实验单独绘制详细的曲线图。
    
    Args:
        experiments: 实验名称到DataFrame的映射
        save_dir: 保存图片的目录
    """
    da_experiments = {
        name: df for name, df in experiments.items() 
        if is_domain_adaptation_exp(df)
    }
    
    for exp_name, df in sorted(da_experiments.items()):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Experiment: {exp_name}', fontsize=14, fontweight='bold')
        
        epochs = df['epoch'].values
        
        # 子图1: 域分类准确率
        ax = axes[0, 0]
        if 'metrics/domain_acc' in df.columns:
            ax.plot(epochs, df['metrics/domain_acc'].values, 
                   label='Source Domain', color='blue', linewidth=2, marker='o', markersize=3)
        if 'target_metrics/domain_acc' in df.columns:
            ax.plot(epochs, df['target_metrics/domain_acc'].values, 
                   label='Target Domain', color='red', linewidth=2, marker='s', markersize=3)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Domain Accuracy', fontsize=11)
        ax.set_title('Domain Classification Accuracy', fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        # 子图2: Domain Loss
        ax = axes[0, 1]
        if 'train/dom_loss' in df.columns:
            ax.plot(epochs, df['train/dom_loss'].values, 
                   label='Train Domain Loss', color='green', linewidth=2, marker='o', markersize=3)
        if 'val/dom_loss' in df.columns:
            ax.plot(epochs, df['val/dom_loss'].values, 
                   label='Val Domain Loss', color='orange', linewidth=2, marker='s', markersize=3)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Domain Loss', fontsize=11)
        ax.set_title('Domain Loss', fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # 子图3: Validation mAP50
        ax = axes[1, 0]
        has_val_data = False
        if 'metrics/mAP50(B)' in df.columns:
            ax.plot(epochs, df['metrics/mAP50(B)'].values, 
                   label='Source Domain (Val)', color='blue', linewidth=2, marker='o', markersize=3)
            has_val_data = True
        if 'target_metrics/mAP50(B)' in df.columns:
            ax.plot(epochs, df['target_metrics/mAP50(B)'].values, 
                   label='Target Domain (Val)', color='red', linewidth=2, marker='s', markersize=3)
            has_val_data = True
        if has_val_data:
            ax.set_xlabel('Epoch', fontsize=11)
            ax.set_ylabel('mAP50', fontsize=11)
            ax.set_title('Validation mAP50', fontsize=12)
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1.05])
        
        # 子图4: 不绘制（保持为空或隐藏）
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # 保存图片
        save_path = save_dir / f'{exp_name}_curves.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  单独曲线图已保存: {save_path}")
        
        plt.close()


def main():
    """主函数。"""
    # 路径设置
    runs_dir = Path('./runs/detect')
    save_dir = Path('./check')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if not runs_dir.exists():
        print(f"错误: 运行目录不存在: {runs_dir}")
        return
    
    # 加载所有实验数据
    experiments = {}
    print(f"扫描实验目录: {runs_dir}")
    
    for exp_path in sorted(runs_dir.iterdir()):
        if exp_path.is_dir() and not exp_path.name.startswith('.'):
            exp_name, df = load_experiment_data(exp_path)
            if df is not None:
                experiments[exp_name] = df
                exp_type = "DA" if is_domain_adaptation_exp(df) else "Standard"
                print(f"  加载: {exp_name} ({exp_type}, {len(df)} epochs)")
    
    if not experiments:
        print("错误: 未找到任何实验结果")
        return
    
    print(f"\n共加载 {len(experiments)} 个实验")
    
    # 绘制对比曲线图（所有实验在同一图中）
    print("\n绘制对比曲线图...")
    plot_domain_curves(experiments, save_dir)
    
    # 为每个实验绘制单独的详细曲线图
    print("\n绘制单独实验曲线图...")
    plot_individual_curves(experiments, save_dir)
    
    print("\n完成！所有曲线图已保存到:", save_dir)


if __name__ == '__main__':
    main()
