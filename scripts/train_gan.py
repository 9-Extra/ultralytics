#!/usr/bin/env python3
"""
YOLO-GAN (GAN-style Domain Adaptation) 训练脚本
使用 CityScape 数据集进行域适应训练 - GAN对抗训练版本

架构:
  - 生成器: YOLOv26 骨干网络 + 检测头
  - 判别器: 独立的域分类器 (DomainDiscriminator)
  
训练策略:
  - 判别器每批次优化 k 次
  - 生成器每批次优化 1 次
  - 交替优化实现对抗训练

源域: 正常天气 (cityscape_yolo)
目标域: 雾天 (cityscape_foggy_yolo)
"""

import argparse
from pathlib import Path

import torch
from ultralytics.models.yolo.domain_adapt.gan_trainer import GANDomainAdaptationTrainer


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Train YOLO-GAN model for domain adaptation using GAN-style adversarial training"
    )

    # 模型配置
    parser.add_argument(
        "--model",
        type=str,
        default="yolo26n-gan.yaml",
        help="模型配置文件路径 (默认: yolo26n-gan.yaml)",
    )
    parser.add_argument(
        "--weights", 
        type=str, 
        default=None, 
        help="预训练权重文件路径 (可选)"
    )

    # 数据集配置
    parser.add_argument(
        "--data",
        type=str,
        default="datasets/cityscape_yolo/cityscapes.yaml",
        help="源域数据集配置文件路径 (默认: datasets/cityscape_yolo/cityscapes.yaml)",
    )
    parser.add_argument(
        "--target-data",
        type=str,
        default="datasets/cityscape_foggy_yolo/cityscapes.yaml",
        help="目标域数据集配置文件路径 (默认: datasets/cityscape_foggy_yolo/cityscapes.yaml)",
    )

    # 训练超参数
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数 (默认: 100)")
    parser.add_argument("--batch", type=int, default=8, help="批次大小 (默认: 8)")
    parser.add_argument(
        "--imgsz", type=int, default=960, help="输入图像尺寸 (默认: 960)"
    )
    parser.add_argument(
        "--lr0", type=float, default=0.01, help="生成器初始学习率 (默认: 0.01)"
    )
    
    # GAN 特有超参数
    parser.add_argument(
        "--d-steps",
        type=int,
        default=3,
        help="每批次判别器优化次数 (默认: 3)",
    )
    parser.add_argument(
        "--d-lr",
        type=float,
        default=0.001,
        help="判别器学习率 (默认: 0.001)",
    )
    parser.add_argument(
        "--lambda-adv",
        type=float,
        default=0.1,
        help="对抗损失权重 (默认: 0.1)",
    )
    parser.add_argument(
        "--discriminator-hidden",
        type=int,
        default=256,
        help="判别器隐藏层维度 (默认: 256)",
    )

    # 其他配置
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="训练设备 (默认: 0, 使用 GPU 0; 可设置为 'cpu' 或 '0,1,2,3' 使用多卡)",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="数据加载器工作进程数 (默认: 4)"
    )
    parser.add_argument(
        "--project", type=str, default="", help="项目保存路径 (默认: " ")"
    )
    parser.add_argument(
        "--name", type=str, default="yolo-gan-da", help="实验名称 (默认: yolo-gan-da)"
    )
    parser.add_argument(
        "--patience", type=int, default=50, help="早停耐心值 (默认: 50)"
    )
    parser.add_argument(
        "--save-period", type=int, default=10, help="每 N 轮保存一次检查点 (默认: 10)"
    )
    parser.add_argument(
        "--compile", action="store_true", help="是否使用 torch.compile"
    )

    return parser.parse_args()


def main():
    """主训练函数"""
    torch.multiprocessing.set_start_method("spawn", force=True)

    args = parse_args()

    # 获取项目根目录
    project_root = Path(__file__).parent.parent.absolute()

    # 构建完整路径
    if Path(args.model).is_absolute():
        model_path = args.model
    elif Path(args.model).exists():
        model_path = args.model
    else:
        # 让 Ultralytics 自动解析模型名称
        model_path = args.model

    # 数据集路径
    data_path = args.data if Path(args.data).is_absolute() else project_root / args.data
    target_data_path = (
        args.target_data
        if Path(args.target_data).is_absolute()
        else project_root / args.target_data
    )

    print("=" * 70)
    print("YOLO-GAN 域适应训练 (GAN-style Domain Adaptation)")
    print("=" * 70)
    print(f"模型: {model_path}")
    print(f"源域数据: {data_path}")
    print(f"目标域数据: {target_data_path}")
    print(f"训练轮数: {args.epochs}")
    print(f"批次大小: {args.batch}")
    print(f"图像尺寸: {args.imgsz}")
    print("-" * 70)
    print("GAN 训练参数:")
    print(f"  判别器每批次迭代: {args.d_steps}")
    print(f"  判别器学习率: {args.d_lr}")
    print(f"  对抗损失权重: {args.lambda_adv}")
    print(f"  判别器隐藏层: {args.discriminator_hidden}")
    print("-" * 70)
    print(f"生成器学习率: {args.lr0}")
    print(f"设备: {args.device}")
    print("=" * 70)

    # 构建训练参数
    overrides = {
        "model": str(model_path),
        "data": str(data_path),
        "target_data": str(target_data_path),
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "lr0": args.lr0,
        "device": args.device,
        "workers": args.workers,
        "project": args.project,
        "name": args.name,
        "patience": args.patience,
        "save_period": args.save_period,
        "compile": args.compile,
        "amp": True,
        "deterministic": False,
        "optimizer": "MuSGD",
        "cache": "disk",
        
        # GAN 特有参数
        "d_steps": args.d_steps,
        "d_lr": args.d_lr,
        "lambda_adv": args.lambda_adv,
        "discriminator_hidden": args.discriminator_hidden,
        
        # 数据增强
        "scale": 0.2,
        "mosaic": 0.8,
        "mixup": 0.1,
    }

    # 如果有预训练权重，添加进去
    if args.weights:
        print(f"\n加载预训练权重: {args.weights}")
        overrides["pretrained"] = args.weights

    # 创建训练器并开始训练
    print("\n初始化 GAN Domain Adaptation Trainer...")
    trainer = GANDomainAdaptationTrainer(overrides=overrides)
    
    print("\n开始训练...\n")
    results = trainer.train()

    print("\n" + "=" * 70)
    print("训练完成!")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
