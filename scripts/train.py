#!/usr/bin/env python3
"""
YOLODA (YOLO Domain Adaptation) 训练脚本
使用 CityScape 数据集进行域适应训练
源域: 正常天气 (cityscape_yolo)
目标域: 雾天 (cityscape_foggy_yolo)
"""

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import yaml_model_load, DetectionModel


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Train YOLODA model for domain adaptation"
    )

    # 模型配置
    parser.add_argument(
        "--model",
        type=str,
        default="yolo26n-da.yaml",
        help="模型配置文件路径 (默认: yolo26n-da.yaml)",
    )
    parser.add_argument(
        "--weights", type=str, default=None, help="预训练权重文件路径 (可选)"
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
        "--lr0", type=float, default=0.01, help="初始学习率 (默认: 0.01)"
    )
    parser.add_argument(
        "--domain-loss-weight",
        type=float,
        default=0.2,
        help="域适应损失初始权重 (默认: 0.2)",
    )
    parser.add_argument(
        "--domain-loss-final-weight",
        type=float,
        default=0.05,
        help="域适应损失最终权重，用于动态权重调度 (默认: 0.05)",
    )
    parser.add_argument(
        "--domain-loss-schedule",
        type=str,
        default="fixed",
        choices=["fixed", "linear", "cosine"],
        help="域损失权重调度策略: 'fixed'=固定, 'linear'=线性衰减, 'cosine'=余弦衰减 (默认: fixed)",
    )
    parser.add_argument(
        "--grl-weight",
        type=float,
        default=-0.1,
        help="梯度反转层系数 (默认: -0.1)",
    )
    parser.add_argument(
        "--bn_freeze",
        action="store_true",
        help="是否在目标域前向传播时冻结BatchNorm参数 (默认: False)",
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
        "--name", type=str, default="yolo-da", help="实验名称 (默认: yolo-da)"
    )
    parser.add_argument(
        "--patience", type=int, default=50, help="早停耐心值 (默认: 50)"
    )
    parser.add_argument(
        "--save-period", type=int, default=10, help="每 N 轮保存一次检查点 (默认: 10)"
    )
    parser.add_argument(
        "--complie", action="store_true", help="是否使用torch.complie"
    )

    return parser.parse_args()


def main():
    """主训练函数"""
    torch.multiprocessing.set_start_method("spawn")

    args = parse_args()

    # 获取项目根目录
    project_root = Path(__file__).parent.parent.absolute()

    # 构建完整路径
    # 模型路径：如果是简单文件名（不含路径分隔符），直接传给 YOLO 类自动解析
    # 如果是实际路径，则构建完整路径
    if Path(args.model).is_absolute():
        model_path = args.model
    elif Path(args.model).exists():
        model_path = args.model
    else:
        # 让 Ultralytics 自动解析模型名称（如 yolo26l-da.yaml）
        model_path = args.model

    # 数据集路径：如果是相对路径，基于项目根目录构建完整路径
    data_path = args.data if Path(args.data).is_absolute() else project_root / args.data
    target_data_path = (
        args.target_data
        if Path(args.target_data).is_absolute()
        else project_root / args.target_data
    )

    print("=" * 60)
    print("YOLODA 域适应训练")
    print("=" * 60)
    print(f"模型: {model_path}")
    print(f"源域数据: {data_path}")
    print(f"目标域数据: {target_data_path}")
    print(f"训练轮数: {args.epochs}")
    print(f"批次大小: {args.batch}")
    print(f"图像尺寸: {args.imgsz}")
    print(f"域适应损失权重: {args.domain_loss_weight}")
    if args.domain_loss_schedule != "fixed":
        print(f"域适应损失最终权重: {args.domain_loss_final_weight}")
        print(f"域适应损失调度策略: {args.domain_loss_schedule}")
    print(f"梯度反转层系数: {args.grl_weight}")
    print(f"目标域冻结BN更新: {args.bn_freeze}")
    print(f"设备: {args.device}")
    print("=" * 60)

    # 加载模型
    if args.weights:
        print(f"加载预训练权重: {args.weights}")
        model = YOLO(args.weights)
    else:
        print(f"加载模型配置: {model_path}")
        model = YOLO(str(model_path))

    # 开始训练
    print("\n开始训练...\n")
    results = model.train(
        data=str(data_path),
        target_data=str(target_data_path),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        lr0=args.lr0,
        optimizer="MuSGD",
        deterministic=False,
        domain_loss_weight=args.domain_loss_weight,
        domain_loss_final_weight=args.domain_loss_final_weight,
        domain_loss_schedule=args.domain_loss_schedule,
        grl_weight=args.grl_weight,
        bn_freeze=args.bn_freeze,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        patience=args.patience,
        save_period=args.save_period,
        amp=True,
        compile=args.complie,
        cache="disk",
        
        # 数据增强
        scale=0.2,
        mosaic=0.8,
        mixup=0.1
    )

    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
