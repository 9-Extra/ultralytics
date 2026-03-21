"""
验证 runs/detect 中所有模型在源域和目标域上的性能 (v2 - 修复版)

修复了以下问题：
1. 模型路径解析问题
2. 添加更多错误检查和日志
3. 确保使用正确的设备
4. 添加权重完整性检查
"""

import os
import sys
import json
import shutil
from pathlib import Path
import torch

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ultralytics import YOLO
from ultralytics.models.yolo.domain_adapt.val import DomainAdaptationValidator
from ultralytics.utils import LOGGER
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils.torch_utils import select_device
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils.checks import check_imgsz
from ultralytics.cfg import get_cfg, get_save_dir


def get_dataloader(dataset_path, args, data, batch_size=16):
    """构建数据加载器"""
    dataset = build_yolo_dataset(args, dataset_path, batch_size, data, mode="val")
    return build_dataloader(
        dataset,
        batch_size,
        args.workers,
        shuffle=False,
        rank=-1,
        drop_last=False,
        pin_memory=False,
    )


def fix_checkpoint(model_path):
    """
    修复训练崩溃的 checkpoint，提取 ema 模型保存为正确的格式
    
    Returns:
        str: 修复后的模型路径（如果是 checkpoint），否则返回原路径
    """
    model_path = Path(model_path).resolve()  # 转为绝对路径
    
    try:
        ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # 检查是否是损坏的 checkpoint（model 为 None 但有 ema）
        if ckpt.get('model') is None and ckpt.get('ema') is not None:
            LOGGER.warning(f"检测到损坏的 checkpoint: {model_path}")
            print(f"Epoch: {ckpt.get('epoch')}, Best fitness: {ckpt.get('best_fitness')}")
            
            # 创建修复后的模型文件
            fixed_path = str(model_path).replace('.pt', '_fixed.pt')
            
            # 从 ema 提取模型并保存完整 checkpoint
            ema_model = ckpt['ema']
            
            # 保存为标准的 YOLO checkpoint 格式（ema 已经是模型对象）
            torch.save({
                'model': ema_model,  # 直接保存模型对象，不是 state_dict
                'ema': None,
                'epoch': ckpt.get('epoch'),
                'best_fitness': ckpt.get('best_fitness'),
                'date': ckpt.get('date'),
            }, fixed_path)
            
            print(f"已修复并保存到: {fixed_path}")
            return fixed_path, True
        
        return str(model_path), False
    except Exception as e:
        LOGGER.warning(f"检查 checkpoint 时出错: {e}")
        return str(model_path), False


def validate_model(model_path, source_dataloader, target_dataloader, batch_size=16, imgsz=640, device=""):
    """
    验证单个模型在源域和目标域上的性能
    
    Args:
        model_path: 模型权重路径
        source_dataloader: 源域数据加载器
        target_dataloader: 目标域数据加载器
        batch_size: 批次大小
        imgsz: 输入图像大小
        device: 设备
    
    Returns:
        dict: 源域和目标域的验证结果
    """
    model_path = Path(model_path).resolve()
    model_name = model_path.parent.parent.name
    
    print(f"\n{'='*60}")
    print(f"正在验证模型: {model_name}")
    print(f"模型路径: {model_path}")
    print(f"设备: {device if device else 'auto'}")
    print(f"{'='*60}\n")
    
    # 修复可能的损坏 checkpoint
    fixed_model_path, do_fixed = fix_checkpoint(model_path)
    fixed_model_path = Path(fixed_model_path).resolve()
    print(f"使用模型路径: {fixed_model_path}")
    
    # 检查文件是否存在
    if not fixed_model_path.exists():
        LOGGER.error(f"模型文件不存在: {fixed_model_path}")
        return None
    
    # 从数据加载器获取数据集路径
    source_data = source_dataloader.dataset.data.get("yaml_file", "")
    
    # 准备参数
    args = get_cfg()
    args.model = str(fixed_model_path)
    args.data = source_data
    args.batch = batch_size
    args.imgsz = imgsz
    args.device = device
    args.workers = 8
    args.conf = 0.001
    args.iou = 0.6
    args.max_det = 300
    args.single_cls = False
    args.augment = False
    args.task = "detect"
    args.split = "val"
    args.plots = True
    args.verbose = True
    args.save_json = True
    args.save_txt = False
    args.half = True
    args.rect = False
    args.dnn = False
    args.end2end = None
    
    save_dir = Path("runs/validate") / model_name
    
    # 创建验证器
    validator = DomainAdaptationValidator(
        dataloader=source_dataloader,
        save_dir=save_dir,
        args=args,
        target_dataloader=target_dataloader
    )
    
    # 运行验证 - 不传 model 参数让验证器自己加载
    results = validator(model=None)
    
    # 检查结果
    if results is None:
        LOGGER.error("验证返回 None！")
        return None
    
    with open(save_dir / "metrics.txt", "w") as m:
        m.write("==================================\nSource domain:\n")
        m.write(validator.metrics.to_csv())
        # for c in validator.metrics.summary():
        #     for k, v in c.items():
        #         m.write(f"{k}: {v}\n")
        m.write("\n\n==================================\nTarget domain:\n")
        # for c in validator.target_metrics.summary():
        #     for k, v in c.items():
        #         m.write(f"{k}: {v}\n")
        m.write(validator.target_metrics.to_csv())
    
    LOGGER.debug(f"Results keys: {list(results.keys())}")
    
    source_results = {
        "mAP50": results.get("metrics/mAP50(B)", 0),
        "mAP50-95": results.get("metrics/mAP50-95(B)", 0),
        "precision": results.get("metrics/precision(B)", 0),
        "recall": results.get("metrics/recall(B)", 0),
    }
    
    target_results = {
        "mAP50": results.get("target_metrics/mAP50(B)", 0),
        "mAP50-95": results.get("target_metrics/mAP50-95(B)", 0),
        "precision": results.get("target_metrics/precision(B)", 0),
        "recall": results.get("target_metrics/recall(B)", 0),
    }
    
    # 检查结果是否合理
    if source_results["mAP50"] < 0.55:  # 随机模型约为 0.50
        LOGGER.warning(f"源域 mAP50 ({source_results['mAP50']:.4f}) 过低，可能使用了随机权重！")
    
    return {
        "model": model_name,
        "model_path": str(model_path),
        "fixed_model_path": str(fixed_model_path) if do_fixed else None,
        "source": source_results,
        "target": target_results,
    }


def main():
    """主函数：验证所有模型"""
    
    # 打印环境信息
    print(f"\n{'='*80}")
    print("验证脚本 v2 - 环境信息")
    print(f"{'='*80}")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"cuDNN 版本: {torch.backends.cudnn.version()}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"工作目录: {Path.cwd()}")
    print(f"项目根目录: {project_root}")
    print(f"{'='*80}\n")
    
    # 配置
    runs_dir = Path("runs/detect")
    source_data = "datasets/cityscape_yolo/cityscapes.yaml"
    target_data = "datasets/cityscape_foggy_yolo/cityscapes.yaml"
    batch_size = 16
    imgsz = 960
    device = ""  # 自动选择，优先GPU
    
    # 检查 runs/detect 目录
    if not runs_dir.exists():
        LOGGER.error(f"目录不存在: {runs_dir}")
        return
    
    # 获取所有模型目录
    model_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    
    if not model_dirs:
        LOGGER.warning(f"在 {runs_dir} 中没有找到模型")
        return
    
    print(f"发现 {len(model_dirs)} 个模型需要验证")
    
    # 准备参数用于数据加载器
    args = get_cfg()
    args.batch = batch_size
    args.imgsz = imgsz
    args.workers = 8
    args.conf = 0.001
    args.iou = 0.6
    args.max_det = 300
    args.single_cls = False
    args.augment = False
    args.task = "detect"
    args.split = "val"
    args.plots = False
    args.verbose = False
    args.save_json = False
    args.save_txt = False
    args.half = False
    args.rect = False
    args.dnn = False
    args.end2end = None
    
    # 检查数据集
    data = check_det_dataset(source_data)
    target_data_dict = check_det_dataset(target_data)
    
    # 构建数据加载器
    source_dataloader = get_dataloader(data.get("val"), args, data, batch_size)
    target_dataloader = get_dataloader(target_data_dict.get("val"), args, target_data_dict, batch_size)
    print("数据集加载完成！\n")
    
    # 存储所有结果
    all_results = []
    
    # 验证每个模型
    for model_dir in sorted(model_dirs):
        best_pt = model_dir / "weights" / "best.pt"
        
        if not best_pt.exists():
            LOGGER.warning(f"跳过 {model_dir.name}: 未找到 best.pt")
            continue
        
        try:
            result = validate_model(
                model_path=str(best_pt),
                source_dataloader=source_dataloader,
                target_dataloader=target_dataloader,
                batch_size=batch_size,
                imgsz=imgsz,
                device=device,
            )
            if result is not None:
                all_results.append(result)
        except Exception as e:
            LOGGER.error(f"验证 {model_dir.name} 时出错: {e}")
            import traceback
            traceback.print_exc()
    
    # 打印汇总结果
    print(f"\n{'='*90}")
    print("验证结果汇总")
    print(f"{'='*90}\n")
    
    print(f"{'模型':<30} {'源域 mAP50':>12} {'源域 mAP50-95':>14} {'目标域 mAP50':>14} {'目标域 mAP50-95':>16}")
    print("-" * 90)
    
    for r in all_results:
        model_name = r["model"]
        src_map50 = r["source"]["mAP50"]
        src_map5095 = r["source"]["mAP50-95"]
        tgt_map50 = r["target"]["mAP50"]
        tgt_map5095 = r["target"]["mAP50-95"]
        
        # 标记是否修复过
        fixed_marker = "*" if r.get("fixed_model_path") else " "
        
        # 标记是否异常（mAP50 < 0.6 可能是随机权重）
        warning_marker = " (!)" if src_map50 < 0.6 else ""
        
        print(f"{model_name:<29}{fixed_marker} {src_map50:>12.4f} {src_map5095:>14.4f} {tgt_map50:>14.4f} {tgt_map5095:>16.4f}{warning_marker}")
    
    print("-" * 90)
    print("* 表示该模型从损坏的 checkpoint 修复后验证")
    print("(!) 表示 mAP50 异常低（可能使用了随机权重）")
    
    # 保存结果到 JSON
    output_file = "runs/validation_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n详细结果已保存到: {output_file}")
    

if __name__ == "__main__":
    main()
