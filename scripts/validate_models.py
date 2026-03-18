"""
验证 runs/detect 中所有模型在源域和目标域上的性能
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


def get_dataloader(dataset_path, args, data, stride, batch_size=16):
    """构建数据加载器"""
    dataset = build_yolo_dataset(args, dataset_path, batch_size, data, mode="val", stride=stride)
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
    try:
        ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # 检查是否是损坏的 checkpoint（model 为 None 但有 ema）
        if ckpt.get('model') is None and ckpt.get('ema') is not None:
            LOGGER.warning(f"检测到损坏的 checkpoint: {model_path}")
            LOGGER.info(f"Epoch: {ckpt.get('epoch')}, Best fitness: {ckpt.get('best_fitness')}")
            
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
            
            LOGGER.info(f"已修复并保存到: {fixed_path}")
            return fixed_path
        
        return model_path
    except Exception as e:
        LOGGER.warning(f"检查 checkpoint 时出错: {e}")
        return model_path


def validate_model(model_path, source_data, target_data, batch_size=16, imgsz=640, device=""):
    """
    验证单个模型在源域和目标域上的性能
    
    Args:
        model_path: 模型权重路径
        source_data: 源域数据集配置路径
        target_data: 目标域数据集配置路径
        batch_size: 批次大小
        imgsz: 输入图像大小
        device: 设备
    
    Returns:
        dict: 源域和目标域的验证结果
    """
    model_name = Path(model_path).parent.parent.name
    LOGGER.info(f"\n{'='*60}")
    LOGGER.info(f"正在验证模型: {model_name}")
    LOGGER.info(f"模型路径: {model_path}")
    LOGGER.info(f"{'='*60}\n")
    
    # 修复可能的损坏 checkpoint
    fixed_model_path = fix_checkpoint(model_path)
    
    # 准备参数
    args = get_cfg()
    args.model = fixed_model_path
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
    
    # 创建临时模型以获取 stride
    device_obj = select_device(device)
    temp_model = AutoBackend(
        model=fixed_model_path,
        device=device_obj,
        dnn=False,
        data=source_data,
        fp16=False,
    )
    stride = temp_model.stride
    
    # 构建数据加载器
    source_dataloader = get_dataloader(data.get("val"), args, data, stride, batch_size)
    target_dataloader = get_dataloader(target_data_dict.get("val"), args, target_data_dict, stride, batch_size)
    
    # 创建验证器
    validator = DomainAdaptationValidator(
        dataloader=source_dataloader,
        save_dir=Path("runs/validate") / model_name,
        args=args,
        target_dataloader=target_dataloader
    )
    
    # 运行验证 - 不传 model 参数让验证器自己加载
    results = validator(model=None)
    
    # 提取结果 - 打印所有键名用于调试
    LOGGER.debug(f"Results keys: {list(results.keys())}")
    
    source_results = {
        "mAP50": results.get("metrics/mAP50(B)", 0),
        "mAP50-95": results.get("metrics/mAP50-95(B)", 0),
        "precision": results.get("metrics/precision(B)", 0),
        "recall": results.get("metrics/recall(B)", 0),
    }
    
    # 目标域结果键名可能是 target_metrics/... 或直接在 source 键名前加 target_
    target_results = {
        "mAP50": results.get("target_metrics/mAP50(B)", 0) or results.get("target_metrics/mAP50(B)", 0),
        "mAP50-95": results.get("target_metrics/mAP50-95(B)", 0) or results.get("target_metrics/mAP50-95(B)", 0),
        "precision": results.get("target_metrics/precision(B)", 0) or results.get("target_metrics/precision(B)", 0),
        "recall": results.get("target_metrics/recall(B)", 0) or results.get("target_metrics/recall(B)", 0),
    }
    
    return {
        "model": model_name,
        "model_path": str(model_path),
        "fixed_model_path": fixed_model_path if fixed_model_path != model_path else None,
        "source": source_results,
        "target": target_results,
    }


def main():
    """主函数：验证所有模型"""
    
    # 配置
    runs_dir = Path("runs/detect")
    source_data = "datasets/cityscape_yolo/cityscapes.yaml"
    target_data = "datasets/cityscape_foggy_yolo/cityscapes.yaml"
    batch_size = 16
    imgsz = 960
    # device = "cpu"  # 强制使用CPU进行验证
    device = "" # 自动选择，优先GPU
    
    # 获取所有模型目录
    model_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    
    if not model_dirs:
        LOGGER.warning(f"在 {runs_dir} 中没有找到模型")
        return
    
    LOGGER.info(f"发现 {len(model_dirs)} 个模型需要验证")
    
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
                source_data=source_data,
                target_data=target_data,
                batch_size=batch_size,
                imgsz=imgsz,
                device=device,
            )
            all_results.append(result)
        except Exception as e:
            LOGGER.error(f"验证 {model_dir.name} 时出错: {e}")
            import traceback
            traceback.print_exc()
    
    # 打印汇总结果
    LOGGER.info(f"\n{'='*90}")
    LOGGER.info("验证结果汇总")
    LOGGER.info(f"{'='*90}\n")
    
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
        
        print(f"{model_name:<29}{fixed_marker} {src_map50:>12.4f} {src_map5095:>14.4f} {tgt_map50:>14.4f} {tgt_map5095:>16.4f}")
    
    print("-" * 90)
    print("* 表示该模型从损坏的 checkpoint 修复后验证")
    
    # 保存结果到 JSON
    output_file = "runs/validation_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    LOGGER.info(f"\n详细结果已保存到: {output_file}")


if __name__ == "__main__":
    main()
