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
    model_path = Path(model_path).resolve()  # 转为绝对路径
    
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
        
        return str(model_path)
    except Exception as e:
        LOGGER.warning(f"检查 checkpoint 时出错: {e}")
        return str(model_path)


def verify_model_weights(model, model_name="model"):
    """验证模型权重是否正确加载"""
    # 获取底层模型
    actual_model = model.model if hasattr(model, 'model') else model
    
    # 找到第一个卷积层
    first_conv = None
    for m in actual_model.modules():
        if hasattr(m, 'weight') and m.weight is not None and first_conv is None:
            first_conv = m.weight
            break
    
    if first_conv is None:
        LOGGER.error(f"{model_name}: 未找到权重！")
        return False
    
    w_std = first_conv.std().item()
    w_mean = first_conv.mean().item()
    
    LOGGER.info(f"{model_name} 第一层权重: mean={w_mean:.6f}, std={w_std:.6f}")
    
    # 检查是否是随机权重
    if w_std < 0.01:
        LOGGER.error(f"{model_name}: 权重 std 过小 ({w_std:.6f})，可能是未初始化的权重！")
        return False
    if abs(w_mean) < 0.001 and 0.01 < w_std < 0.1:
        LOGGER.warning(f"{model_name}: 权重分布接近随机初始化 (std={w_std:.6f})！")
        return False
    
    LOGGER.info(f"{model_name}: 权重检查通过")
    return True


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
    model_path = Path(model_path).resolve()
    model_name = model_path.parent.parent.name
    
    LOGGER.info(f"\n{'='*60}")
    LOGGER.info(f"正在验证模型: {model_name}")
    LOGGER.info(f"模型路径: {model_path}")
    LOGGER.info(f"设备: {device if device else 'auto'}")
    LOGGER.info(f"{'='*60}\n")
    
    # 修复可能的损坏 checkpoint
    fixed_model_path = fix_checkpoint(model_path)
    fixed_model_path = Path(fixed_model_path).resolve()
    LOGGER.info(f"使用模型路径: {fixed_model_path}")
    
    # 检查文件是否存在
    if not fixed_model_path.exists():
        LOGGER.error(f"模型文件不存在: {fixed_model_path}")
        return None
    
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
    args.plots = False
    args.verbose = False
    args.save_json = False
    args.save_txt = False
    args.half = False
    args.rect = False
    args.dnn = False
    args.end2end = None
    
    # 检查数据集
    LOGGER.info(f"加载源域数据集: {source_data}")
    data = check_det_dataset(source_data)
    LOGGER.info(f"源域验证集: {data.get('val')}")
    
    LOGGER.info(f"加载目标域数据集: {target_data}")
    target_data_dict = check_det_dataset(target_data)
    LOGGER.info(f"目标域验证集: {target_data_dict.get('val')}")
    
    # 创建设备
    device_obj = select_device(device)
    LOGGER.info(f"使用设备: {device_obj}")
    
    # 创建临时模型以获取 stride
    LOGGER.info("加载模型获取 stride...")
    temp_model = AutoBackend(
        model=str(fixed_model_path),
        device=device_obj,
        dnn=False,
        data=source_data,
        fp16=False,
    )
    stride = temp_model.stride
    LOGGER.info(f"模型 stride: {stride}")
    
    # 验证模型权重
    if not verify_model_weights(temp_model, "temp_model"):
        LOGGER.error("模型权重验证失败！可能使用了随机权重。")
        return None
    
    # 构建数据加载器
    LOGGER.info("构建数据加载器...")
    source_dataloader = get_dataloader(data.get("val"), args, data, stride, batch_size)
    target_dataloader = get_dataloader(target_data_dict.get("val"), args, target_data_dict, stride, batch_size)
    LOGGER.info(f"源域数据批次: {len(source_dataloader)}")
    LOGGER.info(f"目标域数据批次: {len(target_dataloader)}")
    
    # 创建验证器
    LOGGER.info("创建验证器...")
    validator = DomainAdaptationValidator(
        dataloader=source_dataloader,
        save_dir=Path("runs/validate") / model_name,
        args=args,
        target_dataloader=target_dataloader
    )
    
    # 运行验证 - 不传 model 参数让验证器自己加载
    LOGGER.info("开始验证...")
    results = validator(model=None)
    
    # 检查结果
    if results is None:
        LOGGER.error("验证返回 None！")
        return None
    
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
        "fixed_model_path": str(fixed_model_path) if fixed_model_path != str(model_path) else None,
        "source": source_results,
        "target": target_results,
    }


def main():
    """主函数：验证所有模型"""
    
    # 打印环境信息
    LOGGER.info(f"\n{'='*80}")
    LOGGER.info("验证脚本 v2 - 环境信息")
    LOGGER.info(f"{'='*80}")
    LOGGER.info(f"Python: {sys.version}")
    LOGGER.info(f"PyTorch: {torch.__version__}")
    LOGGER.info(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        LOGGER.info(f"CUDA 版本: {torch.version.cuda}")
        LOGGER.info(f"cuDNN 版本: {torch.backends.cudnn.version()}")
        LOGGER.info(f"GPU: {torch.cuda.get_device_name(0)}")
    LOGGER.info(f"工作目录: {Path.cwd()}")
    LOGGER.info(f"项目根目录: {project_root}")
    LOGGER.info(f"{'='*80}\n")
    
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
            if result is not None:
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
        
        # 标记是否异常（mAP50 < 0.6 可能是随机权重）
        warning_marker = " (!)" if src_map50 < 0.6 else ""
        
        print(f"{model_name:<29}{fixed_marker} {src_map50:>12.4f} {src_map5095:>14.4f} {tgt_map50:>14.4f} {tgt_map5095:>16.4f}{warning_marker}")
    
    print("-" * 90)
    print("* 表示该模型从损坏的 checkpoint 修复后验证")
    print("(!) 表示 mAP50 异常低（可能使用了随机权重）")
    
    # 保存结果到 JSON
    output_file = "runs/validation_results_v2.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    LOGGER.info(f"\n详细结果已保存到: {output_file}")
    
    # 如果有异常结果，给出警告
    abnormal_results = [r for r in all_results if r["source"]["mAP50"] < 0.6]
    if abnormal_results:
        LOGGER.warning(f"\n警告: 发现 {len(abnormal_results)} 个模型的 mAP50 异常低，可能使用了随机权重！")
        for r in abnormal_results:
            LOGGER.warning(f"  - {r['model']}: mAP50 = {r['source']['mAP50']:.4f}")


if __name__ == "__main__":
    main()
