"""
验证 runs/detect 中所有模型在源域和目标域上的性能 (v3 - 支持增量验证)

新增功能：
- 支持 --force-all 参数强制重新验证所有模型
- 支持从 runs/validation_results.json 读取缓存结果，跳过已验证的模型
- 最终表格始终包含所有模型的结果（缓存 + 新验证）

修复了以下问题：
1. 模型路径解析问题
2. 添加更多错误检查和日志
3. 确保使用正确的设备
"""

import sys
import json
import argparse
from pathlib import Path
import torch

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入rich用于美化表格输出
try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

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


def create_base_args(batch_size, imgsz, device="", plots=False, verbose=False, half=False):
    """
    创建基础配置参数
    
    Args:
        batch_size: 批次大小
        imgsz: 输入图像大小
        device: 设备
        plots: 是否保存绘图
        verbose: 是否详细输出
        half: 是否使用半精度
    
    Returns:
        args: 配置好的参数对象
    """
    args = get_cfg()
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
    args.plots = plots
    args.verbose = verbose
    args.save_json = False
    args.save_txt = False
    args.half = half
    args.rect = False
    args.dnn = False
    args.end2end = None
    return args


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
    
    # 检查文件是否存在
    if not model_path.exists():
        LOGGER.error(f"模型文件不存在: {model_path}")
        return None
    
    # 从数据加载器获取数据集路径
    source_data = source_dataloader.dataset.data.get("yaml_file", "")
    
    # 准备参数（基于基础配置，设置验证专用参数）
    args = create_base_args(batch_size, imgsz, device, plots=True, verbose=True, half=True)
    args.model = str(model_path)
    args.data = source_data
    
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
        # 添加域分类指标
        if validator.domain_stats is not None:
            m.write("\n\n==================================\nDomain classification:\n")
            m.write(f"loss,accuracy,precision,recall,total\n")
            stats = validator.domain_stats
            m.write(f"{stats['loss']:.6f},{stats['accuracy']:.6f},{stats['precision']:.6f},{stats['recall']:.6f},{stats['total']}\n")
    
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
    
    # 获取域分类准确率
    domain_accuracy = validator.domain_stats.get("accuracy", 0) if validator.domain_stats else 0
    
    return {
        "model": model_name,
        "model_path": str(model_path),
        "source": source_results,
        "target": target_results,
        "domain_accuracy": domain_accuracy,
    }


def load_cached_results(cache_file):
    """从缓存文件加载之前的验证结果"""
    if not cache_file.exists():
        return {}
    try:
        with open(cache_file, "r") as f:
            results = json.load(f)
        # 转换为以模型名称为键的字典，方便查找
        return {r["model"]: r for r in results}
    except (json.JSONDecodeError, KeyError) as e:
        LOGGER.warning(f"读取缓存文件失败: {e}")
        return {}


def save_results(results, cache_file):
    """保存验证结果到缓存文件"""
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(results, f, indent=2)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="验证 runs/detect 中所有模型在源域和目标域上的性能",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python validate_models.py              # 增量验证（使用缓存）
  python validate_models.py --force-all  # 强制重新验证所有模型
        """
    )
    parser.add_argument(
        "--force-all",
        action="store_true",
        help="强制验证所有模型，忽略之前的缓存结果"
    )
    return parser.parse_args()


def main():
    """主函数：验证所有模型"""
    
    # 解析命令行参数
    args_cmd = parse_args()
    force_all = args_cmd.force_all
    
    # 打印环境信息
    print(f"\n{'='*80}")
    print("验证脚本 v3 - 环境信息")
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
    cache_file = Path("runs/validation_results.json")
    
    # 检查 runs/detect 目录
    if not runs_dir.exists():
        LOGGER.error(f"目录不存在: {runs_dir}")
        return
    
    # 获取所有模型目录
    model_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    
    if not model_dirs:
        LOGGER.warning(f"在 {runs_dir} 中没有找到模型")
        return
    
    # 加载缓存结果（如果不强制重新验证）
    cached_results = {} if force_all else load_cached_results(cache_file)
    if cached_results and not force_all:
        print(f"从缓存加载了 {len(cached_results)} 个模型的验证结果")
    
    # 确定需要验证的模型
    models_to_validate = []
    for model_dir in model_dirs:
        best_pt = model_dir / "weights" / "best.pt"
        if not best_pt.exists():
            LOGGER.warning(f"跳过 {model_dir.name}: 未找到 best.pt")
            continue
        
        if force_all or model_dir.name not in cached_results:
            models_to_validate.append(model_dir)
    
    print(f"发现 {len(model_dirs)} 个模型，其中 {len(models_to_validate)} 个需要验证")
    
    # 准备参数用于数据加载器（使用基础配置）
    args = create_base_args(batch_size, imgsz, device, plots=False, verbose=False, half=False)
    
    # 检查数据集
    data = check_det_dataset(source_data)
    target_data_dict = check_det_dataset(target_data)
    
    # 构建数据加载器
    source_dataloader = get_dataloader(data.get(args.split), args, data, batch_size)
    target_dataloader = get_dataloader(target_data_dict.get(args.split), args, target_data_dict, batch_size)
    print("数据集加载完成！\n")
    
    # 存储新验证的结果
    new_results = []
    
    # 验证需要验证的模型
    if models_to_validate:
        for model_dir in sorted(models_to_validate):
            best_pt = model_dir / "weights" / "best.pt"
            
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
                    new_results.append(result)
                    # 更新缓存
                    cached_results[result["model"]] = result
            except Exception as e:
                LOGGER.error(f"验证 {model_dir.name} 时出错: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n新验证了 {len(new_results)} 个模型")
    else:
        print("\n所有模型都已在缓存中，无需重新验证")
    
    # 合并结果：所有模型目录对应的结果（缓存+新验证）
    all_results = []
    for model_dir in sorted(model_dirs):
        best_pt = model_dir / "weights" / "best.pt"
        if not best_pt.exists():
            continue
        if model_dir.name in cached_results:
            all_results.append(cached_results[model_dir.name])
    
    # 打印汇总结果
    print(f"\n{'='*105}")
    print("验证结果汇总")
    print(f"{'='*105}\n")
    

    # 使用rich表格输出
    console = Console(width=140)  # 设置足够宽的宽度
    table = Table(title="验证结果汇总", box=box.ROUNDED)
    
    # 添加列
    table.add_column("模型", style="cyan", no_wrap=True, min_width=28)
    table.add_column("源域 mAP50", justify="right", style="green", min_width=12)
    table.add_column("源域 mAP50-95", justify="right", style="green", min_width=14)
    table.add_column("目标域 mAP50", justify="right", style="blue", min_width=14)
    table.add_column("目标域 mAP50-95", justify="right", style="blue", min_width=16)
    table.add_column("域分类准确率", justify="right", style="magenta", min_width=14)
    
    # 添加数据行
    for r in all_results:
        model_name = r["model"]
        
        table.add_row(
            model_name,
            f"{r['source']['mAP50']:.4f}",
            f"{r['source']['mAP50-95']:.4f}",
            f"{r['target']['mAP50']:.4f}",
            f"{r['target']['mAP50-95']:.4f}",
            f"{r.get('domain_accuracy', 0):.4f}",
        )
    
    console.print(table)

    # 保存结果到 JSON
    output_file = "runs/validation_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n详细结果已保存到: {output_file}")
    

if __name__ == "__main__":
    main()
