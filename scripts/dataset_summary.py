"""
数据集摘要脚本 - 用于验证两台机器上的数据集是否一致
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from collections import defaultdict, Counter


def compute_file_hash(filepath, algorithm='md5', block_size=65536):
    """计算文件的哈希值"""
    hasher = hashlib.new(algorithm)
    try:
        with open(filepath, 'rb') as f:
            for block in iter(lambda: f.read(block_size), b''):
                hasher.update(block)
        return hasher.hexdigest()
    except Exception as e:
        return f"ERROR: {e}"


def summarize_images(image_dir):
    """摘要图像目录"""
    image_dir = Path(image_dir)
    if not image_dir.exists():
        return {"error": f"Directory not found: {image_dir}"}
    
    # 获取所有图像文件
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.webp']:
        image_files.extend(image_dir.rglob(ext))
        image_files.extend(image_dir.rglob(ext.upper()))
    
    image_files = sorted(set(image_files))
    
    summary = {
        "total_images": len(image_files),
        "directory": str(image_dir),
        "sample_files": [],
        "size_stats": {"total_bytes": 0, "min": float('inf'), "max": 0, "mean": 0},
    }
    
    sizes = []
    for img_path in image_files[:10]:  # 只计算前10个样本的哈希
        stat = img_path.stat()
        size = stat.st_size
        sizes.append(size)
        summary["size_stats"]["total_bytes"] += size
        summary["size_stats"]["min"] = min(summary["size_stats"]["min"], size)
        summary["size_stats"]["max"] = max(summary["size_stats"]["max"], size)
        
        # 计算样本哈希
        file_hash = compute_file_hash(img_path)
        summary["sample_files"].append({
            "name": img_path.name,
            "size": size,
            "md5": file_hash[:16] + "..." if len(file_hash) > 16 else file_hash
        })
    
    if sizes:
        summary["size_stats"]["mean"] = sum(sizes) / len(sizes)
    
    return summary


def summarize_labels(label_dir):
    """摘要标签目录"""
    label_dir = Path(label_dir)
    if not label_dir.exists():
        return {"error": f"Directory not found: {label_dir}"}
    
    label_files = sorted(label_dir.rglob("*.txt"))
    
    summary = {
        "total_label_files": len(label_files),
        "directory": str(label_dir),
        "total_instances": 0,
        "empty_files": 0,
        "class_distribution": Counter(),
        "sample_labels": [],
        "bbox_stats": {
            "widths": [],
            "heights": [],
            "x_centers": [],
            "y_centers": []
        }
    }
    
    for label_file in label_files[:10]:  # 只处理前10个样本
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            if not lines or all(not line.strip() for line in lines):
                summary["empty_files"] += 1
                continue
            
            file_instances = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    x, y, w, h = map(float, parts[1:5])
                    
                    summary["class_distribution"][cls] += 1
                    summary["bbox_stats"]["widths"].append(w)
                    summary["bbox_stats"]["heights"].append(h)
                    summary["bbox_stats"]["x_centers"].append(x)
                    summary["bbox_stats"]["y_centers"].append(y)
                    
                    file_instances.append({
                        "class": cls,
                        "bbox": [x, y, w, h]
                    })
                    summary["total_instances"] += 1
            
            summary["sample_labels"].append({
                "file": label_file.name,
                "instances": len(file_instances),
                "data": file_instances[:3]  # 只显示前3个实例
            })
            
        except Exception as e:
            summary["sample_labels"].append({
                "file": label_file.name,
                "error": str(e)
            })
    
    # 处理所有文件以获取完整统计（正确计算）
    total_instances_correct = 0
    class_dist_correct = Counter()
    for label_file in label_files:
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            file_instances = 0
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    class_dist_correct[cls] += 1
                    file_instances += 1
            total_instances_correct += file_instances
        except:
            pass
    
    summary["total_instances"] = total_instances_correct
    summary["class_distribution"] = dict(class_dist_correct)
    
    # 转换Counter为普通dict以便JSON序列化
    summary["class_distribution"] = dict(summary["class_distribution"])
    
    # 计算bbox统计
    for key in ["widths", "heights", "x_centers", "y_centers"]:
        values = summary["bbox_stats"][key]
        if values:
            summary["bbox_stats"][key] = {
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
                "count": len(values)
            }
        else:
            summary["bbox_stats"][key] = {"count": 0}
    
    return summary


def compare_with_other_machine(current_summary, other_machine_summary_path=None):
    """与另一台机器的数据集摘要做对比"""
    print("\n" + "="*80)
    print("数据集摘要")
    print("="*80)
    print(json.dumps(current_summary, indent=2))
    
    if other_machine_summary_path and Path(other_machine_summary_path).exists():
        with open(other_machine_summary_path, 'r') as f:
            other_summary = json.load(f)
        
        print("\n" + "="*80)
        print("与另一台机器的对比")
        print("="*80)
        
        # 对比源域图像
        current_src_img = current_summary.get("source_images", {})
        other_src_img = other_summary.get("source_images", {})
        
        print(f"\n【源域图像对比】")
        print(f"  当前机器图像数: {current_src_img.get('total_images', 'N/A')}")
        print(f"  另一机器图像数: {other_src_img.get('total_images', 'N/A')}")
        
        if current_src_img.get('total_images') != other_src_img.get('total_images'):
            print(f"  ⚠️ 图像数量不一致!")
        else:
            print(f"  ✓ 图像数量一致")
        
        # 对比源域标签
        current_src_lbl = current_summary.get("source_labels", {})
        other_src_lbl = other_summary.get("source_labels", {})
        
        print(f"\n【源域标签对比】")
        print(f"  当前机器标签文件数: {current_src_lbl.get('total_label_files', 'N/A')}")
        print(f"  另一机器标签文件数: {other_src_lbl.get('total_label_files', 'N/A')}")
        print(f"  当前机器实例总数: {current_src_lbl.get('total_instances', 'N/A')}")
        print(f"  另一机器实例总数: {other_src_lbl.get('total_instances', 'N/A')}")
        
        if current_src_lbl.get('total_label_files') != other_src_lbl.get('total_label_files'):
            print(f"  ⚠️ 标签文件数量不一致!")
        else:
            print(f"  ✓ 标签文件数量一致")
        
        if current_src_lbl.get('total_instances') != other_src_lbl.get('total_instances'):
            print(f"  ⚠️ 实例总数不一致!")
            diff = abs(current_src_lbl.get('total_instances', 0) - other_src_lbl.get('total_instances', 0))
            print(f"    差异: {diff} 个实例")
        else:
            print(f"  ✓ 实例总数一致")
        
        # 对比类别分布
        current_dist = current_src_lbl.get('class_distribution', {})
        other_dist = other_src_lbl.get('class_distribution', {})
        
        print(f"\n【类别分布对比】")
        all_classes = set(current_dist.keys()) | set(other_dist.keys())
        for cls in sorted(all_classes, key=int):
            c_count = current_dist.get(str(cls), 0)
            o_count = other_dist.get(str(cls), 0)
            if c_count != o_count:
                print(f"  ⚠️ 类别 {cls}: 当前={c_count}, 另一台={o_count}, 差异={abs(c_count-o_count)}")
            else:
                print(f"  ✓ 类别 {cls}: {c_count}")


def main():
    """主函数"""
    # 配置路径
    source_data = "datasets/cityscape_yolo"
    target_data = "datasets/cityscape_foggy_yolo"
    
    print("="*80)
    print("数据集摘要生成")
    print("="*80)
    print(f"工作目录: {Path.cwd()}")
    print(f"时间: {__import__('datetime').datetime.now().isoformat()}")
    
    summary = {}
    
    # 源域数据集
    print("\n【源域数据集】")
    print("-"*40)
    source_img_dir = Path(source_data) / "images" / "val"
    source_lbl_dir = Path(source_data) / "labels" / "val"
    
    print(f"图像目录: {source_img_dir}")
    summary["source_images"] = summarize_images(source_img_dir)
    print(f"  总图像数: {summary['source_images'].get('total_images', 'N/A')}")
    print(f"  样本图像: {[f['name'] for f in summary['source_images'].get('sample_files', [])[:3]]}")
    
    print(f"\n标签目录: {source_lbl_dir}")
    summary["source_labels"] = summarize_labels(source_lbl_dir)
    print(f"  总标签文件数: {summary['source_labels'].get('total_label_files', 'N/A')}")
    print(f"  总实例数: {summary['source_labels'].get('total_instances', 'N/A')}")
    print(f"  类别分布: {summary['source_labels'].get('class_distribution', {})}")
    
    # 目标域数据集
    print("\n【目标域数据集】")
    print("-"*40)
    target_img_dir = Path(target_data) / "images" / "val"
    target_lbl_dir = Path(target_data) / "labels" / "val"
    
    print(f"图像目录: {target_img_dir}")
    summary["target_images"] = summarize_images(target_img_dir)
    print(f"  总图像数: {summary['target_images'].get('total_images', 'N/A')}")
    
    print(f"\n标签目录: {target_lbl_dir}")
    summary["target_labels"] = summarize_labels(target_lbl_dir)
    print(f"  总标签文件数: {summary['target_labels'].get('total_label_files', 'N/A')}")
    print(f"  总实例数: {summary['target_labels'].get('total_instances', 'N/A')}")
    print(f"  类别分布: {summary['target_labels'].get('class_distribution', {})}")
    
    # 保存摘要
    output_file = "runs/dataset_summary.json"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n摘要已保存到: {output_file}")
    
    # 检查是否与另一台机器有对比文件
    other_summary = "runs/other_machine_summary.json"
    if Path(other_summary).exists():
        compare_with_other_machine(summary, other_summary)
    else:
        print(f"\n提示: 将另一台机器的摘要保存到 {other_summary} 可以进行对比")


if __name__ == "__main__":
    main()
