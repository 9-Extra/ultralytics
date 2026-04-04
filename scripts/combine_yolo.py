#!/usr/bin/env python3
"""
合并两个YOLO格式数据集，用于测试Oracle模型性能。
合并后的数据集包含源域和目标域的所有图像和标签，用于训练一个Oracle模型作为性能上界。
"""

import os
import shutil
import argparse
from pathlib import Path


YAML_TEMPLATE = """# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]

path: {path}

train:
  - images/train

val:
  - images/val

test:
  - images/val

# Classes
nc: {nc}  # number of classes
names: {names}  # class names
"""


def copy_files(src_dir, dst_dir, suffix="", use_symlink=True, exclude_extensions=None):
    """
    复制或创建符号链接，将源目录中的文件复制到目标目录。
    
    Args:
        src_dir: 源目录路径
        dst_dir: 目标目录路径
        suffix: 文件后缀（用于区分不同域的数据）
        use_symlink: 是否使用符号链接而非复制文件
        exclude_extensions: 要排除的文件扩展名列表（如 ['.npy']）
    """
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    if exclude_extensions is None:
        exclude_extensions = []
    # 统一转换为小写以便比较
    exclude_extensions = [ext.lower() for ext in exclude_extensions]
    
    copied_count = 0
    for src_file in src_dir.iterdir():
        if not src_file.is_file():
            continue
        
        # 排除指定扩展名的文件（如YOLO缓存文件.npy）
        if src_file.suffix.lower() in exclude_extensions:
            continue
        
        # 构造新文件名（添加后缀以区分不同域）
        stem = src_file.stem
        ext = src_file.suffix
        if suffix:
            new_name = f"{stem}_{suffix}{ext}"
        else:
            new_name = f"{stem}{ext}"
        
        dst_file = dst_dir / new_name
        
        # 如果目标文件已存在，跳过
        if dst_file.exists():
            print(f"Warning: {dst_file} already exists, skipping...")
            continue
        
        # 使用符号链接或复制文件
        if use_symlink:
            try:
                os.symlink(src_file.absolute(), dst_file)
            except OSError as e:
                # 如果符号链接失败，则复制文件
                print(f"Symlink failed for {src_file}, copying instead: {e}")
                shutil.copy2(src_file, dst_file)
        else:
            shutil.copy2(src_file, dst_file)
        
        copied_count += 1
    
    return copied_count


def combine_datasets(source_dir, target_dir, output_dir, 
                     source_suffix="source", target_suffix="target",
                     use_symlink=True):
    """
    合并两个YOLO格式数据集。
    
    Args:
        source_dir: 源域数据集路径（带标签的正常天气数据）
        target_dir: 目标域数据集路径（带标签的雾天数据）
        output_dir: 输出目录路径
        source_suffix: 源域文件后缀
        target_suffix: 目标域文件后缀
        use_symlink: 是否使用符号链接
    """
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    output_dir = Path(output_dir)
    
    # 排除YOLO生成的缓存文件扩展名
    exclude_extensions = ['.npy']
    
    print(f"Source dataset: {source_dir}")
    print(f"Target dataset: {target_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Using symlinks: {use_symlink}")
    print(f"Excluding extensions: {exclude_extensions}")
    print()
    
    # 创建输出目录结构
    splits = ['train', 'val']
    for split in splits:
        (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # 合并训练集和验证集
    for split in splits:
        print(f"\nProcessing {split} set...")
        
        # 源域图像和标签
        src_images = source_dir / 'images' / split
        src_labels = source_dir / 'labels' / split
        
        # 目标域图像和标签
        tgt_images = target_dir / 'images' / split
        tgt_labels = target_dir / 'labels' / split
        
        # 输出路径
        out_images = output_dir / 'images' / split
        out_labels = output_dir / 'labels' / split
        
        # 复制源域数据（排除缓存文件）
        if src_images.exists():
            count = copy_files(src_images, out_images, suffix=source_suffix, 
                               use_symlink=use_symlink, exclude_extensions=exclude_extensions)
            print(f"  Copied {count} source images")
        if src_labels.exists():
            count = copy_files(src_labels, out_labels, suffix=source_suffix, 
                               use_symlink=use_symlink, exclude_extensions=exclude_extensions)
            print(f"  Copied {count} source labels")
        
        # 复制目标域数据（排除缓存文件）
        if tgt_images.exists():
            count = copy_files(tgt_images, out_images, suffix=target_suffix, 
                               use_symlink=use_symlink, exclude_extensions=exclude_extensions)
            print(f"  Copied {count} target images")
        if tgt_labels.exists():
            count = copy_files(tgt_labels, out_labels, suffix=target_suffix, 
                               use_symlink=use_symlink, exclude_extensions=exclude_extensions)
            print(f"  Copied {count} target labels")
    
    # 创建YAML配置文件
    # 从源域数据集的YAML中读取类别信息
    source_yaml = source_dir / "cityscapes.yaml"
    nc = 8  # 默认类别数
    names = "['person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']"
    
    if source_yaml.exists():
        try:
            with open(source_yaml, 'r') as f:
                content = f.read()
                # 简单解析YAML内容
                for line in content.split('\n'):
                    line = line.strip()
                    if line.startswith('nc:'):
                        nc = int(line.split(':')[1].split('#')[0].strip())
                    elif line.startswith('names:'):
                        names = line.split(':', 1)[1].split('#')[0].strip()
        except Exception as e:
            print(f"Warning: Failed to parse source YAML: {e}")
    
    yaml_content = YAML_TEMPLATE.format(
        path=str(output_dir.absolute()),
        nc=nc,
        names=names
    )
    
    yaml_path = output_dir / "cityscapes_combined.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\nCreated YAML config: {yaml_path}")
    print(f"\nDataset combination complete!")
    print(f"Output directory: {output_dir}")
    print(f"To use the combined dataset:")
    print(f"  model.train(data='{yaml_path}')")


def main():
    parser = argparse.ArgumentParser(
        description="合并两个YOLO格式数据集，用于测试Oracle模型性能"
    )
    parser.add_argument(
        "--source", "-s",
        type=str,
        default="datasets/cityscape_yolo",
        help="源域数据集路径（带标签的正常天气数据）"
    )
    parser.add_argument(
        "--target", "-t",
        type=str,
        default="datasets/cityscape_foggy_yolo",
        help="目标域数据集路径（带标签的雾天数据）"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="datasets/cityscape_combined_yolo",
        help="输出目录路径"
    )
    parser.add_argument(
        "--source-suffix",
        type=str,
        default="source",
        help="源域文件后缀（默认: source）"
    )
    parser.add_argument(
        "--target-suffix",
        type=str,
        default="target",
        help="目标域文件后缀（默认: target）"
    )
    parser.add_argument(
        "--copy", "-c",
        action="store_true",
        help="复制文件而不是创建符号链接"
    )
    
    args = parser.parse_args()
    
    combine_datasets(
        source_dir=args.source,
        target_dir=args.target,
        output_dir=args.output,
        source_suffix=args.source_suffix,
        target_suffix=args.target_suffix,
        use_symlink=not args.copy
    )


if __name__ == "__main__":
    main()
