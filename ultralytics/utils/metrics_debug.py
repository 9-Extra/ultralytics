"""
Metrics 调试模块 - 用于跨平台对比中间计算结果
"""
import numpy as np
from pathlib import Path
import json
from typing import Dict, Any
import os

# 调试输出目录
DEBUG_DIR = Path("debug_output")


def save_debug_data(name: str, data: Dict[str, Any], step: str):
    """保存调试数据到文件
    
    Args:
        name: 类别名称或标识
        data: 要保存的数据字典
        step: 步骤标识（如 'input', 'sorted', 'class_0' 等）
    """
    DEBUG_DIR.mkdir(exist_ok=True)
    
    # 构建文件名
    filename = f"{step}_{name}.npz"
    filepath = DEBUG_DIR / filename
    
    # 保存为 npz 格式（可存储多个数组）
    np.savez(filepath, **data)
    print(f"[DEBUG] Saved: {filepath}")


def debug_ap_per_class(
    tp: np.ndarray,
    conf: np.ndarray,
    pred_cls: np.ndarray,
    target_cls: np.ndarray,
    plot: bool = False,
    on_plot=None,
    save_dir: Path = Path(),
    names: Dict[int, str] = {},
    eps: float = 1e-16,
    prefix: str = "",
) -> tuple:
    """带调试输出的 ap_per_class 函数
    
    这是原始 ap_per_class 函数的包装，在关键步骤导出中间变量
    """
    from .metrics import compute_ap, smooth
    
    # 步骤 A: 保存输入数据
    save_debug_data("all", {
        "tp": tp,
        "conf": conf,
        "pred_cls": pred_cls,
        "target_cls": target_cls,
    }, "A_input")
    
    # 步骤 B: 排序
    neg_conf = np.negative(conf)
    i = np.argsort(neg_conf, kind='mergesort')
    tp_sorted = tp[i]
    conf_sorted = conf[i]
    pred_cls_sorted = pred_cls[i]
    
    save_debug_data("all", {
        "sort_indices": i,
        "tp_sorted": tp_sorted,
        "conf_sorted": conf_sorted,
        "pred_cls_sorted": pred_cls_sorted,
    }, "B_sorted")
    
    # 找到唯一类别
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]
    
    # 创建 PR 曲线
    x = np.linspace(0, 1, 1000)
    ap = np.zeros((nc, tp.shape[1]))
    p_curve = np.zeros((nc, 1000))
    r_curve = np.zeros((nc, 1000))
    
    # 为每个类别保存数据
    class_data = {}
    
    for ci, c in enumerate(unique_classes):
        mask = pred_cls_sorted == c
        n_l = nt[ci]
        n_p = mask.sum()
        
        if n_p == 0 or n_l == 0:
            continue
        
        tp_c = tp_sorted[mask]
        conf_c = conf_sorted[mask]
        
        # 累积 FPs 和 TPs
        fpc = (1 - tp_c).cumsum(0)
        tpc = tp_c.cumsum(0)
        
        # Recall
        recall = tpc / (n_l + eps)
        r_curve[ci] = np.interp(-x, np.negative(conf_c), recall[:, 0], 
                                left=0, right=recall[-1, 0] if len(recall) > 0 else 0)
        
        # Precision
        precision = tpc / (tpc + fpc)
        p_curve[ci] = np.interp(-x, np.negative(conf_c), precision[:, 0], 
                                left=precision[0, 0] if len(precision) > 0 else 1,
                                right=precision[-1, 0] if len(precision) > 0 else 0)
        
        # 步骤 C: 保存每个类别的中间结果
        class_name = names.get(c, f"class_{c}")
        save_debug_data(class_name, {
            "n_l": n_l,
            "n_p": n_p,
            "tp_c": tp_c,
            "conf_c": conf_c,
            "fpc": fpc,
            "tpc": tpc,
            "recall": recall,
            "precision": precision,
            "p_curve": p_curve[ci],
            "r_curve": r_curve[ci],
        }, f"C_class_{ci}")
        
        # 计算 AP
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
    
    # 计算 F1
    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps)
    
    # 步骤 D: 保存 F1 曲线和 max F1 索引
    smoothed_f1 = smooth(f1_curve.mean(0), 0.1)
    max_f1_idx = smoothed_f1.argmax()
    
    save_debug_data("all", {
        "f1_curve": f1_curve,
        "smoothed_f1": smoothed_f1,
        "max_f1_idx": max_f1_idx,
        "p_curve": p_curve,
        "r_curve": r_curve,
    }, "D_f1_analysis")
    
    # 最终结果
    p = p_curve[:, max_f1_idx]
    r = r_curve[:, max_f1_idx]
    f1 = f1_curve[:, max_f1_idx]
    tp_out = (r * nt).round()
    fp_out = (tp_out / (p + eps) - tp_out).round()
    
    # 步骤 E: 保存最终结果
    save_debug_data("all", {
        "p": p,
        "r": r,
        "f1": f1,
        "tp_out": tp_out,
        "fp_out": fp_out,
        "ap": ap,
        "unique_classes": unique_classes,
        "max_f1_idx": max_f1_idx,
    }, "E_final")
    
    # 同时保存一份人类可读的报告
    report_path = DEBUG_DIR / "report.txt"
    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("Metrics Debug Report\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Input data:\n")
        f.write(f"  tp shape: {tp.shape}\n")
        f.write(f"  conf shape: {conf.shape}\n")
        f.write(f"  pred_cls shape: {pred_cls.shape}\n")
        f.write(f"  target_cls shape: {target_cls.shape}\n\n")
        
        f.write(f"After sorting:\n")
        f.write(f"  conf range: [{conf_sorted.min():.6f}, {conf_sorted.max():.6f}]\n\n")
        
        f.write(f"Per-class results:\n")
        for ci, c in enumerate(unique_classes):
            name = names.get(c, f"class_{c}")
            f.write(f"  {name}:\n")
            f.write(f"    n_l (targets): {nt[ci]}\n")
            f.write(f"    n_p (predictions): {(pred_cls_sorted == c).sum()}\n")
            f.write(f"    precision: {p[ci]:.6f}\n")
            f.write(f"    recall: {r[ci]:.6f}\n")
            f.write(f"    f1: {f1[ci]:.6f}\n")
            f.write(f"    ap50: {ap[ci, 0]:.6f}\n\n")
        
        f.write(f"Max F1 index: {max_f1_idx}\n")
        f.write(f"Smoothed F1 at max: {smoothed_f1[max_f1_idx]:.6f}\n")
    
    print(f"[DEBUG] Report saved: {report_path}")
    
    return tp_out, fp_out, p, r, f1, ap, unique_classes.astype(int), p_curve, r_curve, f1_curve, x, []


def compare_debug_outputs(local_dir: str, remote_dir: str):
    """对比两个系统的调试输出
    
    Args:
        local_dir: 本地系统 debug_output 目录路径
        remote_dir: 外部系统 debug_output 目录路径
    """
    local_path = Path(local_dir)
    remote_path = Path(remote_dir)
    
    print("=" * 80)
    print("Comparing Debug Outputs")
    print("=" * 80)
    
    # 获取所有 npz 文件
    local_files = set(f.name for f in local_path.glob("*.npz"))
    remote_files = set(f.name for f in remote_path.glob("*.npz"))
    
    # 检查文件是否存在
    only_local = local_files - remote_files
    only_remote = remote_files - local_files
    common = local_files & remote_files
    
    if only_local:
        print(f"\nOnly in local: {only_local}")
    if only_remote:
        print(f"\nOnly in remote: {only_remote}")
    
    print(f"\nComparing {len(common)} common files...")
    
    # 对比每个文件
    differences = []
    for filename in sorted(common):
        local_data = np.load(local_path / filename)
        remote_data = np.load(remote_path / filename)
        
        # 检查键是否一致
        local_keys = set(local_data.keys())
        remote_keys = set(remote_data.keys())
        
        if local_keys != remote_keys:
            print(f"\n[WARNING] {filename}: Keys differ!")
            print(f"  Local: {local_keys - remote_keys}")
            print(f"  Remote: {remote_keys - local_keys}")
            continue
        
        # 对比每个数组
        file_diffs = []
        for key in local_keys:
            local_arr = local_data[key]
            remote_arr = remote_data[key]
            
            # 检查形状
            if local_arr.shape != remote_arr.shape:
                file_diffs.append(f"{key}: shape {local_arr.shape} vs {remote_arr.shape}")
                continue
            
            # 检查数值（处理标量）
            if local_arr.shape == ():
                if local_arr != remote_arr:
                    file_diffs.append(f"{key}: {local_arr} vs {remote_arr}")
            else:
                # 数值数组
                if not np.allclose(local_arr, remote_arr, rtol=1e-5, atol=1e-8):
                    diff = np.abs(local_arr - remote_arr).max()
                    file_diffs.append(f"{key}: max diff = {diff:.10f}")
        
        if file_diffs:
            differences.append((filename, file_diffs))
            print(f"\n[DIFF] {filename}:")
            for diff in file_diffs:
                print(f"  - {diff}")
    
    if not differences:
        print("\n[OK] All files are identical!")
    else:
        print(f"\n[SUMMARY] Found differences in {len(differences)} files")
    
    return differences


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        compare_debug_outputs(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python metrics_debug.py <local_dir> <remote_dir>")
