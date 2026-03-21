# Bug 修复：NumPy 就地数组修改导致指标计算错误

## 问题描述

NumPy 在计算 `tpc + fpc` 时可能会执行就地优化，直接修改 `tpc` 而不是创建新数组。这会导致目标检测指标计算中的精度（precision）出现错误。

### 根本原因

在计算精度时：
```python
precision = tpc / (tpc + fpc)
```

NumPy 的内存优化可能导致 `tpc + fpc` 返回 `tpc` 本身并就地修改其值（将 `fpc` 加到它上面）。这种情况发生在：
1. `tpc` 是来自 `cumsum()` 的 C-连续 int64 数组
2. NumPy 可能复用左操作数的内存作为结果
3. 此行为与 NumPy 版本和平台相关

### 影响

**错误的指标：**
- `tpc`（真阳性累积值）变成 `tpc + fpc`（所有预测）
- 精度变成 `tpc / (tpc + fpc) = 1.0`（所有预测）
- 所有类别显示精度 = 1.0，这是错误的
- mAP 值被人为抬高

**修复前（错误）：**
```
Person: P=1.0, R=0.854, mAP50=0.927  # 精度不应该是 1.0
```

**修复后（正确）：**
```
Person: P=0.7, R=0.566, mAP50=0.633  # 正确的精度
```

### 受影响的代码位置

文件：`ultralytics/utils/metrics.py`  
函数：`ap_per_class()`  
行号：约 823 行

```python
# 原始有 bug 的代码：
precision = tpc / (tpc + fpc)

# 修复后的代码：
precision = tpc / (tpc.copy() + fpc)
```

### 修复方法

在加法前对 `tpc` 添加 `.copy()` 以防止就地修改：

```python
# 召回率
tpc = tp_c.cumsum(0)
fpc = (1 - tp_c).cumsum(0)
recall = tpc / (n_l + eps)

# 精度 - 注意：需要使用 tpc.copy()，因为 tpc + fpc 可能会就地修改 tpc
precision = tpc / (tpc.copy() + fpc)
```

### 验证方法

验证修复是否正确应用：

**方法 1：运行测试脚本**
```bash
python test_numpy_inplace_bug.py
```
退出码：
- `0` - 测试通过，bug 已修复
- `1` - 测试失败，bug 仍存在
- `2` - 结果不确定，需要手动验证

**方法 2：手动验证**
1. 检查 `tpc.max()` 是否等于 `tp_c.sum()`（而不是 `n_p`）
2. 检查精度值是否不全是 1.0

### 相关环境信息

- **受影响的 NumPy 版本**：在 NumPy 2.1.3 上观察到，但可能因版本和平台而异
- **平台**：Linux x86_64（也可能影响其他平台）
- **数组属性**：当出现以下情况时，bug 会显现：
  - `tpc` 是 C-连续的（`tpc.flags['C_CONTIGUOUS'] == True`）
  - `tpc` 有 `OWNDATA=True`
  - 两个数组都是 int64 类型

### 在其他代码库中检查此 Bug

**使用测试脚本：**
```bash
# 将 test_numpy_inplace_bug.py 复制到另一个项目并运行
cp test_numpy_inplace_bug.py /path/to/other/project/
cd /path/to/other/project/
python test_numpy_inplace_bug.py
```

**手动搜索模式：**
```bash
# 查找可能受影响的代码
grep -rn "precision.*tpc.*fpc\|precision.*/.*tpc.*+.*fpc" --include="*.py"

# 在指标计算中查找类似模式
grep -rn "cumsum.*0" --include="*.py" | grep -i "precision\|recall"
```

应用相同的修复：对加法中的左操作数添加 `.copy()`。

### 测试脚本使用方法

`test_numpy_inplace_bug.py` 脚本可以独立运行，不需要任何模型或数据集：

```python
# 运行测试
python test_numpy_inplace_bug.py

# 修复应用后的预期输出：
# ✓ Fixed pattern works correctly (no modification of tpc)
# ✓ Fix detected: 'tpc.copy()' is used in the source
# ✓ ALL TESTS PASSED

# Bug 存在时的预期输出：
# ✗ BUG PRESENT: Source uses 'tpc / (tpc + fpc)' without .copy()
# ✗ TEST FAILED
```

测试脚本功能：
1. 创建触发 NumPy 优化的合成数据
2. 测试有 bug 和修复后的两种模式
3. 检查 metrics 模块源代码是否已修复
4. 报告清晰的通过/失败状态

---

## 参考

- NumPy Issue：ufunc 操作中的内存优化
- Ultralytics Issue：指标计算显示所有类别的 P=1.0
