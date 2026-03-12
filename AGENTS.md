# AGENTS.md

本项目在Yolov26中加入GRL梯度反转层以实现域适应（Domain Adaptation），可以将在源域（比如非雾天环境）训练的目标检测模型微调至适用于目标域（比如雾天环境）。而在ultralytics库中加入新模型的正确方法就是fork并修改ultralytics库本身。

## 环境配置

- 使用 `uv run` 来运行 Python 脚本，使用`uv add`来安装依赖，这会自动使用项目配置的虚拟环境，**不要**直接使用系统 Python 或 pip 运行脚本或安装包
- 如果项目根目录下存在ENVIRONMENT.md，请遵循其中的说明
- 无视readme.md中关于环境配置的内容

## 数据集

1. CityScape数据集 
已预先转化为Yolo所需格式。源域数据位于（正常天气）"datasets/cityscape_yolo/cityscapes.yaml"，以及目标域数据（雾天）位于"datasets/cityscape_foggy_yolo/cityscapes.yaml"。如果数据集出现问题，请暂停任务向用户询问。

## YOLODA 模型

YOLODA（YOLO Domain Adaptation）是基于 YOLOv26 的域适应目标检测模型，支持通过梯度反转层（GRL）实现源域到目标域的适应。

### 模型结构

- **配置文件**: `ultralytics/cfg/models/26/yolo26-da.yaml`
- **检测头**: `DetectGRL` - 继承自标准 `Detect` 检测头，目前实现与 `Detect` 一致，预留 GRL 扩展接口
- **支持任务**: 目标检测（detect）
- **支持尺度**: n, s, m, l, x（与 YOLOv26 相同）

### 模块位置

```
ultralytics/
├── cfg/models/26/yolo26-da.yaml      # 模型配置文件
├── models/yolo/
│   ├── model.py                       # YOLODA 模型类定义
│   └── domain_adapt/                  # 域适应专用模块
│       ├── __init__.py
│       ├── train.py                   # DomainAdaptationTrainer
│       ├── val.py                     # DomainAdaptationValidator
│       └── predict.py                 # DomainAdaptationPredictor
└── nn/modules/head.py                 # DetectGRL 检测头定义
```

### 使用方式

```python
from ultralytics import YOLO
from ultralytics.models.yolo import YOLODA

# 方式1: 使用 YOLO 类自动识别（推荐）
model = YOLO("yolo26n-da.yaml")  # 自动识别为 YOLODA，使用 n scale
model = YOLO("yolo26s-da.yaml")  # 使用 s scale
model = YOLO("yolo26m-da.yaml")  # 使用 m scale
model = YOLO("yolo26l-da.yaml")  # 使用 l scale
model = YOLO("yolo26x-da.yaml")  # 使用 x scale

# 方式2: 直接使用 YOLODA 类
model = YOLODA("yolo26n-da.yaml")

# 训练
model.train(data="cityscapes.yaml", epochs=100)

# 验证
model.val(data="cityscapes.yaml")

# 预测
model.predict("image.jpg")
```

### 命名规则

YOLODA 使用 `-da` 后缀来标识域适应模型：

- `yolo26n-da.yaml` → 自动查找 `yolo26-da.yaml` 并应用 n scale
- `yolo26s-da.yaml` → 自动查找 `yolo26-da.yaml` 并应用 s scale
- `yolo26-da.yaml` → 基础配置文件（默认使用 n scale）

## DomainAdaptationTrainer 使用方式

`DomainAdaptationTrainer` 用于训练域自适应目标检测模型，同时加载源域（带标签）和目标域（不带标签）数据集。

### 基本用法

```python
from ultralytics.models.yolo.domain_adapt import DomainAdaptationTrainer

# 同时指定源域和目标域数据集
args = dict(
    model="yolodan.yaml",               # 模型配置文件（使用 DetectGRL 检测头）
    data="source_coco8.yaml",           # 源域数据集（带标签，用于检测任务）
    target_data="target_cityscapes.yaml",  # 目标域数据集（标签可选，用于域分类任务）
    epochs=100
)
trainer = DomainAdaptationTrainer(overrides=args)
trainer.train()
```

### 命令行使用

```bash
yolo detect train model=yolodan.yaml data=source_coco8.yaml target_data=target_cityscapes.yaml epochs=100
```

### 数据格式

每个训练批次包含以下字段：

| 字段 | 说明 |
|------|------|
| `batch["img"]` | 源域图像，用于检测任务 |
| `batch["domain_img"]` | 目标域图像，用于域分类任务 |
| `batch["bboxes"]` | 源域边界框标签 |
| `batch["cls"]` | 源域类别标签 |
| `batch["batch_idx"]` | 批次索引 |

### 目标域数据集配置

目标域数据集使用标准的 YOLO 数据集格式（YAML 文件），但标签文件可以省略或为空：

```yaml
# target_cityscapes.yaml
path: /path/to/cityscapes
train: images/train  # 只需要图像路径
val: images/val      # 验证集（可选）
nc: 80               # 类别数（与源域一致）
names: {0: person, 1: car, ...}  # 类别名称
```

### 超参数

YOLODA 模型支持以下域适应专用超参数：

| 超参数 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `domain_loss_weight` | float | 0.1 | 域适应损失权重，控制域分类损失在总损失中的比重 |

#### 设置方式

**方式1：通过 `model.train()` 设置（推荐）**
```python
from ultralytics import YOLO

model = YOLO("yolo26n-da.yaml")
model.train(
    data="source.yaml",
    target_data="target.yaml",
    domain_loss_weight=0.2,  # 自定义域适应损失权重
    epochs=100
)
```

**方式2：命令行设置**
```bash
yolo detect train model=yolo26n-da.yaml data=source.yaml target_data=target.yaml domain_loss_weight=0.2
```

**方式3：通过自定义配置文件**
创建 `custom.yaml`：
```yaml
domain_loss_weight: 0.2
epochs: 100
```

然后加载配置：
```python
model.train(cfg="custom.yaml", data="source.yaml", target_data="target.yaml")
```

**方式4：直接使用 Trainer**
```python
from ultralytics.models.yolo.domain_adapt import DomainAdaptationTrainer

args = dict(
    model="yolo26n-da.yaml",
    data="source.yaml",
    target_data="target.yaml",
    domain_loss_weight=0.2,
    epochs=100
)
trainer = DomainAdaptationTrainer(overrides=args)
trainer.train()
```

### 注意事项

1. **检测头要求**：模型必须使用 `DetectGRL` 检测头，该检测头包含域分类器
2. **预处理一致性**：源域和目标域图像使用完全相同的预处理流程（包括多尺度增强）
3. **损失函数**：当前使用标准 `E2ELoss`，域分类损失需要在外部处理
4. **批次大小**：源域和目标域使用相同的批次大小

### 模型配置示例 (yolodan.yaml)

```yaml
# 使用 DetectGRL 检测头的模型配置
nc: 80  # 类别数
backbone:
  # ... 标准 backbone 配置
head:
  - [[15, 18, 21], 1, DetectGRL, [nc, 1, True]]  # DetectGRL(nc, reg_max=1, end2end=True)
```

### 实现细节

- `DomainAdaptationTrainer` 继承自 `BaseTrainer`
- 目标域数据加载器在 `get_dataloader()` 中创建
- 目标域图像在 `preprocess_batch()` 中处理并放入 `batch["domain_img"]`
- 目标域迭代器会自动循环（当数据用完时重置）
