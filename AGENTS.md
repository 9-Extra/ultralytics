# AGENTS.md

本项目在Yolov26中加入GRL梯度反转层以实现域适应（Domain Adaptation），可以将在源域（比如非雾天环境）训练的目标检测模型微调至适用于目标域（比如雾天环境）。而在ultralytics库中加入新模型的正确方法就是fork并修改ultralytics库本身。

## 环境配置

- 使用 `uv run` 来运行 Python 脚本，使用`uv add`来安装依赖，这会自动使用项目配置的虚拟环境，**不要**直接使用系统 Python 或 pip 运行脚本或安装包
- 如果项目根目录下存在ENVIRONMENT.md，请遵循其中的说明
- 无视readme.md中关于环境配置的内容

## 数据集

1. CityScape数据集 
已预先转化为Yolo所需格式。源域数据位于（正常天气）"datasets/cityscape_yolo/cityscapes.yaml"，以及目标域数据（雾天）位于"datasets/cityscape_yolo_foggy/cityscapes.yaml"。如果数据集出现问题，请暂停任务向用户询问。

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

### 自定义修改

如需在 DetectGRL 中添加 GRL 相关逻辑：

1. **修改检测头**: 编辑 `ultralytics/nn/modules/head.py` 中的 `DetectGRL` 类
2. **修改训练逻辑**: 编辑 `ultralytics/models/yolo/domain_adapt/train.py` 中的 `DomainAdaptationTrainer`
3. **修改损失函数**: 可创建新的损失函数并在训练器中使用
