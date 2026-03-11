# AGENTS.md

本项目在Yolov26中加入GRL梯度反转层以实现域适应（Domain Adaptation），可以将在源域（比如非雾天环境）训练的目标检测模型微调至适用于目标域（比如雾天环境）。而在ultralytics库中加入新模型的正确方法就是fork并修改ultralytics库本身。

## 环境配置

- 使用 `uv run` 来运行 Python 脚本，使用`uv add`来安装依赖，这会自动使用项目配置的虚拟环境，**不要**直接使用系统 Python 或 pip 运行脚本或安装包
- 如果项目根目录下存在ENVIRONMENT.md，请遵循其中的说明
- 无视readme.md中关于环境配置的内容

## 数据集

1. CityScape数据集 
已预先转化为Yolo所需格式。源域数据位于（正常天气）"datasets/cityscape_yolo/cityscapes.yaml"，以及目标域数据（雾天）位于"datasets/cityscape_yolo_foggy/cityscapes.yaml"。如果数据集出现问题，请暂停任务向用户询问。