# Agent交接文档 - SRAIOT模型改进项目

## 项目背景

这是一个复现论文《Secure routing in the Internet of Things (IoT) with intrusion detection capability based on software-defined networking (SDN) and Machine Learning techniques》(SRAIOT)的项目。

**项目路径**: `E:\桌面\Curriculum_Design`

**核心问题**: KDD数据集存在类别不平衡问题
- 高频率攻击样本数量远大于低频率攻击样本
- 机器学习模型倾向于学习高频率攻击特征，忽视低频率攻击特征
- 导致对低频率攻击的识别率较差

## 改进目标

### 目标1: 使用Focal Loss解决类别不平衡
- **原理**: Focal Loss通过 `(1 - p_t)^gamma` 降低易分类样本权重，提高难分类样本（低频攻击）的关注度
- **实现位置**: `ids/models_improved.py` 中的 `FocalLoss` 和 `FocalLossANNClassifier` 类
- **状态**: ✅ 已完成

### 目标2: 使用AdaBoost改善低频攻击识别
- **原理**: AdaBoost自动对误分类样本（通常是低频攻击）赋予更高权重，迭代训练多个弱分类器
- **实现位置**: `ids/models_improved.py` 中的 `SRAIOTEnsembleImproved` 类
- **状态**: ✅ 已完成

## 已完成的工作

### 1. 核心代码文件

#### `ids/models_improved.py` ✅
- `FocalLoss`: Focal Loss损失函数实现（支持二分类）
- `FocalLossANN`: 使用Focal Loss的PyTorch神经网络
- `FocalLossANNClassifier`: sklearn兼容的包装器类
- `SRAIOTEnsembleImproved`: 改进版集成模型，支持：
  - Focal Loss ANN（可选）
  - AdaBoost分类器（可选）
  - 与原有KNN、SVM的集成

#### `ids/train_improved.py` ✅
- 改进版训练脚本
- 支持命令行参数配置
- 完整的评估指标输出（准确率、敏感性、特异性）
- 混淆矩阵和分类报告

#### `requirements.txt` ✅
- 已添加 `torch` 和 `joblib` 依赖

#### `IMPROVEMENTS.md` ✅
- 详细的改进说明文档
- 使用方法和参数说明
- 实验建议

### 2. 代码特性

- ✅ Focal Loss实现（alpha=0.25, gamma=2.0可调）
- ✅ AdaBoost集成（50个弱分类器）
- ✅ 保持与原始模型的兼容性
- ✅ 支持GPU加速（自动检测CUDA）
- ✅ 完整的评估指标输出

## 待完成的工作

### 优先级1: 创建对比测试脚本 ⚠️

**任务**: 创建一个测试脚本，对比原始模型和改进模型的效果

**要求**:
1. 使用相同的数据集和随机种子，确保公平对比
2. 训练原始模型（`ids/models.py` + `ids/train.py`）
3. 训练改进模型（不同配置）:
   - 仅Focal Loss
   - 仅AdaBoost
   - Focal Loss + AdaBoost
4. 对比指标:
   - 准确率 (Accuracy)
   - 敏感性 (Sensitivity) - **重点关注**（攻击检测率）
   - 特异性 (Specificity)
   - 混淆矩阵
   - 训练时间
5. 生成对比报告（CSV/文本格式）
6. 可视化对比结果（可选，使用matplotlib）

**建议脚本结构**:
```python
# ids/compare_models.py
- 加载数据（统一使用random_state=42）
- 训练原始模型
- 训练改进模型（3种配置）
- 对比所有指标
- 生成对比报告
- 保存结果到 results/comparison_report.csv
```

### 优先级2: 验证和改进（可选）

1. **测试代码运行**: 确保所有代码可以正常运行
2. **参数调优**: 尝试不同的Focal Loss参数组合
3. **结果分析**: 分析改进效果，特别是对低频攻击的识别率提升

## 文件结构

```
Curriculum_Design/
├── ids/
│   ├── models.py              # 原始模型（已完成）
│   ├── models_improved.py     # 改进模型（✅ 已完成）
│   ├── train.py               # 原始训练脚本（已完成）
│   ├── train_improved.py      # 改进训练脚本（✅ 已完成）
│   ├── preprocess.py          # 数据预处理（已完成）
│   └── compare_models.py      # ⚠️ 待创建：对比测试脚本
├── results/                    # 结果输出目录
├── requirements.txt            # ✅ 已更新
├── IMPROVEMENTS.md             # ✅ 改进说明文档
└── AGENT_HANDOFF.md           # 本文档
```

## 使用方法

### 运行原始模型
```bash
python ids/train.py
```

### 运行改进模型（默认：Focal Loss + AdaBoost）
```bash
python ids/train_improved.py
```

### 仅使用Focal Loss
```bash
python ids/train_improved.py --use_adaboost False
```

### 仅使用AdaBoost
```bash
python ids/train_improved.py --use_focal_loss False
```

### 自定义Focal Loss参数
```bash
python ids/train_improved.py --focal_alpha 0.5 --focal_gamma 3.0
```

## 关键代码位置

### Focal Loss实现
- **文件**: `ids/models_improved.py`
- **类**: `FocalLoss` (第12-39行)
- **关键参数**: `alpha=0.25`, `gamma=2.0`

### AdaBoost实现
- **文件**: `ids/models_improved.py`
- **类**: `SRAIOTEnsembleImproved` (第155-231行)
- **配置**: 50个弱分类器，基于决策树

### 数据预处理
- **文件**: `ids/preprocess.py`
- **函数**: `load_combined_data()`
- **注意**: 标签已二值化（0=正常, 1=攻击）

## 注意事项

1. **训练时间**: 
   - 原始模型：SVM训练较慢（约10-15分钟）
   - 改进模型：Focal Loss ANN使用PyTorch，首次运行可能较慢
   - 建议：如果数据量大，考虑使用GPU

2. **内存使用**: 
   - AdaBoost会增加内存占用
   - 如果内存不足，可以减少AdaBoost的n_estimators

3. **随机种子**: 
   - 所有脚本使用 `random_state=42` 确保可复现性
   - 对比测试时务必使用相同的随机种子

4. **模型保存**: 
   - 原始模型：`results/sraiot_ids_model.pkl`
   - 改进模型：`results/sraiot_ids_model_improved.pkl`

## 评估指标说明

- **准确率 (Accuracy)**: 整体分类正确率
- **敏感性 (Sensitivity)**: 攻击检测率 = TP / (TP + FN) - **这是改进的重点**
- **特异性 (Specificity)**: 正常样本识别率 = TN / (TN + FP)

**改进目标**: 在保持或提高准确率的同时，**显著提高敏感性**（特别是对低频攻击的检测率）

## 下一步行动

1. **立即任务**: 创建 `ids/compare_models.py` 对比测试脚本
2. **运行测试**: 执行对比测试，生成对比报告
3. **分析结果**: 评估改进效果，特别是敏感性提升
4. **文档更新**: 根据测试结果更新文档

## 联系信息

如有问题，请参考：
- `IMPROVEMENTS.md`: 详细的改进说明
- `README.md`: 项目整体说明
- `ids/models_improved.py`: 代码注释

---

**最后更新**: 2025年1月
**状态**: 核心功能已完成，等待对比测试脚本


