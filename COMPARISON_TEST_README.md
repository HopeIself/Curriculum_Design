# 模型对比测试脚本使用说明

## 概述

`ids/compare_models.py` 是一个全面的对比测试脚本，用于比较原始SRAIOT模型和改进模型（使用Focal Loss和AdaBoost）的性能。

## 功能特性

### 测试的模型配置

1. **原始模型（基线）**: KNN + SVM + ANN（标准交叉熵损失）
2. **改进模型 - 仅Focal Loss**: KNN + SVM + ANN（Focal Loss）
3. **改进模型 - 仅AdaBoost**: KNN + SVM + ANN + AdaBoost
4. **改进模型 - 完整改进**: KNN + SVM + ANN（Focal Loss）+ AdaBoost

### 评估指标

- **准确率 (Accuracy)**: 整体分类正确率
- **敏感性 (Sensitivity)**: 攻击检测率 = TP / (TP + FN) - **重点关注指标**
- **特异性 (Specificity)**: 正常样本识别率 = TN / (TN + FP)
- **混淆矩阵**: TP, TN, FP, FN
- **训练时间**: 每个模型的训练耗时

### 输出内容

1. **控制台输出**:
   - 每个模型的训练进度和结果
   - 详细对比表
   - 改进幅度分析（相对于原始模型）
   - 关键指标总结

2. **CSV文件**:
   - `results/comparison_report_YYYYMMDD_HHMMSS.csv`: 详细对比报告
   - `results/improvement_analysis_YYYYMMDD_HHMMSS.csv`: 改进幅度分析

## 使用方法

### 基本使用

```bash
python ids/compare_models.py
```

### 运行要求

1. **数据文件**: 确保 `archive/KDDTrain+.txt` 和 `archive/KDDTest+.txt` 存在
2. **依赖库**: 已安装所有requirements.txt中的依赖（包括torch）
3. **时间**: 完整测试可能需要较长时间（取决于硬件配置）
   - 原始模型: ~10-15分钟（SVM训练较慢）
   - 改进模型: 每个配置额外 ~10-15分钟

### 输出示例

```
============================================================
SRAIOT模型对比测试
对比原始模型和改进模型（Focal Loss + AdaBoost）的效果
============================================================

正在加载数据...
训练集规模: 141091
测试集规模: 7426
...

============================================================
对比测试结果汇总
============================================================

详细对比表：
模型名称                         配置                             准确率    敏感性 (攻击检测率)  特异性 (正常识别率)  训练时间(秒)  ...
原始模型 (基线)                  KNN + SVM + ANN (标准交叉熵)     0.9864    0.9830              0.9895              850.23       ...
改进模型 - 仅Focal Loss          KNN + SVM + ANN(Focal Loss)      0.9870    0.9845              0.9900              920.45       ...
...

改进幅度分析（相对于原始模型）
模型名称                         准确率提升(%)  敏感性提升(%)  特异性提升(%)  训练时间变化(秒)
改进模型 - 仅Focal Loss         +0.0600        +0.1500        +0.0500        +70.22
...

对比报告已保存到: results/comparison_report_20250101_120000.csv
改进幅度分析已保存到: results/improvement_analysis_20250101_120000.csv
```

## 注意事项

1. **随机种子**: 所有模型使用相同的随机种子（random_state=42），确保公平对比

2. **训练时间**: 
   - 首次运行可能需要更长时间（PyTorch初始化等）
   - SVM训练通常是最慢的部分
   - 如果有GPU，Focal Loss ANN会自动使用GPU加速

3. **内存使用**: 
   - 完整测试需要足够的内存（建议至少8GB）
   - 如果内存不足，可以分别运行各个模型配置

4. **错误处理**: 
   - 如果某个模型训练失败，脚本会继续运行其他模型
   - 最终只会对比成功训练的模型

5. **结果解读**:
   - **敏感性提升**是最重要的指标，表示对攻击（特别是低频攻击）的检测率提升
   - 理想情况下，改进模型应该在保持或提高准确率的同时，显著提高敏感性

## 快速测试（可选）

如果只想快速测试某个配置，可以使用单独的训练脚本：

```bash
# 仅测试原始模型
python ids/train.py

# 仅测试完整改进模型
python ids/train_improved.py

# 仅测试Focal Loss
python ids/train_improved.py --use_adaboost False

# 仅测试AdaBoost
python ids/train_improved.py --use_focal_loss False
```

## 结果分析建议

1. **重点关注敏感性**: 这是改进的核心目标，应该看到明显的提升
2. **平衡准确率和敏感性**: 理想情况下两者都提高，但敏感性提升更重要
3. **训练时间对比**: 改进模型可能需要更长的训练时间，这是可以接受的权衡
4. **不同配置对比**: 比较仅Focal Loss、仅AdaBoost和两者组合的效果

## 故障排除

### 常见问题

1. **ImportError: No module named 'torch'**
   - 解决: `pip install torch`

2. **内存不足**
   - 解决: 减少batch_size（修改models_improved.py中的batch_size参数）
   - 或分别运行各个配置

3. **训练时间过长**
   - 这是正常的，特别是SVM训练
   - 可以考虑使用更少的迭代次数进行快速测试

4. **CUDA错误（如果有GPU）**
   - 脚本会自动回退到CPU，不影响功能
   - 检查torch是否正确安装GPU支持

## 相关文件

- `ids/models.py`: 原始模型实现
- `ids/models_improved.py`: 改进模型实现
- `ids/train.py`: 原始模型训练脚本
- `ids/train_improved.py`: 改进模型训练脚本
- `IMPROVEMENTS.md`: 改进方法详细说明

