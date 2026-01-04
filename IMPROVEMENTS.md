# SRAIOT模型改进说明

## 改进目标

针对KDD数据集中高频率攻击和低频率攻击不平衡的问题，我们实现了以下两个改进方案：

1. **Focal Loss**: 用于解决类别不平衡问题，特别关注难分类的样本（低频攻击）
2. **AdaBoost**: 通过自适应提升算法，自动对难分类样本赋予更高权重

## 改进内容

### 1. Focal Loss实现

Focal Loss的公式为：
```
FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
```

其中：
- `alpha`: 类别权重参数（默认0.25），用于平衡正负样本
- `gamma`: 聚焦参数（默认2.0），值越大越关注难分类样本
- `p_t`: 模型对真实类别的预测概率

**优势**:
- 自动降低易分类样本的权重
- 增加难分类样本（低频攻击）的关注度
- 不需要手动进行数据平衡处理

### 2. AdaBoost集成

AdaBoost通过以下机制改善低频攻击识别：
- 自动对误分类样本（通常是低频攻击）赋予更高权重
- 迭代训练多个弱分类器，每个弱分类器重点关注前一个分类错误的样本
- 最终通过加权投票组合所有弱分类器的结果

**优势**:
- 自动处理类别不平衡
- 提高整体模型的鲁棒性
- 与现有集成框架兼容良好

## 文件说明

### 新增文件

1. **`ids/models_improved.py`**: 改进版模型实现
   - `FocalLoss`: Focal Loss损失函数实现
   - `FocalLossANN`: 使用Focal Loss的ANN分类器
   - `FocalLossANNClassifier`: sklearn兼容的包装器
   - `SRAIOTEnsembleImproved`: 改进版集成模型

2. **`ids/train_improved.py`**: 改进版训练脚本
   - 支持Focal Loss和AdaBoost的配置
   - 完整的评估指标输出
   - 结果保存功能

### 修改文件

1. **`requirements.txt`**: 添加了`torch`和`joblib`依赖

## 使用方法

### 基本使用（同时启用Focal Loss和AdaBoost）

```bash
python ids/train_improved.py
```

### 仅使用Focal Loss（不使用AdaBoost）

```bash
python ids/train_improved.py --use_adaboost=False
```

### 仅使用AdaBoost（使用标准ANN）

```bash
python ids/train_improved.py --use_focal_loss=False
```

### 自定义Focal Loss参数

```bash
python ids/train_improved.py --focal_alpha 0.5 --focal_gamma 3.0
```

参数说明：
- `--focal_alpha`: Focal Loss的alpha参数（范围0-1，默认0.25）
  - 值越大，对少数类的关注度越高
- `--focal_gamma`: Focal Loss的gamma参数（默认2.0）
  - 值越大，越关注难分类样本
  - 推荐范围：1.0-5.0

### 指定输出文件名后缀

```bash
python ids/train_improved.py --output_suffix focal_adaboost
```

输出文件：
- 模型文件：`results/sraiot_ids_model_{suffix}.pkl`
- 结果文件：`results/training_results_{suffix}.csv`

## 模型对比

改进版模型与原始模型的区别：

| 特性 | 原始模型 | 改进版模型 |
|------|---------|-----------|
| ANN损失函数 | 标准交叉熵 | Focal Loss（可选） |
| 集成分类器数量 | 3个（KNN, SVM, ANN） | 4个（KNN, SVM, ANN, AdaBoost） |
| 类别不平衡处理 | 无 | Focal Loss + AdaBoost |
| 难样本关注度 | 低 | 高（通过Focal Loss和AdaBoost） |

## 预期效果

1. **提高低频攻击识别率**: Focal Loss和AdaBoost都会重点关注难分类的样本，从而提高对低频攻击类型的识别率

2. **保持整体性能**: 通过集成学习和多数投票机制，在提高低频攻击识别率的同时，保持整体准确率

3. **更好的敏感性和特异性平衡**: 改进后的模型应该在识别攻击（特别是低频攻击）方面表现更好

## 实验建议

为了全面评估改进效果，建议进行以下对比实验：

1. **基线对比**: 运行原始模型作为基线
   ```bash
   python ids/train.py
   ```

2. **仅Focal Loss**: 测试Focal Loss单独效果
   ```bash
   python ids/train_improved.py --use_adaboost=False
   ```

3. **仅AdaBoost**: 测试AdaBoost单独效果
   ```bash
   python ids/train_improved.py --use_focal_loss=False
   ```

4. **组合使用**: 测试Focal Loss + AdaBoost的组合效果
   ```bash
   python ids/train_improved.py
   ```

5. **参数调优**: 尝试不同的Focal Loss参数组合
   ```bash
   python ids/train_improved.py --focal_alpha 0.25 --focal_gamma 2.0
   python ids/train_improved.py --focal_alpha 0.5 --focal_gamma 2.0
   python ids/train_improved.py --focal_alpha 0.25 --focal_gamma 3.0
   ```

## 注意事项

1. **训练时间**: 使用PyTorch实现的Focal Loss ANN可能需要更长的训练时间，特别是首次运行时

2. **GPU支持**: 如果系统有CUDA支持的GPU，代码会自动使用GPU加速训练

3. **内存使用**: AdaBoost会增加模型的内存占用，如果数据量很大，可能需要调整batch_size

4. **兼容性**: 改进版模型保存的格式与原始模型兼容，但加载时需要导入`models_improved`模块

## 参考文献

- Focal Loss: Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. ICCV.

- AdaBoost: Freund, Y., & Schapire, R. E. (1997). A decision-theoretic generalization of on-line learning and an application to boosting. Journal of computer and system sciences.

