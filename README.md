# SRAIOT 论文复现项目

本项目是对 Nature 系列期刊 Scientific Reports 上的论文 **《Secure routing in the Internet of Things (IoT) with intrusion detection capability based on software-defined networking (SDN) and Machine Learning techniques》** (SRAIOT) 的完整复现。

## 项目目录结构

项目采用了模块化设计，将入侵检测（IDS）、网络仿真、路由协议实现以及实验评估分离开来。以下是详细的项目结构说明：

### 根目录文件
*   `SRAIOT_Reproduction_Report.pdf`: 最终生成的复现实验报告（PDF格式），包含原理分析、详细复现步骤及结果对比。
*   `SRAIOT_Reproduction_Report.tex`: 复现报告的 LaTeX 源代码。
*   `README.md`: 本文件，提供项目的导航和各个文件的功能说明。
*   `requirements.txt`: 项目运行所需的 Python 依赖库列表（如 scikit-learn, networkx, matplotlib 等）。
*   `Phase1_Summary.md`: 第一阶段（集成 IDS 开发）的任务总结与初步实验数据。
*   `Phase2_3_Summary.md`: 第二、三阶段（网络仿真与路由协议构建）的任务总结与逻辑解析。
*   `Final_Summary_Report.md`: 全过程复现的结题总结报告。
*   `SRAIOT_Reproduction_Plan.md`: 项目启动初期制定的详细复现方案计划书。
*   `s41598-023-44764-6.pdf`: 原论文的 PDF 原始文件。
*   `s41598-023-44764-6.txt`: 原论文的文本提取版本，用于代码逻辑核对。

### 📁 ids/ (入侵检测系统模块)
负责基于集成学习的流量异常检测逻辑。
*   `preprocess.py`: 对 NSL-KDD 数据集进行预处理，包括标签二值化、类别特征编码和数值特征归一化。
*   `models.py`: 定义了集成模型 `SRAIOTEnsemble` 类，实现了 KNN, SVM, ANN 的异构集成及多数投票逻辑。
*   `train.py`: IDS 模型的训练脚本，执行 95/5 随机划分并保存训练好的模型至 `results/`。
*   `models_improved.py`: 改进模型实现（Focal Loss 与 AdaBoost），用于缓解类别不平衡带来的识别偏差。
*   `train_improved.py`: 改进模型训练脚本，支持启用/关闭 Focal Loss 或 AdaBoost。

## 优化思路与实现细节（Focal Loss 与 AdaBoost）

### 问题背景：KDD 数据集的类别不平衡
KDD 数据集存在一个固有缺陷：网络攻击样本分为**高频攻击**和**低频攻击**，高频攻击样本数量远大于低频攻击样本。常规机器学习方法往往更倾向于学习高频攻击的特征，从而**忽略低频攻击**，导致对低频攻击的识别率较差。新版 KDD 数据集似乎通过**欠采样（Undersampling）**降低了高频攻击样本数量以缓解该问题，但其实际效果仍待进一步验证。无论其效果如何，我们依然可以尝试在模型层面进行优化。

### 优化方案 1：Focal Loss（聚焦难分类样本）
**思路**：Focal Loss 通过引入 `(1 - p_t)^γ` 因子降低易分类样本的权重，提高难分类样本（常见于低频攻击）的学习关注度，从而提升对低频攻击的识别能力。  
**实现位置**：
* `ids/models_improved.py` 中的 `FocalLoss` 与 `FocalLossANNClassifier` 类  
**训练脚本**：
* `ids/train_improved.py` 默认启用 Focal Loss  
**可配置参数**：
* `--focal_alpha`：类别权重（默认 0.25）
* `--focal_gamma`：难样本聚焦程度（默认 2.0）
**使用示例**：
```bash
python ids/train_improved.py --use_adaboost False
```

### 优化方案 2：AdaBoost（迭代提升低频攻击权重）
**思路**：AdaBoost 会在训练过程中自动增加被误分类样本的权重，促使后续弱分类器更关注这些难分类样本（通常是低频攻击），从而改善识别效果。  
**实现位置**：
* `ids/models_improved.py` 中的 `SRAIOTEnsembleImproved` 类  
**训练脚本**：
* `ids/train_improved.py` 支持仅启用 AdaBoost  
**使用示例**：
```bash
python ids/train_improved.py --use_focal_loss False
```

### 📁 simulation/ (网络仿真模块)
负责模拟物联网物理环境、节点运动及 SRAIOT 路由协议。
*   `node.py`: 定义 `IoTNode` 类，模拟单个 IoT 节点的物理属性、位置移动、能量损耗和拥塞度。
*   `environment.py`: 定义 `IoTEnvironment` 类，管理区域内所有节点的生命周期、邻居发现及可视化触发。
*   `clustering.py`: 实现 SRAIOT 的核心聚类算法，利用 $T_{ij}$ 连接稳定性预测进行子网划分。
*   `routing.py`: 实现基于加权树的层次化路由逻辑，计算 $W_{ij}$ 链路权值并构建 Dijkstra 最短路径树。
*   `simulator.py`: 仿真运行器，整合物理模拟与 IDS 检测引擎，驱动每一时间步的流量交互。
*   `experiments.py`: 综合实验脚本，自动化运行不同参数（节点数、发送速率）下的仿真，生成对比图表。
*   `utils.py`: 存放仿真过程中使用的通用辅助函数。

### 📁 results/ (实验结果与数据)
存放运行过程中产生的所有日志、图表和持久化模型。
*   `sraiot_ids_model.pkl`: 经过训练的集成学习 IDS 模型文件。
*   `experiments/`: 专门存放与原论文 Figure 5-13 对标的对比实验图表（PNG）及 JSON 原始数据。
*   `sim_log_*.txt`: 每次仿真运行的详细控制台日志。
*   `security_step_*.png`: 仿真过程中实时生成的网络拓扑分布图，标注了普通节点、恶意节点及 SDN 控制器。
*   `final_experiment_stats.csv`: 实验过程中记录的所有网络性能指标统计表。

### 📁 archive/ & 📁 data/ (数据集)
*   `KDDTrain+.txt` / `KDDTest+.txt`: 原始 NSL-KDD 数据集文件，作为 IDS 训练和仿真流量模拟的数据源。

---
**项目复现人：周睿敏**  
**优化：周宇航**

**日期：2025年12月31日**
