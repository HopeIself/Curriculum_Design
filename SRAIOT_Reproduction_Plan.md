# SRAIOT 算法复现计划：基于 SDN 和机器学习的 IoT 安全路由与入侵检测

本项目旨在复现论文 *“Secure routing in the Internet of Things (IoT) with intrusion detection capability based on software-defined networking (SDN) and Machine Learning techniques”* 中提出的 SRAIOT 算法。

## 1. 项目目标
- 实现 SRAIOT 的核心组件：网络聚类、层次化路由树构建、集成学习入侵检测。
- 验证算法在数据包交付率、能耗、延迟及攻击检测准确率方面的表现。
- 对比集成学习模型与单一模型（SVM, KNN, ANN）的效果。

## 2. 核心算法组件
### A. 集成学习入侵检测 (IDS)
- **模型**：KNN (K=5), SVM (Linear Kernel), ANN (10 neurons hidden layer).
- **数据集**：NSL-KDD。
- **策略**：多数投票机制（Majority Voting）。

### B. SDN 网络聚类与拓扑管理
- **聚类依据**：节点移动模式稳定性 $T_{ij}$。
- **角色分配**：选取邻居度最高的节点作为 SDN 控制器。
- **路由权重**：$W_{ij} = (C_j \times D_j) / E_j$（拥塞、距离、剩余能量）。

## 3. 复现路线图

### 第一阶段：集成学习 IDS 开发 (当前任务)
1. [ ] 下载并处理 NSL-KDD 数据集。
2. [ ] 实现数据标准化与特征编码。
3. [ ] 训练并评估单一模型 (KNN, SVM, ANN)。
4. [ ] 实现多数投票集成模型并对比效果。

### 第二阶段：IoT 网络仿真环境搭建
1. [ ] 定义节点类（包含位置、能量、速度等属性）。
2. [ ] 实现移动性模型及连接稳定性预测公式。
3. [ ] 实现基于 RSSI 的邻居发现机制。

### 第三阶段：SRAIOT 路由协议实现
1. [ ] 实现 SDN 驱动的聚类算法。
2. [ ] 构建层次化最短路径路由树。
3. [ ] 实现跨子网数据传输逻辑。

### 第四阶段：实验仿真与结果分析
1. [ ] 运行不同节点规模（100-300）的仿真。
2. [ ] 运行不同数据包速率（80-120 pkts/s）的仿真。
3. [ ] 统计并绘制性能曲线图。

## 4. 环境要求
- Python 3.8+
- Scikit-learn, PyTorch (或 TensorFlow)
- NetworkX, Pandas, Matplotlib, Numpy

