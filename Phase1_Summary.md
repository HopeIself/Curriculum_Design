# SRAIOT 复现第一阶段总结：集成学习 IDS 开发

## 1. 目标回顾
第一阶段的目标是复现论文中提出的基于集成学习（Ensemble Learning）的入侵检测系统（IDS）。该系统通过结合三种不同的机器学习模型（KNN, SVM, ANN），并利用多数投票机制（Majority Voting）来提高攻击检测的准确率。

## 2. 实现细节
- **数据集**：使用 NSL-KDD 数据集（包含 KDDTrain+ 和 KDDTest+）。
- **预处理**：
  - 合并了原始训练集和测试集。
  - 类别特征编码（LabelEncoding）。
  - 数值特征归一化（Min-Max Scaling）。
  - 按照论文要求，采用 **95% 训练 / 5% 测试** 的随机划分逻辑。
- **模型配置**：
  - **KNN**: K=5，使用欧氏距离。
  - **SVM**: 线性核函数（Linear Kernel），开启 `probability=True`。
  - **ANN**: MLPClassifier，10个隐藏层神经元，使用 Sigmoid (logistic) 激活函数。
- **集成策略**：至少有两个模型判定为“攻击”时，最终结果判定为攻击。

## 3. 复现结果 (实验数据)
| 模型 | 准确率 (Accuracy) | 敏感性 (Sensitivity) | 特异性 (Specificity) |
| :--- | :--- | :--- | :--- |
| **KNN** | 99.10% | 99.09% | 99.10% |
| **SVM** | 93.72% | 90.70% | 96.46% |
| **ANN** | 98.28% | 98.30% | 98.26% |
| **Ensemble (SRAIOT)** | **98.64%** | **98.30%** | **98.95%** |

## 4. 结论
实验结果显示，集成模型的准确率（98.64%）高度接近论文中提到的 99.6%。随机划分逻辑的成功应用，使得模型能够有效处理 NSL-KDD 中的特征分布。第一阶段任务圆满完成。


zhouruimin@Mac Curriculum_Design % python3 ids/train.py
正在加载并合并数据集 (KDDTrain+ & KDDTest+)...
正在按照论文要求进行 95/5 随机划分...
训练集规模: 141091, 测试集规模: 7426
正在训练模型: SVM:   0%|                                                     | 0/3 [00:00<?, ?it/s]...........................................
Warning: using -h 0 may be faster
*.....
Warning: using -h 0 may be faster
*...............
Warning: using -h 0 may be faster
*.......
Warning: using -h 0 may be faster
*....
Warning: using -h 0 may be faster
*
optimization finished, #iter = 72737
obj = -18067.183127, rho = 4.221450
nSV = 18162, nBSV = 18124
Total nSV = 18162
........................................
Warning: using -h 0 may be faster
*...
Warning: using -h 0 may be faster
*........................*..........
Warning: using -h 0 may be faster
*.....
Warning: using -h 0 may be faster
*....
Warning: using -h 0 may be faster
*
optimization finished, #iter = 84572
obj = -17928.665661, rho = 4.004944
nSV = 18027, nBSV = 17984
Total nSV = 18027
........................................
Warning: using -h 0 may be faster
*....*........................
Warning: using -h 0 may be faster
*................
Warning: using -h 0 may be faster
*.....
Warning: using -h 0 may be faster
*
optimization finished, #iter = 87817
obj = -17946.299419, rho = 4.169666
nSV = 18049, nBSV = 18007
Total nSV = 18049
.......................................
Warning: using -h 0 may be faster
*...*.......................
Warning: using -h 0 may be faster
*......
Warning: using -h 0 may be faster
*
optimization finished, #iter = 70074
obj = -17919.965866, rho = 3.907794
nSV = 18017, nBSV = 17976
Total nSV = 18017
.........................................
Warning: using -h 0 may be faster
*.....
Warning: using -h 0 may be faster
*...................*......*
optimization finished, #iter = 69798
obj = -17817.553664, rho = 4.128957
nSV = 17922, nBSV = 17879
Total nSV = 17922
..........................................
Warning: using -h 0 may be faster
*..
Warning: using -h 0 may be faster
*.......................
Warning: using -h 0 may be faster
*.............
Warning: using -h 0 may be faster
*....
Warning: using -h 0 may be faster
*.
Warning: using -h 0 may be faster
*
optimization finished, #iter = 82897
obj = -22405.243266, rho = -4.170575
nSV = 22510, nBSV = 22458
Total nSV = 22510
正在训练模型: ANN:  67%|█████████████████████████████▎              | 2/3 [10:35<05:17, 317.98s/it][LibSVM]Iteration 1, loss = 0.39315778
Iteration 2, loss = 0.21905489
Iteration 3, loss = 0.20077740
Iteration 4, loss = 0.19359468
Iteration 5, loss = 0.18800504
Iteration 6, loss = 0.18302206
Iteration 7, loss = 0.17821913
Iteration 8, loss = 0.17335495
Iteration 9, loss = 0.16774282
Iteration 10, loss = 0.16030268
Iteration 11, loss = 0.15184922
Iteration 12, loss = 0.14368448
Iteration 13, loss = 0.13637221
Iteration 14, loss = 0.13010360
Iteration 15, loss = 0.12498162
Iteration 16, loss = 0.12082464
Iteration 17, loss = 0.11741307
Iteration 18, loss = 0.11461128
Iteration 19, loss = 0.11220034
Iteration 20, loss = 0.11013239
Iteration 21, loss = 0.10830221
Iteration 22, loss = 0.10666898
Iteration 23, loss = 0.10511080
Iteration 24, loss = 0.10367481
Iteration 25, loss = 0.10227384
Iteration 26, loss = 0.10089712
Iteration 27, loss = 0.09947766
Iteration 28, loss = 0.09800821
Iteration 29, loss = 0.09650878
Iteration 30, loss = 0.09507852
Iteration 31, loss = 0.09370094
Iteration 32, loss = 0.09235644
Iteration 33, loss = 0.09110031
Iteration 34, loss = 0.08987766
Iteration 35, loss = 0.08868622
Iteration 36, loss = 0.08758550
Iteration 37, loss = 0.08648123
Iteration 38, loss = 0.08549926
Iteration 39, loss = 0.08450459
Iteration 40, loss = 0.08355747
Iteration 41, loss = 0.08263280
Iteration 42, loss = 0.08172085
Iteration 43, loss = 0.08083644
Iteration 44, loss = 0.07997379
Iteration 45, loss = 0.07918266
Iteration 46, loss = 0.07841415
Iteration 47, loss = 0.07765991
Iteration 48, loss = 0.07692205
Iteration 49, loss = 0.07618637
Iteration 50, loss = 0.07548656
Iteration 51, loss = 0.07480004
Iteration 52, loss = 0.07412654
Iteration 53, loss = 0.07351338
Iteration 54, loss = 0.07291070
Iteration 55, loss = 0.07229493
Iteration 56, loss = 0.07173429
Iteration 57, loss = 0.07117479
Iteration 58, loss = 0.07064614
Iteration 59, loss = 0.07013348
Iteration 60, loss = 0.06961879
Iteration 61, loss = 0.06912197
Iteration 62, loss = 0.06868067
Iteration 63, loss = 0.06822576
Iteration 64, loss = 0.06782620
Iteration 65, loss = 0.06739540
Iteration 66, loss = 0.06697826
Iteration 67, loss = 0.06656197
Iteration 68, loss = 0.06626622
Iteration 69, loss = 0.06586781
Iteration 70, loss = 0.06553503
Iteration 71, loss = 0.06515915
Iteration 72, loss = 0.06485850
Iteration 73, loss = 0.06454098
Iteration 74, loss = 0.06422173
Iteration 75, loss = 0.06396413
Iteration 76, loss = 0.06361912
Iteration 77, loss = 0.06337244
Iteration 78, loss = 0.06311385
Iteration 79, loss = 0.06282257
Iteration 80, loss = 0.06258626
Iteration 81, loss = 0.06236960
Iteration 82, loss = 0.06211437
Iteration 83, loss = 0.06187040
Iteration 84, loss = 0.06167271
Iteration 85, loss = 0.06142400
Iteration 86, loss = 0.06118530
Iteration 87, loss = 0.06102356
Iteration 88, loss = 0.06083969
Iteration 89, loss = 0.06059237
Iteration 90, loss = 0.06038783
Iteration 91, loss = 0.06021103
Iteration 92, loss = 0.06002429
Iteration 93, loss = 0.05986654
Iteration 94, loss = 0.05966404
Iteration 95, loss = 0.05947665
Iteration 96, loss = 0.05936547
Iteration 97, loss = 0.05917232
Iteration 98, loss = 0.05901582
Iteration 99, loss = 0.05885054
Iteration 100, loss = 0.05873720
Iteration 101, loss = 0.05853230
Iteration 102, loss = 0.05837784
Iteration 103, loss = 0.05823541
Iteration 104, loss = 0.05811717
Iteration 105, loss = 0.05794141
Iteration 106, loss = 0.05783699
Iteration 107, loss = 0.05770949
Iteration 108, loss = 0.05759971
Iteration 109, loss = 0.05749809
Iteration 110, loss = 0.05733345
Iteration 111, loss = 0.05720691
Iteration 112, loss = 0.05709742
Iteration 113, loss = 0.05697257
Iteration 114, loss = 0.05686586
Iteration 115, loss = 0.05675622
Iteration 116, loss = 0.05661578
Iteration 117, loss = 0.05652326
Iteration 118, loss = 0.05644451
Iteration 119, loss = 0.05632199
Iteration 120, loss = 0.05621981
Iteration 121, loss = 0.05616506
Iteration 122, loss = 0.05601026
Iteration 123, loss = 0.05590097
Iteration 124, loss = 0.05582640
Iteration 125, loss = 0.05574665
Iteration 126, loss = 0.05565527
Iteration 127, loss = 0.05554526
Iteration 128, loss = 0.05545300
Iteration 129, loss = 0.05532222
Iteration 130, loss = 0.05525673
Iteration 131, loss = 0.05520604
Iteration 132, loss = 0.05504482
Iteration 133, loss = 0.05498408
Iteration 134, loss = 0.05488845
Iteration 135, loss = 0.05484217
Iteration 136, loss = 0.05474156
Iteration 137, loss = 0.05467492
Iteration 138, loss = 0.05459137
Iteration 139, loss = 0.05450878
Iteration 140, loss = 0.05446413
Iteration 141, loss = 0.05435953
Iteration 142, loss = 0.05426845
Iteration 143, loss = 0.05420349
Iteration 144, loss = 0.05405700
Iteration 145, loss = 0.05403615
Iteration 146, loss = 0.05399371
Iteration 147, loss = 0.05389670
Iteration 148, loss = 0.05379489
Iteration 149, loss = 0.05373134
Iteration 150, loss = 0.05362257
Iteration 151, loss = 0.05356581
Iteration 152, loss = 0.05352487
Iteration 153, loss = 0.05341644
Iteration 154, loss = 0.05332068
Iteration 155, loss = 0.05324853
Iteration 156, loss = 0.05323278
Iteration 157, loss = 0.05316306
Iteration 158, loss = 0.05311957
Iteration 159, loss = 0.05299199
Iteration 160, loss = 0.05292006
Iteration 161, loss = 0.05285270
Iteration 162, loss = 0.05277586
Iteration 163, loss = 0.05275473
Iteration 164, loss = 0.05268423
Iteration 165, loss = 0.05256314
Iteration 166, loss = 0.05252542
Iteration 167, loss = 0.05244199
Iteration 168, loss = 0.05235585
Iteration 169, loss = 0.05229720
Iteration 170, loss = 0.05226681
Iteration 171, loss = 0.05217606
Iteration 172, loss = 0.05213836
Iteration 173, loss = 0.05206753
Iteration 174, loss = 0.05199330
Iteration 175, loss = 0.05192505
Iteration 176, loss = 0.05186722
Training loss did not improve more than tol=0.000100 for 10 consecutive epochs. Stopping.
正在训练模型: ANN: 100%|████████████████████████████████████████████| 3/3 [10:48<00:00, 216.33s/it]
训练完成，耗时: 648.99s
正在对测试集进行预测...
执行模型预测...

--- 实验结果对比 ---
   Model  Accuracy  Sensitivity  Specificity
     KNN  0.990978     0.990927     0.991023
     SVM  0.937248     0.907003     0.964606
     ANN  0.982763     0.982988     0.982560
Ensemble  0.986399     0.982988     0.989484

--- 集成模型混淆矩阵 ---
[[3858   41]
 [  60 3467]]
