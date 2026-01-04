import pandas as pd
from preprocess import load_combined_data
from models_improved import SRAIOTEnsembleImproved
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import time
import joblib
import os
import argparse

def calculate_metrics(y_true, y_pred):
    """计算准确率、敏感性和特异性"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return accuracy, sensitivity, specificity

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    parser = argparse.ArgumentParser(description='训练改进版SRAIOT模型')
    parser.add_argument('--use_focal_loss', type=str2bool, default=True, const=True, nargs='?',
                       help='使用Focal Loss ANN（默认True）')
    parser.add_argument('--use_adaboost', type=str2bool, default=True, const=True, nargs='?',
                       help='使用AdaBoost分类器（默认True）')
    parser.add_argument('--focal_alpha', type=float, default=0.25,
                       help='Focal Loss的alpha参数（默认0.25）')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                       help='Focal Loss的gamma参数（默认2.0）')
    parser.add_argument('--output_suffix', type=str, default='improved',
                       help='输出模型文件名后缀（默认improved）')
    
    args = parser.parse_args()
    
    # 1. 加载数据并合并
    print("正在加载并合并数据集 (KDDTrain+ & KDDTest+)...")
    df, scaler, encoders = load_combined_data()
    
    # 按照论文要求：随机选取 5% 作为测试集，其余 95% 作为训练集
    print("正在按照论文要求进行 95/5 随机划分...")
    X = df.drop('label', axis=1)
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)
    
    print(f"训练集规模: {len(X_train)}, 测试集规模: {len(X_test)}")
    print(f"训练集中正常样本: {(y_train == 0).sum()}, 攻击样本: {(y_train == 1).sum()}")
    print(f"测试集中正常样本: {(y_test == 0).sum()}, 攻击样本: {(y_test == 1).sum()}")
    
    # 2. 初始化并训练模型
    print(f"\n模型配置:")
    print(f"  - 使用Focal Loss: {args.use_focal_loss}")
    print(f"  - 使用AdaBoost: {args.use_adaboost}")
    if args.use_focal_loss:
        print(f"  - Focal Loss alpha: {args.focal_alpha}, gamma: {args.focal_gamma}")
    
    ensemble = SRAIOTEnsembleImproved(
        use_focal_loss=args.use_focal_loss,
        use_adaboost=args.use_adaboost,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma
    )
    
    start_time = time.time()
    ensemble.train(X_train.values, y_train.values)
    training_time = time.time() - start_time
    print(f"\n训练完成，耗时: {training_time:.2f}s")
    
    # 保存模型和预处理器以便仿真使用
    print("\n保存模型与预处理器中...")
    model_filename = f'results/sraiot_ids_model_{args.output_suffix}.pkl'
    model_data = {
        'model': ensemble,
        'scaler': scaler,
        'encoders': encoders,
        'feature_cols': X.columns.tolist(),
        'use_focal_loss': args.use_focal_loss,
        'use_adaboost': args.use_adaboost
    }
    joblib.dump(model_data, model_filename)
    print(f"保存成功: {model_filename}")
    
    # 3. 预测与评估
    print("\n正在对测试集进行预测...")
    y_pred_ensemble, individual_preds = ensemble.predict(X_test.values)
    
    if args.use_adaboost:
        knn_pred, svm_pred, ann_pred, adaboost_pred = individual_preds
        model_names = ["KNN", "SVM", "ANN", "AdaBoost", "Ensemble"]
        predictions = [knn_pred, svm_pred, ann_pred, adaboost_pred, y_pred_ensemble]
    else:
        knn_pred, svm_pred, ann_pred = individual_preds
        model_names = ["KNN", "SVM", "ANN", "Ensemble"]
        predictions = [knn_pred, svm_pred, ann_pred, y_pred_ensemble]
    
    # 4. 输出结果
    results = []
    for name, pred in zip(model_names, predictions):
        acc, sens, spec = calculate_metrics(y_test.values, pred)
        results.append({
            "Model": name,
            "Accuracy": f"{acc:.4f}",
            "Sensitivity": f"{sens:.4f}",
            "Specificity": f"{spec:.4f}"
        })
        
    results_df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("实验结果对比")
    print("="*60)
    print(results_df.to_string(index=False))
    
    # 打印集成模型的详细混淆矩阵
    print("\n" + "="*60)
    print("集成模型混淆矩阵")
    print("="*60)
    cm = confusion_matrix(y_test.values, y_pred_ensemble)
    print(f"\n真实\\预测     正常(0)    攻击(1)")
    print(f"正常(0)      {cm[0,0]:8d}  {cm[0,1]:8d}")
    print(f"攻击(1)      {cm[1,0]:8d}  {cm[1,1]:8d}")
    
    # 打印分类报告
    print("\n" + "="*60)
    print("集成模型详细分类报告")
    print("="*60)
    print(classification_report(y_test.values, y_pred_ensemble, 
                              target_names=['正常', '攻击']))
    
    # 保存结果到CSV
    results_csv = f'results/training_results_{args.output_suffix}.csv'
    results_df.to_csv(results_csv, index=False)
    print(f"\n结果已保存到: {results_csv}")

if __name__ == "__main__":
    main()

