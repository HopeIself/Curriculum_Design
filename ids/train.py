import pandas as pd
from preprocess import load_combined_data
from models import SRAIOTEnsemble
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import time
import joblib
import os

def calculate_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return accuracy, sensitivity, specificity

def main():
    # 1. 加载数据并合并
    print("正在加载并合并数据集 (KDDTrain+ & KDDTest+)...")
    df, scaler, encoders = load_combined_data()
    
    # 按照论文要求：随机选取 5% 作为测试集，其余 95% 作为训练集
    print("正在按照论文要求进行 95/5 随机划分...")
    X = df.drop('label', axis=1)
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)
    
    print(f"训练集规模: {len(X_train)}, 测试集规模: {len(X_test)}")
    
    # 2. 初始化并训练模型
    ensemble = SRAIOTEnsemble()
    start_time = time.time()
    ensemble.train(X_train, y_train)
    print(f"训练完成，耗时: {time.time() - start_time:.2f}s")
    
    # 保存模型和预处理器以便仿真使用
    print("保存模型与预处理器中...")
    model_data = {
        'model': ensemble,
        'scaler': scaler,
        'encoders': encoders,
        'feature_cols': X.columns.tolist()
    }
    joblib.dump(model_data, 'results/sraiot_ids_model.pkl')
    print("保存成功: results/sraiot_ids_model.pkl")
    
    # 3. 预测与评估
    print("正在对测试集进行预测...")
    y_pred_ensemble, individual_preds = ensemble.predict(X_test)
    knn_pred, svm_pred, ann_pred = individual_preds
    
    # 4. 输出结果
    results = []
    for name, pred in [("KNN", knn_pred), ("SVM", svm_pred), ("ANN", ann_pred), ("Ensemble", y_pred_ensemble)]:
        acc, sens, spec = calculate_metrics(y_test, pred)
        results.append({
            "Model": name,
            "Accuracy": acc,
            "Sensitivity": sens,
            "Specificity": spec
        })
        
    results_df = pd.DataFrame(results)
    print("\n--- 实验结果对比 ---")
    print(results_df.to_string(index=False))
    
    # 打印集成模型的详细混淆矩阵
    print("\n--- 集成模型混淆矩阵 ---")
    print(confusion_matrix(y_test, y_pred_ensemble))

if __name__ == "__main__":
    main()

