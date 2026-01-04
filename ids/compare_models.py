"""
对比测试脚本：原始模型 vs 改进模型
对比不同配置下的模型性能，重点关注对低频攻击的识别率提升
"""

import argparse
import pandas as pd
import numpy as np
from preprocess import load_combined_data
from models import SRAIOTEnsemble
from models_improved import SRAIOTEnsembleImproved
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import time
import os
from datetime import datetime

def calculate_metrics(y_true, y_pred):
    """计算准确率、敏感性和特异性"""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }

def train_and_evaluate_model(model, model_name, X_train, X_test, y_train, y_test):
    """训练模型并评估性能"""
    print(f"\n{'='*60}")
    print(f"正在训练: {model_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # 训练模型
    if isinstance(X_train, pd.DataFrame):
        X_train_values = X_train.values
        X_test_values = X_test.values
        y_train_values = y_train.values
        y_test_values = y_test.values
    else:
        X_train_values = X_train
        X_test_values = X_test
        y_train_values = y_train
        y_test_values = y_test
    
    try:
        model.train(X_train_values, y_train_values)
        training_time = time.time() - start_time
        
        # 预测
        predictions = model.predict(X_test_values)
        if isinstance(predictions, tuple):
            y_pred, _ = predictions
        else:
            y_pred = predictions
        
        # 计算指标
        metrics = calculate_metrics(y_test_values, y_pred)
        metrics['training_time'] = training_time
        
        print(f"训练完成，耗时: {training_time:.2f}秒")
        print(f"准确率: {metrics['accuracy']:.4f}")
        print(f"敏感性 (攻击检测率): {metrics['sensitivity']:.4f}")
        print(f"特异性 (正常识别率): {metrics['specificity']:.4f}")
        print(f"混淆矩阵:")
        print(f"  真实\\预测     正常(0)    攻击(1)")
        print(f"  正常(0)      {metrics['tn']:8d}  {metrics['fp']:8d}")
        print(f"  攻击(1)      {metrics['fn']:8d}  {metrics['tp']:8d}")
        
        return metrics
        
    except Exception as e:
        print(f"训练失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def parse_args():
    parser = argparse.ArgumentParser(description="SRAIOT模型对比测试")
    parser.add_argument("--sample-fraction", type=float, default=0.3,
                        help="使用数据集的比例（0-1），默认0.3以加快对比速度")
    parser.add_argument("--test-size", type=float, default=0.05,
                        help="测试集比例，默认0.05")
    parser.add_argument("--random-state", type=int, default=42,
                        help="随机种子，默认42")
    parser.add_argument("--skip-full", action="store_true",
                        help="跳过Focal Loss + AdaBoost的完整改进配置")
    return parser.parse_args()


def main():
    args = parse_args()
    print("="*60)
    print("SRAIOT模型对比测试")
    print("对比原始模型和改进模型（Focal Loss + AdaBoost）的效果")
    print("="*60)
    print(f"运行参数: sample_fraction={args.sample_fraction}, test_size={args.test_size}, random_state={args.random_state}")
    
    # 1. 加载数据（使用固定的随机种子确保公平对比）
    print("\n正在加载数据...")
    df, scaler, encoders = load_combined_data()
    
    X = df.drop('label', axis=1)
    y = df['label']

    if not 0 < args.sample_fraction <= 1:
        raise ValueError("sample_fraction 必须在 (0, 1] 之间")

    if args.sample_fraction < 1:
        print(f"使用数据子集进行快速对比（比例={args.sample_fraction:.2f}）")
        X, _, y, _ = train_test_split(
            X, y, test_size=1 - args.sample_fraction,
            random_state=args.random_state, stratify=y
        )
    
    # 使用相同的随机种子划分数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )
    
    print(f"训练集规模: {len(X_train)}")
    print(f"测试集规模: {len(X_test)}")
    print(f"训练集中 - 正常样本: {(y_train == 0).sum()}, 攻击样本: {(y_train == 1).sum()}")
    print(f"测试集中 - 正常样本: {(y_test == 0).sum()}, 攻击样本: {(y_test == 1).sum()}")
    
    results = []
    
    # 2. 训练原始模型
    print("\n" + "="*60)
    print("第1组：原始模型（基线）")
    print("="*60)
    original_model = SRAIOTEnsemble()
    original_metrics = train_and_evaluate_model(
        original_model, "原始模型 (KNN+SVM+ANN)",
        X_train, X_test, y_train, y_test
    )
    if original_metrics:
        results.append({
            '模型名称': '原始模型 (基线)',
            '配置': 'KNN + SVM + ANN (标准交叉熵)',
            '准确率': original_metrics['accuracy'],
            '敏感性 (攻击检测率)': original_metrics['sensitivity'],
            '特异性 (正常识别率)': original_metrics['specificity'],
            '训练时间(秒)': original_metrics['training_time'],
            'TP': original_metrics['tp'],
            'TN': original_metrics['tn'],
            'FP': original_metrics['fp'],
            'FN': original_metrics['fn']
        })
    
    # 3. 训练改进模型 - 仅Focal Loss
    print("\n" + "="*60)
    print("第2组：改进模型 - 仅使用Focal Loss")
    print("="*60)
    improved_focal_only = SRAIOTEnsembleImproved(
        use_focal_loss=True,
        use_adaboost=False,
        focal_alpha=0.25,
        focal_gamma=2.0
    )
    focal_metrics = train_and_evaluate_model(
        improved_focal_only, "改进模型 (仅Focal Loss)",
        X_train, X_test, y_train, y_test
    )
    if focal_metrics:
        results.append({
            '模型名称': '改进模型 - 仅Focal Loss',
            '配置': 'KNN + SVM + ANN(Focal Loss)',
            '准确率': focal_metrics['accuracy'],
            '敏感性 (攻击检测率)': focal_metrics['sensitivity'],
            '特异性 (正常识别率)': focal_metrics['specificity'],
            '训练时间(秒)': focal_metrics['training_time'],
            'TP': focal_metrics['tp'],
            'TN': focal_metrics['tn'],
            'FP': focal_metrics['fp'],
            'FN': focal_metrics['fn']
        })
    
    # 4. 训练改进模型 - 仅AdaBoost
    print("\n" + "="*60)
    print("第3组：改进模型 - 仅使用AdaBoost")
    print("="*60)
    improved_adaboost_only = SRAIOTEnsembleImproved(
        use_focal_loss=False,
        use_adaboost=True,
        focal_alpha=0.25,
        focal_gamma=2.0
    )
    adaboost_metrics = train_and_evaluate_model(
        improved_adaboost_only, "改进模型 (仅AdaBoost)",
        X_train, X_test, y_train, y_test
    )
    if adaboost_metrics:
        results.append({
            '模型名称': '改进模型 - 仅AdaBoost',
            '配置': 'KNN + SVM + ANN + AdaBoost',
            '准确率': adaboost_metrics['accuracy'],
            '敏感性 (攻击检测率)': adaboost_metrics['sensitivity'],
            '特异性 (正常识别率)': adaboost_metrics['specificity'],
            '训练时间(秒)': adaboost_metrics['training_time'],
            'TP': adaboost_metrics['tp'],
            'TN': adaboost_metrics['tn'],
            'FP': adaboost_metrics['fp'],
            'FN': adaboost_metrics['fn']
        })
    
    # 5. 训练改进模型 - Focal Loss + AdaBoost
    full_metrics = None
    if not args.skip_full:
        print("\n" + "="*60)
        print("第4组：改进模型 - Focal Loss + AdaBoost (完整改进)")
        print("="*60)
        improved_full = SRAIOTEnsembleImproved(
            use_focal_loss=True,
            use_adaboost=True,
            focal_alpha=0.25,
            focal_gamma=2.0
        )
        full_metrics = train_and_evaluate_model(
            improved_full, "改进模型 (Focal Loss + AdaBoost)",
            X_train, X_test, y_train, y_test
        )
        if full_metrics:
            results.append({
                '模型名称': '改进模型 - Focal Loss + AdaBoost',
                '配置': 'KNN + SVM + ANN(Focal Loss) + AdaBoost',
                '准确率': full_metrics['accuracy'],
                '敏感性 (攻击检测率)': full_metrics['sensitivity'],
                '特异性 (正常识别率)': full_metrics['specificity'],
                '训练时间(秒)': full_metrics['training_time'],
                'TP': full_metrics['tp'],
                'TN': full_metrics['tn'],
                'FP': full_metrics['fp'],
                'FN': full_metrics['fn']
            })
    
    # 6. 生成对比报告
    if results:
        print("\n" + "="*60)
        print("对比测试结果汇总")
        print("="*60)
        
        results_df = pd.DataFrame(results)

        if original_metrics:
            results_df['敏感性提升(%)'] = (results_df['敏感性 (攻击检测率)'] - original_metrics['sensitivity']) * 100
            results_df['准确率提升(%)'] = (results_df['准确率'] - original_metrics['accuracy']) * 100
            results_df['特异性提升(%)'] = (results_df['特异性 (正常识别率)'] - original_metrics['specificity']) * 100
        
        # 格式化输出
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 50)
        
        print("\n详细对比表：")
        print(results_df.to_string(index=False))
        
        # 计算改进幅度（相对于原始模型）
        improvement_df = []  # 初始化改进幅度分析列表
        if original_metrics:
            print("\n" + "="*60)
            print("改进幅度分析（相对于原始模型）")
            print("="*60)
            
            baseline_acc = original_metrics['accuracy']
            baseline_sens = original_metrics['sensitivity']
            baseline_spec = original_metrics['specificity']
            
            for result in results[1:]:  # 跳过原始模型本身
                acc_improve = (result['准确率'] - baseline_acc) * 100
                sens_improve = (result['敏感性 (攻击检测率)'] - baseline_sens) * 100
                spec_improve = (result['特异性 (正常识别率)'] - baseline_spec) * 100
                
                improvement_df.append({
                    '模型名称': result['模型名称'],
                    '准确率提升(%)': f"{acc_improve:+.4f}",
                    '敏感性提升(%)': f"{sens_improve:+.4f}",
                    '特异性提升(%)': f"{spec_improve:+.4f}",
                    '训练时间变化(秒)': f"{result['训练时间(秒)'] - original_metrics['training_time']:+.2f}"
                })
            
            if improvement_df:
                improvement_results = pd.DataFrame(improvement_df)
                print(improvement_results.to_string(index=False))
        
        # 保存结果到CSV
        output_dir = 'results'
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f'comparison_report_{timestamp}.csv')
        results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n对比报告已保存到: {output_file}")
        
        # 保存改进幅度分析
        if improvement_df:
            improvement_file = os.path.join(output_dir, f'improvement_analysis_{timestamp}.csv')
            pd.DataFrame(improvement_df).to_csv(improvement_file, index=False, encoding='utf-8-sig')
            print(f"改进幅度分析已保存到: {improvement_file}")
        
        # 关键指标总结
        if original_metrics:
            print("\n" + "="*60)
            print("关键指标总结")
            print("="*60)
            print(f"\n重点关注指标：敏感性（攻击检测率）")
            print(f"原始模型敏感性: {original_metrics['sensitivity']:.4f}")
            if full_metrics:
                print(f"完整改进模型敏感性: {full_metrics['sensitivity']:.4f}")
                improvement = (full_metrics['sensitivity'] - original_metrics['sensitivity']) * 100
                print(f"提升幅度: {improvement:+.4f}%")

            best_row = results_df.loc[results_df['敏感性 (攻击检测率)'].idxmax()]
            print("\n最佳敏感性模型:")
            print(f"  模型: {best_row['模型名称']}")
            print(f"  敏感性: {best_row['敏感性 (攻击检测率)']:.4f}")
            if '敏感性提升(%)' in best_row:
                print(f"  相对原始模型提升: {best_row['敏感性提升(%)']:+.4f}%")
        
        print("\n测试完成！")
    else:
        print("\n错误：所有模型训练都失败了，无法生成对比报告。")

if __name__ == "__main__":
    main()

