import os
import sys
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
from .environment import IoTEnvironment
from .routing import build_hierarchical_tree

# 将 ids 目录加入路径，以便 joblib 能够找到 models 模块
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ids"))

class SRAIOTSimulator:
    def __init__(self, num_nodes=100, area_size=(500, 500), results_dir="results"):
        self.results_dir = results_dir
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            
        self.env = IoTEnvironment(num_nodes=num_nodes, area_size=area_size)
        self.log_file = os.path.join(results_dir, f"sim_log_{int(time.time())}.txt")
        
        # 加载 IDS 模型
        model_path = os.path.join("results", "sraiot_ids_model.pkl")
        if os.path.exists(model_path):
            try:
                self.ids_data = joblib.load(model_path)
                self.model = self.ids_data['model']
                self.scaler = self.ids_data['scaler']
                self.encoders = self.ids_data['encoders']
                self.log("成功加载 IDS 模型")
            except Exception as e:
                self.log(f"加载模型失败: {e}")
                self.model = None
        else:
            self.log("警告: 未找到 IDS 模型文件！请先运行 ids/train.py")
            self.model = None

        # 加载攻击流量样本 (用于模拟)
        try:
            from ids.preprocess import load_combined_data
            df, _, _ = load_combined_data()
            self.attack_samples = df[df['label'] == 1]
            self.normal_samples = df[df['label'] == 0]
            self.log("成功加载流量样本库")
        except Exception as e:
            self.log(f"加载流量样本失败: {e}")

        # 随机指定恶意节点 (10%)
        self.malicious_ids = np.random.choice(range(num_nodes), size=int(num_nodes * 0.1), replace=False)
        self.stats = []

    def log(self, message):
        print(message)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(message + "\n")

    def simulate_traffic(self, step):
        """模拟网络流量并执行 IDS 检测"""
        total_packets = 0
        blocked_packets = 0
        correct_detections = 0
        
        active_nodes = self.env.get_active_nodes()
        if not active_nodes: return 0, 0, 0

        # 每个节点尝试发送 1 个数据包
        for node in active_nodes:
            is_malicious = node.node_id in self.malicious_ids
            # 随机抽取一个样本
            if is_malicious:
                sample = self.attack_samples.sample(1)
            else:
                sample = self.normal_samples.sample(1)
            
            # 特征准备
            X = sample.drop('label', axis=1)
            y_true = sample['label'].values[0]
            
            # 执行集成 IDS 检测
            if self.model:
                y_pred, _ = self.model.predict(X)
                is_attack_detected = (y_pred[0] == 1)
                
                total_packets += 1
                
                # 评估检测结果
                if is_attack_detected:
                    blocked_packets += 1
                    if is_malicious: # 正确识别攻击
                        correct_detections += 1
                else:
                    if not is_malicious: # 正确放行正常流量
                        correct_detections += 1
            
            # 模拟能耗 (基础消耗 + 发送消耗)
            node.consume_energy(0.001)

        return total_packets, blocked_packets, correct_detections

    def run(self, total_steps=20):
        self.log(f"--- SRAIOT 完整实验开始 (总节点: {len(self.env.nodes)}, 恶意节点: {len(self.malicious_ids)}) ---")
        
        for step in range(total_steps):
            # 1. 节点移动
            self.env.step(dt=1.0)
            
            # 2. 聚类与拓扑更新
            num_clusters = self.env.run_clustering()
            ct, tree_edges = build_hierarchical_tree(self.env.controllers, self.env.nodes)
            
            # 3. 流量模拟与安全防御
            total, blocked, correct = self.simulate_traffic(step)
            
            # 4. 指标统计
            acc = correct / total if total > 0 else 0
            pdr = (total - blocked) / total if total > 0 else 0 # 模拟交付率
            
            active_nodes = len(self.env.get_active_nodes())
            msg = f"Step {step}: 存活节点={active_nodes}, 簇={num_clusters}, 检测准确率={acc:.2%}, PDR={pdr:.2%}"
            self.log(msg)
            
            self.stats.append({
                "step": step,
                "active_nodes": active_nodes,
                "detection_accuracy": acc,
                "pdr": pdr,
                "energy_mean": np.mean([n.energy for n in self.env.nodes])
            })
            
            # 5. 保存可视化 (每 10 步)
            if step % 10 == 0 or step == total_steps - 1:
                self.save_visualization(step, ct, tree_edges)

        self.save_stats()
        self.log("--- 仿真结束 ---")

    def save_visualization(self, step, ct, tree_edges):
        plt.figure(figsize=(10, 10))
        active_nodes = self.env.get_active_nodes()
        
        # 绘制正常节点
        nx_val = [n.x for n in active_nodes if n.node_id not in self.malicious_ids]
        ny_val = [n.y for n in active_nodes if n.node_id not in self.malicious_ids]
        plt.scatter(nx_val, ny_val, c='blue', s=30, alpha=0.5, label='Healthy Nodes')
        
        # 绘制恶意节点
        mx_val = [n.x for n in active_nodes if n.node_id in self.malicious_ids]
        my_val = [n.y for n in active_nodes if n.node_id in self.malicious_ids]
        plt.scatter(mx_val, my_val, c='orange', s=60, marker='x', label='Malicious Nodes')
        
        # 绘制控制器
        controllers = [n for n in active_nodes if n.is_controller]
        plt.scatter([n.x for n in controllers], [n.y for n in controllers], 
                    c='red', s=120, marker='*', label='SDN Controllers')
        
        plt.xlim(0, self.env.area_size[0])
        plt.ylim(0, self.env.area_size[1])
        plt.title(f"SRAIOT Security Simulation at Step {step}")
        plt.legend()
        plt.grid(True)
        
        img_path = os.path.join(self.results_dir, f"security_step_{step}.png")
        plt.savefig(img_path)
        plt.close()

    def save_stats(self):
        df = pd.DataFrame(self.stats)
        csv_path = os.path.join(self.results_dir, "final_experiment_stats.csv")
        df.to_csv(csv_path, index=False)
        self.log(f"保存最终统计数据至: {csv_path}")

if __name__ == "__main__":
    sim = SRAIOTSimulator(num_nodes=100)
    sim.run(total_steps=20)
