import matplotlib.pyplot as plt
from .node import IoTNode, calculate_stability
from .clustering import perform_sraiot_clustering
import numpy as np

class IoTEnvironment:
    def __init__(self, num_nodes=100, area_size=(500, 500), radio_range=100, delta_t=10.0):
        self.area_size = area_size
        self.radio_range = radio_range
        self.delta_t = delta_t
        self.nodes = [IoTNode(i, area_size, radio_range) for i in range(num_nodes)]
        self.time = 0.0
        self.clusters = []
        self.controllers = []

    def discover_neighbors(self):
        """模拟 Hello 报文交换，发现邻居并记录 RSSI"""
        for i, node_i in enumerate(self.nodes):
            if not node_i.is_alive: continue
            node_i.neighbors = []
            node_i.rssi_table = {}
            # 重置控制器状态，等待重新竞选
            node_i.is_controller = False
            
            for j, node_j in enumerate(self.nodes):
                if i == j or not node_j.is_alive: continue
                
                dist = node_i.get_distance(node_j)
                if dist <= self.radio_range:
                    node_i.neighbors.append(node_j.node_id)
                    node_i.rssi_table[node_j.node_id] = node_i.estimate_rssi(node_j)

    def run_clustering(self):
        """执行 SRAIOT 聚类过程"""
        active_nodes = self.get_active_nodes()
        self.clusters, self.controllers = perform_sraiot_clustering(active_nodes, self.delta_t)
        return len(self.clusters)

    def step(self, dt=1.0):
        """仿真前进一步"""
        self.time += dt
        for node in self.nodes:
            node.move(dt)
        self.discover_neighbors()
        # 按照论文，聚类是周期性（delta_t）进行的
        if int(self.time) % int(self.delta_t) == 0:
            self.run_clustering()

    def get_active_nodes(self):
        return [n for n in self.nodes if n.is_alive]

    def visualize(self):
        """可视化当前节点分布及通信链路"""
        plt.figure(figsize=(10, 10))
        active_nodes = self.get_active_nodes()
        
        # 绘制普通节点
        x = [node.x for node in active_nodes if not node.is_controller]
        y = [node.y for node in active_nodes if not node.is_controller]
        plt.scatter(x, y, c='blue', s=30, label='Normal Nodes')
        
        # 绘制控制器
        controllers = [n for n in active_nodes if n.is_controller]
        if controllers:
            cx = [node.x for node in controllers]
            cy = [node.y for node in controllers]
            plt.scatter(cx, cy, c='red', s=100, marker='*', label='SDN Controllers')

        plt.xlim(0, self.area_size[0])
        plt.ylim(0, self.area_size[1])
        plt.title(f"SRAIOT Clustering at T={self.time:.1f}s (Clusters: {len(self.clusters)})")
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    env = IoTEnvironment(num_nodes=100)
    print("开始运行聚类仿真...")
    env.discover_neighbors()
    num_clusters = env.run_clustering()
    print(f"聚类完成，生成了 {num_clusters} 个子网，选举了 {len(env.controllers)} 个控制器。")
    # env.visualize()
