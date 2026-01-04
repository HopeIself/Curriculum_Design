import numpy as np
import networkx as nx

def build_hierarchical_tree(controllers, all_nodes):
    """
    实现 SRAIOT 层次树构建
    1. 选取邻居最多的控制器作为中心节点 Ct
    2. 计算链路权重 Wij = (Cj * Dj) / Ej
    3. 构建最短路径树
    """
    if not controllers:
        return None, []

    # 1. 确定中心节点 Ct (邻居最多的控制器)
    ct = max(controllers, key=lambda c: len(c.neighbors))
    
    # 2. 构建图并计算权重
    # 注意：论文中提到通过中间节点（网关）连接，这里我们建立控制器间的逻辑图
    # 权重 Wij = (Cj * Dj) / Ej
    G = nx.Graph()
    
    # 获取所有参数的范围以便归一化 (公式 4)
    all_c = [n.get_congestion() for n in all_nodes]
    all_e = [n.energy for n in all_nodes]
    
    c_min, c_max = min(all_c), max(all_c)
    e_min, e_max = min(all_e), max(all_e)
    
    def normalize(val, v_min, v_max):
        if v_max == v_min: return 1.0
        return (val - v_min) / (v_max - v_min)

    # 为了简化仿真，我们假设控制器之间如果距离在 radio_range * 3 以内即可通过网关连接
    for i, c_i in enumerate(controllers):
        for j, c_j in enumerate(controllers):
            if i >= j: continue
            
            dist = c_i.get_distance(c_j)
            if dist < c_i.radio_range * 5: # 假设 5 跳内能达
                # 归一化参数
                norm_c = normalize(c_j.get_congestion(), c_min, c_max)
                norm_d = dist / (500 * np.sqrt(2)) # 距离归一化
                norm_e = normalize(c_j.energy, e_min, e_max) + 0.01 # 避免除 0
                
                # 公式 (2): Wij
                weight = (norm_c * norm_d) / norm_e
                G.add_edge(c_i.node_id, c_j.node_id, weight=weight)

    # 3. 计算从各控制器到 Ct 的最短路径 (层次树)
    tree_edges = []
    try:
        paths = nx.single_source_dijkstra_path(G, ct.node_id)
        for target_id, path in paths.items():
            if len(path) > 1:
                # 记录树的边：(子, 父)
                tree_edges.append((path[-1], path[-2]))
    except Exception as e:
        print(f"构建树时出错: {e}")

    return ct, tree_edges

