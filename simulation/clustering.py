import numpy as np
from .node import calculate_stability

def perform_sraiot_clustering(nodes, delta_t=10.0):
    """
    实现 SRAIOT 聚类算法
    :param nodes: 存活的节点列表
    :param delta_t: 论文中的时间间隔参数
    :return: clusters (list of lists of nodes), controllers (list of nodes)
    """
    unassigned_nodes = list(nodes)
    clusters = []
    
    while unassigned_nodes:
        # 1. 随机选取一个未分配节点
        x = unassigned_nodes.pop(np.random.randint(len(unassigned_nodes)))
        current_cluster = [x]
        
        # 2. 查找满足条件的邻居
        to_remove = []
        for y in unassigned_nodes:
            # 条件 1: 在通信范围内 (已经在 neighbors 列表中体现)
            # 条件 2: Tij >= delta_t
            if y.node_id in x.neighbors:
                t_ij = calculate_stability(x, y, delta_t)
                if t_ij >= delta_t:
                    current_cluster.append(y)
                    to_remove.append(y)
        
        # 从未分配列表中移除已加入簇的节点
        for node in to_remove:
            unassigned_nodes.remove(node)
            
        clusters.append(current_cluster)
    
    # 3. 选取控制器 (Cluster Head)
    controllers = []
    for cluster in clusters:
        # 选取簇内邻居度最高的节点
        # 邻居度定义：在同一个簇内，该节点拥有的邻居数量
        best_node = None
        max_degree = -1
        
        cluster_ids = [n.node_id for n in cluster]
        for node in cluster:
            # 计算该节点在簇内的度
            degree = len([nid for nid in node.neighbors if nid in cluster_ids])
            if degree > max_degree:
                max_degree = degree
                best_node = node
        
        if best_node:
            best_node.is_controller = True
            controllers.append(best_node)
            
    return clusters, controllers

