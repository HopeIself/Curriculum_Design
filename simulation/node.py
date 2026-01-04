import numpy as np

class IoTNode:
    def __init__(self, node_id, area_size=(500, 500), radio_range=100):
        self.node_id = node_id
        self.area_size = area_size
        self.x = np.random.uniform(0, area_size[0])
        self.y = np.random.uniform(0, area_size[1])
        
        self.v = np.random.uniform(0, 10)
        self.phi = np.random.uniform(0, 2 * np.pi)
        
        self.energy = np.random.uniform(0.5, 1.0)
        self.radio_range = radio_range
        
        self.is_alive = True
        self.is_controller = False
        self.neighbors = []
        self.rssi_table = {}
        
        # 路由与负载相关 (阶段三)
        self.t_service = np.random.uniform(0.01, 0.05) # 模拟服务时间
        self.t_arrival = np.random.uniform(0.05, 0.1)  # 模拟到达时间
        self.parent = None  # 层次树中的父节点
        self.children = []  # 层次树中的子节点

    def get_congestion(self):
        """公式 (3): Ci = T_service / T_arrival"""
        return self.t_service / self.t_arrival

    def move(self, dt, area_size=None):
        if not self.is_alive: return
        target_area = area_size if area_size else self.area_size
        self.x += self.v * np.cos(self.phi) * dt
        self.y += self.v * np.sin(self.phi) * dt
        if self.x < 0: self.x = 0; self.phi = np.pi - self.phi
        if self.x > target_area[0]: self.x = target_area[0]; self.phi = np.pi - self.phi
        if self.y < 0: self.y = 0; self.phi = -self.phi
        if self.y > target_area[1]: self.y = target_area[1]; self.phi = -self.phi

    def get_distance(self, other_node):
        return np.sqrt((self.x - other_node.x)**2 + (self.y - other_node.y)**2)

    def estimate_rssi(self, other_node):
        d = self.get_distance(other_node)
        return -20 * np.log10(d) if d > 0 else 0

    def consume_energy(self, amount):
        if self.is_alive:
            self.energy -= amount
            if self.energy <= 0:
                self.energy = 0
                self.is_alive = False

def calculate_stability(node_i, node_j, delta_t):
    d = node_i.get_distance(node_j)
    r = node_i.radio_range
    v_ix, v_iy = node_i.v * np.cos(node_i.phi), node_i.v * np.sin(node_i.phi)
    v_jx, v_jy = node_j.v * np.cos(node_j.phi), node_j.v * np.sin(node_j.phi)
    v_ij_x, v_ij_y = v_ix - v_jx, v_iy - v_jy
    v_ij = np.sqrt(v_ij_x**2 + v_ij_y**2)
    if v_ij == 0: return float('inf') if d <= r else 0
    dx, dy = node_i.x - node_j.x, node_i.y - node_j.y
    cos_theta = -(dx * v_ij_x + dy * v_ij_y) / (d * v_ij) if d > 0 else 1
    sin_theta_sq = 1 - cos_theta**2
    discriminant = r**2 - (d**2 * sin_theta_sq)
    if discriminant < 0: return 0
    return (d * cos_theta + np.sqrt(discriminant)) / v_ij
