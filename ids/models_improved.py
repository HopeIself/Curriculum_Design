import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    where p_t is the model's estimated probability for the true class.
    For binary classification: alpha_t = alpha if t=1, else 1-alpha
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        # 计算交叉熵（输入是logits，不需要softmax）
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        # 获取预测概率 p_t
        pt = torch.exp(-ce_loss)
        
        # 对于二分类，alpha_t = alpha if target=1, else 1-alpha
        # 由于我们使用的是多类交叉熵（2类），我们需要根据target选择alpha
        alpha_t = torch.where(targets == 1, 
                              torch.tensor(self.alpha, device=inputs.device),
                              torch.tensor(1.0 - self.alpha, device=inputs.device))
        
        # 计算focal loss
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class FocalLossANN(nn.Module):
    """
    使用Focal Loss的ANN分类器
    """
    def __init__(self, input_dim, hidden_dim=10, num_classes=2):
        super(FocalLossANN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.Sigmoid()  # logistic activation
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class TensorDataset(Dataset):
    """将numpy数组转换为PyTorch Dataset"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class FocalLossANNClassifier:
    """
    Focal Loss ANN的sklearn兼容包装器
    """
    def __init__(self, hidden_dim=10, alpha=0.25, gamma=2.0, 
                 max_iter=500, batch_size=256, lr=0.001, verbose=True):
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.gamma = gamma
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.lr = lr
        self.verbose = verbose
        self.model = None
        self.input_dim = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y).ravel()
        
        self.input_dim = X.shape[1]
        self.model = FocalLossANN(self.input_dim, self.hidden_dim, num_classes=2)
        self.model.to(self.device)
        
        criterion = FocalLoss(alpha=self.alpha, gamma=self.gamma)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model.train()
        for epoch in tqdm(range(self.max_iter), desc="Focal Loss ANN训练", disable=not self.verbose):
            epoch_loss = 0
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            if self.verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{self.max_iter}, Loss: {epoch_loss / len(dataloader):.4f}")
        
    def predict(self, X):
        X = np.array(X)
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
            return predicted.cpu().numpy()
    
    def predict_proba(self, X):
        X = np.array(X)
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            probs = nn.functional.softmax(outputs, dim=1)
            return probs.cpu().numpy()


class SRAIOTEnsembleImproved:
    """
    改进版SRAIOT集成模型
    支持Focal Loss ANN和AdaBoost
    """
    def __init__(self, use_focal_loss=True, use_adaboost=True, focal_alpha=0.25, focal_gamma=2.0):
        """
        参数:
        use_focal_loss: 是否使用Focal Loss ANN替代标准ANN
        use_adaboost: 是否添加AdaBoost分类器
        focal_alpha: Focal Loss的alpha参数（类别权重）
        focal_gamma: Focal Loss的gamma参数（聚焦参数，越大越关注难分类样本）
        """
        self.use_focal_loss = use_focal_loss
        self.use_adaboost = use_adaboost
        
        # KNN: K=5, Euclidean distance
        self.knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
        
        # SVM: Linear kernel
        self.svm = SVC(kernel='linear', probability=True, verbose=False)
        
        # ANN: 根据配置选择标准ANN或Focal Loss ANN
        if use_focal_loss:
            self.ann = FocalLossANNClassifier(hidden_dim=10, alpha=focal_alpha, 
                                             gamma=focal_gamma, max_iter=500, verbose=False)
        else:
            from sklearn.neural_network import MLPClassifier
            self.ann = MLPClassifier(hidden_layer_sizes=(10,), activation='logistic', 
                                    solver='adam', max_iter=500, verbose=False)
        
        # AdaBoost: 使用决策树作为基分类器
        if use_adaboost:
            base_estimator = DecisionTreeClassifier(max_depth=3)
            self.adaboost = AdaBoostClassifier(
                estimator=base_estimator,
                n_estimators=50,
                learning_rate=1.0,
                algorithm='SAMME.R'
            )
        else:
            self.adaboost = None
        
    def train(self, X_train, y_train):
        """训练所有模型"""
        from tqdm import tqdm
        
        models = [
            ("KNN", self.knn),
            ("SVM", self.svm),
            ("ANN", self.ann)
        ]
        
        if self.use_adaboost:
            models.append(("AdaBoost", self.adaboost))
        
        pbar = tqdm(models, desc="整体训练进度")
        for name, model in pbar:
            pbar.set_description(f"正在训练模型: {name}")
            model.fit(X_train, y_train)
        
    def predict(self, X):
        """
        实现多数投票机制
        如果使用AdaBoost，则使用4个模型的投票；否则使用3个模型的投票
        """
        # KNN 预测
        knn_pred = self.knn.predict(X)
        # SVM 预测
        svm_pred = self.svm.predict(X)
        # ANN 预测
        ann_pred = self.ann.predict(X)
        
        preds = [knn_pred, svm_pred, ann_pred]
        
        # AdaBoost 预测（如果启用）
        if self.use_adaboost:
            adaboost_pred = self.adaboost.predict(X)
            preds.append(adaboost_pred)
        
        # 将结果堆叠并计算多数票
        preds_stack = np.stack(preds, axis=1)
        # 至少有一半的模型判定为攻击时，判定为攻击
        threshold = len(preds) // 2 + 1
        ensemble_pred = (np.sum(preds_stack, axis=1) >= threshold).astype(int)
        
        if self.use_adaboost:
            return ensemble_pred, (knn_pred, svm_pred, ann_pred, adaboost_pred)
        else:
            return ensemble_pred, (knn_pred, svm_pred, ann_pred)

