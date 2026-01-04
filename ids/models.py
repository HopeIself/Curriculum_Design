import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

class SRAIOTEnsemble:
    def __init__(self):
        # KNN: K=5, Euclidean distance
        self.knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
        
        # SVM: Linear kernel. 设置 verbose=True 以观察进度
        # 注意：对于 12万条数据，SVC 的训练会非常缓慢
        self.svm = SVC(kernel='linear', probability=True, verbose=True)
        
        # ANN: 10 neurons hidden layer. 设置 verbose=True 以观察迭代进度
        self.ann = MLPClassifier(hidden_layer_sizes=(10,), activation='logistic', 
                                 solver='adam', max_iter=500, verbose=True)
        
    def train(self, X_train, y_train):
        from tqdm import tqdm
        models = [
            ("KNN", self.knn),
            ("SVM", self.svm),
            ("ANN", self.ann)
        ]
        
        pbar = tqdm(models, desc="整体训练进度")
        for name, model in pbar:
            pbar.set_description(f"正在训练模型: {name}")
            model.fit(X_train, y_train)
        
    def predict(self, X):
        """实现多数投票机制"""
        # KNN 预测
        knn_pred = self.knn.predict(X)
        # SVM 预测
        svm_pred = self.svm.predict(X)
        # ANN 预测
        ann_pred = self.ann.predict(X)
        
        # 将结果堆叠并计算多数票
        preds = np.stack([knn_pred, svm_pred, ann_pred], axis=1)
        ensemble_pred = (np.sum(preds, axis=1) >= 2).astype(int)
        
        return ensemble_pred, (knn_pred, svm_pred, ann_pred)

