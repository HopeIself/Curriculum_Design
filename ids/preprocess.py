import pandas as pd
import numpy as np
import os
import requests
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# NSL-KDD 字段名称
COL_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login",
    "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty_level"
]

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "archive")

def load_combined_data():
    """加载并合并训练集与测试集，统一预处理"""
    train_path = os.path.join(DATA_DIR, "KDDTrain+.txt")
    test_path = os.path.join(DATA_DIR, "KDDTest+.txt")
    
    df_train = pd.read_csv(train_path, names=COL_NAMES, header=None)
    df_test = pd.read_csv(test_path, names=COL_NAMES, header=None)
    df = pd.concat([df_train, df_test], axis=0)
    
    # 丢弃不需要的 difficulty_level
    df.drop('difficulty_level', axis=1, inplace=True)
    
    # 将标签映射为二分类
    df['label'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)
    
    # 对类别特征进行编码
    cat_cols = ['protocol_type', 'service', 'flag']
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
        
    # 数值特征归一化
    num_cols = df.columns.drop('label')
    scaler = MinMaxScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    
    return df, scaler, encoders

if __name__ == "__main__":
    train_df = load_and_preprocess_data(is_train=True)
    print(f"训练集形状: {train_df.shape}")
    print(train_df['label'].value_counts())
    
    test_df = load_and_preprocess_data(is_train=False)
    print(f"测试集形状: {test_df.shape}")
    print(test_df['label'].value_counts())

