import numpy as np
from sklearn.decomposition import PCA
import logging
from utils import progress_bar

def sliding_window_segmentation(raw_features, window_size=50, step=25):
    """对原始环境特征集进行滑动窗口分割"""
    num_features = raw_features.shape[1]
    windows = []
    
    for i in range(0, len(raw_features) - window_size + 1, step):
        window = raw_features[i:i+window_size, :]
        windows.append(window)
    
    return np.array(windows)

def calculate_correlation_matrix(windows):
    """计算相邻时间片段特征集之间的相关性系数"""
    num_windows = len(windows)
    corr_matrix = np.zeros((num_windows, num_windows))
    
    for i in range(num_windows):
        for j in range(num_windows):
            # 计算窗口间的相关系数
            flat_i = windows[i].flatten()
            flat_j = windows[j].flatten()
            min_len = min(len(flat_i), len(flat_j))
            corr = np.corrcoef(flat_i[:min_len], flat_j[:min_len])[0, 1]
            corr_matrix[i, j] = corr
    
    return corr_matrix

def build_dynamic_feature_matrix(raw_features, duration=8):
    """构建动态环境特征矩阵"""
    # 滑动窗口分割
    windows = sliding_window_segmentation(raw_features)
    logging.info(f"滑动窗口分割完成，窗口数量: {len(windows)}")
    
    # 计算相关性系数
    corr_matrix = calculate_correlation_matrix(windows)
    
    # 主成分分析降维
    pca = PCA(n_components=10)  
    dynamic_matrix = pca.fit_transform(corr_matrix)
    
    # 显示进度
    progress_bar(duration, "动态环境特征矩阵构建")
    
    logging.info(f"动态特征矩阵构建完成，维度: {dynamic_matrix.shape}")
    logging.info(f"主成分解释方差比例: {np.sum(pca.explained_variance_ratio_):.2%}")
    
    return dynamic_matrix, windows, corr_matrix, pca