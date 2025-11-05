import numpy as np
from sklearn.decomposition import PCA
from typing import Dict, List
from config import SystemConfig

class EnvironmentAnalyzer:
    """环境特征时空关联分析"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.window_size = config.WINDOW_SIZE
        self.slide_step = config.SLIDE_STEP
        
    def build_dynamic_feature_matrix(self, raw_features: Dict[str, np.ndarray]) -> np.ndarray:
        """
        构建动态环境特征矩阵
        Args:
            raw_features: 原始环境特征集
        Returns:
            动态环境特征矩阵
        """
        # 步骤2.1: 滑动窗口分割
        time_segments = self._segment_time_windows(raw_features)
        
        # 步骤2.2: 计算时空关联矩阵
        correlation_matrix = self._compute_correlation_matrix(time_segments)
        
        # 步骤2.3: 主成分分析降维
        feature_matrix = self._apply_pca(correlation_matrix)
        
        return feature_matrix
    
    def _segment_time_windows(self, raw_features: Dict[str, np.ndarray]) -> List[Dict]:
        """滑动窗口分割形成时间片段特征集"""
        signal_data = raw_features['signal_strength']
        total_samples = signal_data.shape[1]
        
        time_segments = []
        start_idx = 0
        
        while start_idx + self.window_size <= total_samples:
            segment = {}
            
            # 提取当前窗口的数据
            for key, data in raw_features.items():
                if key == 'signal_strength':
                    segment[key] = data[:, start_idx:start_idx + self.window_size]
                elif key == 'noise_power':
                    segment[key] = data  
                elif key == 'multipath_interference':
                    segment[key] = data  
            
            time_segments.append(segment)
            start_idx += self.slide_step
        
        return time_segments
    
    def _compute_correlation_matrix(self, time_segments: List[Dict]) -> np.ndarray:
        """计算相邻时间片段的相关性系数，构建时空关联矩阵"""
        segment_count = len(time_segments)
        
        # 提取第一个片段的特征以确定维度
        first_features = self._extract_features(time_segments[0])
        feature_dim = first_features.shape[1]  # 使用实际提取的特征维度
        
        correlation_matrix = np.zeros((segment_count - 1, feature_dim))
        
        for i in range(segment_count - 1):
            current_features = self._extract_features(time_segments[i])
            next_features = self._extract_features(time_segments[i + 1])
            
            # 确保特征维度一致
            actual_dim = min(feature_dim, current_features.shape[1], next_features.shape[1])
            
            # 计算特征相关性
            for j in range(actual_dim):
                curr_feat = current_features[0, j]
                next_feat = next_features[0, j]
                # 简单的相关性：归一化差值
                corr = 1.0 / (1.0 + abs(curr_feat - next_feat))
                correlation_matrix[i, j] = corr
        
        return correlation_matrix
    
    def _calculate_feature_dimension(self, segment: Dict) -> int:
        """计算特征维度"""
        dim = 0
        if 'signal_strength' in segment:
            dim += segment['signal_strength'].size
        if 'noise_power' in segment:
            dim += segment['noise_power'].size
        if 'multipath_interference' in segment:
            dim += segment['multipath_interference'].size
        return dim
    
    def _extract_features(self, segment: Dict) -> np.ndarray:
        """从时间片段中提取特征向量"""
        features = []
        
        if 'signal_strength' in segment:
            signal_features = self._extract_signal_features(segment['signal_strength'])
            features.extend(signal_features.flatten())
        
        if 'noise_power' in segment:
            features.extend(segment['noise_power'].flatten())
            
        if 'multipath_interference' in segment:
            features.extend(segment['multipath_interference'].flatten())
        
        return np.array(features).reshape(1, -1)
    
    def _extract_signal_features(self, signal_data: np.ndarray) -> np.ndarray:
        """提取信号特征"""
        features = []
        
        for channel in range(signal_data.shape[0]):
            channel_data = signal_data[channel]
            
            # 计算统计特征
            power = np.mean(np.abs(channel_data)**2)
            variance = np.var(channel_data)
            spectral_centroid = self._compute_spectral_centroid(channel_data)
            
            features.extend([power, variance, spectral_centroid])
        
        return np.array(features)
    
    def _compute_spectral_centroid(self, signal: np.ndarray) -> float:
        """计算频谱质心"""
        spectrum = np.fft.fft(signal)
        magnitude = np.abs(spectrum[:len(spectrum)//2])
        frequencies = np.fft.fftfreq(len(signal))[:len(signal)//2]
        
        if np.sum(magnitude) == 0:
            return 0.0
            
        centroid = np.sum(frequencies * magnitude) / np.sum(magnitude)
        return centroid
    
    def _apply_pca(self, correlation_matrix: np.ndarray, n_components: int = 10) -> np.ndarray:
        """应用主成分分析进行降维"""
        if correlation_matrix.shape[0] < 2:
            # 数据量太少，无法进行PCA，直接返回
            return correlation_matrix
        
        # 确保n_components不超过数据维度
        max_components = min(n_components, correlation_matrix.shape[0], correlation_matrix.shape[1])
        
        if max_components < 1:
            return correlation_matrix
        
        try:
            pca = PCA(n_components=max_components)
            reduced_features = pca.fit_transform(correlation_matrix)
            return reduced_features
        except Exception as e:
            print(f"PCA降维失败: {e}，返回原始特征")
            return correlation_matrix