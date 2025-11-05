import numpy as np
from typing import Dict, Tuple
from config import SystemConfig

class QualityAssessor:
    """信号质量评估器"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        
    def assess_signal_quality(self, enhanced_signal: np.ndarray, 
                            reference_signal: np.ndarray = None) -> Dict[str, np.ndarray]:
        """
        对增强目标信号进行多维质量评估
        Args:
            enhanced_signal: 增强目标信号
            reference_signal: 参考信号
        Returns:
            信号质量评估矩阵
        """
        # 步骤6.1: 计算质量指标
        quality_metrics = self._compute_quality_metrics(enhanced_signal, reference_signal)
        
        # 步骤6.2: 构建三维评估矩阵
        assessment_matrix = self._build_assessment_matrix(quality_metrics)
        
        # 步骤6.3: 滑动加权平均
        final_matrix = self._apply_sliding_weighted_average(assessment_matrix)
        
        return {
            'quality_matrix': final_matrix,
            'metrics': quality_metrics
        }
    
    def _compute_quality_metrics(self, signal: np.ndarray, reference: np.ndarray = None) -> Dict[str, float]:
        """计算信号质量指标"""
        metrics = {}
        
        # 计算信噪比
        metrics['snr'] = self._calculate_snr(signal, reference)
        
        # 计算误码率（模拟）
        metrics['ber'] = self._estimate_ber(signal, reference)
        
        # 计算相位一致性
        metrics['phase_consistency'] = self._calculate_phase_consistency(signal)
        
        # 计算频谱平坦度
        metrics['spectral_flatness'] = self._calculate_spectral_flatness(signal)
        
        # 计算信号幅度稳定性
        metrics['amplitude_stability'] = self._calculate_amplitude_stability(signal)
        
        return metrics
    
    def _build_assessment_matrix(self, quality_metrics: Dict[str, float]) -> np.ndarray:
        """构建包含时间维度和指标维度的三维评估矩阵"""
        # 创建基础评估矩阵
        time_points = 10  # 时间维度
        metric_count = len(quality_metrics)
        
        # 初始化三维矩阵 [时间, 指标, 评估值]
        assessment_matrix = np.zeros((time_points, metric_count, 3))
        
        # 填充评估值
        metrics_list = list(quality_metrics.keys())
        for t in range(time_points):
            for m, metric_name in enumerate(metrics_list):
                metric_value = quality_metrics[metric_name]
                
                # 添加时间变化（模拟）
                time_variation = np.random.normal(0, 0.1 * metric_value)
                current_value = metric_value + time_variation
                
                # 计算评估分数
                score = self._metric_to_score(metric_name, current_value)
                confidence = self._calculate_confidence(current_value)
                
                assessment_matrix[t, m, 0] = current_value  # 原始值
                assessment_matrix[t, m, 1] = score          # 评估分数
                assessment_matrix[t, m, 2] = confidence     # 置信度
        
        return assessment_matrix
    
    def _apply_sliding_weighted_average(self, matrix: np.ndarray, window_size: int = 3) -> np.ndarray:
        """通过滑动加权平均算法生成最终评估矩阵"""
        time_points, metrics, dimensions = matrix.shape
        smoothed_matrix = np.zeros_like(matrix)
        
        # 定义权重窗口（高斯权重）
        weights = np.exp(-0.5 * np.arange(-window_size//2, window_size//2 + 1)**2)
        weights /= np.sum(weights)
        
        for t in range(time_points):
            for m in range(metrics):
                for d in range(dimensions):
                    # 滑动窗口
                    start_idx = max(0, t - window_size // 2)
                    end_idx = min(time_points, t + window_size // 2 + 1)
                    
                    window_values = matrix[start_idx:end_idx, m, d]
                    window_weights = weights[:len(window_values)]
                    
                    # 加权平均
                    smoothed_value = np.average(window_values, weights=window_weights)
                    smoothed_matrix[t, m, d] = smoothed_value
        
        return smoothed_matrix
    
    def _calculate_snr(self, signal: np.ndarray, reference: np.ndarray = None) -> float:
        """计算信噪比"""
        if reference is not None:
            # 有参考信号的情况
            noise = signal - reference
            signal_power = np.mean(reference**2)
            noise_power = np.mean(noise**2)
        else:
            # 无参考信号的情况（估计）
            signal_power = np.mean(signal**2)
            # 使用高通滤波估计噪声
            from scipy import signal as sp_signal
            b, a = sp_signal.butter(4, 0.1, 'high')
            noise = sp_signal.filtfilt(b, a, signal)
            noise_power = np.mean(noise**2)
        
        if noise_power == 0:
            return float('inf')
        
        snr = 10 * np.log10(signal_power / noise_power)
        return snr
    
    def _estimate_ber(self, received_signal: np.ndarray, reference: np.ndarray = None) -> float:
        """估计误码率"""
        if reference is not None:
            # 有参考信号的情况
            decision_threshold = 0.0
            received_bits = (received_signal > decision_threshold).astype(int)
            reference_bits = (reference > decision_threshold).astype(int)
            
            error_count = np.sum(received_bits != reference_bits)
            total_bits = len(received_bits)
            
            ber = error_count / total_bits if total_bits > 0 else 0.0
        else:
            # 无参考信号的情况（基于信噪比估计）
            snr = self._calculate_snr(received_signal)
            ber = 0.5 * np.exp(-0.5 * snr)  # AWGN信道下的理论BER
            
        return ber
    
    def _calculate_phase_consistency(self, signal: np.ndarray) -> float:
        """计算相位一致性"""
        analytic_signal = self._compute_analytic_signal(signal)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        phase_diff = np.diff(instantaneous_phase)
        
        # 相位变化的一致性
        phase_consistency = 1.0 / (1.0 + np.std(phase_diff))
        return phase_consistency
    
    def _calculate_spectral_flatness(self, signal: np.ndarray) -> float:
        """计算频谱平坦度"""
        spectrum = np.abs(np.fft.fft(signal))**2
        geometric_mean = np.exp(np.mean(np.log(spectrum + 1e-10)))
        arithmetic_mean = np.mean(spectrum)
        
        flatness = geometric_mean / arithmetic_mean
        return flatness
    
    def _calculate_amplitude_stability(self, signal: np.ndarray) -> float:
        """计算幅度稳定性"""
        amplitude = np.abs(signal)
        stability = 1.0 / (1.0 + np.std(amplitude) / np.mean(amplitude))
        return stability
    
    def _compute_analytic_signal(self, signal: np.ndarray) -> np.ndarray:
        """计算解析信号"""
        from scipy import signal as sp_signal
        return sp_signal.hilbert(signal)
    
    def _metric_to_score(self, metric_name: str, value: float) -> float:
        """将指标值转换为评估分数"""
        if metric_name == 'snr':
            return min(value / 30.0, 1.0)  # 30dB为满分
        elif metric_name == 'ber':
            return max(0.0, 1.0 - value * 1000)  # BER越低分数越高
        elif metric_name == 'phase_consistency':
            return value  # 0-1范围
        elif metric_name == 'spectral_flatness':
            return value  # 0-1范围
        elif metric_name == 'amplitude_stability':
            return value  # 0-1范围
        else:
            return 0.0
    
    def _calculate_confidence(self, value: float) -> float:
        """计算评估置信度"""
        # 基于值的稳定性计算置信度
        return 1.0 / (1.0 + np.abs(value - 0.5) * 2.0)  # 中间值置信度较低