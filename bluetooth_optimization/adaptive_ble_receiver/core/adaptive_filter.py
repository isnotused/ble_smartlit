import numpy as np
from scipy import signal
from typing import Dict, Tuple
from config import SystemConfig

class AdaptiveFilter:
    """自适应滤波算法实现"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        
    def apply_filter_strategy(self, received_signal: np.ndarray, strategy_id: int) -> np.ndarray:
        """
        应用滤波策略对接收信号进行处理
        Args:
            received_signal: 接收信号
            strategy_id: 滤波策略标识
        Returns:
            初步优化信号
        """
        strategy_config = self.config.FILTER_STRATEGIES.get(strategy_id)
        if not strategy_config:
            raise ValueError(f"未知的滤波策略标识: {strategy_id}")
        
        # 步骤4.1: 解析滤波参数
        time_params, freq_params = self._parse_filter_parameters(strategy_config)
        
        # 步骤4.2: 动态时域滤波
        time_filtered = self._apply_time_domain_filter(received_signal, time_params)
        
        # 步骤4.3: 自适应频域滤波
        freq_filtered = self._apply_frequency_domain_filter(time_filtered, freq_params)
        
        # 步骤4.4: 信号重构
        optimized_signal = self._reconstruct_signal(freq_filtered)
        
        return optimized_signal
    
    def _parse_filter_parameters(self, strategy_config: Dict) -> Tuple[Dict, Dict]:
        """解析时域和频域处理参数"""
        filter_type = strategy_config['type']
        params = strategy_config['params']
        
        time_params = {}
        freq_params = {}
        
        if filter_type == 'kalman':
            time_params = {'q': params['q'], 'r': params['r']}
            freq_params = {'smoothing_factor': 0.1}
        elif filter_type == 'wiener':
            time_params = {'window_size': params['window_size']}
            freq_params = {'noise_estimate': 0.01}
        elif filter_type == 'adaptive_lms':
            time_params = {'mu': params['mu'], 'order': params['order']}
            freq_params = {'adaptive_rate': 0.05}
        elif filter_type == 'butterworth':
            time_params = {'order': params['order'], 'cutoff': params['cutoff']}
            freq_params = {'stopband_attenuation': 40}
        
        return time_params, freq_params
    
    def _apply_time_domain_filter(self, signal_data: np.ndarray, params: Dict) -> np.ndarray:
        """采用可变长度滑动窗口进行动态时域滤波"""
        window_size = params.get('window_size', 32)
        filtered_signal = np.zeros_like(signal_data)
        
        for i in range(len(signal_data)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(signal_data), i + window_size // 2)
            
            # 当前窗口
            window = signal_data[start_idx:end_idx]
            
            # 应用时域滤波
            if 'q' in params and 'r' in params:
                # Kalman滤波
                filtered_signal[i] = self._kalman_filter(window, params['q'], params['r'])
            elif 'mu' in params and 'order' in params:
                # LMS自适应滤波
                filtered_signal[i] = self._lms_filter(window, params['mu'], params['order'])
            else:
                # 移动平均滤波
                filtered_signal[i] = np.mean(window)
        
        return filtered_signal
    
    def _apply_frequency_domain_filter(self, signal_data: np.ndarray, params: Dict) -> np.ndarray:
        """对时域处理后的信号进行自适应频域变换，实施动态频域滤波"""
        # 执行FFT
        spectrum = np.fft.fft(signal_data)
        frequencies = np.fft.fftfreq(len(signal_data))
        
        # 设计频域滤波器
        filter_response = self._design_frequency_filter(frequencies, params)
        
        # 应用频域滤波
        filtered_spectrum = spectrum * filter_response
        
        return filtered_spectrum
    
    def _reconstruct_signal(self, filtered_spectrum: np.ndarray) -> np.ndarray:
        """将时频处理结果进行信号重构"""
        # 执行逆FFT
        reconstructed_signal = np.fft.ifft(filtered_spectrum)
        
        # 确保输出为实数信号（如果输入是实数）
        if np.all(np.isreal(reconstructed_signal)):
            reconstructed_signal = np.real(reconstructed_signal)
        
        return reconstructed_signal
    
    def _kalman_filter(self, data: np.ndarray, q: float, r: float) -> float:
        """Kalman滤波实现"""
        x = data[0]  # 初始状态
        p = 1.0      # 初始协方差
        
        for measurement in data[1:]:
            # 预测步骤
            x_pred = x
            p_pred = p + q
            
            # 更新步骤
            k = p_pred / (p_pred + r)
            x = x_pred + k * (measurement - x_pred)
            p = (1 - k) * p_pred
        
        return x
    
    def _lms_filter(self, data: np.ndarray, mu: float, order: int) -> float:
        """LMS自适应滤波实现"""
        if len(data) <= order:
            return np.mean(data)
            
        # 初始化权重
        w = np.zeros(order)
        y = 0.0  # 初始化y
        
        # LMS迭代
        for i in range(order, len(data)):
            x = data[i-order:i]
            y = np.dot(w, x)
            e = data[i] - y
            w = w + mu * e * x
        
        # 返回最后一个输出
        return y
    
    def _design_frequency_filter(self, frequencies: np.ndarray, params: Dict) -> np.ndarray:
        """设计频域滤波器"""
        if 'cutoff' in params:
            # 低通滤波器
            cutoff = params['cutoff']
            filter_response = np.where(np.abs(frequencies) < cutoff, 1.0, 0.0)
        else:
            # 默认全通滤波器
            filter_response = np.ones_like(frequencies)
        
        return filter_response