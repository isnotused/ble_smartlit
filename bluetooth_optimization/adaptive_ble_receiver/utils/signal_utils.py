import numpy as np
from scipy import signal as sp_signal
from typing import Union, Tuple

class SignalUtils:
    """信号处理工具函数"""
    
    @staticmethod
    def compute_spectrogram(signal_data: np.ndarray, 
                          fs: float = 1.0,
                          nperseg: int = 256,
                          noverlap: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        计算信号的频谱图
        Args:
            signal_data: 输入信号
            fs: 采样率
            nperseg: 每段长度
            noverlap: 重叠长度
        Returns:
            f: 频率数组
            t: 时间数组
            Sxx: 频谱图
        """
        if noverlap is None:
            noverlap = nperseg // 2
            
        f, t, Sxx = sp_signal.spectrogram(signal_data, fs=fs, 
                                         nperseg=nperseg, noverlap=noverlap)
        return f, t, Sxx
    
    @staticmethod
    def apply_bandpass_filter(signal_data: np.ndarray, 
                            lowcut: float, 
                            highcut: float, 
                            fs: float, 
                            order: int = 4) -> np.ndarray:
        """
        应用带通滤波器
        Args:
            signal_data: 输入信号
            lowcut: 低频截止频率
            highcut: 高频截止频率
            fs: 采样率
            order: 滤波器阶数
        Returns:
            滤波后的信号
        """
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        
        b, a = sp_signal.butter(order, [low, high], btype='band')
        filtered_signal = sp_signal.filtfilt(b, a, signal_data)
        
        return filtered_signal
    
    @staticmethod
    def estimate_power_spectral_density(signal_data: np.ndarray, 
                                      fs: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        估计功率谱密度
        Args:
            signal_data: 输入信号
            fs: 采样率
        Returns:
            f: 频率数组
            Pxx: 功率谱密度
        """
        f, Pxx = sp_signal.welch(signal_data, fs=fs, nperseg=1024)
        return f, Pxx
    
    @staticmethod
    def compute_correlation(signal1: np.ndarray, 
                          signal2: np.ndarray, 
                          max_lag: int = None) -> np.ndarray:
        """
        计算信号互相关
        Args:
            signal1: 第一个信号
            signal2: 第二个信号
            max_lag: 最大延迟
        Returns:
            互相关序列
        """
        if max_lag is None:
            max_lag = len(signal1) // 2
            
        correlation = np.correlate(signal1, signal2, mode='full')
        center = len(correlation) // 2
        return correlation[center - max_lag:center + max_lag + 1]
    
    @staticmethod
    def normalize_signal(signal_data: np.ndarray, 
                        method: str = 'zero_mean') -> np.ndarray:
        """
        信号归一化
        Args:
            signal_data: 输入信号
            method: 归一化方法
        Returns:
            归一化后的信号
        """
        if method == 'zero_mean':
            # 零均值归一化
            normalized = signal_data - np.mean(signal_data)
            if np.std(normalized) > 0:
                normalized = normalized / np.std(normalized)
            return normalized
            
        elif method == 'minmax':
            # 最小最大归一化
            signal_min = np.min(signal_data)
            signal_max = np.max(signal_data)
            if signal_max - signal_min > 0:
                return (signal_data - signal_min) / (signal_max - signal_min)
            else:
                return signal_data - signal_min
                
        elif method == 'unit_power':
            # 单位功率归一化
            power = np.mean(signal_data**2)
            if power > 0:
                return signal_data / np.sqrt(power)
            else:
                return signal_data
        else:
            raise ValueError(f"不支持的归一化方法: {method}")
    
    @staticmethod
    def detect_peaks(signal_data: np.ndarray, 
                    height: float = None, 
                    distance: int = None,
                    prominence: float = None) -> np.ndarray:
        """
        检测信号峰值
        Args:
            signal_data: 输入信号
            height: 峰值高度阈值
            distance: 峰值最小距离
            prominence: 峰值显著性
        Returns:
            峰值位置数组
        """
        peaks, _ = sp_signal.find_peaks(signal_data, height=height, 
                                       distance=distance, prominence=prominence)
        return peaks
    
    @staticmethod
    def compute_signal_envelope(signal_data: np.ndarray, 
                              method: str = 'hilbert') -> np.ndarray:
        """
        计算信号包络
        Args:
            signal_data: 输入信号
            method: 包络检测方法
        Returns:
            信号包络
        """
        if method == 'hilbert':
            # 希尔伯特变换法
            analytic_signal = sp_signal.hilbert(signal_data)
            envelope = np.abs(analytic_signal)
            return envelope
            
        elif method == 'moving_avg':
            # 移动平均法
            window_size = min(50, len(signal_data) // 10)
            envelope = np.convolve(np.abs(signal_data), 
                                 np.ones(window_size)/window_size, 
                                 mode='same')
            return envelope
            
        else:
            raise ValueError(f"不支持的包络检测方法: {method}")
    
    @staticmethod
    def estimate_signal_bandwidth(psd: np.ndarray, 
                                frequencies: np.ndarray, 
                                threshold: float = 0.5) -> float:
        """
        估计信号带宽
        Args:
            psd: 功率谱密度
            frequencies: 频率数组
            threshold: 带宽阈值（相对于峰值）
        Returns:
            估计的带宽
        """
        peak_power = np.max(psd)
        threshold_power = peak_power * threshold
        
        # 找到超过阈值的频率范围
        above_threshold = psd >= threshold_power
        if np.any(above_threshold):
            min_freq = frequencies[above_threshold][0]
            max_freq = frequencies[above_threshold][-1]
            bandwidth = max_freq - min_freq
        else:
            bandwidth = 0.0
            
        return bandwidth