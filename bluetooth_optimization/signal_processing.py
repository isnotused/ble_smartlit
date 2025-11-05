import numpy as np
import logging
from scipy.fft import fft, ifft
from utils import progress_bar

def dynamic_time_filter(signal, param, window_base=30):
    """动态时域滤波"""
    # 根据参数调整窗口大小
    window_size = int(window_base * (1 + param))
    filtered = np.zeros_like(signal)
    
    for i in range(len(signal)):
        start = max(0, i - window_size // 2)
        end = min(len(signal), i + window_size // 2 + 1)
        filtered[i] = np.mean(signal[start:end])
    
    return filtered

def adaptive_freq_filter(signal, param):
    """自适应频域滤波"""
    # 傅里叶变换
    freq_domain = fft(signal)
    freq = np.fft.fftfreq(len(signal))
    
    # 根据参数调整截止频率
    cutoff = 0.1 + 0.3 * param
    freq_domain[np.abs(freq) > cutoff] = 0
    
    # 逆傅里叶变换
    filtered_signal = np.real(ifft(freq_domain))
    return filtered_signal

def optimize_received_signal(raw_features, strategy, duration=8):
    """对接收信号进行时频联合处理"""
    # 提取原始信号强度
    original_signal = raw_features[:, 1]  
    
    # 时域处理
    time_filtered = dynamic_time_filter(original_signal, strategy["time_param"])
    
    # 频域处理
    freq_filtered = adaptive_freq_filter(time_filtered, strategy["freq_param"])
    
    # 信号重构
    optimized_signal = freq_filtered
    
    # 显示进度
    progress_bar(duration, "接收信号优化")
    
    logging.info(f"信号优化完成，原始信号均值: {np.mean(original_signal):.2f}")
    logging.info(f"优化后信号均值: {np.mean(optimized_signal):.2f}")
    
    return original_signal, optimized_signal