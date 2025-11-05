import numpy as np
import logging
from utils import progress_bar

def calculate_error_rate(original, enhanced):
    """计算误码率"""
    return np.mean(np.abs(enhanced - original)) / (np.max(original) - np.min(original))

def calculate_snr(signal, original):
    """计算信噪比"""
    noise = signal - original
    signal_power = np.mean(original ** 2)
    noise_power = np.mean(noise ** 2)
    return 10 * np.log10(signal_power / noise_power)

def calculate_phase_consistency(original, enhanced):
    """计算相位一致性"""
    original_fft = np.fft.fft(original)
    enhanced_fft = np.fft.fft(enhanced)
    
    original_phase = np.angle(original_fft)
    enhanced_phase = np.angle(enhanced_fft)

    phase_diff = np.abs(original_phase - enhanced_phase)
    return np.mean(np.cos(phase_diff))

def evaluate_signal_quality(original_signal, optimized_signal, enhanced_signal, duration=8):
    """评估信号质量并生成评估矩阵"""
    segment_size = 50
    num_segments = len(original_signal) // segment_size
    
    # 计算各项指标
    error_rates = []
    snrs = []
    phase_consistencies = []
    
    for i in range(num_segments):
        start = i * segment_size
        end = (i + 1) * segment_size
        
        orig_seg = original_signal[start:end]
        opt_seg = optimized_signal[start:end]
        enh_seg = enhanced_signal[start:end]
        
        # 计算优化后的指标
        error_rates.append(calculate_error_rate(orig_seg, enh_seg))
        snrs.append(calculate_snr(enh_seg, orig_seg))
        phase_consistencies.append(calculate_phase_consistency(orig_seg, enh_seg))
    
    # 构建评估矩阵并进行滑动加权平均
    eval_matrix = np.column_stack((error_rates, snrs, phase_consistencies))
    
    # 滑动加权平均
    window = 3
    weights = np.arange(1, window + 1) / np.sum(np.arange(1, window + 1))
    smoothed_matrix = np.zeros_like(eval_matrix)
    
    for i in range(len(eval_matrix)):
        start = max(0, i - window + 1)
        end = i + 1
        window_data = eval_matrix[start:end]

        if len(window_data) < window:
            w = weights[-len(window_data):] / np.sum(weights[-len(window_data):])
        else:
            w = weights
        
        smoothed_matrix[i] = np.sum(window_data * w[:, np.newaxis], axis=0)
    
    # 显示进度
    progress_bar(duration, "信号质量评估")
    
    logging.info(f"信号质量评估完成，平均误码率: {np.mean(error_rates):.4f}")
    logging.info(f"平均信噪比: {np.mean(snrs):.2f} dB")
    logging.info(f"平均相位一致性: {np.mean(phase_consistencies):.4f}")
    
    return smoothed_matrix, error_rates, snrs, phase_consistencies