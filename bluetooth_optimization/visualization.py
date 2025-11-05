import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_signal_comparison(original, optimized, enhanced, time_points):
    """绘制原始信号、优化信号和增强信号的对比图"""
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(time_points, original)
    plt.title('原始蓝牙信号强度')
    plt.ylabel('信号强度 (dBm)')
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(time_points, optimized)
    plt.title('优化后蓝牙信号强度')
    plt.ylabel('信号强度 (dBm)')
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(time_points, enhanced)
    plt.title('增强后蓝牙信号强度')
    plt.xlabel('时间 (秒)')
    plt.ylabel('信号强度 (dBm)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join('results', '信号对比图.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_environment_features(raw_features):
    """绘制环境特征图"""
    time_points = raw_features[:, 0]
    signal_strength = raw_features[:, 1]
    noise_power = raw_features[:, 2]
    multipath = raw_features[:, 3]
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('时间 (秒)')
    ax1.set_ylabel('信号强度 (dBm)', color=color)
    ax1.plot(time_points, signal_strength, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('噪声功率', color=color)
    ax2.plot(time_points, noise_power, color=color, linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)
    
    ax3 = ax1.twinx()
    color = 'tab:green'
    ax3.spines['right'].set_position(('outward', 60))
    ax3.set_ylabel('多径干扰', color=color)
    ax3.plot(time_points, multipath, color=color, linestyle=':')
    ax3.tick_params(axis='y', labelcolor=color)
    
    plt.title('环境特征随时间变化')
    plt.grid(True, axis='x')
    plt.tight_layout()
    plt.savefig(os.path.join('results', '环境特征图.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_correlation_matrix(corr_matrix):
    """绘制相关性矩阵热力图"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, annot=False)
    plt.title('时间片段特征相关性矩阵')
    plt.tight_layout()
    plt.savefig(os.path.join('results', '相关性矩阵.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_attention_weights(attention_weights):
    """绘制注意力权重分布图"""
    plt.figure(figsize=(12, 6))
    plt.imshow(attention_weights.T, aspect='auto', cmap='viridis')
    plt.colorbar(label='权重值')
    plt.title('环境特征注意力权重分布')
    plt.xlabel('样本索引')
    plt.ylabel('特征维度')
    plt.tight_layout()
    plt.savefig(os.path.join('results', '注意力权重.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_signal_quality_metrics(error_rates, snrs, phase_consistencies):
    """绘制信号质量评估指标"""
    segments = np.arange(len(error_rates))
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(segments, error_rates, 'r-')
    plt.title('误码率')
    plt.ylabel('值')
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(segments, snrs, 'g-')
    plt.title('信噪比 (dB)')
    plt.ylabel('值')
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(segments, phase_consistencies, 'b-')
    plt.title('相位一致性')
    plt.xlabel('信号片段')
    plt.ylabel('值')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join('results', '信号质量指标.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_parameter_adjustments(params, future_params):
    """绘制参数调整对比图"""
    current_gain = params['gain']
    current_bw = params['bandwidth']
    future_gain = future_params['gain']
    future_bw = future_params['bandwidth']

    mod_mapping = {"GFSK": 0, "2-FSK": 1}
    current_mod = mod_mapping[params['modulation']]
    future_mod = mod_mapping[future_params['modulation']]
    
    x = np.arange(3)
    width = 0.35
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, [current_gain, current_bw, current_mod], width, label='当前参数')
    plt.bar(x + width/2, [future_gain, future_bw, future_mod], width, label='预测参数')
    
    plt.xticks(x, ['增益', '带宽', '调制方式 (0=GFSK, 1=2-FSK)'])
    plt.title('接收参数调整对比')
    plt.ylabel('参数值')
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join('results', '参数调整对比.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_pca_components(pca):
    """绘制PCA主成分解释方差比例"""
    plt.figure(figsize=(10, 6))
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    plt.bar(range(1, len(explained_variance)+1), explained_variance, alpha=0.7, label='单个主成分解释方差')
    plt.plot(range(1, len(cumulative_variance)+1), cumulative_variance, 'r-', marker='o', label='累计解释方差')
    plt.axhline(y=0.9, color='g', linestyle='--', label='90%方差阈值')
    
    plt.title('主成分分析解释方差比例')
    plt.xlabel('主成分数量')
    plt.ylabel('解释方差比例')
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join('results', 'PCA方差解释.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_segment_comparison(original_segments, enhanced_segments, segment_idx=0):
    """绘制信号片段增强前后对比"""
    plt.figure(figsize=(12, 6))
    plt.plot(original_segments[segment_idx], 'b-', label='优化后信号片段')
    plt.plot(enhanced_segments[segment_idx], 'r-', label='增强后信号片段')
    plt.title(f'信号片段 {segment_idx} 增强前后对比')
    plt.xlabel('样本点')
    plt.ylabel('信号强度')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join('results', '信号片段增强对比.png'), dpi=300, bbox_inches='tight')
    plt.close()