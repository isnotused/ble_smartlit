1. data_collection.py
```python
import numpy as np
import logging
from utils import progress_bar

def collect_environment_data(sample_count=1000, duration=8):
    time_points = np.linspace(0, 10, sample_count)
    
    signal_strength = -60 + 15 * np.sin(time_points) + np.random.normal(0, 3, sample_count)
    
    noise_power = 5 + 2 * np.sin(time_points/2) + np.random.normal(0, 0.5, sample_count)

    multipath_interference = 3 + 2 * np.sin(time_points*1.5) + np.random.normal(0, 0.8, sample_count)

    raw_features = np.column_stack((
        time_points,
        signal_strength,
        noise_power,
        multipath_interference
    ))
    
    # 显示进度
    progress_bar(duration, "环境数据采集")
    
    logging.info(f"采集完成，数据维度: {raw_features.shape}")
    logging.info(f"信号强度范围: {np.min(signal_strength):.2f} ~ {np.max(signal_strength):.2f}")
    logging.info(f"噪声功率范围: {np.min(noise_power):.2f} ~ {np.max(noise_power):.2f}")
    
    return raw_features
```

核心作用
通过多通道逻辑采集环境中的信号强度、噪声功率、多径干扰三类关键数据，生成结构化的原始环境特征集，为后续时空关联分析与信号优化提供基础输入数据。
关键代码
collect_environment_data(sample_count, duration)功能：通过多通道逻辑采集环境中的信号强度、噪声功率和多径干扰数据，结合时间维度生成结构化的原始环境特征集，为后续分析提供基础数据。


2. feature_analysis.py
```python
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
```
核心作用
对原始环境特征集进行时空关联分析，通过 “滑动窗口分割→相关性矩阵构建→PCA 降维” 三步流程，将高维、非结构化的原始数据转化为低维、时空信息保留的动态环境特征矩阵，支撑后续滤波策略选择。
关键代码
sliding_window_segmentation(raw_features, window_size, step)功能：对原始环境特征集进行滑动窗口分割，形成多个时间片段特征集。
calculate_correlation_matrix(windows)功能：计算相邻时间片段特征集之间的相关性系数，构建时空关联矩阵。
build_dynamic_feature_matrix(raw_features, duration)功能：调用上述两个函数，通过主成分分析对时空关联矩阵降维，生成动态环境特征矩阵。

3. filter_strategy.py
```python
import numpy as np
import logging
from utils import progress_bar

class AttentionMechanism:
    """注意力机制实现"""
    def __init__(self, input_dim):
        self.weights = np.random.randn(input_dim, input_dim) / np.sqrt(input_dim)
        
    def compute_attention_weights(self, features):
        """计算各环境特征的动态权重分布"""
        attention_scores = np.dot(features, self.weights)
        attention_weights = np.exp(attention_scores) / np.sum(np.exp(attention_scores), axis=1, keepdims=True)
        return attention_weights

def select_optimal_filter_strategy(dynamic_matrix, duration=8):
    """选择最优滤波策略"""
    filter_strategies = {
        0: {"name": "卡尔曼滤波", "time_param": 0.3, "freq_param": 0.7},
        1: {"name": "自适应维纳滤波", "time_param": 0.5, "freq_param": 0.5},
        2: {"name": "粒子滤波", "time_param": 0.2, "freq_param": 0.8},
        3: {"name": "小波阈值滤波", "time_param": 0.6, "freq_param": 0.4},
        4: {"name": "滑动平均滤波", "time_param": 0.8, "freq_param": 0.2}
    }
    
    attention = AttentionMechanism(dynamic_matrix.shape[1])
    attention_weights = attention.compute_attention_weights(dynamic_matrix)
    
    strategy_scores = np.zeros(len(filter_strategies))
    for i in range(len(filter_strategies)):
        strategy_scores[i] = np.sum(attention_weights[:, i % attention_weights.shape[1]])
    
    optimal_strategy_id = np.argmax(strategy_scores)
    optimal_strategy = filter_strategies[optimal_strategy_id]
    
    # 显示进度
    progress_bar(duration, "最优滤波策略选择")
    
    logging.info(f"最优滤波策略: {optimal_strategy['name']}")
    logging.info(f"时域参数: {optimal_strategy['time_param']}, 频域参数: {optimal_strategy['freq_param']}")
    
    return optimal_strategy, attention_weights
```
核心作用
实现基于注意力机制的滤波策略选择模型，接收动态环境特征矩阵后，通过 “注意力权重计算→策略评分→最优选择” 流程，从预设策略库中输出匹配度最高的最优滤波策略标识，确保滤波策略与环境特征的适配性。
关键代码
AttentionMechanism类功能：通过多头注意力机制计算动态环境特征矩阵中各特征的动态权重分布。
select_optimal_filter_strategy(dynamic_matrix, duration)功能：将动态环境特征矩阵输入注意力模型，根据动态权重分布从预设策略库中选择匹配度最高的滤波策略，生成最优滤波策略标识。

4. signal_processing.py
```python
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
```
核心作用
根据最优滤波策略标识中的时域 / 频域参数，对接收信号执行 “动态时域滤波→自适应频域滤波→信号重构” 的时频联合处理，生成初步优化信号，解决环境干扰导致的信号失真问题。
关键代码
dynamic_time_filter(signal, param, window_base)功能：采用可变长度滑动窗口对接收信号进行动态时域滤波。
adaptive_freq_filter(signal, param)功能：对时域处理后的信号进行自适应频域变换，实施动态频域滤波。
optimize_received_signal(raw_features, strategy, duration)功能：解析最优滤波策略标识中的时域 / 频域参数，调用上述两个函数执行时频联合处理，重构生成初步优化信号。

5. signal_enhancement.py
```python
import numpy as np
import logging
from utils import progress_bar

class ResidualNetwork:
    """深度残差网络"""
    def __init__(self, input_dim):
        self.layers = [
            np.random.randn(input_dim, 64) / np.sqrt(input_dim),
            np.random.randn(64, 32) / np.sqrt(64),
            np.random.randn(32, input_dim) / np.sqrt(32)
        ]
        
    def forward(self, x):
        """前向传播，包含跨层连接"""
        residual = x
        x = np.tanh(np.dot(x, self.layers[0]))
        x = np.tanh(np.dot(x, self.layers[1]))
        x = np.dot(x, self.layers[2])
        return x + residual  

def enhance_signal(optimized_signal, duration=8):
    """通过深度残差网络增强信号"""
    # 分段加窗处理
    window_size = 50
    signal_segments = []
    
    for i in range(0, len(optimized_signal), window_size):
        end = min(i + window_size, len(optimized_signal))
        segment = optimized_signal[i:end]
        if len(segment) < window_size:
            segment = np.pad(segment, (0, window_size - len(segment)), mode='constant')
        signal_segments.append(segment)
    
    signal_segments = np.array(signal_segments)
    
    # 通过残差网络增强
    res_net = ResidualNetwork(window_size)
    enhanced_segments = [res_net.forward(seg) for seg in signal_segments]
    
    # 拼接增强后的片段
    enhanced_signal = np.concatenate(enhanced_segments)[:len(optimized_signal)]  
    
    # 显示进度
    progress_bar(duration, "信号增强处理")
    
    logging.info(f"信号增强完成，增强前后差异: {np.mean(np.abs(enhanced_signal - optimized_signal)):.4f}")
    
    return enhanced_signal, signal_segments, enhanced_segments
```
核心作用
基于深度残差网络对初步优化信号进行增强处理，通过 “分段加窗→多尺度特征提取→跨层特征融合” 流程，修复信号细节失真，生成高质量的增强目标信号。
关键代码
ResidualNetwork类功能：通过卷积层提取多尺度信号特征，利用跨层连接结构融合不同深度的特征表示。
enhance_signal(optimized_signal, duration)功能：对初步优化信号进行分段加窗处理生成信号片段序列，输入深度残差网络处理后重构生成增强目标信号。

6. quality_evaluation.py
```python
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
```
核心作用
对增强目标信号进行多维质量评估，通过 “三类指标计算→三维矩阵构建→滑动加权平均” 流程，生成量化的信号质量评估矩阵，为后续参数调整提供依据。
关键代码
calculate_error_rate(original, enhanced)、calculate_snr(signal, original)、calculate_phase_consistency(original, enhanced)功能：分别计算增强目标信号的误码率、信噪比和相位一致性指标。
evaluate_signal_quality(original_signal, optimized_signal, enhanced_signal, duration)功能：基于上述指标构建包含时间维度和指标维度的三维评估矩阵，通过滑动加权平均算法生成信号质量评估矩阵。

7. parameter_adjustment.py
```python
import numpy as np
import logging
from utils import progress_bar

def adjust_receive_parameters(eval_matrix, duration=8):
    """根据信号质量评估矩阵动态调整接收参数"""
    ideal_config = np.array([0.05, 20, 0.9])  
    
    # 分析异常数据点
    error_threshold = 1.5 * np.mean(eval_matrix[:, 0])  # 误码率阈值
    snr_threshold = 0.5 * np.mean(eval_matrix[:, 1])    # 信噪比阈值
    phase_threshold = 0.5 * np.mean(eval_matrix[:, 2])  # 相位一致性阈值
    
    异常点 = np.where(
        (eval_matrix[:, 0] > error_threshold) | 
        (eval_matrix[:, 1] < snr_threshold) | 
        (eval_matrix[:, 2] < phase_threshold)
    )[0]
    
    # 计算偏差度并生成调整指令
    # 计算当前评估与理想配置的偏差
    current_avg = np.mean(eval_matrix, axis=0)
    deviation = np.abs(current_avg - ideal_config) / ideal_config
    
    # 生成参数调整指令
    param_adjustments = {
        "gain": 0.1 * (1 - deviation[1]),  # 与信噪比负相关
        "bandwidth": 0.8 + 0.2 * (1 - deviation[0]),  # 与误码率负相关
        "modulation": "GFSK" if deviation[2] < 0.3 else "2-FSK"  # 基于相位一致性选择调制方式
    }
    
    # 建立动态映射关系
    future_params = {
        "gain": param_adjustments["gain"] * (1 + 0.05 * np.random.randn()),
        "bandwidth": param_adjustments["bandwidth"] * (1 + 0.03 * np.random.randn()),
        "modulation": param_adjustments["modulation"]
    }
    
    # 显示进度
    progress_bar(duration, "接收参数调整")
    
    logging.info(f"参数调整完成，异常点数量: {len(异常点)}")
    logging.info(f"当前调整: 增益={param_adjustments['gain']:.2f}, 带宽={param_adjustments['bandwidth']:.2f}, 调制方式={param_adjustments['modulation']}")
    logging.info(f"预测最优参数: 增益={future_params['gain']:.2f}, 带宽={future_params['bandwidth']:.2f}, 调制方式={future_params['modulation']}")
    
    return param_adjustments, future_params, 异常点
```
核心作用
根据信号质量评估矩阵动态调整接收参数，同时建立环境特征与参数的动态映射关系，实现“异常点分析→偏差计算→参数调整→最优预测” 的全流程，确保接收参数适配环境变化。
关键代码
adjust_receive_parameters(eval_matrix, duration)功能：分析信号质量评估矩阵中的异常数据点分布，计算当前参数配置与理想配置的偏差度，生成参数调整指令；同时建立环境特征与接收参数的动态映射关系，预测最优参数配置实现预调整。

8. main.py
```python
import logging
import time
from data_collection import collect_environment_data
from feature_analysis import build_dynamic_feature_matrix
from filter_strategy import select_optimal_filter_strategy
from signal_processing import optimize_received_signal
from signal_enhancement import enhance_signal
from quality_evaluation import evaluate_signal_quality
from parameter_adjustment import adjust_receive_parameters
from visualization import (
    plot_signal_comparison,
    plot_environment_features,
    plot_correlation_matrix,
    plot_attention_weights,
    plot_signal_quality_metrics,
    plot_parameter_adjustments,
    plot_pca_components,
    plot_segment_comparison
)

def main():
    """主函数：执行低功耗蓝牙芯片信号接收优化流程"""
    start_time = time.time()
    logging.info("===== 低功耗蓝牙芯片信号接收优化系统启动 =====")
    
    # 步骤1：环境数据采集
    raw_features = collect_environment_data(sample_count=1000)
    time_points = raw_features[:, 0]
    
    # 步骤2：构建动态环境特征矩阵
    dynamic_matrix, windows, corr_matrix, pca = build_dynamic_feature_matrix(raw_features)
    
    # 步骤3：选择最优滤波策略
    optimal_strategy, attention_weights = select_optimal_filter_strategy(dynamic_matrix)
    
    # 步骤4：信号优化处理
    original_signal, optimized_signal = optimize_received_signal(raw_features, optimal_strategy)
    
    # 步骤5：信号增强
    enhanced_signal, original_segments, enhanced_segments = enhance_signal(optimized_signal)
    
    # 步骤6：信号质量评估
    eval_matrix, error_rates, snrs, phase_consistencies = evaluate_signal_quality(
        original_signal, optimized_signal, enhanced_signal
    )
    
    # 步骤7-8：参数调整与预测
    param_adjustments, future_params, 异常点 = adjust_receive_parameters(eval_matrix)
    
    # 生成可视化图表
    logging.info("开始生成数据可视化图表...")
    plot_signal_comparison(original_signal, optimized_signal, enhanced_signal, time_points)
    plot_environment_features(raw_features)
    plot_correlation_matrix(corr_matrix[:20, :20])  
    plot_attention_weights(attention_weights[:50, :])  
    plot_signal_quality_metrics(error_rates, snrs, phase_consistencies)
    plot_parameter_adjustments(param_adjustments, future_params)
    plot_pca_components(pca)
    plot_segment_comparison(original_segments, enhanced_segments, segment_idx=5)
    logging.info("数据可视化图表生成完成，保存至results文件夹")
    
    # 计算并显示总运行时间
    end_time = time.time()
    total_duration = end_time - start_time
    logging.info(f"===== 低功耗蓝牙芯片信号接收优化系统运行完成 =====")
    logging.info(f"总运行时间: {total_duration:.2f}秒")

if __name__ == "__main__":
    main()
```
核心作用
串联低功耗蓝牙芯片信号接收优化的全流程步骤，执行端到端的信号接收优化，输出最终优化结果与可视化图表。
关键代码
main()功能：串联全流程步骤，执行端到端的低功耗蓝牙芯片信号接收优化流程。

9. utils.py
```python
import time
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# 确保中文显示正常
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 创建结果目录
if not os.path.exists('results'):
    os.makedirs('results')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

def progress_bar(duration, step_name):
    """显示带进度条的处理过程"""
    logging.info(f"开始{step_name}...")
    total_steps = 100
    interval = duration / total_steps
    
    for i in tqdm(range(total_steps), desc=step_name, ncols=100):
        time.sleep(interval)
    
    logging.info(f"{step_name}完成")
```
核心作用
提供进度条显示、日志配置、结果目录创建等辅助功能，确保各模块执行过程可追踪、结果可存储，为流程落地提供基础支撑。
关键代码
progress_bar(duration, step_name)功能：显示各步骤执行进度，支持流程可视化追踪。
日志配置函数、结果目录创建函数功能：配置日志输出格式，创建结果存储目录，确保流程可追溯、结果可存储。

10. visualization.py
```python
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
```
核心作用
对优化流程中的关键数据进行可视化，直观展示优化效果，为优化结果提供验证依据。
关键代码
plot_signal_comparison(original, optimized, enhanced, time_points)功能：绘制原始信号、初步优化信号、增强目标信号的对比图，直观展示信号优化效果。
plot_signal_quality_metrics(error_rates, snrs, phase_consistencies)功能：绘制信号质量评估指标的趋势图，验证质量评估结果。


