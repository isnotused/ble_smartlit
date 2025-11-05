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