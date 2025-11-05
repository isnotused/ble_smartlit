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