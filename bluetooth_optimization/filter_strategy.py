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