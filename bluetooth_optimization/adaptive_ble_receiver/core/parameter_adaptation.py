import numpy as np
from typing import Dict, List, Tuple
from config import SystemConfig

class ParameterAdapter:
    """参数自适应调整器"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.parameter_history = []
        self.quality_history = []
        
    def adapt_parameters(self, quality_matrix: np.ndarray, 
                        current_parameters: Dict) -> Dict:
        """
        根据信号质量评估矩阵动态调整接收参数配置
        Args:
            quality_matrix: 信号质量评估矩阵
            current_parameters: 当前参数配置
        Returns:
            新的接收参数组合
        """
        # 步骤7.1: 分析异常数据点分布
        anomaly_distribution = self._analyze_anomaly_distribution(quality_matrix)
        
        # 步骤7.2: 计算参数调整指令
        adjustment_instruction = self._compute_adjustment_instruction(
            quality_matrix, current_parameters, anomaly_distribution)
        
        # 生成新的参数组合
        new_parameters = self._generate_new_parameters(
            current_parameters, adjustment_instruction)
        
        # 更新历史记录
        self._update_history(new_parameters, quality_matrix)
        
        return new_parameters
    
    def _analyze_anomaly_distribution(self, quality_matrix: np.ndarray) -> Dict:
        """分析评估矩阵中的异常数据点分布"""
        time_points, metrics, dimensions = quality_matrix.shape
        
        anomaly_info = {
            'anomaly_count': 0,
            'metric_anomalies': np.zeros(metrics),
            'time_anomalies': np.zeros(time_points),
            'severity_levels': []
        }
        
        threshold_low = 0.3  # 低质量阈值
        threshold_high = 0.8  # 高质量阈值
        
        for t in range(time_points):
            for m in range(metrics):
                score = quality_matrix[t, m, 1]  # 评估分数维度
                confidence = quality_matrix[t, m, 2]  # 置信度维度
                
                if score < threshold_low and confidence > 0.5:
                    # 检测到异常点
                    anomaly_info['anomaly_count'] += 1
                    anomaly_info['metric_anomalies'][m] += 1
                    anomaly_info['time_anomalies'][t] += 1
                    
                    severity = (threshold_low - score) / threshold_low
                    anomaly_info['severity_levels'].append({
                        'time': t,
                        'metric': m,
                        'severity': severity,
                        'score': score
                    })
        
        return anomaly_info
    
    def _compute_adjustment_instruction(self, quality_matrix: np.ndarray,
                                      current_parameters: Dict,
                                      anomaly_info: Dict) -> Dict:
        """计算当前参数配置与理想配置的偏差度，生成参数调整指令"""
        # 计算整体质量评分
        overall_quality = self._compute_overall_quality(quality_matrix)
        
        # 分析各指标表现
        metric_performance = self._analyze_metric_performance(quality_matrix)
        
        # 生成调整方向和幅度
        adjustment = {
            'direction': {},  # 调整方向
            'magnitude': {},  # 调整幅度
            'priority': {}    # 调整优先级
        }
        
        # 根据各指标表现调整相应参数
        for metric_name, performance in metric_performance.items():
            if performance < 0.6:  # 性能较差
                param_adjustments = self._get_parameter_adjustments_for_metric(
                    metric_name, performance, current_parameters)
                adjustment['direction'].update(param_adjustments['direction'])
                adjustment['magnitude'].update(param_adjustments['magnitude'])
                adjustment['priority'].update(param_adjustments['priority'])
        
        return adjustment
    
    def _generate_new_parameters(self, current_parameters: Dict,
                               adjustment: Dict) -> Dict:
        """生成新的参数组合"""
        new_parameters = current_parameters.copy()
        
        for param_name, direction in adjustment['direction'].items():
            magnitude = adjustment['magnitude'].get(param_name, 0.1)
            
            if param_name in new_parameters:
                current_value = new_parameters[param_name]
                
                if isinstance(current_value, (int, float)):
                    # 数值参数调整
                    adjustment_value = direction * magnitude * current_value
                    new_parameters[param_name] = current_value + adjustment_value
                
                elif isinstance(current_value, list):
                    # 列表参数调整（如滤波器系数）
                    adjustment_array = np.array([direction * magnitude] * len(current_value))
                    new_parameters[param_name] = (np.array(current_value) + adjustment_array).tolist()
        
        return new_parameters
    
    def _compute_overall_quality(self, quality_matrix: np.ndarray) -> float:
        """计算整体质量评分"""
        scores = quality_matrix[:, :, 1]  # 评估分数维度
        confidences = quality_matrix[:, :, 2]  # 置信度维度
        
        # 加权平均
        weighted_scores = scores * confidences
        overall_quality = np.mean(weighted_scores) / np.mean(confidences)
        
        return overall_quality
    
    def _analyze_metric_performance(self, quality_matrix: np.ndarray) -> Dict[str, float]:
        """分析各指标的性能表现"""
        metric_names = ['snr', 'ber', 'phase_consistency', 'spectral_flatness', 'amplitude_stability']
        performance = {}
        
        for i, metric_name in enumerate(metric_names):
            if i < quality_matrix.shape[1]:
                metric_scores = quality_matrix[:, i, 1]
                metric_confidences = quality_matrix[:, i, 2]
                
                # 置信度加权平均
                weighted_score = np.sum(metric_scores * metric_confidences) / np.sum(metric_confidences)
                performance[metric_name] = weighted_score
        
        return performance
    
    def _get_parameter_adjustments_for_metric(self, metric_name: str,
                                            performance: float,
                                            current_parameters: Dict) -> Dict:
        """根据指标性能获取参数调整方案"""
        adjustments = {
            'direction': {},
            'magnitude': {},
            'priority': {}
        }
        
        if metric_name == 'snr':
            # 信噪比低，需要调整增益和滤波参数
            adjustments['direction']['rf_gain'] = 1  # 增加增益
            adjustments['magnitude']['rf_gain'] = 0.15
            adjustments['priority']['rf_gain'] = 1
            
            adjustments['direction']['filter_cutoff'] = -1  # 降低截止频率
            adjustments['magnitude']['filter_cutoff'] = 0.1
            adjustments['priority']['filter_cutoff'] = 2
            
        elif metric_name == 'ber':
            # 误码率高，需要调整解调参数
            adjustments['direction']['demodulation_threshold'] = 1
            adjustments['magnitude']['demodulation_threshold'] = 0.2
            adjustments['priority']['demodulation_threshold'] = 1
            
        elif metric_name == 'phase_consistency':
            # 相位一致性差，需要调整同步参数
            adjustments['direction']['phase_lock_gain'] = 1
            adjustments['magnitude']['phase_lock_gain'] = 0.25
            adjustments['priority']['phase_lock_gain'] = 2
            
        elif metric_name == 'spectral_flatness':
            # 频谱平坦度差，需要调整均衡参数
            adjustments['direction']['equalizer_coeffs'] = 1
            adjustments['magnitude']['equalizer_coeffs'] = 0.1
            adjustments['priority']['equalizer_coeffs'] = 3
            
        elif metric_name == 'amplitude_stability':
            # 幅度稳定性差，需要调整AGC参数
            adjustments['direction']['agc_attack_time'] = -1
            adjustments['magnitude']['agc_attack_time'] = 0.3
            adjustments['priority']['agc_attack_time'] = 1
        
        return adjustments
    
    def _update_history(self, parameters: Dict, quality_matrix: np.ndarray):
        """更新参数和质量历史记录"""
        self.parameter_history.append(parameters)
        self.quality_history.append(quality_matrix)
        
        # 保持历史记录长度
        max_history = 100
        if len(self.parameter_history) > max_history:
            self.parameter_history.pop(0)
            self.quality_history.pop(0)