import numpy as np
from typing import Dict, List, Tuple
from sklearn.ensemble import RandomForestRegressor
from config import SystemConfig

class ParameterPredictor:
    """参数预测器"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.mapping_models = {}
        self.prediction_history = []
        self.is_trained = False
        
    def establish_dynamic_mapping(self, environment_features: np.ndarray,
                                optimal_parameters: List[Dict],
                                quality_scores: List[float]) -> bool:
        """
        建立环境特征与接收参数的动态映射关系
        Args:
            environment_features: 环境特征序列
            optimal_parameters: 最优参数序列
            quality_scores: 质量评分序列
        Returns:
            训练是否成功
        """
        if len(environment_features) < 10:
            return False
        
        try:
            # 准备训练数据
            X, y = self._prepare_training_data(environment_features, optimal_parameters)
            
            if len(X) == 0:
                return False
            
            # 为每个关键参数训练预测模型
            parameter_names = list(optimal_parameters[0].keys())
            
            for param_name in parameter_names:
                if param_name in ['rf_gain', 'filter_cutoff', 'demodulation_threshold']:
                    # 提取该参数的目标值
                    y_param = np.array([params[param_name] for params in optimal_parameters])
                    
                    # 训练随机森林回归模型
                    model = RandomForestRegressor(n_estimators=50, random_state=42)
                    model.fit(X, y_param)
                    
                    self.mapping_models[param_name] = model
            
            self.is_trained = True
            return True
            
        except Exception as e:
            print(f"动态映射训练失败: {e}")
            return False
    
    def predict_optimal_parameters(self, current_environment: np.ndarray,
                                 environment_trend: np.ndarray) -> Dict:
        """
        根据环境变化趋势预测最优参数配置
        Args:
            current_environment: 当前环境特征
            environment_trend: 环境变化趋势
        Returns:
            预测的最优参数配置
        """
        if not self.is_trained or not self.mapping_models:
            return self._get_default_parameters()
        
        try:
            # 构建预测特征
            prediction_features = self._build_prediction_features(
                current_environment, environment_trend)
            
            # 预测各参数值
            predicted_parameters = {}
            
            for param_name, model in self.mapping_models.items():
                predicted_value = model.predict(prediction_features.reshape(1, -1))[0]
                predicted_parameters[param_name] = float(predicted_value)
            
            # 记录预测结果
            self.prediction_history.append({
                'environment': current_environment,
                'trend': environment_trend,
                'predicted_parameters': predicted_parameters
            })
            
            return predicted_parameters
            
        except Exception as e:
            print(f"参数预测失败: {e}")
            return self._get_default_parameters()
    
    def _prepare_training_data(self, environment_features: np.ndarray,
                             optimal_parameters: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """准备训练数据"""
        X = []
        y = []
        
        for i in range(len(environment_features)):
            # 环境特征
            env_feature = environment_features[i]
            
            # 参数向量
            if i < len(optimal_parameters):
                param_vector = self._parameters_to_vector(optimal_parameters[i])
                X.append(env_feature.flatten())
                y.append(param_vector)
        
        return np.array(X), np.array(y)
    
    def _parameters_to_vector(self, parameters: Dict) -> np.ndarray:
        """将参数字典转换为向量"""
        vector = []
        important_params = ['rf_gain', 'filter_cutoff', 'demodulation_threshold', 
                          'phase_lock_gain', 'agc_attack_time']
        
        for param_name in important_params:
            if param_name in parameters:
                vector.append(parameters[param_name])
        
        return np.array(vector)
    
    def _build_prediction_features(self, current_environment: np.ndarray,
                                 environment_trend: np.ndarray) -> np.ndarray:
        """构建预测特征向量"""
        # 当前环境特征
        current_features = current_environment.flatten()
        
        # 趋势特征（差分、斜率等）
        trend_features = self._extract_trend_features(environment_trend)
        
        # 组合特征
        combined_features = np.concatenate([current_features, trend_features])
        
        return combined_features
    
    def _extract_trend_features(self, trend_data: np.ndarray) -> np.ndarray:
        """提取趋势特征"""
        if len(trend_data) < 2:
            return np.zeros(5)
        
        features = []
        
        # 近期变化率
        recent_changes = np.diff(trend_data[-5:]) if len(trend_data) >= 5 else np.diff(trend_data)
        if len(recent_changes) > 0:
            features.append(np.mean(recent_changes))  # 平均变化率
            features.append(np.std(recent_changes))   # 变化稳定性
        
        # 趋势方向
        if len(trend_data) >= 3:
            slope = self._compute_slope(trend_data[-3:])
            features.append(slope)
        else:
            features.append(0.0)
        
        # 填充到固定维度
        while len(features) < 5:
            features.append(0.0)
        
        return np.array(features[:5])
    
    def _compute_slope(self, data: np.ndarray) -> float:
        """计算数据序列的斜率"""
        x = np.arange(len(data))
        slope = np.polyfit(x, data, 1)[0]
        return slope
    
    def _get_default_parameters(self) -> Dict:
        """获取默认参数配置"""
        return {
            'rf_gain': 20.0,
            'filter_cutoff': 0.1,
            'demodulation_threshold': 0.0,
            'phase_lock_gain': 1.0,
            'agc_attack_time': 0.01,
            'equalizer_coeffs': [1.0, 0.0, 0.0]
        }
    
    def update_mapping_with_feedback(self, actual_parameters: Dict,
                                   actual_quality: float,
                                   predicted_parameters: Dict):
        pass