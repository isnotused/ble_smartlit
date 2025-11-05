import numpy as np
from typing import Dict, List, Optional
from config import SystemConfig, ModelConfig
from core.rf_frontend import RFFrontend
from core.environment_analyzer import EnvironmentAnalyzer
from core.attention_filter_selector import AttentionFilterManager
from core.adaptive_filter import AdaptiveFilter
from core.residual_enhancement import SignalEnhancementProcessor
from core.quality_assessor import QualityAssessor
from core.parameter_adaptation import ParameterAdapter
from core.parameter_predictor import ParameterPredictor

class BLESignalOptimizer:
    """低功耗蓝牙信号接收优化系统主控制器"""
    
    def __init__(self):
        # 初始化配置
        self.system_config = SystemConfig()
        self.model_config = ModelConfig()
        
        # 初始化各个模块
        self.rf_frontend = RFFrontend(self.system_config)
        self.env_analyzer = EnvironmentAnalyzer(self.system_config)
        self.filter_selector = AttentionFilterManager(self.model_config, self.system_config)
        self.adaptive_filter = AdaptiveFilter(self.system_config)
        self.signal_enhancer = SignalEnhancementProcessor(self.model_config)
        self.quality_assessor = QualityAssessor(self.system_config)
        self.parameter_adapter = ParameterAdapter(self.system_config)
        self.parameter_predictor = ParameterPredictor(self.system_config)
        
        # 系统状态
        self.current_parameters = self._initialize_parameters()
        self.optimization_history = []
        self.is_initialized = False
        
    def initialize_system(self) -> bool:
        """初始化系统"""
        try:
            # 执行系统自检
            self._system_self_test()
            self.is_initialized = True
            return True
        except Exception as e:
            print(f"系统初始化失败: {e}")
            return False
    
    def optimize_signal_reception(self, duration: float = 0.1) -> Dict:
        """
        执行完整的信号接收优化流程
        Args:
            duration: 信号采集时长(秒)
        Returns:
            优化结果
        """
        if not self.is_initialized:
            raise RuntimeError("系统未初始化")
        
        optimization_result = {}
        
        try:
            # 步骤1: 采集环境数据
            raw_features = self.rf_frontend.collect_environment_data(duration)
            optimization_result['raw_features'] = raw_features
            
            # 步骤2: 环境特征分析
            feature_matrix = self.env_analyzer.build_dynamic_feature_matrix(raw_features)
            optimization_result['feature_matrix'] = feature_matrix
            
            # 步骤3: 选择滤波策略
            optimal_strategy = self.filter_selector.select_optimal_filter_strategy(feature_matrix)
            optimization_result['optimal_strategy'] = optimal_strategy
            
            # 提取接收信号
            received_signal = raw_features['signal_strength'][0]  # 使用第一个通道
            
            # 步骤4: 自适应滤波处理
            filtered_signal = self.adaptive_filter.apply_filter_strategy(
                received_signal, optimal_strategy)
            optimization_result['filtered_signal'] = filtered_signal
            
            # 步骤5: 信号增强
            enhanced_signal = self.signal_enhancer.enhance_signal(filtered_signal)
            optimization_result['enhanced_signal'] = enhanced_signal
            
            # 步骤6: 质量评估
            quality_result = self.quality_assessor.assess_signal_quality(enhanced_signal)
            optimization_result['quality_assessment'] = quality_result
            
            # 步骤7: 参数自适应调整
            new_parameters = self.parameter_adapter.adapt_parameters(
                quality_result['quality_matrix'], self.current_parameters)
            optimization_result['new_parameters'] = new_parameters
            
            # 更新当前参数
            self.current_parameters = new_parameters
            
            # 步骤8: 参数预测预调整
            predicted_parameters = self.parameter_predictor.predict_optimal_parameters(
                feature_matrix, self._extract_environment_trend())
            optimization_result['predicted_parameters'] = predicted_parameters
            
            # 记录优化历史
            self._record_optimization_history(optimization_result)
            
            return optimization_result
            
        except Exception as e:
            print(f"信号优化过程出错: {e}")
            return self._get_fallback_result()
    
    def continuous_optimization(self, duration: float = 0.1, cycles: int = 10) -> List[Dict]:
        """
        执行连续优化循环
        Args:
            duration: 每次采集时长
            cycles: 优化循环次数
        Returns:
            优化结果列表
        """
        results = []
        
        for cycle in range(cycles):
            print(f"执行优化循环 {cycle + 1}/{cycles}")
            
            result = self.optimize_signal_reception(duration)
            results.append(result)
            
            # 更新参数预测器的映射关系
            if cycle >= 5:  # 积累一定数据后开始训练
                self._update_parameter_mapping()
        
        return results
    
    def get_system_status(self) -> Dict:
        """获取系统状态信息"""
        return {
            'initialized': self.is_initialized,
            'current_parameters': self.current_parameters,
            'optimization_count': len(self.optimization_history),
            'module_status': {
                'rf_frontend': True,
                'environment_analyzer': True,
                'filter_selector': True,
                'adaptive_filter': True,
                'signal_enhancer': True,
                'quality_assessor': True,
                'parameter_adapter': True,
                'parameter_predictor': self.parameter_predictor.is_trained
            }
        }
    
    def _initialize_parameters(self) -> Dict:
        """初始化系统参数"""
        return {
            'rf_gain': 20.0,
            'filter_cutoff': 0.1,
            'demodulation_threshold': 0.0,
            'phase_lock_gain': 1.0,
            'agc_attack_time': 0.01,
            'equalizer_coeffs': [1.0, 0.0, 0.0],
            'sampling_rate': self.system_config.SAMPLE_RATE
        }
    
    def _system_self_test(self):
        """系统自检"""
        # 测试各个模块的基本功能
        test_signal = np.random.normal(0, 1.0, 1000) + 1j * np.random.normal(0, 1.0, 1000)
        
        # 测试环境分析器
        test_features = {'signal_strength': test_signal.reshape(1, -1),
                        'noise_power': np.array([-90.0]),
                        'multipath_interference': np.random.normal(0, 1.0, (1, 10))}
        feature_matrix = self.env_analyzer.build_dynamic_feature_matrix(test_features)
        
        # 测试滤波选择器
        if feature_matrix.size > 0:
            strategy = self.filter_selector.select_optimal_filter_strategy(feature_matrix)
            assert strategy in self.system_config.FILTER_STRATEGIES
        
        # 测试信号增强器
        enhanced = self.signal_enhancer.enhance_signal(test_signal.real)
        assert len(enhanced) == len(test_signal)
        
        print("系统自检完成")
    
    def _extract_environment_trend(self) -> np.ndarray:
        """提取环境变化趋势"""
        if len(self.optimization_history) < 2:
            return np.zeros(5)
        
        # 从历史记录中提取环境特征趋势
        recent_features = []
        for result in self.optimization_history[-5:]:
            if 'feature_matrix' in result:
                recent_features.append(result['feature_matrix'].flatten())
        
        if len(recent_features) > 1:
            trend = np.diff(np.array(recent_features), axis=0)
            return trend.flatten()
        else:
            return np.zeros(5)
    
    def _update_parameter_mapping(self):
        """更新参数映射关系"""
        if len(self.optimization_history) < 10:
            return
        
        # 准备训练数据
        environment_features = []
        optimal_parameters = []
        quality_scores = []
        
        for result in self.optimization_history[-20:]:  # 使用最近20个样本
            if all(key in result for key in ['feature_matrix', 'new_parameters', 'quality_assessment']):
                environment_features.append(result['feature_matrix'])
                optimal_parameters.append(result['new_parameters'])
                
                # 提取质量评分
                quality_matrix = result['quality_assessment']['quality_matrix']
                overall_quality = np.mean(quality_matrix[:, :, 1])  # 平均评估分数
                quality_scores.append(overall_quality)
        
        if len(environment_features) >= 10:
            success = self.parameter_predictor.establish_dynamic_mapping(
                np.array(environment_features), optimal_parameters, quality_scores)
            
            if success:
                print("参数映射关系已更新")
    
    def _record_optimization_history(self, result: Dict):
        """记录优化历史"""
        self.optimization_history.append(result)
        
        # 保持历史记录长度
        max_history = 100
        if len(self.optimization_history) > max_history:
            self.optimization_history.pop(0)
    
    def _get_fallback_result(self) -> Dict:
        """获取降级处理结果"""
        return {
            'status': 'fallback',
            'enhanced_signal': np.zeros(1000),
            'quality_assessment': {'quality_matrix': np.zeros((10, 5, 3))},
            'new_parameters': self.current_parameters
        }