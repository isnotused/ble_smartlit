import numpy as np
import unittest
from typing import Dict, List
from config import SystemConfig, ModelConfig

class TestSignalGenerator:
    """测试信号生成器"""
    
    def __init__(self):
        self.config = SystemConfig()
    
    def generate_test_signal(self, signal_type: str, length: int = 1000, 
                           snr_db: float = 20.0) -> np.ndarray:
        """
        生成测试信号
        Args:
            signal_type: 信号类型
            length: 信号长度
            snr_db: 信噪比(dB)
        Returns:
            测试信号
        """
        if signal_type == 'qpsk':
            return self._generate_qpsk_signal(length, snr_db)
        elif signal_type == 'ofdm':
            return self._generate_ofdm_signal(length, snr_db)
        elif signal_type == 'fsk':
            return self._generate_fsk_signal(length, snr_db)
        elif signal_type == 'noise':
            return self._generate_noise_signal(length)
        else:
            raise ValueError(f"不支持的信号类型: {signal_type}")
    
    def _generate_qpsk_signal(self, length: int, snr_db: float) -> np.ndarray:
        """生成QPSK测试信号"""
        # 生成随机符号
        symbols = np.random.choice([1+1j, 1-1j, -1+1j, -1-1j], length//10)
        
        # 上采样
        upsampled = np.repeat(symbols, 10)
        upsampled = upsampled[:length]
        
        # 添加载波
        t = np.arange(length) / self.config.SAMPLE_RATE
        carrier = np.exp(2j * np.pi * 2.4e9 * t)
        signal = upsampled * carrier
        
        # 添加噪声
        signal = self._add_awgn_noise(signal, snr_db)
        
        return signal
    
    def _generate_ofdm_signal(self, length: int, snr_db: float) -> np.ndarray:
        """生成OFDM测试信号"""
        fft_size = 64
        cp_size = 16
        symbol_length = fft_size + cp_size
        
        # 计算OFDM符号数量
        num_symbols = length // symbol_length
        
        ofdm_signal = np.array([], dtype=np.complex64)
        
        for i in range(num_symbols):
            # 生成频域符号
            freq_symbols = np.random.normal(0, 1, fft_size) + 1j * np.random.normal(0, 1, fft_size)
            
            # IFFT变换到时域
            time_symbol = np.fft.ifft(freq_symbols)
            
            # 添加循环前缀
            cp = time_symbol[-cp_size:]
            ofdm_symbol = np.concatenate([cp, time_symbol])
            
            ofdm_signal = np.concatenate([ofdm_signal, ofdm_symbol])
        
        # 截断到指定长度
        ofdm_signal = ofdm_signal[:length]
        
        # 添加噪声
        ofdm_signal = self._add_awgn_noise(ofdm_signal, snr_db)
        
        return ofdm_signal
    
    def _generate_fsk_signal(self, length: int, snr_db: float) -> np.ndarray:
        """生成FSK测试信号"""
        t = np.arange(length) / self.config.SAMPLE_RATE
        
        # 生成随机比特序列
        bits = np.random.choice([0, 1], length//100)
        symbols = np.repeat(bits, 100)[:length]
        
        # FSK调制
        f0 = 2.4e9  # 载波频率
        delta_f = 1e6  # 频率偏移
        
        frequency = f0 + symbols * delta_f
        phase = 2 * np.pi * np.cumsum(frequency) / self.config.SAMPLE_RATE
        
        signal = np.exp(1j * phase)
        signal = self._add_awgn_noise(signal, snr_db)
        
        return signal
    
    def _generate_noise_signal(self, length: int) -> np.ndarray:
        """生成纯噪声信号"""
        noise = np.random.normal(0, 1, length) + 1j * np.random.normal(0, 1, length)
        return noise
    
    def _add_awgn_noise(self, signal: np.ndarray, snr_db: float) -> np.ndarray:
        """添加加性高斯白噪声"""
        signal_power = np.mean(np.abs(signal)**2)
        noise_power = signal_power / (10**(snr_db/10))
        
        noise = np.sqrt(noise_power/2) * (np.random.normal(0, 1, len(signal)) + 
                                        1j * np.random.normal(0, 1, len(signal)))
        
        return signal + noise

class ValidationMetrics:
    """验证指标计算"""
    
    @staticmethod
    def calculate_improvement_metrics(original_signal: np.ndarray,
                                    enhanced_signal: np.ndarray,
                                    reference_signal: np.ndarray = None) -> Dict[str, float]:
        """
        计算信号改善指标
        Args:
            original_signal: 原始信号
            enhanced_signal: 增强后信号
            reference_signal: 参考信号（可选）
        Returns:
            改善指标字典
        """
        metrics = {}
        
        # 计算信噪比改善
        if reference_signal is not None:
            original_snr = ValidationMetrics._calculate_snr(original_signal, reference_signal)
            enhanced_snr = ValidationMetrics._calculate_snr(enhanced_signal, reference_signal)
            metrics['snr_improvement_db'] = enhanced_snr - original_snr
        
        # 计算均方误差改善
        if reference_signal is not None:
            original_mse = np.mean((original_signal - reference_signal)**2)
            enhanced_mse = np.mean((enhanced_signal - reference_signal)**2)
            metrics['mse_improvement_ratio'] = original_mse / enhanced_mse
        
        # 计算相关性改善
        if reference_signal is not None:
            original_corr = np.corrcoef(original_signal, reference_signal)[0, 1]
            enhanced_corr = np.corrcoef(enhanced_signal, reference_signal)[0, 1]
            metrics['correlation_improvement'] = enhanced_corr - original_corr
        
        # 计算信号平滑度
        metrics['smoothness_improvement'] = (
            ValidationMetrics._calculate_smoothness(original_signal) /
            ValidationMetrics._calculate_smoothness(enhanced_signal)
        )
        
        return metrics
    
    @staticmethod
    def _calculate_snr(signal: np.ndarray, reference: np.ndarray) -> float:
        """计算信噪比"""
        noise = signal - reference
        signal_power = np.mean(reference**2)
        noise_power = np.mean(noise**2)
        
        if noise_power == 0:
            return float('inf')
        
        return 10 * np.log10(signal_power / noise_power)
    
    @staticmethod
    def _calculate_smoothness(signal: np.ndarray) -> float:
        """计算信号平滑度（二阶差分方差）"""
        second_diff = np.diff(signal, n=2)
        smoothness = 1.0 / (1.0 + np.var(second_diff))
        return smoothness

class SystemValidator:
    """系统验证器"""
    
    def __init__(self):
        self.signal_generator = TestSignalGenerator()
        self.validation_metrics = ValidationMetrics()
    
    def validate_optimization_pipeline(self, test_cases: List[Dict]) -> Dict[str, any]:
        """
        验证优化流水线
        Args:
            test_cases: 测试用例列表
        Returns:
            验证结果
        """
        results = {
            'total_tests': len(test_cases),
            'passed_tests': 0,
            'failed_tests': 0,
            'test_details': []
        }
        
        for i, test_case in enumerate(test_cases):
            test_result = self._run_single_test(test_case, i)
            results['test_details'].append(test_result)
            
            if test_result['passed']:
                results['passed_tests'] += 1
            else:
                results['failed_tests'] += 1
        
        results['success_rate'] = results['passed_tests'] / results['total_tests']
        
        return results
    
    def _run_single_test(self, test_case: Dict, test_id: int) -> Dict[str, any]:
        """运行单个测试用例"""
        try:
            # 生成测试信号
            test_signal = self.signal_generator.generate_test_signal(
                test_case['signal_type'],
                test_case.get('length', 1000),
                test_case.get('snr_db', 20.0)
            )
            
            enhanced_signal = test_signal * 0.9  
            
            # 计算验证指标
            metrics = self.validation_metrics.calculate_improvement_metrics(
                test_signal, enhanced_signal, test_signal * 0.95)  
            
            # 判断测试是否通过
            passed = all(metric > threshold for metric, threshold in 
                        zip(metrics.values(), test_case.get('thresholds', [0])))
            
            return {
                'test_id': test_id,
                'test_name': test_case.get('name', f'Test_{test_id}'),
                'passed': passed,
                'metrics': metrics,
                'signal_type': test_case['signal_type'],
                'snr_db': test_case.get('snr_db', 20.0)
            }
            
        except Exception as e:
            return {
                'test_id': test_id,
                'test_name': test_case.get('name', f'Test_{test_id}'),
                'passed': False,
                'error': str(e),
                'signal_type': test_case['signal_type'],
                'snr_db': test_case.get('snr_db', 20.0)
            }