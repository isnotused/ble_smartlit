import numpy as np
from typing import Tuple, Dict
from config import SystemConfig

class RFFrontend:
    """多通道射频前端数据采集"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.channel_count = config.RF_CHANNELS
        self.sample_rate = config.SAMPLE_RATE
        
    def collect_environment_data(self, duration: float) -> Dict[str, np.ndarray]:
        """
        采集环境特征数据
        Args:
            duration: 采集时长(秒)
        Returns:
            原始环境特征集
        """
        sample_count = int(duration * self.sample_rate)
        
        # 多通道信号采集
        signal_data = np.zeros((self.channel_count, sample_count), dtype=np.complex64)
        noise_power = np.zeros(self.channel_count)
        multipath_data = np.zeros((self.channel_count, 10))  # 10个多径分量
        
        for channel in range(self.channel_count):
            # 采集信号强度
            signal_power = self._measure_signal_power(channel)
            # 测量噪声功率
            noise_power[channel] = self._measure_noise_power(channel)
            # 分析多径干扰
            multipath_data[channel] = self._analyze_multipath(channel)
            
            signal_data[channel] = self._generate_signal_samples(channel, sample_count)
        
        return {
            'signal_strength': signal_data,
            'noise_power': noise_power,
            'multipath_interference': multipath_data,
            'timestamp': np.array([self._get_timestamp()])
        }
    
    def _measure_signal_power(self, channel: int) -> float:
        """测量指定通道的信号强度"""
        base_power = -70.0  # dBm
        channel_variation = np.random.normal(0, 2.0)
        return base_power + channel_variation
    
    def _measure_noise_power(self, channel: int) -> float:
        """测量指定通道的噪声功率"""
        base_noise = -90.0  # dBm
        channel_variation = np.random.normal(0, 1.0)
        return base_noise + channel_variation
    
    def _analyze_multipath(self, channel: int) -> np.ndarray:
        """分析多径干扰特征"""
        multipath_profile = np.zeros(10)
        # 生成典型的多径延迟分布
        for i in range(10):
            delay = i * 0.1  # 微秒
            amplitude = np.exp(-delay * 2.0)  # 指数衰减
            multipath_profile[i] = amplitude + np.random.normal(0, 0.1)
        return multipath_profile
    
    def _generate_signal_samples(self, channel: int, sample_count: int) -> np.ndarray:
        t = np.arange(sample_count) / self.sample_rate
        carrier_freq = 2400000000 + channel * 2000000  # 2.4GHz + 通道偏移
        
        # 生成QPSK调制信号
        symbols = np.random.choice([1+1j, 1-1j, -1+1j, -1-1j], sample_count//100)
        symbol_waveform = np.repeat(symbols, 100)
        
        # 添加载波
        signal = symbol_waveform[:sample_count] * np.exp(2j * np.pi * carrier_freq * t)
        
        # 添加噪声
        noise = np.random.normal(0, 0.1, sample_count) + 1j * np.random.normal(0, 0.1, sample_count)
        return signal + noise
    
    def _get_timestamp(self) -> float:
        """获取当前时间戳"""
        import time
        return time.time()