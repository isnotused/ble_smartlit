import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
from config import ModelConfig

class ResidualBlock(nn.Module):
    """残差块实现"""
    
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class MultiScaleFeatureExtractor(nn.Module):
    """多尺度特征提取器"""
    
    def __init__(self, config: ModelConfig, input_channels: int = 64):
        super().__init__()
        self.config = config
        
        # 多尺度卷积层 - 接受64通道输入
        self.conv_layers = nn.ModuleList()
        for kernel_size in config.CONV_KERNEL_SIZES:
            padding = kernel_size // 2
            conv = nn.Conv1d(input_channels, 32, kernel_size, padding=padding)
            self.conv_layers.append(conv)
        
        self.feature_fusion = nn.Conv1d(32 * len(config.CONV_KERNEL_SIZES), 64, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = []
        for conv in self.conv_layers:
            feature = conv(x)
            features.append(feature)
        
        # 特征融合
        fused_features = torch.cat(features, dim=1)
        output = self.feature_fusion(fused_features)
        
        return output

class DeepResidualEnhancement(nn.Module):
    """基于深度残差网络的信号增强系统"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # 输入处理层
        self.input_conv = nn.Conv1d(1, 64, 7, padding=3)
        self.input_bn = nn.BatchNorm1d(64)
        self.input_relu = nn.ReLU(inplace=True)
        
        # 多尺度特征提取 - 传入正确的输入通道数
        self.multiscale_extractor = MultiScaleFeatureExtractor(config, input_channels=64)
        
        # 残差块
        self.residual_blocks = nn.Sequential(*[
            ResidualBlock(64) for _ in range(config.RESIDUAL_BLOCKS)
        ])
        
        # 特征重构网络
        self.reconstruction_net = nn.Sequential(
            nn.Conv1d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 1, 3, padding=1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            x: 输入信号 [batch_size, 1, signal_length]
        Returns:
            增强目标信号 [batch_size, 1, signal_length]
        """
        # 步骤5.1: 输入预处理
        x = self.input_conv(x)
        x = self.input_bn(x)
        x = self.input_relu(x)
        
        # 步骤5.2: 多尺度特征提取
        multiscale_features = self.multiscale_extractor(x)
        
        # 步骤5.3: 残差特征处理
        residual_features = self.residual_blocks(multiscale_features)
        
        # 特征融合与重构
        enhanced_signal = self.reconstruction_net(residual_features)
        
        return enhanced_signal

class SignalEnhancementProcessor:
    """信号增强处理器"""
    
    def __init__(self, config: ModelConfig):
        self.model = DeepResidualEnhancement(config)
        
    def enhance_signal(self, input_signal: np.ndarray, window_size: int = 1024) -> np.ndarray:
        """
        对初步优化信号进行增强处理
        Args:
            input_signal: 初步优化信号
            window_size: 分段窗口大小
        Returns:
            增强目标信号
        """
        # 分段加窗处理
        signal_segments = self._segment_and_window(input_signal, window_size)
        
        enhanced_segments = []
        
        for segment in signal_segments:
            # 转换为模型输入格式
            input_tensor = torch.FloatTensor(segment).unsqueeze(0).unsqueeze(0)
            
            # 模型推理
            with torch.no_grad():
                enhanced_tensor = self.model(input_tensor)
                enhanced_segment = enhanced_tensor.squeeze().numpy()
            
            enhanced_segments.append(enhanced_segment)
        
        # 信号重构
        enhanced_signal = self._reconstruct_from_segments(enhanced_segments, len(input_signal))
        
        return enhanced_signal
    
    def _segment_and_window(self, signal: np.ndarray, window_size: int) -> list:
        """对信号进行分段加窗处理"""
        segments = []
        length = len(signal)
        
        for i in range(0, length, window_size // 2):
            end_idx = min(i + window_size, length)
            segment = signal[i:end_idx]
            
            # 零填充不足的段
            if len(segment) < window_size:
                segment = np.pad(segment, (0, window_size - len(segment)))
            
            # 应用汉宁窗
            window = np.hanning(window_size)
            windowed_segment = segment * window
            
            segments.append(windowed_segment)
        
        return segments
    
    def _reconstruct_from_segments(self, segments: list, original_length: int) -> np.ndarray:
        """从分段中重构完整信号"""
        window_size = len(segments[0])
        hop_size = window_size // 2
        
        reconstructed = np.zeros(original_length)
        weights = np.zeros(original_length)
        
        for i, segment in enumerate(segments):
            start_idx = i * hop_size
            end_idx = start_idx + window_size
            
            if end_idx > original_length:
                segment = segment[:original_length - start_idx]
                end_idx = original_length
            
            reconstructed[start_idx:end_idx] += segment
            weights[start_idx:end_idx] += np.hanning(len(segment))
        
        # 避免除零
        weights[weights == 0] = 1
        reconstructed = reconstructed / weights
        
        return reconstructed