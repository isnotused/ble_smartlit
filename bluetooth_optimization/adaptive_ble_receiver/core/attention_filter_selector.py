import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import ModelConfig, SystemConfig

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.heads = config.ATTENTION_HEADS
        self.attention_dim = config.ATTENTION_DIM
        self.head_dim = config.ATTENTION_DIM // config.ATTENTION_HEADS
        
        self.query = nn.Linear(config.ATTENTION_DIM, config.ATTENTION_DIM)
        self.key = nn.Linear(config.ATTENTION_DIM, config.ATTENTION_DIM)
        self.value = nn.Linear(config.ATTENTION_DIM, config.ATTENTION_DIM)
        self.output = nn.Linear(config.ATTENTION_DIM, config.ATTENTION_DIM)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # 线性变换
        Q = self.query(x).view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力权重
        attention_weights = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attention_weights = F.softmax(attention_weights, dim=-1)
        
        # 应用注意力权重
        attention_output = torch.matmul(attention_weights, V)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.attention_dim)
        
        return self.output(attention_output)

class FilterStrategySelector(nn.Module):
    """基于注意力机制的滤波策略选择模型"""
    
    def __init__(self, config: ModelConfig, system_config: SystemConfig):
        super().__init__()
        self.config = config
        self.system_config = system_config
        
        # 特征编码层 - 使用自适应输入维度
        # PCA降维后的特征维度可能不固定，这里使用灵活的编码层
        self.feature_encoder = None  # 延迟初始化
        
        # 注意力层
        self.attention = MultiHeadAttention(config)
        
        # 策略选择层
        self.strategy_predictor = nn.Sequential(
            nn.Linear(config.ATTENTION_DIM, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, len(system_config.FILTER_STRATEGIES))
        )
    
    def _initialize_encoder(self, input_dim: int):
        """根据输入维度初始化编码器"""
        if self.feature_encoder is None:
            self.feature_encoder = nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.ReLU(),
                nn.Linear(32, self.config.ATTENTION_DIM)
            )
        
    def forward(self, feature_matrix: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            feature_matrix: 动态环境特征矩阵 [batch_size, seq_len, feature_dim]
        Returns:
            滤波策略概率分布 [batch_size, num_strategies]
        """
        # 动态初始化编码器
        if self.feature_encoder is None:
            input_dim = feature_matrix.shape[-1]
            self._initialize_encoder(input_dim)
        
        # 特征编码
        encoded_features = self.feature_encoder(feature_matrix)
        
        # 注意力计算
        attention_output = self.attention(encoded_features)
        
        # 全局平均池化
        global_features = torch.mean(attention_output, dim=1)
        
        # 策略预测
        strategy_logits = self.strategy_predictor(global_features)
        
        return strategy_logits

class AttentionFilterManager:
    """滤波策略管理器"""
    
    def __init__(self, config: ModelConfig, system_config: SystemConfig):
        self.model = FilterStrategySelector(config, system_config)
        self.strategy_count = len(system_config.FILTER_STRATEGIES)
        
    def select_optimal_filter_strategy(self, feature_matrix: np.ndarray) -> int:
        """
        选择最优滤波策略
        Args:
            feature_matrix: 动态环境特征矩阵
        Returns:
            最优滤波策略标识
        """
        # 转换为模型输入格式
        input_tensor = torch.FloatTensor(feature_matrix).unsqueeze(0)
        
        # 模型推理
        with torch.no_grad():
            strategy_logits = self.model(input_tensor)
            strategy_probs = F.softmax(strategy_logits, dim=-1)
            optimal_strategy = torch.argmax(strategy_probs, dim=-1).item()
        
        return optimal_strategy
    
    def get_dynamic_weights(self, feature_matrix: np.ndarray) -> np.ndarray:
        """
        获取环境特征的动态权重分布
        Args:
            feature_matrix: 动态环境特征矩阵
        Returns:
            动态权重分布
        """
        input_tensor = torch.FloatTensor(feature_matrix).unsqueeze(0)
        
        # 动态初始化编码器
        if self.model.feature_encoder is None:
            input_dim = input_tensor.shape[-1]
            self.model._initialize_encoder(input_dim)
        
        # 提取注意力权重
        encoded_features = self.model.feature_encoder(input_tensor)
        Q = self.model.attention.query(encoded_features)
        K = self.model.attention.key(encoded_features)
        
        attention_weights = torch.matmul(Q, K.transpose(-2, -1))
        attention_weights = F.softmax(attention_weights, dim=-1)
        
        # 计算平均注意力权重
        mean_weights = torch.mean(attention_weights, dim=[0, 1])
        
        return mean_weights.detach().numpy()