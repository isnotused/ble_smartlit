# 系统配置参数
class SystemConfig:
    # 射频前端参数
    RF_CHANNELS = 8
    SAMPLE_RATE = 2000000  # 2MHz采样率
    ADC_RESOLUTION = 12
    
    # 滑动窗口参数
    WINDOW_SIZE = 256
    SLIDE_STEP = 64
    
    # 滤波策略参数
    FILTER_STRATEGIES = {
        0: {'type': 'kalman', 'params': {'q': 0.1, 'r': 1.0}},
        1: {'type': 'wiener', 'params': {'window_size': 32}},
        2: {'type': 'adaptive_lms', 'params': {'mu': 0.01, 'order': 16}},
        3: {'type': 'butterworth', 'params': {'order': 4, 'cutoff': 0.1}}
    }
    
    # 信号质量评估阈值
    SNR_THRESHOLD = 15.0
    BER_THRESHOLD = 1e-4
    PHASE_CONSISTENCY_THRESHOLD = 0.8

# 神经网络模型配置
class ModelConfig:
    ATTENTION_HEADS = 8
    ATTENTION_DIM = 64
    RESIDUAL_BLOCKS = 12
    CONV_KERNEL_SIZES = [3, 5, 7]