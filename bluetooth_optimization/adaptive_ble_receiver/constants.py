# 系统常量定义
class SignalConstants:
    BLE_CHANNEL_COUNT = 40
    CHANNEL_SPACING = 2000000  # 2MHz
    SYMBOL_RATE = 1000000  # 1Msymbol/s
    
    # 信号处理常量
    FFT_SIZE = 1024
    OVERLAP_RATIO = 0.75
    MAX_DOPPLER_SHIFT = 50000  # 50kHz

class ErrorConstants:
    SUCCESS = 0
    INVALID_PARAMETER = -1
    PROCESSING_TIMEOUT = -2
    MEMORY_ALLOCATION_FAILED = -3