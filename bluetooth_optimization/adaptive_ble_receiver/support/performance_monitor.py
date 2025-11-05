import time
import numpy as np
from typing import Dict, List, Optional
from threading import Thread, Lock
from collections import deque

class PerformanceMonitor:
    """系统性能监控器"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics_lock = Lock()
        
        # 性能指标存储
        self.latency_history = deque(maxlen=window_size)
        self.memory_usage_history = deque(maxlen=window_size)
        self.cpu_usage_history = deque(maxlen=window_size)
        self.quality_history = deque(maxlen=window_size)
        
        # 实时监控状态
        self.is_monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """开始性能监控"""
        if self.is_monitoring:
            return
            
        self.is_monitoring = True
        self.monitor_thread = Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """停止性能监控"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
    
    def record_optimization_metrics(self, optimization_result: Dict, latency: float):
        """
        记录优化过程指标
        Args:
            optimization_result: 优化结果
            latency: 处理延迟
        """
        with self.metrics_lock:
            self.latency_history.append(latency)
            
            # 记录质量指标
            if 'quality_assessment' in optimization_result:
                quality_matrix = optimization_result['quality_assessment']['quality_matrix']
                overall_quality = np.mean(quality_matrix[:, :, 1])  # 平均评估分数
                self.quality_history.append(overall_quality)
            
            # 记录系统资源使用
            self._record_system_resources()
    
    def get_performance_report(self) -> Dict:
        """获取性能报告"""
        with self.metrics_lock:
            latency_array = np.array(self.latency_history)
            quality_array = np.array(self.quality_history)
            memory_array = np.array(self.memory_usage_history)
            cpu_array = np.array(self.cpu_usage_history)
            
            report = {
                'latency': {
                    'current': latency_array[-1] if len(latency_array) > 0 else 0.0,
                    'average': np.mean(latency_array) if len(latency_array) > 0 else 0.0,
                    'std_dev': np.std(latency_array) if len(latency_array) > 0 else 0.0,
                    'percentile_95': np.percentile(latency_array, 95) if len(latency_array) > 0 else 0.0
                },
                'quality': {
                    'current': quality_array[-1] if len(quality_array) > 0 else 0.0,
                    'average': np.mean(quality_array) if len(quality_array) > 0 else 0.0,
                    'trend': self._compute_trend(quality_array)
                },
                'system': {
                    'memory_usage_mb': np.mean(memory_array) if len(memory_array) > 0 else 0.0,
                    'cpu_usage_percent': np.mean(cpu_array) if len(cpu_array) > 0 else 0.0
                },
                'stability': self._assess_system_stability()
            }
            
            return report
    
    def get_real_time_metrics(self) -> Dict:
        """获取实时性能指标"""
        return {
            'latency': self.latency_history[-1] if self.latency_history else 0.0,
            'quality': self.quality_history[-1] if self.quality_history else 0.0,
            'memory_usage': self.memory_usage_history[-1] if self.memory_usage_history else 0.0,
            'cpu_usage': self.cpu_usage_history[-1] if self.cpu_usage_history else 0.0
        }
    
    def _monitoring_loop(self):
        """监控循环"""
        while self.is_monitoring:
            self._record_system_resources()
            time.sleep(1.0)  # 每秒记录一次
    
    def _record_system_resources(self):
        """记录系统资源使用情况"""
        try:
            import psutil
            process = psutil.Process()
            
            # 内存使用(MB)
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.memory_usage_history.append(memory_mb)
            
            # CPU使用率(%)
            cpu_percent = process.cpu_percent()
            self.cpu_usage_history.append(cpu_percent)
            
        except ImportError:
            # 如果没有psutil，使用模拟数据
            self.memory_usage_history.append(100.0)
            self.cpu_usage_history.append(25.0)
    
    def _compute_trend(self, data: np.ndarray) -> float:
        """计算数据趋势"""
        if len(data) < 2:
            return 0.0
        
        x = np.arange(len(data))
        slope = np.polyfit(x, data, 1)[0]
        return slope
    
    def _assess_system_stability(self) -> Dict:
        """评估系统稳定性"""
        with self.metrics_lock:
            latency_array = np.array(self.latency_history)
            quality_array = np.array(self.quality_history)
            
            if len(latency_array) < 10:
                return {'status': 'insufficient_data', 'score': 0.0}
            
            stability_metrics = {}
            
            # 延迟稳定性
            latency_cv = np.std(latency_array) / np.mean(latency_array)  # 变异系数
            stability_metrics['latency_stability'] = 1.0 / (1.0 + latency_cv)
            
            # 质量稳定性
            if len(quality_array) > 0:
                quality_std = np.std(quality_array)
                stability_metrics['quality_stability'] = 1.0 / (1.0 + quality_std)
            else:
                stability_metrics['quality_stability'] = 0.0
            
            # 综合稳定性评分
            overall_stability = np.mean(list(stability_metrics.values()))
            
            # 稳定性状态
            if overall_stability > 0.8:
                status = 'excellent'
            elif overall_stability > 0.6:
                status = 'good'
            elif overall_stability > 0.4:
                status = 'fair'
            else:
                status = 'poor'
            
            stability_metrics.update({
                'overall_score': overall_stability,
                'status': status
            })
            
            return stability_metrics
    
    def generate_alert(self, metric: str, threshold: float) -> Optional[Dict]:
        """生成性能告警"""
        current_value = getattr(self, f'{metric}_history')
        if not current_value:
            return None
        
        latest_value = current_value[-1]
        if latest_value > threshold:
            return {
                'metric': metric,
                'value': latest_value,
                'threshold': threshold,
                'timestamp': time.time(),
                'severity': 'high' if latest_value > threshold * 1.5 else 'medium'
            }
        
        return None