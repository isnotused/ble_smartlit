import traceback
import logging
from typing import Optional, Dict, Any
from enum import Enum

class ErrorSeverity(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class ErrorCode(Enum):
    # 系统错误
    SYSTEM_INIT_FAILED = 1001
    MODULE_LOAD_FAILED = 1002
    MEMORY_ALLOCATION_ERROR = 1003
    
    # 信号处理错误
    SIGNAL_ACQUISITION_ERROR = 2001
    FILTER_PROCESSING_ERROR = 2002
    ENHANCEMENT_PROCESSING_ERROR = 2003
    QUALITY_ASSESSMENT_ERROR = 2004
    
    # 参数错误
    INVALID_PARAMETER = 3001
    PARAMETER_ADJUSTMENT_ERROR = 3002
    PREDICTION_MODEL_ERROR = 3003
    
    # 数据错误
    DATA_SAVE_ERROR = 4001
    DATA_LOAD_ERROR = 4002
    DATA_CORRUPTION = 4003

class OptimizationError(Exception):
    """优化系统专用异常类"""
    
    def __init__(self, error_code: ErrorCode, message: str, 
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 module: str = None, details: Dict[str, Any] = None):
        self.error_code = error_code
        self.message = message
        self.severity = severity
        self.module = module
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'error_code': self.error_code.value,
            'message': self.message,
            'severity': self.severity.name,
            'module': self.module,
            'details': self.details,
            'timestamp': self.details.get('timestamp', '')
        }

class ErrorHandler:
    """错误处理器"""
    
    def __init__(self, log_file: str = "optimization_errors.log"):
        self.log_file = log_file
        self.setup_logging()
        
        # 错误统计
        self.error_stats = {
            severity: 0 for severity in ErrorSeverity
        }
        self.module_errors = {}
        
    def setup_logging(self):
        """设置日志系统"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('BLEOptimization')
    
    def handle_error(self, error: OptimizationError, 
                    context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        处理错误
        Args:
            error: 优化错误对象
            context: 错误上下文信息
        Returns:
            错误处理结果
        """
        # 更新错误统计
        self.error_stats[error.severity] += 1
        
        module = error.module or 'unknown'
        if module not in self.module_errors:
            self.module_errors[module] = 0
        self.module_errors[module] += 1
        
        # 记录错误详情
        error_details = error.to_dict()
        error_details['context'] = context or {}
        error_details['traceback'] = traceback.format_exc()
        
        # 根据严重级别记录日志
        if error.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(f"[{error.error_code.name}] {error.message}", 
                               extra=error_details)
        elif error.severity == ErrorSeverity.HIGH:
            self.logger.error(f"[{error.error_code.name}] {error.message}", 
                            extra=error_details)
        elif error.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(f"[{error.error_code.name}] {error.message}", 
                              extra=error_details)
        else:
            self.logger.info(f"[{error.error_code.name}] {error.message}", 
                           extra=error_details)
        
        # 生成错误响应
        response = {
            'success': False,
            'error': error_details,
            'recovery_action': self._get_recovery_action(error),
            'should_retry': self._should_retry(error)
        }
        
        return response
    
    def handle_generic_exception(self, exception: Exception, 
                               module: str = None,
                               context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        处理通用异常
        Args:
            exception: 通用异常对象
            module: 发生异常的模块
            context: 错误上下文
        Returns:
            错误处理结果
        """
        # 转换为优化错误
        optimization_error = OptimizationError(
            error_code=ErrorCode.SYSTEM_INIT_FAILED,
            message=f"未处理的异常: {str(exception)}",
            severity=ErrorSeverity.HIGH,
            module=module,
            details={'original_exception': type(exception).__name__}
        )
        
        return self.handle_error(optimization_error, context)
    
    def _get_recovery_action(self, error: OptimizationError) -> str:
        """根据错误类型获取恢复动作"""
        if error.error_code in [ErrorCode.MEMORY_ALLOCATION_ERROR, 
                              ErrorCode.SYSTEM_INIT_FAILED]:
            return "restart_system"
        
        elif error.error_code in [ErrorCode.SIGNAL_ACQUISITION_ERROR,
                                ErrorCode.FILTER_PROCESSING_ERROR]:
            return "fallback_processing"
        
        elif error.error_code in [ErrorCode.INVALID_PARAMETER,
                                ErrorCode.PARAMETER_ADJUSTMENT_ERROR]:
            return "use_default_parameters"
        
        elif error.error_code in [ErrorCode.DATA_SAVE_ERROR,
                                ErrorCode.DATA_LOAD_ERROR]:
            return "use_alternative_storage"
        
        else:
            return "continue_with_limitations"
    
    def _should_retry(self, error: OptimizationError) -> bool:
        """判断是否应该重试"""
        if error.severity in [ErrorSeverity.LOW, ErrorSeverity.MEDIUM]:
            return True
        
        if error.error_code in [ErrorCode.SIGNAL_ACQUISITION_ERROR,
                              ErrorCode.DATA_LOAD_ERROR]:
            return True
        
        return False
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """获取错误统计信息"""
        total_errors = sum(self.error_stats.values())
        
        return {
            'total_errors': total_errors,
            'severity_distribution': {
                severity.name: count for severity, count in self.error_stats.items()
            },
            'module_distribution': self.module_errors,
            'error_rate': total_errors / max(1, len(self.module_errors))
        }
    
    def reset_statistics(self):
        """重置错误统计"""
        self.error_stats = {severity: 0 for severity in ErrorSeverity}
        self.module_errors = {}
    
    def check_system_health(self) -> Dict[str, Any]:
        """检查系统健康状态"""
        stats = self.get_error_statistics()
        critical_errors = self.error_stats[ErrorSeverity.CRITICAL]
        high_errors = self.error_stats[ErrorSeverity.HIGH]
        
        if critical_errors > 0:
            health_status = 'critical'
        elif high_errors > 5:
            health_status = 'poor'
        elif high_errors > 0:
            health_status = 'degraded'
        else:
            health_status = 'healthy'
        
        return {
            'health_status': health_status,
            'critical_errors': critical_errors,
            'high_errors': high_errors,
            'total_errors': stats['total_errors'],
            'recommendation': self._get_health_recommendation(health_status)
        }
    
    def _get_health_recommendation(self, health_status: str) -> str:
        """获取健康状态建议"""
        recommendations = {
            'healthy': "系统运行正常，无需干预",
            'degraded': "建议监控系统性能，准备维护",
            'poor': "需要立即进行系统维护和错误排查",
            'critical': "系统面临严重问题，建议重启或紧急维护"
        }
        return recommendations.get(health_status, "未知状态")