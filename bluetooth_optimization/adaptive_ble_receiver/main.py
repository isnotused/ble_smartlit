"""
低功耗蓝牙信号接收优化系统主程序
"""

import argparse
import sys
import time
import numpy as np
from utils.ble_signal_optimizer import BLESignalOptimizer
from support.performance_monitor import PerformanceMonitor
from support.data_manager import DataManager
from support.error_handler import ErrorHandler, ErrorSeverity, ErrorCode
from support.demo_runner import DemoRunner

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='低功耗蓝牙信号接收优化系统')
    parser.add_argument('--mode', choices=['optimize', 'demo', 'monitor', 'test'], 
                       default='optimize', help='运行模式')
    parser.add_argument('--duration', type=float, default=0.1, 
                       help='每次信号采集时长(秒)')
    parser.add_argument('--cycles', type=int, default=10, 
                       help='优化循环次数')
    parser.add_argument('--output', type=str, default='optimization_results.h5',
                       help='输出文件名')
    
    args = parser.parse_args()
    
    # 初始化错误处理器
    error_handler = ErrorHandler()
    
    try:
        if args.mode == 'demo':
            # 演示模式
            run_demo_mode()
            
        elif args.mode == 'monitor':
            # 监控模式
            run_monitor_mode(args.duration, args.cycles)
            
        elif args.mode == 'test':
            # 测试模式
            run_test_mode()
            
        else:
            # 优化模式
            run_optimization_mode(args.duration, args.cycles, args.output)
            
    except KeyboardInterrupt:
        print("\n程序被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"程序执行出错: {e}")
        sys.exit(1)

def run_demo_mode():
    """运行演示模式"""
    print("启动演示模式...")
    demo = DemoRunner()
    demo.run_complete_demo(cycles=10)
    demo.run_comparison_demo()

def run_monitor_mode(duration: float, cycles: int):
    """运行监控模式"""
    print("启动监控模式...")
    
    optimizer = BLESignalOptimizer()
    performance_monitor = PerformanceMonitor()
    
    if not optimizer.initialize_system():
        print("系统初始化失败!")
        return
    
    performance_monitor.start_monitoring()
    
    print(f"开始连续优化监控，周期: {cycles}次")
    print("按 Ctrl+C 停止监控")
    
    try:
        for cycle in range(cycles):
            start_time = time.time()
            
            # 执行优化
            result = optimizer.optimize_signal_reception(duration)
            
            # 记录性能
            latency = time.time() - start_time
            performance_monitor.record_optimization_metrics(result, latency)
            
            # 显示实时指标
            if cycle % 5 == 0:
                metrics = performance_monitor.get_real_time_metrics()
                print(f"周期 {cycle+1}/{cycles} - "
                      f"延迟: {metrics['latency']*1000:.1f}ms, "
                      f"质量: {metrics['quality']:.3f}")
            
            time.sleep(0.1)  # 避免过载
            
    except KeyboardInterrupt:
        print("\n监控被用户中断")
    
    finally:
        performance_monitor.stop_monitoring()
        
        # 生成最终报告
        report = performance_monitor.get_performance_report()
        display_performance_summary(report)

def run_optimization_mode(duration: float, cycles: int, output_file: str):
    """运行优化模式"""
    print("启动优化模式...")
    
    optimizer = BLESignalOptimizer()
    data_manager = DataManager()
    
    if not optimizer.initialize_system():
        print("系统初始化失败!")
        return
    
    print(f"开始信号接收优化，采集时长: {duration}秒，循环次数: {cycles}")
    
    results = []
    for cycle in range(cycles):
        print(f"执行优化循环 {cycle+1}/{cycles}...")
        
        result = optimizer.optimize_signal_reception(duration)
        results.append(result)
        
        # 显示当前质量
        if 'quality_assessment' in result:
            quality_matrix = result['quality_assessment']['quality_matrix']
            overall_quality = np.mean(quality_matrix[:, :, 1])
            print(f"  信号质量评分: {overall_quality:.3f}")
    
    # 保存结果
    if results:
        data_manager.save_optimization_result(results[-1], output_file)
        print(f"优化结果已保存到: {output_file}")
    
    # 显示系统状态
    status = optimizer.get_system_status()
    print(f"系统状态: 已初始化={status['initialized']}, "
          f"优化次数={status['optimization_count']}")

def run_test_mode():
    """运行测试模式"""
    print("启动测试模式...")
    
    from support.test_utils import SystemValidator, TestSignalGenerator
    
    validator = SystemValidator()
    signal_generator = TestSignalGenerator()
    
    test_cases = [
        {
            'name': 'QPSK信号优化测试',
            'signal_type': 'qpsk',
            'snr_db': 15.0,
            'length': 2000,
            'thresholds': [0.5, 0.6, 0.7]
        },
        {
            'name': 'OFDM信号优化测试', 
            'signal_type': 'ofdm',
            'snr_db': 10.0,
            'length': 3000,
            'thresholds': [0.4, 0.5, 0.6]
        },
        {
            'name': 'FSK信号优化测试',
            'signal_type': 'fsk', 
            'snr_db': 20.0,
            'length': 1500,
            'thresholds': [0.6, 0.7, 0.8]
        }
    ]
    
    # 执行验证
    results = validator.validate_optimization_pipeline(test_cases)
    
    # 显示测试结果
    print("\n测试结果汇总:")
    print("=" * 40)
    print(f"总测试数: {results['total_tests']}")
    print(f"通过数: {results['passed_tests']}")
    print(f"失败数: {results['failed_tests']}")
    print(f"成功率: {results['success_rate']*100:.1f}%")
    
    # 显示详细结果
    print("\n详细结果:")
    for test in results['test_details']:
        status = "通过" if test['passed'] else "失败"
        print(f"  {test['test_name']}: {status}")
        if not test['passed'] and 'error' in test:
            print(f"    错误: {test['error']}")

def display_performance_summary(report: dict):
    """显示性能摘要"""
    print("\n性能摘要:")
    print("=" * 30)
    print(f"平均延迟: {report['latency']['average']*1000:.2f}ms")
    print(f"延迟稳定性: {report['stability']['latency_stability']:.3f}")
    print(f"质量稳定性: {report['stability']['quality_stability']:.3f}")
    print(f"系统状态: {report['stability']['status']}")

if __name__ == "__main__":
    main()