import logging
import time
from data_collection import collect_environment_data
from feature_analysis import build_dynamic_feature_matrix
from filter_strategy import select_optimal_filter_strategy
from signal_processing import optimize_received_signal
from signal_enhancement import enhance_signal
from quality_evaluation import evaluate_signal_quality
from parameter_adjustment import adjust_receive_parameters
from visualization import (
    plot_signal_comparison,
    plot_environment_features,
    plot_correlation_matrix,
    plot_attention_weights,
    plot_signal_quality_metrics,
    plot_parameter_adjustments,
    plot_pca_components,
    plot_segment_comparison
)

def main():
    """主函数：执行低功耗蓝牙芯片信号接收优化流程"""
    start_time = time.time()
    logging.info("===== 低功耗蓝牙芯片信号接收优化系统启动 =====")
    
    # 步骤1：环境数据采集
    raw_features = collect_environment_data(sample_count=1000)
    time_points = raw_features[:, 0]
    
    # 步骤2：构建动态环境特征矩阵
    dynamic_matrix, windows, corr_matrix, pca = build_dynamic_feature_matrix(raw_features)
    
    # 步骤3：选择最优滤波策略
    optimal_strategy, attention_weights = select_optimal_filter_strategy(dynamic_matrix)
    
    # 步骤4：信号优化处理
    original_signal, optimized_signal = optimize_received_signal(raw_features, optimal_strategy)
    
    # 步骤5：信号增强
    enhanced_signal, original_segments, enhanced_segments = enhance_signal(optimized_signal)
    
    # 步骤6：信号质量评估
    eval_matrix, error_rates, snrs, phase_consistencies = evaluate_signal_quality(
        original_signal, optimized_signal, enhanced_signal
    )
    
    # 步骤7-8：参数调整与预测
    param_adjustments, future_params, 异常点 = adjust_receive_parameters(eval_matrix)
    
    # 生成可视化图表
    logging.info("开始生成数据可视化图表...")
    plot_signal_comparison(original_signal, optimized_signal, enhanced_signal, time_points)
    plot_environment_features(raw_features)
    plot_correlation_matrix(corr_matrix[:20, :20])  
    plot_attention_weights(attention_weights[:50, :])  
    plot_signal_quality_metrics(error_rates, snrs, phase_consistencies)
    plot_parameter_adjustments(param_adjustments, future_params)
    plot_pca_components(pca)
    plot_segment_comparison(original_segments, enhanced_segments, segment_idx=5)
    logging.info("数据可视化图表生成完成，保存至results文件夹")
    
    # 计算并显示总运行时间
    end_time = time.time()
    total_duration = end_time - start_time
    logging.info(f"===== 低功耗蓝牙芯片信号接收优化系统运行完成 =====")
    logging.info(f"总运行时间: {total_duration:.2f}秒")

if __name__ == "__main__":
    main()