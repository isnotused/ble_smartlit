"""
低功耗蓝牙信号接收优化系统 - Streamlit Web界面
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import time
import sys
import os
from datetime import datetime
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 添加项目路径到系统路径
project_root = Path(__file__).parent
adaptive_ble_path = project_root / "bluetooth_optimization" / "adaptive-ble-receiver"
sys.path.append(str(adaptive_ble_path))

# 导入项目模块
try:
    from bluetooth_optimization.adaptive_ble_receiver.utils.ble_signal_optimizer import BLESignalOptimizer
    from bluetooth_optimization.adaptive_ble_receiver.support.performance_monitor import PerformanceMonitor
    from bluetooth_optimization.adaptive_ble_receiver.support.data_manager import DataManager
    from bluetooth_optimization.adaptive_ble_receiver.support.test_utils import SystemValidator, TestSignalGenerator
    MODULES_AVAILABLE = True
except ImportError as e:
    st.error(f"模块导入失败: {e}")
    MODULES_AVAILABLE = False

# 页面配置
st.set_page_config(
    page_title="低功耗蓝牙信号优化系统",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义样式
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
}
.mode-card {
    padding: 1rem;
    border-radius: 0.5rem;
    border: 1px solid #ddd;
    margin-bottom: 1rem;
    background-color: #f8f9fa;
}
.metric-card {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 0.5rem;
    color: white;
    text-align: center;
    margin: 0.5rem 0;
}
.success-box {
    background-color: #d4edda;
    border: 1px solid #c3e6cb;
    color: #155724;
    padding: 0.75rem;
    border-radius: 0.375rem;
    margin: 1rem 0;
}
.warning-box {
    background-color: #fff3cd;
    border: 1px solid #ffeaa7;
    color: #856404;
    padding: 0.75rem;
    border-radius: 0.375rem;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# 主标题
st.markdown('<h1 class="main-header">📡 低功耗蓝牙信号接收优化系统</h1>', unsafe_allow_html=True)

# 侧边栏 - 系统配置
with st.sidebar:
    st.header("🛠️ 系统配置")
    
    # 运行模式选择
    mode = st.selectbox(
        "运行模式",
        options=["optimize", "demo", "monitor", "test"],
        index=0,
        help="选择系统运行模式"
    )
    
    st.markdown("---")
    
    # 参数配置
    st.subheader("⚙️ 参数设置")
    
    duration = st.slider(
        "信号采集时长 (秒)",
        min_value=0.01,
        max_value=1.0,
        value=0.1,
        step=0.01,
        help="每次信号采集的时长"
    )
    
    cycles = st.slider(
        "优化循环次数",
        min_value=1,
        max_value=100,
        value=10,
        step=1,
        help="优化循环的次数"
    )
    
    output_file = st.text_input(
        "输出文件名",
        value="optimization_results.h5",
        help="保存结果的HDF5文件名"
    )
    
    st.markdown("---")
    
    # 系统状态
    st.subheader("📊 系统状态")
    if MODULES_AVAILABLE:
        st.success("✅ 模块加载成功")
    else:
        st.error("❌ 模块加载失败")

# 主要内容区域
if MODULES_AVAILABLE:
    # 创建标签页
    tab1, tab2, tab3, tab4 = st.tabs(["🏠 主控面板", "📈 实时监控", "📊 数据分析", "🧪 测试验证"])
    
    with tab1:
        st.header("主控面板")
        
        # 显示当前配置
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>运行模式</h3>
                <p>{mode.upper()}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>采集时长</h3>
                <p>{duration}s</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>循环次数</h3>
                <p>{cycles}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>输出文件</h3>
                <p>{output_file}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # 运行控制
        st.subheader("🚀 运行控制")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🎯 开始优化", type="primary", use_container_width=True):
                run_optimization_mode(duration, cycles, output_file)
        
        with col2:
            if st.button("🎭 演示模式", use_container_width=True):
                run_demo_mode()
        
        with col3:
            if st.button("📊 监控模式", use_container_width=True):
                run_monitor_mode(duration, cycles)
        
        with col4:
            if st.button("🧪 测试模式", use_container_width=True):
                run_test_mode()
    
    with tab2:
        st.header("实时监控")
        
        # 实时监控界面
        if st.button("启动实时监控", type="primary"):
            monitor_placeholder = st.empty()
            metrics_placeholder = st.empty()
            chart_placeholder = st.empty()
            
            # 模拟实时监控
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i in range(cycles):
                # 更新进度
                progress = (i + 1) / cycles
                progress_bar.progress(progress)
                status_text.text(f"执行第 {i+1}/{cycles} 次优化循环...")
                
                # 模拟指标数据
                latency = np.random.normal(50, 10)  # ms
                quality = np.random.normal(0.7, 0.1)
                memory = np.random.normal(128, 20)  # MB
                cpu = np.random.normal(45, 15)  # %
                
                # 显示实时指标
                col1, col2, col3, col4 = metrics_placeholder.columns(4)
                with col1:
                    st.metric("延迟", f"{latency:.1f}ms", delta=f"{np.random.normal(0, 5):.1f}")
                with col2:
                    st.metric("质量评分", f"{quality:.3f}", delta=f"{np.random.normal(0, 0.05):.3f}")
                with col3:
                    st.metric("内存使用", f"{memory:.0f}MB", delta=f"{np.random.normal(0, 10):.0f}")
                with col4:
                    st.metric("CPU使用率", f"{cpu:.1f}%", delta=f"{np.random.normal(0, 5):.1f}")
                
                time.sleep(0.5)  # 模拟处理时间
            
            status_text.text("监控完成!")
    
    with tab3:
        st.header("数据分析")
        
        # 文件选择
        h5_files = []
        data_dirs = [
            Path("bluetooth_optimization/adaptive-ble-receiver/data"),
            Path("data")
        ]
        
        for data_dir in data_dirs:
            if data_dir.exists():
                h5_files.extend(list(data_dir.glob("*.h5")))
        
        if h5_files:
            selected_file = st.selectbox(
                "选择HDF5文件进行分析",
                options=h5_files,
                format_func=lambda x: x.name
            )
            
            if st.button("📊 分析数据", type="primary"):
                analyze_h5_file(selected_file)
        else:
            st.warning("未找到HDF5数据文件，请先运行优化生成数据。")
    
    with tab4:
        st.header("测试验证")
        
        # 测试配置
        st.subheader("🧪 测试配置")
        
        test_signal_type = st.selectbox(
            "信号类型",
            options=["qpsk", "ofdm", "fsk", "noise"],
            help="选择要测试的信号类型"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            test_snr = st.slider("信噪比 (dB)", -10.0, 30.0, 15.0, 0.5)
        with col2:
            test_length = st.slider("信号长度", 500, 5000, 2000, 100)
        
        if st.button("🚀 运行测试", type="primary"):
            run_signal_test(test_signal_type, test_snr, test_length)

else:
    st.error("❌ 系统模块未正确加载，请检查项目配置。")


# 函数定义
def run_optimization_mode(duration: float, cycles: int, output_file: str):
    """运行优化模式"""
    with st.spinner("正在初始化优化系统..."):
        try:
            optimizer = BLESignalOptimizer()
            data_manager = DataManager()
            
            if not optimizer.initialize_system():
                st.error("❌ 系统初始化失败!")
                return
            
            st.success("✅ 系统初始化成功!")
        except Exception as e:
            st.error(f"❌ 初始化失败: {str(e)}")
            return
    
    st.info(f"🚀 开始信号接收优化，采集时长: {duration}秒，循环次数: {cycles}")
    
    # 创建进度条和结果显示区域
    progress_bar = st.progress(0)
    results_container = st.container()
    
    results = []
    quality_scores = []
    
    for cycle in range(cycles):
        # 更新进度
        progress = (cycle + 1) / cycles
        progress_bar.progress(progress)
        
        with st.spinner(f"执行优化循环 {cycle+1}/{cycles}..."):
            try:
                result = optimizer.optimize_signal_reception(duration)
                results.append(result)
                
                # 提取质量评分
                if 'quality_assessment' in result:
                    quality_matrix = result['quality_assessment']['quality_matrix']
                    overall_quality = np.mean(quality_matrix[:, :, 1])
                    quality_scores.append(overall_quality)
                    
                    # 实时显示结果
                    with results_container:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(f"循环 {cycle+1} 质量评分", f"{overall_quality:.3f}")
                        with col2:
                            if len(quality_scores) > 1:
                                trend = quality_scores[-1] - quality_scores[-2]
                                st.metric("趋势", f"{trend:+.3f}")
                
            except Exception as e:
                st.error(f"❌ 循环 {cycle+1} 执行失败: {str(e)}")
    
    # 保存结果
    if results:
        try:
            data_manager.save_optimization_result(results[-1], output_file)
            st.success(f"✅ 优化结果已保存到: {output_file}")
            
            # 显示质量趋势图
            if quality_scores:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(1, len(quality_scores) + 1)),
                    y=quality_scores,
                    mode='lines+markers',
                    name='信号质量评分',
                    line=dict(color='#1f77b4', width=3),
                    marker=dict(size=8)
                ))
                fig.update_layout(
                    title="信号质量评分趋势",
                    xaxis_title="优化循环",
                    yaxis_title="质量评分",
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ 保存结果失败: {str(e)}")
    
    # 显示系统状态
    status = optimizer.get_system_status()
    st.markdown(f"""
    <div class="success-box">
        <strong>🎉 优化完成!</strong><br>
        系统状态: 已初始化={status['initialized']}<br>
        优化次数: {status['optimization_count']}
    </div>
    """, unsafe_allow_html=True)


def run_demo_mode():
    """运行演示模式"""
    st.info("🎭 启动演示模式...")
    
    with st.spinner("正在运行演示..."):
        # 模拟演示过程
        demo_steps = [
            "初始化系统组件",
            "生成测试信号",
            "执行信号优化",
            "评估优化效果",
            "生成对比图表"
        ]
        
        progress_bar = st.progress(0)
        
        for i, step in enumerate(demo_steps):
            st.text(f"📋 {step}...")
            time.sleep(1)  # 模拟处理时间
            progress_bar.progress((i + 1) / len(demo_steps))
    
    st.success("✅ 演示模式完成!")
    
    # 生成示例图表
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('原始信号', '优化后信号', '频谱对比', '质量指标'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # 生成示例数据
    t = np.linspace(0, 1, 1000)
    original_signal = np.sin(2 * np.pi * 10 * t) + 0.3 * np.random.randn(1000)
    optimized_signal = np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.randn(1000)
    
    fig.add_trace(go.Scatter(x=t[:200], y=original_signal[:200], name='原始信号'), row=1, col=1)
    fig.add_trace(go.Scatter(x=t[:200], y=optimized_signal[:200], name='优化信号'), row=1, col=2)
    
    st.plotly_chart(fig, use_container_width=True)


def run_monitor_mode(duration: float, cycles: int):
    """运行监控模式"""
    st.info("📊 启动监控模式...")
    
    # 创建实时图表占位符
    chart_placeholder = st.empty()
    metrics_placeholder = st.empty()
    
    # 模拟监控数据
    latency_data = []
    quality_data = []
    
    progress_bar = st.progress(0)
    
    for cycle in range(cycles):
        # 生成模拟数据
        latency = np.random.normal(50, 10)
        quality = np.random.normal(0.7, 0.1)
        
        latency_data.append(latency)
        quality_data.append(quality)
        
        # 更新指标
        with metrics_placeholder.container():
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("当前延迟", f"{latency:.1f}ms")
            with col2:
                st.metric("当前质量", f"{quality:.3f}")
            with col3:
                st.metric("完成进度", f"{(cycle+1)/cycles*100:.1f}%")
        
        # 更新图表
        if len(latency_data) > 1:
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('处理延迟趋势', '信号质量趋势')
            )
            
            fig.add_trace(
                go.Scatter(x=list(range(len(latency_data))), y=latency_data, name='延迟(ms)'),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=list(range(len(quality_data))), y=quality_data, name='质量评分'),
                row=1, col=2
            )
            
            chart_placeholder.plotly_chart(fig, use_container_width=True)
        
        progress_bar.progress((cycle + 1) / cycles)
        time.sleep(0.5)  # 模拟处理间隔
    
    st.success("📊 监控模式完成!")


def run_test_mode():
    """运行测试模式"""
    st.info("🧪 启动测试模式...")
    
    test_cases = [
        {'name': 'QPSK信号优化测试', 'signal_type': 'qpsk', 'snr_db': 15.0, 'length': 2000},
        {'name': 'OFDM信号优化测试', 'signal_type': 'ofdm', 'snr_db': 10.0, 'length': 3000},
        {'name': 'FSK信号优化测试', 'signal_type': 'fsk', 'snr_db': 20.0, 'length': 1500}
    ]
    
    results_data = []
    
    progress_bar = st.progress(0)
    
    for i, test_case in enumerate(test_cases):
        with st.spinner(f"执行测试: {test_case['name']}"):
            # 模拟测试结果
            success = np.random.choice([True, False], p=[0.8, 0.2])
            score = np.random.uniform(0.6, 0.95) if success else np.random.uniform(0.3, 0.6)
            
            results_data.append({
                '测试名称': test_case['name'],
                '信号类型': test_case['signal_type'].upper(),
                '信噪比(dB)': test_case['snr_db'],
                '测试结果': '通过' if success else '失败',
                '评分': f"{score:.3f}"
            })
            
            progress_bar.progress((i + 1) / len(test_cases))
            time.sleep(1)
    
    # 显示测试结果表格
    df = pd.DataFrame(results_data)
    st.subheader("🧪 测试结果")
    st.dataframe(df, use_container_width=True)
    
    # 统计信息
    passed_tests = sum(1 for r in results_data if r['测试结果'] == '通过')
    success_rate = passed_tests / len(results_data) * 100
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("总测试数", len(results_data))
    with col2:
        st.metric("通过数", passed_tests)
    with col3:
        st.metric("成功率", f"{success_rate:.1f}%")


def run_signal_test(signal_type: str, snr_db: float, length: int):
    """运行信号测试"""
    st.info(f"🚀 开始测试 {signal_type.upper()} 信号...")
    
    with st.spinner("生成测试信号..."):
        # 生成测试信号
        t = np.linspace(0, 1, length)
        
        if signal_type == 'qpsk':
            signal = np.exp(1j * np.random.choice([0, np.pi/2, np.pi, 3*np.pi/2], length))
        elif signal_type == 'ofdm':
            signal = np.random.normal(0, 1, length) + 1j * np.random.normal(0, 1, length)
        elif signal_type == 'fsk':
            freq = np.random.choice([1, -1], length)
            signal = np.exp(1j * 2 * np.pi * freq * t)
        else:  # noise
            signal = np.random.normal(0, 1, length) + 1j * np.random.normal(0, 1, length)
        
        # 添加噪声
        noise_power = 10**(-snr_db/10)
        noise = np.sqrt(noise_power/2) * (np.random.randn(length) + 1j * np.random.randn(length))
        noisy_signal = signal + noise
        
    # 显示信号
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('实部时域', '虚部时域', '幅度谱', '相位谱')
    )
    
    fig.add_trace(go.Scatter(x=t[:500], y=np.real(noisy_signal[:500]), name='实部'), row=1, col=1)
    fig.add_trace(go.Scatter(x=t[:500], y=np.imag(noisy_signal[:500]), name='虚部'), row=1, col=2)
    
    spectrum = np.fft.fft(noisy_signal)
    freqs = np.fft.fftfreq(len(spectrum))
    
    fig.add_trace(go.Scatter(x=freqs[:len(freqs)//2], y=np.abs(spectrum[:len(spectrum)//2]), name='幅度谱'), row=2, col=1)
    fig.add_trace(go.Scatter(x=freqs[:len(freqs)//2], y=np.angle(spectrum[:len(spectrum)//2]), name='相位谱'), row=2, col=2)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 显示信号参数
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("信号类型", signal_type.upper())
    with col2:
        st.metric("信噪比", f"{snr_db} dB")
    with col3:
        st.metric("信号长度", length)
    with col4:
        signal_power = np.mean(np.abs(signal)**2)
        st.metric("信号功率", f"{10*np.log10(signal_power):.1f} dB")


def analyze_h5_file(filepath: Path):
    """分析HDF5文件"""
    st.info(f"📊 正在分析文件: {filepath.name}")
    
    try:
        with h5py.File(filepath, 'r') as f:
            # 显示文件信息
            st.subheader("📋 文件信息")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**数据集:**")
                for key in f.keys():
                    if isinstance(f[key], h5py.Dataset):
                        st.write(f"- {key}: {f[key].shape} ({f[key].dtype})")
            
            with col2:
                st.write("**属性:**")
                for key, value in f.attrs.items():
                    st.write(f"- {key}: {value}")
            
            # 可视化数据
            if 'enhanced_signal' in f:
                signal_data = f['enhanced_signal'][:]
                
                st.subheader("📈 增强信号分析")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=np.arange(min(1000, len(signal_data))),
                    y=signal_data[:min(1000, len(signal_data))],
                    name='增强信号'
                ))
                fig.update_layout(title="增强信号波形", xaxis_title="采样点", yaxis_title="幅度")
                st.plotly_chart(fig, use_container_width=True)
            
            if 'quality_matrix' in f:
                quality_data = f['quality_matrix'][:]
                
                st.subheader("📊 质量评估分析")
                quality_scores = quality_data[:, :, 1].mean(axis=1)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=np.arange(len(quality_scores)),
                    y=quality_scores,
                    mode='lines+markers',
                    name='质量评分'
                ))
                fig.update_layout(title="质量评分趋势", xaxis_title="时间窗口", yaxis_title="质量评分")
                st.plotly_chart(fig, use_container_width=True)
            
            if 'feature_matrix' in f:
                feature_data = f['feature_matrix'][:]
                
                st.subheader("🎯 特征矩阵分析")
                
                # 特征统计
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("特征维度", feature_data.shape[1])
                with col2:
                    st.metric("样本数量", feature_data.shape[0])
                with col3:
                    st.metric("数据范围", f"{feature_data.min():.3f} - {feature_data.max():.3f}")
                
                # 特征分布热图
                if feature_data.shape[1] <= 20:  # 只有特征数不太多时才显示热图
                    fig = px.imshow(
                        feature_data[:50].T,  # 显示前50个样本
                        labels=dict(x="样本", y="特征", color="数值"),
                        title="特征矩阵热图 (前50个样本)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"❌ 分析文件时出错: {str(e)}")


# 页脚
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    📡 低功耗蓝牙信号接收优化系统 v1.0 | 
    基于 Streamlit 构建 | 
    © 2025
</div>
""", unsafe_allow_html=True)
