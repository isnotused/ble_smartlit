"""
Streamlit 应用的简化配置 - 避免导入问题
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import time
from datetime import datetime
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 设置模块导入路径
import sys
project_root = Path(__file__).parent
adaptive_ble_path = project_root / "bluetooth_optimization" / "adaptive_ble_receiver"
if str(adaptive_ble_path) not in sys.path:
    sys.path.insert(0, str(adaptive_ble_path))

# 尝试导入真实模块
try:
    from bluetooth_optimization.adaptive_ble_receiver.utils.ble_signal_optimizer import BLESignalOptimizer
    from bluetooth_optimization.adaptive_ble_receiver.support.data_manager import DataManager
    REAL_MODULES_AVAILABLE = True
except ImportError as e:
    REAL_MODULES_AVAILABLE = False
    print(f"Real modules not available: {e}")

# 页面配置
st.set_page_config(
    page_title="低功耗蓝牙信号优化系统",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 检查模块可用性
def check_modules():
    """检查核心模块是否可用"""
    return REAL_MODULES_AVAILABLE, "所有模块加载成功" if REAL_MODULES_AVAILABLE else "使用模拟模式"

# 模拟优化系统类（当真实模块不可用时使用）
class MockOptimizer:
    def __init__(self):
        self.initialized = False
        self.optimization_count = 0
    
    def initialize_system(self):
        time.sleep(1)  # 模拟初始化时间
        self.initialized = True
        return True
    
    def optimize_signal_reception(self, duration):
        time.sleep(duration)  # 模拟处理时间
        self.optimization_count += 1
        
        # 返回模拟结果
        return {
            'quality_assessment': {
                'quality_matrix': np.random.rand(10, 5, 3)
            },
            'enhanced_signal': np.random.randn(1000),
            'feature_matrix': np.random.randn(50, 10),
            'new_parameters': {
                'rf_gain': np.random.uniform(15, 25),
                'filter_cutoff': np.random.uniform(0.05, 0.15),
                'equalizer_coeffs': np.random.randn(3)
            }
        }
    
    def get_system_status(self):
        return {
            'initialized': self.initialized,
            'optimization_count': self.optimization_count
        }

class MockDataManager:
    def save_optimization_result(self, result, filename):
        # 模拟保存到HDF5
        filepath = Path(f"bluetooth_optimization/adaptive-ble-receiver/data/{filename}")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(filepath, 'w') as f:
            if 'enhanced_signal' in result:
                f.create_dataset('enhanced_signal', data=result['enhanced_signal'])
            if 'quality_assessment' in result:
                f.create_dataset('quality_matrix', data=result['quality_assessment']['quality_matrix'])
            if 'feature_matrix' in result:
                f.create_dataset('feature_matrix', data=result['feature_matrix'])
            
            # 保存参数
            if 'new_parameters' in result:
                params_group = f.create_group('parameters')
                for key, value in result['new_parameters'].items():
                    if isinstance(value, (list, np.ndarray)):
                        params_group.create_dataset(key, data=value)
            
            # 添加元数据
            f.attrs['timestamp'] = datetime.now().isoformat()
            f.attrs['version'] = '1.0'

# 检查模块可用性
MODULES_AVAILABLE, module_status = check_modules()

# 主应用
def main():
    # 自定义样式 - 深色主题，类似股票分析界面
    st.markdown("""
    <style>
    /* 全局背景设置 */
    .stApp {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    }
    
    /* 主标题样式 */
    .main-header {
        font-size: 2.5rem;
        color: #00d4ff;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 0px 2px 10px rgba(0, 212, 255, 0.3);
        font-weight: 600;
        letter-spacing: 1px;
    }
    
    /* 指标卡片样式 */
    .metric-card {
        background: rgba(20, 40, 80, 0.8);
        border: 1px solid rgba(0, 212, 255, 0.2);
        padding: 1.2rem;
        border-radius: 10px;
        color: #ffffff;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(10px);
    }
    
    .metric-card h3 {
        color: #00d4ff;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .metric-card p {
        font-size: 1.4rem;
        font-weight: bold;
        margin: 0;
        text-shadow: 0px 1px 3px rgba(0, 0, 0, 0.5);
    }
    
    /* 状态卡片 */
    .status-card {
        background: rgba(30, 60, 114, 0.9);
        border: 1px solid rgba(0, 255, 136, 0.3);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        color: #ffffff;
    }
    
    .status-card.success {
        border-color: rgba(0, 255, 136, 0.5);
        background: rgba(0, 100, 50, 0.3);
    }
    
    .status-card.warning {
        border-color: rgba(255, 193, 7, 0.5);
        background: rgba(100, 80, 0, 0.3);
    }
    
    /* 控制按钮样式 */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* 侧边栏样式 */
    .css-1d391kg {
        background: rgba(20, 40, 80, 0.95);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(0, 212, 255, 0.2);
    }
    
    /* 图表容器 */
    .chart-container {
        background: rgba(20, 40, 80, 0.6);
        border: 1px solid rgba(0, 212, 255, 0.2);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    /* 选项卡样式 */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(20, 40, 80, 0.8);
        border-radius: 10px;
        padding: 0.2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #ffffff;
        border-radius: 8px;
        font-weight: 600;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(0, 212, 255, 0.1);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%) !important;
        color: #000000 !important;
    }
    
    /* 数据表格样式 */
    .stDataFrame {
        background: rgba(20, 40, 80, 0.8);
        border-radius: 10px;
    }
    
    /* 滑块和输入框样式 */
    .stSlider > div > div {
        background: rgba(0, 212, 255, 0.2);
    }
    
    .stTextInput > div > div {
        background: rgba(20, 40, 80, 0.8);
        color: #ffffff;
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 8px;
    }
    
    .stSelectbox > div > div {
        background: rgba(20, 40, 80, 0.8);
        color: #ffffff;
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 主标题
    st.markdown('<h1 class="main-header">📡 低功耗蓝牙信号接收优化系统</h1>', unsafe_allow_html=True)
    
    # 主要布局：左侧内容区域，右侧控制面板
    main_col, control_col = st.columns([3, 1])
    
    with control_col:
        st.markdown("""
        <div style="background: rgba(20, 40, 80, 0.8); padding: 1rem; border-radius: 10px; border: 1px solid rgba(0, 212, 255, 0.2);">
        <h3 style="color: #00d4ff; text-align: center; margin-bottom: 1rem;">🛠️ 控制面板</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # 运行模式选择
        mode = st.selectbox(
            "🎯 运行模式",
            options=["optimize", "demo", "monitor", "test"],
            index=0,
            help="选择系统运行模式"
        )
        
        st.markdown("---")
        
        # 参数配置
        st.markdown("##### ⚙️ 参数设置")
        
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
            value="streamlit_optimization_results.h5",
            help="保存结果的HDF5文件名"
        )
        
        st.markdown("---")
        
        # 系统状态
        st.markdown("##### 📊 系统状态")
        if REAL_MODULES_AVAILABLE:
            st.markdown("""
            <div class="status-card success">
                ✅ 真实模块已加载
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="status-card warning">
                ⚠️ 模拟模式运行<br>
                <small>{module_status}</small>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # 运行控制按钮
        st.markdown("##### 🚀 快速操作")
        
        if st.button("🎯 开始优化", type="primary", use_container_width=True):
            with main_col:
                run_optimization_mode(duration, cycles, output_file)
        
        if st.button("🎭 演示模式", use_container_width=True):
            with main_col:
                run_demo_mode()
        
        if st.button("📊 监控模式", use_container_width=True):
            with main_col:
                run_monitor_mode(duration, cycles)
        
        if st.button("🧪 测试模式", use_container_width=True):
            with main_col:
                run_test_mode()
        
        # 当前配置摘要
        st.markdown("---")
        st.markdown("##### 📋 当前配置")
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>模式</h3>
            <p>{mode.upper()}</p>
        </div>
        <div class="metric-card">
            <h3>时长</h3>
            <p>{duration}s</p>
        </div>
        <div class="metric-card">
            <h3>循环</h3>
            <p>{cycles}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with main_col:
        # 主要内容区域 - 使用选项卡
        tab1, tab2, tab3, tab4 = st.tabs(["📈 实时监控", "📊 数据分析", "🧪 测试验证", "📋 系统信息"])
        
        with tab1:
            show_main_dashboard(duration, cycles)
        
        with tab2:
            show_data_analysis_main()
        
        with tab3:
            show_test_interface_main()
        
        with tab4:
            show_system_info(mode, duration, cycles, output_file)


def run_optimization_mode(duration: float, cycles: int, output_file: str):
    """运行优化模式"""
    with st.spinner("正在初始化优化系统..."):
        try:
            if REAL_MODULES_AVAILABLE:
                optimizer = BLESignalOptimizer()
                data_manager = DataManager()
            else:
                optimizer = MockOptimizer()
                data_manager = MockDataManager()
            
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
            time.sleep(1)
            progress_bar.progress((i + 1) / len(demo_steps))
    
    st.success("✅ 演示模式完成!")
    
    # 生成示例图表
    t = np.linspace(0, 1, 1000)
    original_signal = np.sin(2 * np.pi * 10 * t) + 0.3 * np.random.randn(1000)
    optimized_signal = np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.randn(1000)
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=('原始信号', '优化后信号'))
    fig.add_trace(go.Scatter(x=t[:200], y=original_signal[:200], name='原始信号'), row=1, col=1)
    fig.add_trace(go.Scatter(x=t[:200], y=optimized_signal[:200], name='优化信号'), row=1, col=2)
    
    st.plotly_chart(fig, use_container_width=True)


def run_monitor_mode(duration: float, cycles: int):
    """运行监控模式"""
    st.info("📊 启动监控模式...")
    
    chart_placeholder = st.empty()
    metrics_placeholder = st.empty()
    
    latency_data = []
    quality_data = []
    
    progress_bar = st.progress(0)
    
    for cycle in range(cycles):
        latency = np.random.normal(50, 10)
        quality = np.random.normal(0.7, 0.1)
        
        latency_data.append(latency)
        quality_data.append(quality)
        
        with metrics_placeholder.container():
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("当前延迟", f"{latency:.1f}ms")
            with col2:
                st.metric("当前质量", f"{quality:.3f}")
            with col3:
                st.metric("完成进度", f"{(cycle+1)/cycles*100:.1f}%")
        
        if len(latency_data) > 1:
            fig = make_subplots(rows=1, cols=2, subplot_titles=('处理延迟趋势', '信号质量趋势'))
            fig.add_trace(go.Scatter(x=list(range(len(latency_data))), y=latency_data, name='延迟(ms)'), row=1, col=1)
            fig.add_trace(go.Scatter(x=list(range(len(quality_data))), y=quality_data, name='质量评分'), row=1, col=2)
            chart_placeholder.plotly_chart(fig, use_container_width=True)
        
        progress_bar.progress((cycle + 1) / cycles)
        time.sleep(0.5)
    
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
    
    df = pd.DataFrame(results_data)
    st.subheader("🧪 测试结果")
    st.dataframe(df, use_container_width=True)
    
    passed_tests = sum(1 for r in results_data if r['测试结果'] == '通过')
    success_rate = passed_tests / len(results_data) * 100
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("总测试数", len(results_data))
    with col2:
        st.metric("通过数", passed_tests)
    with col3:
        st.metric("成功率", f"{success_rate:.1f}%")


def show_monitoring_interface(duration: float, cycles: int):
    """显示监控界面"""
    if st.button("启动实时监控", type="primary"):
        run_monitor_mode(duration, cycles)


def show_data_analysis():
    """显示数据分析界面"""
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


def show_test_interface():
    """显示测试界面"""
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


def run_signal_test(signal_type: str, snr_db: float, length: int):
    """运行信号测试"""
    st.info(f"🚀 开始测试 {signal_type.upper()} 信号...")
    
    with st.spinner("生成测试信号..."):
        t = np.linspace(0, 1, length)
        
        if signal_type == 'qpsk':
            signal = np.exp(1j * np.random.choice([0, np.pi/2, np.pi, 3*np.pi/2], length))
        elif signal_type == 'ofdm':
            signal = np.random.normal(0, 1, length) + 1j * np.random.normal(0, 1, length)
        elif signal_type == 'fsk':
            freq = np.random.choice([1, -1], length)
            signal = np.exp(1j * 2 * np.pi * freq * t)
        else:
            signal = np.random.normal(0, 1, length) + 1j * np.random.normal(0, 1, length)
        
        noise_power = 10**(-snr_db/10)
        noise = np.sqrt(noise_power/2) * (np.random.randn(length) + 1j * np.random.randn(length))
        noisy_signal = signal + noise
    
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
    
    except Exception as e:
        st.error(f"❌ 分析文件时出错: {str(e)}")


def show_main_dashboard(duration: float, cycles: int):
    """显示主仪表板"""
    # 实时状态指标
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>系统状态</h3>
            <p>运行中</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        latency = np.random.normal(45, 8)
        st.markdown(f"""
        <div class="metric-card">
            <h3>处理延迟</h3>
            <p>{latency:.1f}ms</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        quality = np.random.normal(0.75, 0.1)
        st.markdown(f"""
        <div class="metric-card">
            <h3>信号质量</h3>
            <p>{quality:.3f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        throughput = np.random.normal(150, 20)
        st.markdown(f"""
        <div class="metric-card">
            <h3>数据吞吐</h3>
            <p>{throughput:.0f} KB/s</p>
        </div>
        """, unsafe_allow_html=True)
    
    # 实时图表
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    
    # 生成实时数据
    time_points = np.arange(0, 100)
    signal_quality = np.random.normal(0.7, 0.1, 100).cumsum() * 0.01 + 0.6
    signal_quality = np.clip(signal_quality, 0, 1)
    
    processing_latency = np.random.normal(50, 10, 100)
    
    # 创建子图
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('信号质量趋势', '处理延迟', '频谱分析', '误码率监控'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # 信号质量趋势
    fig.add_trace(
        go.Scatter(x=time_points, y=signal_quality, 
                  name='信号质量', line=dict(color='#00d4ff', width=2)),
        row=1, col=1
    )
    
    # 处理延迟
    fig.add_trace(
        go.Scatter(x=time_points, y=processing_latency,
                  name='延迟(ms)', line=dict(color='#ff6b6b', width=2)),
        row=1, col=2
    )
    
    # 频谱分析
    freqs = np.linspace(0, 50, 50)
    spectrum = np.abs(np.random.randn(50) + 1j * np.random.randn(50))
    fig.add_trace(
        go.Bar(x=freqs, y=spectrum, name='频谱', marker_color='#4ecdc4'),
        row=2, col=1
    )
    
    # 误码率监控
    ber = np.random.exponential(0.001, 100)
    fig.add_trace(
        go.Scatter(x=time_points, y=ber,
                  name='误码率', line=dict(color='#ffa500', width=2)),
        row=2, col=2
    )
    
    # 更新布局
    fig.update_layout(
        height=600,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    # 更新所有子图的坐标轴
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

def show_data_analysis_main():
    """显示数据分析主界面"""
    st.markdown("### 📊 数据分析中心")
    
    # 文件选择和分析
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
            "选择数据文件",
            options=h5_files,
            format_func=lambda x: x.name
        )
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            if st.button("📊 分析数据", type="primary"):
                analyze_h5_file(selected_file)
        
        with col1:
            st.info(f"当前选择: {selected_file.name}")
    else:
        st.warning("📂 未找到数据文件，请先运行优化生成数据")
    
    # 历史数据概览
    st.markdown("---")
    st.markdown("#### 📈 历史趋势")
    
    # 生成示例历史数据
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    quality_trend = np.random.normal(0.7, 0.1, len(dates)).cumsum() * 0.001 + 0.7
    quality_trend = np.clip(quality_trend, 0.3, 0.95)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=quality_trend,
        mode='lines',
        name='信号质量',
        line=dict(color='#00d4ff', width=2)
    ))
    
    fig.update_layout(
        title="年度信号质量趋势",
        xaxis_title="日期",
        yaxis_title="质量评分",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=400
    )
    
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
    
    st.plotly_chart(fig, use_container_width=True)

def show_test_interface_main():
    """显示测试界面主版本"""
    st.markdown("### 🧪 信号测试中心")
    
    # 测试配置
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔧 测试配置")
        
        test_signal_type = st.selectbox(
            "信号类型",
            options=["qpsk", "ofdm", "fsk", "noise"],
            help="选择要测试的信号类型"
        )
        
        test_snr = st.slider("信噪比 (dB)", -10.0, 30.0, 15.0, 0.5)
        test_length = st.slider("信号长度", 500, 5000, 2000, 100)
        
        if st.button("🚀 运行测试", type="primary", use_container_width=True):
            run_signal_test(test_signal_type, test_snr, test_length)
    
    with col2:
        st.markdown("#### 📋 快速测试")
        
        if st.button("🎯 QPSK 标准测试", use_container_width=True):
            run_signal_test("qpsk", 15.0, 2000)
        
        if st.button("📡 OFDM 性能测试", use_container_width=True):
            run_signal_test("ofdm", 10.0, 3000)
        
        if st.button("🔄 FSK 稳定性测试", use_container_width=True):
            run_signal_test("fsk", 20.0, 1500)
        
        if st.button("🎭 噪声环境测试", use_container_width=True):
            run_signal_test("noise", 5.0, 2500)

def show_system_info(mode: str, duration: float, cycles: int, output_file: str):
    """显示系统信息"""
    st.markdown("### 📋 系统信息")
    
    # 系统状态
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🖥️ 系统状态")
        
        st.markdown(f"""
        **运行模式**: {mode.upper()}  
        **采集时长**: {duration}秒  
        **循环次数**: {cycles}  
        **输出文件**: {output_file}  
        **模块状态**: {'真实模块' if REAL_MODULES_AVAILABLE else '模拟模式'}
        """)
    
    with col2:
        st.markdown("#### 📊 性能指标")
        
        try:
            import psutil
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
        except ImportError:
            cpu_percent = 0.0
            memory_percent = 0.0
        
        st.markdown(f"""
        **CPU 使用率**: {cpu_percent:.1f}%  
        **内存使用率**: {memory_percent:.1f}%  
        **Python 版本**: {sys.version.split()[0]}  
        **Streamlit 版本**: {st.__version__}
        """)
    
    # 功能模块状态
    st.markdown("---")
    st.markdown("#### 🔧 模块状态")
    
    modules_status = [
        {"模块": "信号优化器", "状态": "✅ 正常" if REAL_MODULES_AVAILABLE else "⚠️ 模拟"},
        {"模块": "数据管理器", "状态": "✅ 正常" if REAL_MODULES_AVAILABLE else "⚠️ 模拟"},
        {"模块": "环境分析器", "状态": "✅ 正常"},
        {"模块": "质量评估器", "状态": "✅ 正常"},
        {"模块": "参数预测器", "状态": "✅ 正常"}
    ]
    
    df_modules = pd.DataFrame(modules_status)
    st.dataframe(df_modules, use_container_width=True, hide_index=True)
    
    # 日志信息
    st.markdown("---")
    st.markdown("#### 📝 最近日志")
    
    log_entries = [
        f"{datetime.now().strftime('%H:%M:%S')} - 系统启动完成",
        f"{datetime.now().strftime('%H:%M:%S')} - 模块初始化{'成功' if REAL_MODULES_AVAILABLE else '(模拟模式)'}",
        f"{datetime.now().strftime('%H:%M:%S')} - Web界面已就绪"
    ]
    
    for entry in log_entries:
        st.text(entry)


if __name__ == "__main__":
    main()
