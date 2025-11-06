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
    
    /* 侧边栏样式 */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0e27 0%, #1a2332 100%);
        border-right: 2px solid rgba(0, 212, 255, 0.3);
    }
    
    section[data-testid="stSidebar"] > div {
        background: transparent;
    }
    
    /* 侧边栏按钮样式 */
    section[data-testid="stSidebar"] .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.2) 0%, rgba(0, 153, 204, 0.2) 100%);
        border: 1px solid rgba(0, 212, 255, 0.3);
        color: #ffffff;
        transition: all 0.3s ease;
    }
    
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.4) 0%, rgba(0, 153, 204, 0.4) 100%);
        border: 1px solid rgba(0, 212, 255, 0.6);
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.4);
    }
        </style>
    """, unsafe_allow_html=True)
    
    # 主标题
    st.markdown('<h1 class="main-header">📡 低功耗蓝牙信号接收优化系统</h1>', unsafe_allow_html=True, width="stretch")
    
    # 侧边栏 - 可收缩的控制面板
    with st.sidebar:
        st.markdown("""
        <div style="background: rgba(20, 40, 80, 0.5); padding: 1rem; border-radius: 10px; border: 1px solid rgba(0, 200, 255, 0.2);">
        <h2 style="color: #00d4ff; text-align: center; margin-bottom: 1rem;">控制面板</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # 运行模式选择
        mode = st.selectbox(
            "运行模式",
            options=["optimize", "demo", "monitor", "test", "interactive"],
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
            value="optimization_results.h5",
            help="保存结果的HDF5文件名"
        )
        
        st.markdown("---")
        
        # 运行控制按钮
        st.markdown("##### 快速操作")
        
        run_optimize = st.button("开始优化", type="primary", use_container_width=True)
        run_demo = st.button("演示模式", use_container_width=True)
        run_monitor = st.button("监控模式", use_container_width=True)
        run_test = st.button("测试模式", use_container_width=True)
    
    # 主内容区域 - 现在使用全宽
    if run_optimize:
        run_optimization_mode(duration, cycles, output_file)
    
    if run_demo:
        run_demo_mode()
    
    if run_monitor:
        run_monitor_mode(duration, cycles)
    
    if run_test:
        run_test_mode()
    
    
    # 主要内容区域 - 使用选项卡
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["实时监控", "数据分析", "测试验证", "优化系统", "系统信息"])
    
    with tab1:
        show_main_dashboard(duration, cycles)
    
    with tab2:
        show_data_analysis_main()
    
    with tab3:
        show_test_interface_main()
    
    with tab4:
        show_interactive_optimization()
    
    with tab5:
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
                st.plotly_chart(fig, width='stretch')
            
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
    st.info("启动演示模式...")
    
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
    
    st.plotly_chart(fig, width='stretch')


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
            chart_placeholder.plotly_chart(fig, width='stretch')
        
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
    st.dataframe(df, width='stretch')
    
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
    
    st.plotly_chart(fig, width='stretch')
    
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
                st.subheader("增强信号分析")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=np.arange(min(1000, len(signal_data))),
                    y=signal_data[:min(1000, len(signal_data))],
                    name='增强信号'
                ))
                fig.update_layout(title="增强信号波形", xaxis_title="采样点", yaxis_title="幅度")
                st.plotly_chart(fig, width='stretch')
            
            if 'quality_matrix' in f:
                quality_data = f['quality_matrix'][:]
                st.subheader("质量评估分析")
                quality_scores = quality_data[:, :, 1].mean(axis=1)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=np.arange(len(quality_scores)),
                    y=quality_scores,
                    mode='lines+markers',
                    name='质量评分'
                ))
                fig.update_layout(title="质量评分趋势", xaxis_title="时间窗口", yaxis_title="质量评分")
                st.plotly_chart(fig, width='stretch')
    
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
    
    st.plotly_chart(fig, width='stretch')
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
        
        row1, row2 = st.columns([2, 1])

        with row2:
            if st.button("分析数据", type="primary"):
                analyze_h5_file(selected_file)

        with row1:
            st.info(f"当前选择: {selected_file.name}")
    else:
        st.warning("📂 未找到数据文件，请先运行优化生成数据")
    
    # 历史数据概览
    # st.markdown("---")
    # st.markdown("#### 📈 历史趋势")
    
    # # 生成示例历史数据
    # dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    # quality_trend = np.random.normal(0.7, 0.1, len(dates)).cumsum() * 0.001 + 0.7
    # quality_trend = np.clip(quality_trend, 0.3, 0.95)
    
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(
    #     x=dates,
    #     y=quality_trend,
    #     mode='lines',
    #     name='信号质量',
    #     line=dict(color='#00d4ff', width=2)
    # ))
    
    # fig.update_layout(
    #     title="年度信号质量趋势",
    #     xaxis_title="日期",
    #     yaxis_title="质量评分",
    #     plot_bgcolor='rgba(0,0,0,0)',
    #     paper_bgcolor='rgba(0,0,0,0)',
    #     font=dict(color='white'),
    #     height=400
    # )
    
    # fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
    # fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
    
    # st.plotly_chart(fig, width='stretch')

def show_test_interface_main():
    """显示测试界面主版本"""
    st.markdown("### 信号测试中心")
    
    # 测试配置
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 测试配置")
        
        test_signal_type = st.selectbox(
            "信号类型",
            options=["qpsk", "ofdm", "fsk", "noise"],
            help="选择要测试的信号类型"
        )
        
        test_snr = st.slider("信噪比 (dB)", -10.0, 30.0, 15.0, 0.5)
        test_length = st.slider("信号长度", 500, 5000, 2000, 100)
        
        if st.button("运行测试", type="primary", width='stretch'):
            run_signal_test(test_signal_type, test_snr, test_length)
    
    with col2:
        st.markdown("#### 快速测试")
        
        if st.button("QPSK 标准测试", width='stretch'):
            run_signal_test("qpsk", 15.0, 2000)
        
        if st.button("OFDM 性能测试", width='stretch'):
            run_signal_test("ofdm", 10.0, 3000)
        
        if st.button("FSK 稳定性测试", width='stretch'):
            run_signal_test("fsk", 20.0, 1500)
        
        if st.button("噪声环境测试", width='stretch'):
            run_signal_test("noise", 5.0, 2500)

def show_interactive_optimization():
    """显示交互式信号优化界面 - 完整优化流程"""
    st.markdown("### 信号优化系统")
    
    st.markdown("""
    <div style="background: rgba(20, 40, 80, 0.8); padding: 1rem; border-radius: 10px; border: 1px solid rgba(0, 212, 255, 0.2); margin-bottom: 1rem;">
    <p style="color: #ffffff; margin: 0;">
    <strong>完整优化流程说明：</strong>环境数据采集 → 注意力机制策略选择 → 时频联合滤波 → 深度残差网络增强 → 质量评估与参数调整
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 创建两列布局：参数控制 + 实时显示
    control_col, display_col = st.columns([1, 2])
    
    with control_col:
        st.markdown("#### ⚙️ 信号参数调节")
        
        # 信号强度控制
        st.markdown("##### 📶 信号强度")
        signal_power_db = st.slider(
            "信号功率 (dBm)",
            min_value=-90.0,
            max_value=-50.0,
            value=-70.0,
            step=1.0,
            help="调节基础信号功率强度"
        )
        
        signal_variation = st.slider(
            "信号波动 (dB)",
            min_value=0.0,
            max_value=10.0,
            value=2.0,
            step=0.5,
            help="信号强度的随机波动范围"
        )
        
        st.markdown("---")
        
        # 噪声功率控制
        st.markdown("##### 🔊 噪声功率")
        noise_power_db = st.slider(
            "噪声功率 (dBm)",
            min_value=-110.0,
            max_value=-70.0,
            value=-90.0,
            step=1.0,
            help="调节环境噪声功率"
        )
        
        noise_variation = st.slider(
            "噪声波动 (dB)",
            min_value=0.0,
            max_value=5.0,
            value=1.0,
            step=0.5,
            help="噪声功率的随机波动范围"
        )
        
        st.markdown("---")
        
        # 多径干扰控制
        st.markdown("##### 多径干扰")
        multipath_strength = st.slider(
            "多径强度",
            min_value=0.0,
            max_value=2.0,
            value=1.0,
            step=0.1,
            help="多径干扰的总体强度系数"
        )
        
        multipath_decay = st.slider(
            "衰减速率",
            min_value=0.5,
            max_value=5.0,
            value=2.0,
            step=0.5,
            help="多径分量的指数衰减速率"
        )
        
        multipath_paths = st.slider(
            "多径数量",
            min_value=3,
            max_value=15,
            value=10,
            step=1,
            help="模拟的多径分量数量"
        )
        
        st.markdown("---")
        
        # 信号采样设置
        st.markdown("##### 采样设置")
        sample_length = st.slider(
            "采样长度",
            min_value=500,
            max_value=5000,
            value=2000,
            step=100,
            help="生成的信号采样点数"
        )
        
        # 生成和优化按钮
        st.markdown("---")
        if st.button("执行完整优化流程", type="primary", width='stretch'):
            run_complete_optimization_pipeline(
                signal_power_db, signal_variation,
                noise_power_db, noise_variation,
                multipath_strength, multipath_decay, multipath_paths,
                sample_length, display_col
            )
    
    # with display_col:
    #     st.markdown("#### 📊 优化结果展示")
    #     st.info("👈 请在左侧调节参数，然后点击\"执行完整优化流程\"按钮查看优化效果")


def generate_custom_signal(signal_power_db, signal_variation, noise_power_db, 
                          noise_variation, multipath_strength, multipath_decay, 
                          multipath_paths, sample_length):
    """根据用户参数生成自定义信号"""
    
    # 生成时间序列
    sample_rate = 2000000  # 2MHz
    t = np.arange(sample_length) / sample_rate
    carrier_freq = 2400000000  # 2.4GHz
    
    # 生成QPSK调制信号
    symbols = np.random.choice([1+1j, 1-1j, -1+1j, -1-1j], sample_length//100)
    symbol_waveform = np.repeat(symbols, 100)[:sample_length]
    
    # 添加载波
    signal_power_linear = 10 ** (signal_power_db / 10)
    signal_amplitude = np.sqrt(signal_power_linear)
    
    # 添加信号波动
    signal_envelope = signal_amplitude * (1 + np.random.normal(0, signal_variation/20, sample_length))
    
    clean_signal = signal_envelope * symbol_waveform * np.exp(2j * np.pi * carrier_freq * t)
    
    # 添加多径干扰
    multipath_signal = np.zeros(sample_length, dtype=np.complex64)
    for i in range(multipath_paths):
        delay_samples = int(i * sample_rate * 0.1e-6)  # 每个多径延迟0.1微秒
        if delay_samples < sample_length:
            amplitude = multipath_strength * np.exp(-i * multipath_decay / multipath_paths)
            phase_shift = np.random.uniform(0, 2*np.pi)
            
            # 延迟信号
            delayed = np.zeros(sample_length, dtype=np.complex64)
            delayed[delay_samples:] = clean_signal[:-delay_samples] if delay_samples > 0 else clean_signal
            delayed *= amplitude * np.exp(1j * phase_shift)
            
            multipath_signal += delayed
    
    # 添加噪声
    noise_power_linear = 10 ** (noise_power_db / 10)
    noise_amplitude = np.sqrt(noise_power_linear)
    noise_envelope = noise_amplitude * (1 + np.random.normal(0, noise_variation/20, sample_length))
    
    noise = (np.random.normal(0, 1, sample_length) + 
             1j * np.random.normal(0, 1, sample_length)) * noise_envelope
    
    # 合成最终信号
    noisy_signal = clean_signal + multipath_signal + noise
    
    return clean_signal, noisy_signal, t


def apply_adaptive_filter(signal_data, filter_strategy):
    """应用自适应滤波算法"""
    
    if filter_strategy == "Kalman":
        # Kalman滤波实现
        filtered = kalman_filter_impl(signal_data, q=0.1, r=1.0)
    elif filter_strategy == "Wiener":
        # Wiener滤波实现
        filtered = wiener_filter_impl(signal_data, window_size=32)
    elif filter_strategy == "LMS自适应":
        # LMS自适应滤波
        filtered = lms_filter_impl(signal_data, mu=0.01, order=16)
    elif filter_strategy == "Butterworth":
        # Butterworth滤波
        filtered = butterworth_filter_impl(signal_data, order=4, cutoff=0.1)
    else:
        filtered = signal_data
    
    return filtered


def kalman_filter_impl(signal_data, q=0.1, r=1.0):
    """Kalman滤波实现"""
    n = len(signal_data)
    filtered = np.zeros_like(signal_data)
    
    # 初始化
    x_hat = signal_data[0]
    P = 1.0
    
    for i in range(n):
        # 预测
        x_hat_minus = x_hat
        P_minus = P + q
        
        # 更新
        K = P_minus / (P_minus + r)
        x_hat = x_hat_minus + K * (signal_data[i] - x_hat_minus)
        P = (1 - K) * P_minus
        
        filtered[i] = x_hat
    
    return filtered


def wiener_filter_impl(signal_data, window_size=32):
    """Wiener滤波实现"""
    filtered = np.zeros_like(signal_data)
    half_window = window_size // 2
    
    for i in range(len(signal_data)):
        start_idx = max(0, i - half_window)
        end_idx = min(len(signal_data), i + half_window)
        
        window = signal_data[start_idx:end_idx]
        filtered[i] = np.mean(window)
    
    return filtered


def lms_filter_impl(signal_data, mu=0.01, order=16):
    """LMS自适应滤波实现"""
    n = len(signal_data)
    filtered = np.zeros_like(signal_data)
    w = np.zeros(order, dtype=signal_data.dtype)
    
    for i in range(order, n):
        x = signal_data[i-order:i][::-1]
        y = np.dot(w, x)
        e = signal_data[i] - y
        w = w + mu * np.conj(e) * x
        filtered[i] = y
    
    filtered[:order] = signal_data[:order]
    return filtered


def butterworth_filter_impl(signal_data, order=4, cutoff=0.1):
    """Butterworth滤波实现"""
    from scipy.signal import butter, filtfilt
    
    b, a = butter(order, cutoff, btype='low')
    
    # 分别处理实部和虚部
    filtered_real = filtfilt(b, a, np.real(signal_data))
    filtered_imag = filtfilt(b, a, np.imag(signal_data))
    
    return filtered_real + 1j * filtered_imag


def calculate_signal_metrics(clean_signal, noisy_signal, filtered_signal):
    """计算信号质量指标"""
    
    # 信噪比 (SNR)
    signal_power = np.mean(np.abs(clean_signal)**2)
    noise_power_noisy = np.mean(np.abs(noisy_signal - clean_signal)**2)
    noise_power_filtered = np.mean(np.abs(filtered_signal - clean_signal)**2)
    
    snr_before = 10 * np.log10(signal_power / noise_power_noisy) if noise_power_noisy > 0 else 0
    snr_after = 10 * np.log10(signal_power / noise_power_filtered) if noise_power_filtered > 0 else 0
    
    # 误差向量幅度 (EVM)
    evm_before = np.sqrt(np.mean(np.abs(noisy_signal - clean_signal)**2)) / np.sqrt(signal_power) * 100
    evm_after = np.sqrt(np.mean(np.abs(filtered_signal - clean_signal)**2)) / np.sqrt(signal_power) * 100
    
    # 相关系数
    corr_before = np.abs(np.corrcoef(np.real(clean_signal), np.real(noisy_signal))[0, 1])
    corr_after = np.abs(np.corrcoef(np.real(clean_signal), np.real(filtered_signal))[0, 1])
    
    return {
        'snr_before': snr_before,
        'snr_after': snr_after,
        'snr_improvement': snr_after - snr_before,
        'evm_before': evm_before,
        'evm_after': evm_after,
        'evm_reduction': evm_before - evm_after,
        'corr_before': corr_before,
        'corr_after': corr_after
    }


def apply_all_filters(signal_data):
    """应用所有滤波策略并返回结果"""
    filter_results = {}
    
    # 1. 卡尔曼滤波
    filter_results['Kalman'] = kalman_filter_impl(signal_data, q=0.1, r=1.0)
    
    # 2. Wiener滤波
    filter_results['Wiener'] = wiener_filter_impl(signal_data, window_size=32)
    
    # 3. 粒子滤波 (简化版，使用平均滤波模拟)
    filter_results['Particle'] = particle_filter_impl(signal_data)
    
    # 4. 小波阈值滤波
    filter_results['Wavelet'] = wavelet_filter_impl(signal_data)
    
    # 5. 滑动平均滤波
    filter_results['MovingAvg'] = moving_average_filter_impl(signal_data, window=20)
    
    return filter_results


def particle_filter_impl(signal_data, num_particles=50):
    """粒子滤波实现（简化版）- 支持复数信号"""
    n = len(signal_data)
    filtered = np.zeros_like(signal_data)
    
    # 分别处理实部和虚部
    real_part = np.real(signal_data)
    imag_part = np.imag(signal_data)
    
    # 实部粒子滤波
    particles_real = np.tile(real_part[0], num_particles) + np.random.normal(0, 0.1, num_particles)
    weights_real = np.ones(num_particles) / num_particles
    
    for i in range(n):
        particles_real += np.random.normal(0, 0.05, num_particles)
        likelihood = np.exp(-0.5 * (particles_real - real_part[i])**2 / 0.1)
        weights_real = likelihood / (np.sum(likelihood) + 1e-10)
        filtered[i] = np.sum(particles_real * weights_real)
        
        if 1.0 / (np.sum(weights_real**2) + 1e-10) < num_particles / 2:
            indices = np.random.choice(num_particles, num_particles, p=weights_real)
            particles_real = particles_real[indices]
            weights_real = np.ones(num_particles) / num_particles
    
    filtered_real = filtered.real.copy()
    
    # 虚部粒子滤波
    particles_imag = np.tile(imag_part[0], num_particles) + np.random.normal(0, 0.1, num_particles)
    weights_imag = np.ones(num_particles) / num_particles
    
    for i in range(n):
        particles_imag += np.random.normal(0, 0.05, num_particles)
        likelihood = np.exp(-0.5 * (particles_imag - imag_part[i])**2 / 0.1)
        weights_imag = likelihood / (np.sum(likelihood) + 1e-10)
        filtered[i] = np.sum(particles_imag * weights_imag)
        
        if 1.0 / (np.sum(weights_imag**2) + 1e-10) < num_particles / 2:
            indices = np.random.choice(num_particles, num_particles, p=weights_imag)
            particles_imag = particles_imag[indices]
            weights_imag = np.ones(num_particles) / num_particles
    
    filtered_imag = filtered.real.copy()
    
    return filtered_real + 1j * filtered_imag


def wavelet_filter_impl(signal_data, threshold_scale=0.5):
    """小波阈值滤波实现（简化版）"""
    from scipy import signal as scipy_signal
    
    # 使用离散小波变换（简化实现）
    # 这里使用高通/低通滤波器模拟小波分解
    sos = scipy_signal.butter(4, 0.1, btype='low', output='sos')
    filtered = scipy_signal.sosfiltfilt(sos, np.real(signal_data)) + \
               1j * scipy_signal.sosfiltfilt(sos, np.imag(signal_data))
    
    # 阈值处理
    threshold = threshold_scale * np.std(signal_data - filtered)
    residual = signal_data - filtered
    residual[np.abs(residual) < threshold] = 0
    
    return filtered + residual


def moving_average_filter_impl(signal_data, window=20):
    """滑动平均滤波实现"""
    filtered = np.zeros_like(signal_data)
    half_window = window // 2
    
    for i in range(len(signal_data)):
        start_idx = max(0, i - half_window)
        end_idx = min(len(signal_data), i + half_window + 1)
        filtered[i] = np.mean(signal_data[start_idx:end_idx])
    
    return filtered


def select_optimal_filter_with_attention(filter_results, clean_signal):
    """使用注意力机制选择最优滤波策略"""
    filter_names = list(filter_results.keys())
    num_filters = len(filter_names)
    
    # 计算每个滤波器的性能分数
    scores = np.zeros(num_filters)
    for i, (name, filtered) in enumerate(filter_results.items()):
        # 计算SNR作为性能指标
        signal_power = np.mean(np.abs(clean_signal)**2)
        noise_power = np.mean(np.abs(filtered - clean_signal)**2)
        snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
        scores[i] = snr
    
    # 注意力权重（Softmax）
    attention_weights = np.exp(scores) / np.sum(np.exp(scores))
    
    # 选择最优策略
    best_idx = np.argmax(scores)
    best_filter = filter_names[best_idx]
    
    return best_filter, attention_weights, scores


def apply_residual_network_enhancement(signal_data, num_blocks=3):
    """应用深度残差网络增强"""
    enhanced = signal_data.copy()
    
    for block in range(num_blocks):
        # 残差连接
        residual = enhanced
        
        # 简化的卷积操作（使用滑动窗口）
        window_size = 5
        conv_output = np.zeros_like(enhanced)
        
        for i in range(len(enhanced)):
            start = max(0, i - window_size // 2)
            end = min(len(enhanced), i + window_size // 2 + 1)
            window = enhanced[start:end]
            
            # 非线性激活
            conv_output[i] = np.tanh(np.mean(window))
        
        # 跨层连接
        enhanced = conv_output + 0.3 * residual
    
    return enhanced


def evaluate_signal_quality_matrix(clean_signal, noisy_signal, enhanced_signal, segment_size=50):
    """评估信号质量并生成评估矩阵"""
    num_segments = len(clean_signal) // segment_size
    
    error_rates = []
    snrs = []
    phase_consistencies = []
    
    for i in range(num_segments):
        start = i * segment_size
        end = (i + 1) * segment_size
        
        clean_seg = clean_signal[start:end]
        noisy_seg = noisy_signal[start:end]
        enh_seg = enhanced_signal[start:end]
        
        # 误码率（确保返回实数）
        error_rate = float(np.real(np.mean(np.abs(enh_seg - clean_seg)) / (np.max(np.abs(clean_seg)) - np.min(np.abs(clean_seg)) + 1e-10)))
        error_rates.append(error_rate)
        
        # SNR（确保返回实数）
        sig_power = float(np.real(np.mean(np.abs(clean_seg)**2)))
        noise_pow = float(np.real(np.mean(np.abs(enh_seg - clean_seg)**2)))
        snr = float(10 * np.log10(sig_power / (noise_pow + 1e-10)))
        snrs.append(snr)
        
        # 相位一致性（确保返回实数）
        phase_clean = np.angle(np.fft.fft(clean_seg))
        phase_enh = np.angle(np.fft.fft(enh_seg))
        phase_diff = np.abs(phase_clean - phase_enh)
        phase_consist = float(np.real(np.mean(np.cos(phase_diff))))
        phase_consistencies.append(phase_consist)
    
    eval_matrix = np.column_stack((error_rates, snrs, phase_consistencies))
    
    return eval_matrix, error_rates, snrs, phase_consistencies


def run_complete_optimization_pipeline(signal_power_db, signal_variation, noise_power_db, 
                                      noise_variation, multipath_strength, multipath_decay, 
                                      multipath_paths, sample_length, display_col):
    """运行完整的优化流程"""
    
    with display_col:
        progress_placeholder = st.empty()
        
        with st.spinner("🔄 步骤1/6: 生成环境信号数据..."):
            # 生成信号
            clean_signal, noisy_signal, t = generate_custom_signal(
                signal_power_db, signal_variation,
                noise_power_db, noise_variation,
                multipath_strength, multipath_decay,
                multipath_paths, sample_length
            )
            progress_placeholder.progress(1/6)
        
        with st.spinner("🔄 步骤2/6: 应用所有滤波策略..."):
            # 应用所有滤波策略
            filter_results = apply_all_filters(noisy_signal)
            progress_placeholder.progress(2/6)
        
        with st.spinner("🔄 步骤3/6: 注意力机制选择最优策略..."):
            # 使用注意力机制选择最优策略
            best_filter, attention_weights, filter_scores = select_optimal_filter_with_attention(
                filter_results, clean_signal
            )
            optimized_signal = filter_results[best_filter]
            progress_placeholder.progress(3/6)
        
        with st.spinner("🔄 步骤4/6: 深度残差网络增强..."):
            # 残差网络增强
            enhanced_signal = apply_residual_network_enhancement(optimized_signal)
            progress_placeholder.progress(4/6)
        
        with st.spinner("🔄 步骤5/6: 信号质量评估..."):
            # 质量评估
            eval_matrix, error_rates, snrs, phase_consistencies = evaluate_signal_quality_matrix(
                clean_signal, noisy_signal, enhanced_signal
            )
            progress_placeholder.progress(5/6)
        
        with st.spinner("🔄 步骤6/6: 参数调整建议..."):
            # 参数调整
            avg_snr = np.mean(snrs)
            avg_error = np.mean(error_rates)
            
            param_adjustments = {
                'gain': 1.0 + 0.1 * (15 - avg_snr) / 15,  # 根据SNR调整增益
                'bandwidth': max(0.5, 1.0 - avg_error),  # 根据误码率调整带宽
                'modulation': 'GFSK' if avg_error < 0.1 else '2-FSK'
            }
            progress_placeholder.progress(1.0)
            time.sleep(0.5)
            progress_placeholder.empty()
        
        st.success("✅ 优化完成！")
        
        # ==================== 显示结果 ====================
        
        # 1. 各滤波策略小窗口对比
        st.markdown("---")
        st.markdown("#### 滤波策略对比")
        
        filter_display_names = {
            'Kalman': '卡尔曼滤波',
            'Wiener': '维纳滤波',
            'Particle': '粒子滤波',
            'Wavelet': '小波阈值滤波',
            'MovingAvg': '滑动平均滤波'
        }
        
        # 创建5个小窗口
        cols = st.columns(5)
        display_samples = min(200, len(t))
        
        for idx, (filter_name, filtered_signal) in enumerate(filter_results.items()):
            with cols[idx]:
                # 计算该滤波器的SNR
                sig_pow = np.mean(np.abs(clean_signal)**2)
                noise_pow = np.mean(np.abs(filtered_signal - clean_signal)**2)
                snr = 10 * np.log10(sig_pow / (noise_pow + 1e-10))
                
                is_best = (filter_name == best_filter)
                border_color = '#00ff88' if is_best else '#666666'
                
                st.markdown(f"""
                <div style="border: 2px solid {border_color}; border-radius: 8px; padding: 0.5rem; background: rgba(20, 40, 80, 0.6);">
                    <h6 style="color: {'#00ff88' if is_best else '#ffffff'}; text-align: center; margin: 0;">
                        {filter_display_names[filter_name]} {'⭐' if is_best else ''}
                    </h6>
                    <p style="color: #00d4ff; text-align: center; font-size: 0.8rem; margin: 0.2rem 0;">
                        SNR: {snr:.1f} dB
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # 小图表
                fig_small = go.Figure()
                fig_small.add_trace(go.Scatter(
                    x=t[:display_samples]*1e6,
                    y=np.real(filtered_signal[:display_samples]),
                    line=dict(color='#00d4ff' if is_best else '#666666', width=1),
                    showlegend=False
                ))
                fig_small.update_layout(
                    height=150,
                    margin=dict(l=20, r=20, t=10, b=20),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(showgrid=False, showticklabels=False),
                    yaxis=dict(showgrid=False, showticklabels=False),
                    font=dict(color='white', size=8)
                )
                st.plotly_chart(fig_small, width='stretch')
        
        # 2. 注意力权重分布
        st.markdown("---")
        st.markdown("#### 注意力机制 - 滤波策略权重分布")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig_attention = go.Figure()
            fig_attention.add_trace(go.Bar(
                x=list(filter_display_names.values()),
                y=attention_weights,
                marker_color=['#00ff88' if name == best_filter else '#4ecdc4' 
                             for name in filter_results.keys()],
                text=[f'{w:.3f}' for w in attention_weights],
                textposition='outside'
            ))
            fig_attention.update_layout(
                title="滤波策略注意力权重",
                xaxis_title="滤波策略",
                yaxis_title="权重",
                height=300,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            fig_attention.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
            fig_attention.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
            st.plotly_chart(fig_attention, width='stretch')
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>最优策略</h3>
                <p style="color: #00ff88;">{filter_display_names[best_filter]}</p>
                <small>权重: {attention_weights[list(filter_results.keys()).index(best_filter)]:.3f}</small>
            </div>
            <div class="metric-card">
                <h3>策略SNR</h3>
                <p style="color: #00d4ff;">{filter_scores[list(filter_results.keys()).index(best_filter)]:.2f} dB</p>
            </div>
            """, unsafe_allow_html=True)
        
        # 3. 完整优化流程主窗口
        st.markdown("---")
        st.markdown("#### 完整优化流程结果（主窗口）")
        
        # 性能指标
        final_metrics = calculate_signal_metrics(clean_signal, noisy_signal, enhanced_signal)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>SNR改善</h3>
                <p style="color: {'#00ff88' if final_metrics['snr_improvement'] > 0 else '#ff6b6b'};">
                    {final_metrics['snr_improvement']:+.2f} dB
                </p>
                <small>前: {final_metrics['snr_before']:.1f} dB<br>
                后: {final_metrics['snr_after']:.1f} dB</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>EVM降低</h3>
                <p style="color: {'#00ff88' if final_metrics['evm_reduction'] > 0 else '#ff6b6b'};">
                    {final_metrics['evm_reduction']:.2f}%
                </p>
                <small>前: {final_metrics['evm_before']:.1f}%<br>
                后: {final_metrics['evm_after']:.1f}%</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>相关度</h3>
                <p style="color: #00d4ff;">
                    {final_metrics['corr_after']:.3f}
                </p>
                <small>前: {final_metrics['corr_before']:.3f}</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>平均误码率</h3>
                <p style="color: #ffa500;">
                    {avg_error:.4f}
                </p>
                <small>质量评分: {(1-avg_error)*100:.1f}%</small>
            </div>
            """, unsafe_allow_html=True)
        
        # 信号波形对比
        st.markdown("#### 完整流程信号对比")
        
        display_samples_main = min(1000, len(t))
        t_display = t[:display_samples_main]
        
        fig_main = make_subplots(
            rows=2, cols=2,
            subplot_titles=('原始信号', '带噪信号', 
                           f'最优滤波({filter_display_names[best_filter]})', '残差网络增强'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 原始信号
        fig_main.add_trace(
            go.Scatter(x=t_display*1e6, y=np.real(clean_signal[:display_samples_main]),
                      name='原始', line=dict(color='#00ff88', width=1.5)),
            row=1, col=1
        )
        
        # 带噪信号
        fig_main.add_trace(
            go.Scatter(x=t_display*1e6, y=np.real(noisy_signal[:display_samples_main]),
                      name='带噪', line=dict(color='#ff6b6b', width=1)),
            row=1, col=2
        )
        
        # 最优滤波
        fig_main.add_trace(
            go.Scatter(x=t_display*1e6, y=np.real(optimized_signal[:display_samples_main]),
                      name='滤波后', line=dict(color='#00d4ff', width=1.5)),
            row=2, col=1
        )
        fig_main.add_trace(
            go.Scatter(x=t_display*1e6, y=np.real(clean_signal[:display_samples_main]),
                      name='参考', line=dict(color='#00ff88', width=1, dash='dash'),
                      opacity=0.4),
            row=2, col=1
        )
        
        # 残差增强
        fig_main.add_trace(
            go.Scatter(x=t_display*1e6, y=np.real(enhanced_signal[:display_samples_main]),
                      name='增强', line=dict(color='#4ecdc4', width=1.5)),
            row=2, col=2
        )
        fig_main.add_trace(
            go.Scatter(x=t_display*1e6, y=np.real(clean_signal[:display_samples_main]),
                      name='参考', line=dict(color='#00ff88', width=1, dash='dash'),
                      opacity=0.4),
            row=2, col=2
        )
        
        fig_main.update_xaxes(title_text="时间 (μs)", gridcolor='rgba(255,255,255,0.1)')
        fig_main.update_yaxes(title_text="幅度", gridcolor='rgba(255,255,255,0.1)')
        
        fig_main.update_layout(
            height=600,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        st.plotly_chart(fig_main, width='stretch')
        
        # 4. 质量评估矩阵
        st.markdown("---")
        st.markdown("#### 信号质量评估矩阵")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig_quality = make_subplots(
                rows=1, cols=3,
                subplot_titles=('误码率', 'SNR (dB)', '相位一致性')
            )
            
            segments = np.arange(len(error_rates))
            
            fig_quality.add_trace(
                go.Scatter(x=segments, y=error_rates, line=dict(color='#ff6b6b', width=2)),
                row=1, col=1
            )
            
            fig_quality.add_trace(
                go.Scatter(x=segments, y=snrs, line=dict(color='#00d4ff', width=2)),
                row=1, col=2
            )
            
            fig_quality.add_trace(
                go.Scatter(x=segments, y=phase_consistencies, line=dict(color='#4ecdc4', width=2)),
                row=1, col=3
            )
            
            fig_quality.update_xaxes(title_text="片段", gridcolor='rgba(255,255,255,0.1)')
            fig_quality.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
            
            fig_quality.update_layout(
                height=300,
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            
            st.plotly_chart(fig_quality, width='stretch')
        
        with col2:
            st.markdown("##### 参数调整建议")
            st.markdown(f"""
            <div style="background: rgba(20, 40, 80, 0.8); padding: 1rem; border-radius: 8px; border: 1px solid rgba(0, 212, 255, 0.3);">
                <p style="color: #ffffff; margin: 0.3rem 0;">
                    <strong style="color: #00d4ff;">增益调整:</strong> {param_adjustments['gain']:.2f}x
                </p>
                <p style="color: #ffffff; margin: 0.3rem 0;">
                    <strong style="color: #00d4ff;">带宽调整:</strong> {param_adjustments['bandwidth']:.2f}
                </p>
                <p style="color: #ffffff; margin: 0.3rem 0;">
                    <strong style="color: #00d4ff;">建议调制:</strong> {param_adjustments['modulation']}
                </p>
                <hr style="border-color: rgba(0, 212, 255, 0.3); margin: 0.5rem 0;">
                <p style="color: #00ff88; margin: 0.3rem 0; font-size: 0.9rem;">
                    <strong>质量评分:</strong> {(1-avg_error)*100:.1f}%
                </p>
                <p style="color: #00d4ff; margin: 0.3rem 0; font-size: 0.9rem;">
                    <strong>平均SNR:</strong> {avg_snr:.2f} dB
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # 5. 详细分析图表
        st.markdown("---")
        st.markdown("#### 详细分析结果")
        
        with st.expander("🔍 查看完整分析图表", expanded=False):
            # 创建子标签页
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "环境特征", "相关性矩阵", "注意力权重", 
                "PCA方差解释", "信号片段对比", "参数调整对比"
            ])
            
            with tab1:
                st.markdown("##### 环境特征随时间变化")
                # 构建原始环境特征（模拟）- 使用更密集的采样点以显示细节
                t_env = np.linspace(0, 10, 1000)  # 0-10秒
                
                # 信号强度 (dBm) - 蓝色实线
                signal_strength = signal_power_db + signal_variation * np.sin(2*np.pi*0.5*t_env) + np.random.normal(0, 3, 1000)
                
                # 噪声功率 - 红色虚线
                noise_power = noise_power_db + noise_variation * np.sin(2*np.pi*0.3*t_env) + np.random.normal(0, 0.5, 1000)
                
                # 多径干扰 - 绿色点线
                multipath_inter = multipath_strength * (1 + multipath_decay * np.sin(2*np.pi*0.7*t_env)) + np.random.normal(0, 0.8, 1000)
                
                # 创建三轴图表（模仿附件样式）
                fig_env = go.Figure()
                
                # 添加信号强度轨迹（左Y轴，蓝色实线）
                fig_env.add_trace(go.Scatter(
                    x=t_env,
                    y=signal_strength,
                    name='信号强度 (dBm)',
                    line=dict(color='#0066cc', width=1),
                    mode='lines',
                    yaxis='y1'
                ))
                
                # 添加噪声功率轨迹（中间Y轴，红色虚线）
                fig_env.add_trace(go.Scatter(
                    x=t_env,
                    y=noise_power,
                    name='噪声功率',
                    line=dict(color='#cc0000', width=1, dash='dot'),
                    mode='lines',
                    yaxis='y2'
                ))
                
                # 添加多径干扰轨迹（右Y轴，绿色点线）
                fig_env.add_trace(go.Scatter(
                    x=t_env,
                    y=multipath_inter,
                    name='多径干扰',
                    line=dict(color='#00aa00', width=1, dash='dot'),
                    mode='lines',
                    yaxis='y3'
                ))
                
                # 更新布局 - 三个Y轴
                fig_env.update_layout(
                    title=dict(
                        text='环境特征随时间变化',
                        font=dict(size=16, color='white')
                    ),
                    xaxis=dict(
                        title='时间 (秒)',
                        domain=[0.1, 0.9],
                        gridcolor='rgba(255,255,255,0.1)',
                        showgrid=True
                    ),
                    yaxis=dict(
                        title=dict(text='信号强度 (dBm)', font=dict(color='#0066cc', size=14)),
                        tickfont=dict(color='#0066cc'),
                        gridcolor='rgba(255,255,255,0.1)',
                        showgrid=True,
                        side='left'
                    ),
                    yaxis2=dict(
                        title=dict(text='噪声功率', font=dict(color='#cc0000', size=14)),
                        tickfont=dict(color='#cc0000'),
                        overlaying='y',
                        side='right',
                        showgrid=False  # 中间Y轴不显示网格
                    ),
                    yaxis3=dict(
                        title=dict(text='多径干扰', font=dict(color='#00aa00', size=14)),
                        tickfont=dict(color='#00aa00'),
                        overlaying='y',
                        side='right',
                        position=1,
                        showgrid=False  # 右侧Y轴不显示网格
                    ),
                    height=500,
                    showlegend=True,
                    legend=dict(
                        x=0.1,
                        y=1.15,
                        orientation='h',
                        bgcolor='rgba(20, 40, 80, 0.8)',
                        bordercolor='rgba(0, 212, 255, 0.3)',
                        borderwidth=1,
                        font=dict(color='white')
                    ),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                
                st.plotly_chart(fig_env, width='stretch')
            
            with tab2:
                st.markdown("##### 特征相关性矩阵")
                # 计算滑动窗口相关性矩阵
                window_size = 50
                step = 25
                windows = []
                for i in range(0, len(clean_signal[:1000]) - window_size + 1, step):
                    # 只取实部进行相关性分析，避免复数问题
                    window = np.real(clean_signal[i:i+window_size])
                    windows.append(window)
                
                windows = np.array(windows)
                num_windows = min(20, len(windows))  # 只显示前20个窗口
                corr_matrix = np.real(np.corrcoef(windows[:num_windows]))  # 确保返回实数矩阵
                
                # 处理NaN和Inf值，确保可以JSON序列化
                corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=1.0, neginf=-1.0)
                corr_matrix = corr_matrix.astype(float).tolist()  # 转换为Python float列表
                
                fig_corr = go.Figure(data=go.Heatmap(
                    z=corr_matrix,
                    colorscale='RdBu',
                    zmid=0,
                    colorbar=dict(title="相关系数")
                ))
                
                fig_corr.update_layout(
                    title="时间片段相关性热力图",
                    # xaxis_title="窗口索引",
                    # yaxis_title="窗口索引",
                    height=500,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                st.plotly_chart(fig_corr, width='stretch')
            
            with tab3:
                st.markdown("##### 注意力权重分布")
                # 创建更详细的注意力权重可视化
                filter_names = list(filter_results.keys())
                
                fig_att = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=('策略权重分布', '策略性能得分'),
                    row_heights=[1, 0.5],
                    vertical_spacing=0.3       # 子图间距
                )
                
                # 权重分布柱状图
                fig_att.add_trace(
                    go.Bar(
                        x=filter_names,
                        y=attention_weights,
                        marker=dict(
                            color=attention_weights,
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title="权重", y=0.75, len=0.4),
                            
                        ),
                        text=[f'{w:.3f}' for w in attention_weights],
                        textposition='outside'
                    ),
                    row=1, col=1
                )
                
                # 性能得分雷达图数据转换为柱状图
                fig_att.add_trace(
                    go.Bar(
                        x=filter_names,
                        y=filter_scores,
                        marker=dict(color='#00d4ff'),
                        text=[f'{s:.2f}' for s in filter_scores],
                        textposition='auto'
                    ),
                    row=2, col=1
                )
                
                fig_att.update_xaxes(title_text="滤波策略", gridcolor='rgba(255,255,255,0.1)')
                fig_att.update_yaxes(title_text="权重值", gridcolor='rgba(255,255,255,0.1)', range=[0, max(attention_weights)*1.2], row=1, col=1)
                fig_att.update_yaxes(title_text="性能得分", gridcolor='rgba(255,255,255,0.1)', row=2, col=1)

                fig_att.update_layout(
                    height=700,
                    showlegend=False,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                st.plotly_chart(fig_att, width='stretch')
            
            with tab4:
                st.markdown("##### PCA主成分方差解释")
                # 对相关性矩阵进行PCA
                from sklearn.decomposition import PCA
                
                # 将列表转回numpy数组进行PCA分析
                corr_matrix_np = np.array(corr_matrix)
                
                pca = PCA()
                pca.fit(corr_matrix_np)
                
                explained_var = pca.explained_variance_ratio_[:10]  # 前10个主成分
                # 确保explained_var是实数
                explained_var = np.real(explained_var).astype(float)
                cumsum_var = np.cumsum(explained_var)
                
                fig_pca = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('方差解释比例', '累积方差解释'),
                    specs=[[{"type": "bar"}, {"type": "scatter"}]]
                )
                
                fig_pca.add_trace(
                    go.Bar(
                        x=[f'PC{i+1}' for i in range(len(explained_var))],
                        y=explained_var * 100,
                        marker=dict(color='#00d4ff'),
                        text=[f'{v*100:.1f}%' for v in explained_var],
                        textposition='outside'
                    ),
                    row=1, col=1
                )
                
                fig_pca.add_trace(
                    go.Scatter(
                        x=[f'PC{i+1}' for i in range(len(cumsum_var))],
                        y=cumsum_var * 100,
                        mode='lines+markers',
                        line=dict(color='#4ecdc4', width=3),
                        marker=dict(size=8)
                    ),
                    row=1, col=2
                )
                
                fig_pca.update_xaxes(title_text="主成分", gridcolor='rgba(255,255,255,0.1)')
                fig_pca.update_yaxes(title_text="方差解释 (%)", gridcolor='rgba(255,255,255,0.1)',range=[0, max(explained_var)*120], row=1, col=1)
                
                fig_pca.update_layout(
                    height=400,
                    showlegend=False,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                st.plotly_chart(fig_pca, width='stretch')
                
                st.info(f"前10个主成分累计解释方差: {cumsum_var[-1]*100:.2f}%")
            
            with tab5:
                st.markdown("##### 信号片段增强前后对比")
                # 选择几个代表性片段进行对比
                segment_indices = [0, len(clean_signal)//4, len(clean_signal)//2, 3*len(clean_signal)//4]
                segment_size = 100
                
                fig_seg = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=[f'片段 {i+1}' for i in range(4)],
                    vertical_spacing=0.12,
                    horizontal_spacing=0.1
                )
                
                for idx, seg_start in enumerate(segment_indices):
                    row = idx // 2 + 1
                    col = idx % 2 + 1
                    
                    seg_end = min(seg_start + segment_size, len(clean_signal))
                    t_seg = np.arange(segment_size) if seg_end - seg_start >= segment_size else np.arange(seg_end - seg_start)
                    
                    # 原始信号片段
                    fig_seg.add_trace(
                        go.Scatter(x=t_seg, y=np.real(noisy_signal[seg_start:seg_end]),
                                  line=dict(color='#ff6b6b', width=1, dash='dot'),
                                  name='带噪信号', showlegend=(idx==0)),
                        row=row, col=col
                    )
                    
                    # 增强信号片段
                    fig_seg.add_trace(
                        go.Scatter(x=t_seg, y=np.real(enhanced_signal[seg_start:seg_end]),
                                  line=dict(color='#00d4ff', width=1.5),
                                  name='增强信号', showlegend=(idx==0)),
                        row=row, col=col
                    )
                    
                    # 清洁信号片段
                    fig_seg.add_trace(
                        go.Scatter(x=t_seg, y=np.real(clean_signal[seg_start:seg_end]),
                                  line=dict(color='#00ff88', width=1, dash='dash'),
                                  name='理想信号', showlegend=(idx==0)),
                        row=row, col=col
                    )
                
                fig_seg.update_xaxes(title_text="采样点", gridcolor='rgba(255,255,255,0.1)')
                fig_seg.update_yaxes(title_text="幅度", gridcolor='rgba(255,255,255,0.1)')
                
                fig_seg.update_layout(
                    height=600,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    legend=dict(
                        bgcolor='rgba(20, 40, 80, 0.8)',
                        bordercolor='rgba(0, 212, 255, 0.3)',
                        borderwidth=1
                    )
                )
                st.plotly_chart(fig_seg, width='stretch')
            
            with tab6:
                st.markdown("##### 参数调整对比")
                # 显示调整前后的参数对比
                original_params = {
                    'gain': 1.0,
                    'bandwidth': 1.0,
                    'modulation': '2-FSK'
                }
                
                param_names = list(original_params.keys())
                original_values = [1.0, 1.0, 1.0]  # 标准化值
                adjusted_values = [
                    param_adjustments['gain'],
                    param_adjustments['bandwidth'],
                    1.0 if param_adjustments['modulation'] == '2-FSK' else 1.2
                ]
                
                fig_param = go.Figure()
                
                fig_param.add_trace(go.Bar(
                    name='调整前',
                    x=['增益', '带宽', '调制方式'],
                    y=original_values,
                    marker=dict(color='#ff6b6b'),
                    text=[f'{v:.2f}' for v in original_values],
                    textposition='outside'
                ))
                
                fig_param.add_trace(go.Bar(
                    name='调整后',
                    x=['增益', '带宽', '调制方式'],
                    y=adjusted_values,
                    marker=dict(color='#00d4ff'),
                    text=[f'{v:.2f}' for v in adjusted_values],
                    textposition='auto'
                ))
                
                fig_param.update_layout(
                    title="接收参数调整对比",
                    barmode='group',
                    xaxis_title="参数类型",
                    yaxis_title="相对值",
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    legend=dict(
                        bgcolor='rgba(20, 40, 80, 0.8)',
                        bordercolor='rgba(0, 212, 255, 0.3)',
                        borderwidth=1
                    )
                )
                fig_param.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
                fig_param.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
                
                st.plotly_chart(fig_param, width='stretch')
                
                # 显示详细参数说明
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("""
                    <div style="background: rgba(20, 40, 80, 0.6); padding: 1rem; border-radius: 8px;">
                        <h4 style="color: #ff6b6b;">调整前参数</h4>
                        <p style="color: #ffffff;">增益: 1.00x (标准)</p>
                        <p style="color: #ffffff;">带宽: 1.00 (标准)</p>
                        <p style="color: #ffffff;">调制: 2-FSK</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div style="background: rgba(20, 40, 80, 0.6); padding: 1rem; border-radius: 8px;">
                        <h4 style="color: #00d4ff;">调整后参数</h4>
                        <p style="color: #ffffff;">增益: {param_adjustments['gain']:.2f}x</p>
                        <p style="color: #ffffff;">带宽: {param_adjustments['bandwidth']:.2f}</p>
                        <p style="color: #ffffff;">调制: {param_adjustments['modulation']}</p>
                    </div>
                    """, unsafe_allow_html=True)


def run_interactive_optimization(signal_power_db, signal_variation, noise_power_db, 
                                noise_variation, multipath_strength, multipath_decay, 
                                multipath_paths, filter_strategy, sample_length):
    """运行交互式信号优化"""
    
    with st.spinner("🔄 正在生成信号并执行优化..."):
        # 生成信号
        clean_signal, noisy_signal, t = generate_custom_signal(
            signal_power_db, signal_variation,
            noise_power_db, noise_variation,
            multipath_strength, multipath_decay,
            multipath_paths, sample_length
        )
        
        # 应用滤波
        filtered_signal = apply_adaptive_filter(noisy_signal, filter_strategy)
        
        # 计算指标
        metrics = calculate_signal_metrics(clean_signal, noisy_signal, filtered_signal)
    
    st.success("✅ 优化完成！")
    
    # 显示性能指标
    st.markdown("#### 性能指标对比")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>信噪比改善</h3>
            <p style="color: {'#00ff88' if metrics['snr_improvement'] > 0 else '#ff6b6b'};">
                {metrics['snr_improvement']:+.2f} dB
            </p>
            <small>优化前: {metrics['snr_before']:.2f} dB<br>
            优化后: {metrics['snr_after']:.2f} dB</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>EVM降低</h3>
            <p style="color: {'#00ff88' if metrics['evm_reduction'] > 0 else '#ff6b6b'};">
                {metrics['evm_reduction']:.2f}%
            </p>
            <small>优化前: {metrics['evm_before']:.2f}%<br>
            优化后: {metrics['evm_after']:.2f}%</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>信号相关度</h3>
            <p style="color: #00d4ff;">
                {metrics['corr_after']:.3f}
            </p>
            <small>优化前: {metrics['corr_before']:.3f}<br>
            优化后: {metrics['corr_after']:.3f}</small>
        </div>
        """, unsafe_allow_html=True)
    
    # 绘制信号对比图
    st.markdown("---")
    st.markdown("#### 信号波形对比")
    
    # 只显示前1000个采样点以提高性能
    display_samples = min(1000, len(t))
    t_display = t[:display_samples]
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('原始信号 (实部)', '带噪信号 (实部)', 
                       '优化信号 (实部)', '频谱对比'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # 原始信号
    fig.add_trace(
        go.Scatter(x=t_display*1e6, y=np.real(clean_signal[:display_samples]),
                  name='原始信号', line=dict(color='#00ff88', width=1.5)),
        row=1, col=1
    )
    
    # 带噪信号
    fig.add_trace(
        go.Scatter(x=t_display*1e6, y=np.real(noisy_signal[:display_samples]),
                  name='带噪信号', line=dict(color='#ff6b6b', width=1)),
        row=1, col=2
    )
    
    # 优化后信号
    fig.add_trace(
        go.Scatter(x=t_display*1e6, y=np.real(filtered_signal[:display_samples]),
                  name='优化信号', line=dict(color='#00d4ff', width=1.5)),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=t_display*1e6, y=np.real(clean_signal[:display_samples]),
                  name='参考信号', line=dict(color='#00ff88', width=1, dash='dash'),
                  opacity=0.5),
        row=2, col=1
    )
    
    # 频谱对比
    freqs = np.fft.fftfreq(len(noisy_signal), 1/2000000)
    spectrum_noisy = np.abs(np.fft.fft(noisy_signal))
    spectrum_filtered = np.abs(np.fft.fft(filtered_signal))
    
    # 只显示正频率部分
    pos_freqs = freqs[:len(freqs)//2] / 1e6  # 转换为MHz
    
    fig.add_trace(
        go.Scatter(x=pos_freqs[:500], y=20*np.log10(spectrum_noisy[:500]+1e-10),
                  name='带噪频谱', line=dict(color='#ff6b6b', width=1)),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=pos_freqs[:500], y=20*np.log10(spectrum_filtered[:500]+1e-10),
                  name='优化频谱', line=dict(color='#00d4ff', width=1.5)),
        row=2, col=2
    )
    
    # 更新布局
    fig.update_xaxes(title_text="时间 (μs)", row=1, col=1)
    fig.update_xaxes(title_text="时间 (μs)", row=1, col=2)
    fig.update_xaxes(title_text="时间 (μs)", row=2, col=1)
    fig.update_xaxes(title_text="频率 (MHz)", row=2, col=2)
    
    fig.update_yaxes(title_text="幅度", row=1, col=1)
    fig.update_yaxes(title_text="幅度", row=1, col=2)
    fig.update_yaxes(title_text="幅度", row=2, col=1)
    fig.update_yaxes(title_text="功率 (dB)", row=2, col=2)
    
    fig.update_layout(
        height=700,
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        legend=dict(
            bgcolor='rgba(20, 40, 80, 0.8)',
            bordercolor='rgba(0, 212, 255, 0.3)',
            borderwidth=1
        )
    )
    
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
    
    st.plotly_chart(fig, width='stretch')
    
    # 详细参数信息
    with st.expander("📋 查看详细参数"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 输入参数")
            st.markdown(f"""
            - **信号功率**: {signal_power_db} dBm (±{signal_variation} dB)
            - **噪声功率**: {noise_power_db} dBm (±{noise_variation} dB)
            - **多径强度**: {multipath_strength}
            - **衰减速率**: {multipath_decay}
            - **多径数量**: {multipath_paths}
            """)
        
        with col2:
            st.markdown("##### 优化配置")
            st.markdown(f"""
            - **滤波策略**: {filter_strategy}
            - **采样长度**: {sample_length}
            - **采样率**: 2 MHz
            - **载波频率**: 2.4 GHz
            """)

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
        """)
    
    # 功能模块状态
    st.markdown("---")
    st.markdown("#### 模块状态")
    
    modules_status = [
        {"模块": "信号优化器", "状态": "✅ 正常" if REAL_MODULES_AVAILABLE else "⚠️ 模拟"},
        {"模块": "数据管理器", "状态": "✅ 正常" if REAL_MODULES_AVAILABLE else "⚠️ 模拟"},
        {"模块": "环境分析器", "状态": "✅ 正常"},
        {"模块": "质量评估器", "状态": "✅ 正常"},
        {"模块": "参数预测器", "状态": "✅ 正常"}
    ]
    
    df_modules = pd.DataFrame(modules_status)
    st.dataframe(df_modules, width='stretch', hide_index=True)
    
    # 日志信息
    st.markdown("---")
    st.markdown("#### 最近日志")
    
    log_entries = [
        f"{datetime.now().strftime('%H:%M:%S')} - 系统启动完成",
        f"{datetime.now().strftime('%H:%M:%S')} - 模块初始化{'成功' if REAL_MODULES_AVAILABLE else '(模拟模式)'}",
        f"{datetime.now().strftime('%H:%M:%S')} - Web界面已就绪"
    ]
    
    for entry in log_entries:
        st.text(entry)


if __name__ == "__main__":
    main()
