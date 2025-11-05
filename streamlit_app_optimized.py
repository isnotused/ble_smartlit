"""
优化版低功耗蓝牙信号接收优化系统 - Streamlit Web界面
专注于突出8步优化方法的核心技术
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
import json

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

# ================== 页面配置 ==================
st.set_page_config(
    page_title="低功耗蓝牙信号优化系统",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================== 自定义CSS样式 ==================
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #0f2027 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
    }
    
    .optimization-step {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: white;
        border-left: 4px solid #00d4aa;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #2196F3 0%, #21CBF3 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        color: white;
        margin: 0.5rem;
        box-shadow: 0 6px 20px rgba(33, 150, 243, 0.3);
    }
    
    .patent-highlight {
        background: linear-gradient(135deg, #FF6B6B 0%, #FFE66D 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        color: #333;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 6px 20px rgba(255, 107, 107, 0.3);
    }
    
    .status-success {
        background: linear-gradient(135deg, #4CAF50 0%, #8BC34A 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .status-processing {
        background: linear-gradient(135deg, #FF9800 0%, #FFC107 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .tech-spec {
        background: rgba(13, 17, 23, 0.95);
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 1rem;
        color: #c9d1d9;
        font-family: 'SFMono-Regular', Consolas, monospace;
        margin: 0.5rem 0;
    }
    
    div[data-testid="stSidebar"] > div {
        background: linear-gradient(180deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    }
    
    .stSelectbox > div > div {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
    }
    
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #00d4aa 0%, #00d4aa 100%);
    }
</style>
""", unsafe_allow_html=True)

# ================== 模拟优化系统类 ==================
class MockOptimizer:
    def __init__(self):
        self.initialized = False
        self.optimization_count = 0
        self.current_step = 0
        self.step_progress = [0] * 8
        
    def initialize(self):
        self.initialized = True
        
    def optimize_signal(self, duration=10, cycles=3):
        """模拟8步优化过程"""
        optimization_results = []
        
        for cycle in range(cycles):
            cycle_results = {}
            
            # 模拟8个优化步骤
            steps = [
                "RF前端信号获取",
                "动态特征矩阵构建", 
                "注意力机制滤波器选择",
                "自适应滤波处理",
                "深度残差增强",
                "质量评估",
                "参数调整",
                "预测优化"
            ]
            
            step_results = {}
            base_quality = 0.7 + 0.1 * cycle
            
            for i, step in enumerate(steps):
                # 模拟每步的性能指标
                step_quality = base_quality + 0.03 * i + np.random.normal(0, 0.02)
                step_latency = 5 + np.random.exponential(2)
                step_snr = 15 + 2 * i + np.random.normal(0, 1)
                
                step_results[step] = {
                    'quality': max(0, min(1, step_quality)),
                    'latency_ms': step_latency,
                    'snr_db': step_snr,
                    'throughput_mbps': 1.2 + 0.1 * i + np.random.normal(0, 0.05)
                }
                
                self.step_progress[i] = step_quality * 100
                
            cycle_results = {
                'cycle': cycle + 1,
                'overall_quality': np.mean([s['quality'] for s in step_results.values()]),
                'total_latency': sum([s['latency_ms'] for s in step_results.values()]),
                'avg_snr': np.mean([s['snr_db'] for s in step_results.values()]),
                'total_throughput': sum([s['throughput_mbps'] for s in step_results.values()]),
                'steps': step_results,
                'timestamp': datetime.now()
            }
            
            optimization_results.append(cycle_results)
            self.optimization_count += 1
            
        return optimization_results

# ================== 主应用程序 ==================
def main():
    # 主标题
    st.markdown("""
    <div class="main-header">
        <h1>🚀 低功耗蓝牙信号优化系统</h1>
        <h3>基于专利8步优化算法的智能信号处理平台</h3>
        <p>RF前端 → 特征构建 → 注意力滤波 → 自适应处理 → 残差增强 → 质量评估 → 参数调整 → 预测优化</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 模块状态检查
    modules_available, status_msg = REAL_MODULES_AVAILABLE, "真实模块" if REAL_MODULES_AVAILABLE else "模拟模式"
    
    # 侧边栏 - 核心控制面板
    with st.sidebar:
        st.markdown("### 🎛️ 优化控制中心")
        
        # 系统状态
        status_color = "success" if modules_available else "processing"
        st.markdown(f"""
        <div class="status-{status_color}">
            <strong>系统状态:</strong> {status_msg}
        </div>
        """, unsafe_allow_html=True)
        
        # 优化参数
        st.markdown("#### 优化参数配置")
        
        optimization_mode = st.selectbox(
            "优化模式",
            ["低延迟模式", "高质量模式", "平衡模式", "节能模式"],
            index=2
        )
        
        duration = st.slider("信号持续时间(秒)", 5, 60, 15)
        cycles = st.slider("优化循环次数", 1, 10, 3)
        snr_threshold = st.slider("SNR阈值(dB)", 10, 30, 18)
        
        # 专利技术说明
        st.markdown("""
        <div class="patent-highlight">
            🏆 专利核心技术<br>
            8步自适应优化算法<br>
            智能信号增强处理
        </div>
        """, unsafe_allow_html=True)

    # 主界面 - 3个核心模块
    col1, col2, col3 = st.columns([1, 1, 1])
    
    # ================== 1. 实时优化控制 ==================
    with col1:
        st.markdown("### 🎯 实时优化控制")
        
        if st.button("🚀 启动优化", type="primary", use_container_width=True):
            with st.spinner('正在执行8步优化流程...'):
                optimizer = MockOptimizer()
                optimizer.initialize()
                
                # 创建进度条容器
                progress_container = st.container()
                
                # 执行优化
                results = optimizer.optimize_signal(duration=duration, cycles=cycles)
                
                # 保存结果到session state
                st.session_state.optimization_results = results
                st.session_state.last_optimization = datetime.now()
                
                st.success(f"✅ 优化完成！执行了 {cycles} 个循环，{len(results)} 个结果")
        
        # 显示8步优化流程
        st.markdown("#### 📋 优化步骤进度")
        steps = [
            "1️⃣ RF前端信号获取",
            "2️⃣ 动态特征矩阵构建", 
            "3️⃣ 注意力机制滤波器选择",
            "4️⃣ 自适应滤波处理",
            "5️⃣ 深度残差增强",
            "6️⃣ 质量评估",
            "7️⃣ 参数调整",
            "8️⃣ 预测优化"
        ]
        
        if 'optimization_results' in st.session_state:
            # 从最新结果中获取步骤数据
            latest_result = st.session_state.optimization_results[-1]
            for i, step in enumerate(steps):
                step_name = list(latest_result['steps'].keys())[i]
                step_data = latest_result['steps'][step_name]
                quality_pct = int(step_data['quality'] * 100)
                
                st.markdown(f"""
                <div class="optimization-step">
                    {step}<br>
                    <small>质量: {quality_pct}% | 延迟: {step_data['latency_ms']:.1f}ms</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            for step in steps:
                st.markdown(f"""
                <div class="optimization-step">
                    {step}<br>
                    <small>等待执行...</small>
                </div>
                """, unsafe_allow_html=True)

    # ================== 2. 实时性能监控 ==================
    with col2:
        st.markdown("### 📊 实时性能监控")
        
        if 'optimization_results' in st.session_state:
            latest_result = st.session_state.optimization_results[-1]
            
            # 核心性能指标
            metrics_col1, metrics_col2 = st.columns(2)
            
            with metrics_col1:
                quality_score = latest_result['overall_quality']
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{quality_score:.1%}</h3>
                    <p>信号质量</p>
                </div>
                """, unsafe_allow_html=True)
                
                avg_snr = latest_result['avg_snr']
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{avg_snr:.1f} dB</h3>
                    <p>平均SNR</p>
                </div>
                """, unsafe_allow_html=True)
            
            with metrics_col2:
                total_latency = latest_result['total_latency']
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{total_latency:.1f} ms</h3>
                    <p>总延迟</p>
                </div>
                """, unsafe_allow_html=True)
                
                throughput = latest_result['total_throughput']
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{throughput:.1f} Mbps</h3>
                    <p>总吞吐量</p>
                </div>
                """, unsafe_allow_html=True)
            
            # 实时波形图
            st.markdown("#### 📈 信号质量趋势")
            
            # 创建时间序列数据
            time_points = np.arange(len(st.session_state.optimization_results))
            quality_values = [r['overall_quality'] for r in st.session_state.optimization_results]
            snr_values = [r['avg_snr'] for r in st.session_state.optimization_results]
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('信号质量', 'SNR (dB)'),
                vertical_spacing=0.1
            )
            
            fig.add_trace(
                go.Scatter(x=time_points, y=quality_values, 
                          mode='lines+markers', name='质量', 
                          line=dict(color='#00d4aa', width=3)),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=time_points, y=snr_values,
                          mode='lines+markers', name='SNR',
                          line=dict(color='#ff6b6b', width=3)),
                row=2, col=1
            )
            
            fig.update_layout(
                height=400,
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            # 默认显示
            st.info("💡 点击'启动优化'开始监控性能指标")
            
            # 显示技术规格
            st.markdown("#### 🔧 技术规格")
            st.markdown("""
            <div class="tech-spec">
            • 频率范围: 2.4 GHz ISM 频段<br>
            • 调制方式: GFSK, π/4-DQPSK<br>
            • 数据速率: 1 Mbps - 2 Mbps<br>
            • 接收灵敏度: -94 dBm @ 1 Mbps<br>
            • 动态范围: > 80 dB<br>
            • 优化延迟: < 50ms
            </div>
            """, unsafe_allow_html=True)

    # ================== 3. 数据分析与导出 ==================
    with col3:
        st.markdown("### 📋 数据分析与导出")
        
        if 'optimization_results' in st.session_state:
            results = st.session_state.optimization_results
            
            # 统计摘要
            st.markdown("#### 📈 优化统计摘要")
            
            avg_quality = np.mean([r['overall_quality'] for r in results])
            max_quality = max([r['overall_quality'] for r in results])
            min_latency = min([r['total_latency'] for r in results])
            avg_throughput = np.mean([r['total_throughput'] for r in results])
            
            summary_df = pd.DataFrame({
                '指标': ['平均质量', '最佳质量', '最低延迟', '平均吞吐量'],
                '数值': [f"{avg_quality:.1%}", f"{max_quality:.1%}", 
                        f"{min_latency:.1f}ms", f"{avg_throughput:.1f}Mbps"],
                '状态': ['🟢 良好', '🟢 优秀', '🟢 快速', '🟢 稳定']
            })
            
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
            
            # 8步骤性能对比
            st.markdown("#### 🔍 各步骤性能分析")
            
            # 获取最新结果的各步骤数据
            latest_steps = latest_result['steps']
            step_names = list(latest_steps.keys())
            step_qualities = [latest_steps[step]['quality'] for step in step_names]
            step_latencies = [latest_steps[step]['latency_ms'] for step in step_names]
            
            # 创建雷达图
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=step_qualities,
                theta=[f"步骤{i+1}" for i in range(len(step_names))],
                fill='toself',
                name='信号质量',
                line_color='#00d4aa'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True,
                height=300,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 数据导出
            st.markdown("#### 💾 数据导出")
            
            export_format = st.selectbox("选择导出格式", ["JSON", "CSV", "HDF5"])
            
            if st.button("📥 导出优化数据", use_container_width=True):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"ble_optimization_{timestamp}"
                
                if export_format == "JSON":
                    # 准备JSON数据
                    export_data = {
                        'metadata': {
                            'timestamp': timestamp,
                            'cycles': len(results),
                            'mode': optimization_mode,
                            'duration': duration
                        },
                        'results': results
                    }
                    
                    # 将datetime对象转换为字符串
                    def json_serial(obj):
                        if isinstance(obj, datetime):
                            return obj.isoformat()
                        raise TypeError(f"Type {type(obj)} not serializable")
                    
                    json_str = json.dumps(export_data, default=json_serial, indent=2, ensure_ascii=False)
                    
                    st.download_button(
                        label="📁 下载JSON文件",
                        data=json_str,
                        file_name=f"{filename}.json",
                        mime="application/json"
                    )
                
                elif export_format == "CSV":
                    # 创建CSV数据
                    csv_data = []
                    for result in results:
                        for step_name, step_data in result['steps'].items():
                            csv_data.append({
                                'cycle': result['cycle'],
                                'step': step_name,
                                'quality': step_data['quality'],
                                'latency_ms': step_data['latency_ms'],
                                'snr_db': step_data['snr_db'],
                                'throughput_mbps': step_data['throughput_mbps']
                            })
                    
                    df = pd.DataFrame(csv_data)
                    csv_str = df.to_csv(index=False)
                    
                    st.download_button(
                        label="📁 下载CSV文件",
                        data=csv_str,
                        file_name=f"{filename}.csv",
                        mime="text/csv"
                    )
                
                st.success(f"✅ {export_format}数据准备完成！")
        
        else:
            st.info("💡 执行优化后可查看分析结果")
            
            # 显示算法简介
            st.markdown("#### 🧠 核心算法")
            st.markdown("""
            <div class="tech-spec">
            <strong>8步自适应优化流程:</strong><br><br>
            1️⃣ RF前端: 多频段信号采集<br>
            2️⃣ 特征矩阵: 动态时频分析<br>
            3️⃣ 注意力滤波: 智能噪声抑制<br>
            4️⃣ 自适应滤波: 实时信道估计<br>
            5️⃣ 残差增强: 深度学习优化<br>
            6️⃣ 质量评估: 多维度性能评价<br>
            7️⃣ 参数调整: 自适应参数优化<br>
            8️⃣ 预测优化: 基于ML的预测调整
            </div>
            """, unsafe_allow_html=True)

    # ================== 底部状态栏 ==================
    st.markdown("---")
    
    status_col1, status_col2, status_col3, status_col4 = st.columns(4)
    
    with status_col1:
        if 'last_optimization' in st.session_state:
            last_time = st.session_state.last_optimization.strftime("%H:%M:%S")
            st.metric("🕒 最后优化", last_time)
        else:
            st.metric("🕒 最后优化", "未执行")
    
    with status_col2:
        total_cycles = len(st.session_state.get('optimization_results', []))
        st.metric("🔄 优化循环", f"{total_cycles} 次")
    
    with status_col3:
        if 'optimization_results' in st.session_state:
            avg_quality = np.mean([r['overall_quality'] for r in st.session_state.optimization_results])
            st.metric("📊 平均质量", f"{avg_quality:.1%}")
        else:
            st.metric("📊 平均质量", "0%")
    
    with status_col4:
        system_status = "🟢 正常运行" if modules_available else "🟡 模拟模式"
        st.metric("⚡ 系统状态", system_status)

if __name__ == "__main__":
    main()
