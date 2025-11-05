#!/bin/bash

# 低功耗蓝牙信号优化系统 Streamlit 启动脚本

echo "🚀 启动低功耗蓝牙信号优化系统 Web 界面..."
echo "================================"

# 检查虚拟环境
if [ -d ".venv" ]; then
    echo "✅ 发现虚拟环境，正在激活..."
    source .venv/bin/activate
else
    echo "⚠️  未发现虚拟环境，使用系统 Python"
fi

# 检查 Streamlit 是否安装
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit 未安装，正在安装..."
    pip install streamlit plotly
fi

# 启动 Streamlit 应用
echo "🌐 启动 Web 界面..."
echo "访问地址: http://localhost:8501"
echo "按 Ctrl+C 停止服务"
echo "================================"

# 命令行输入：
streamlit run streamlit_app_simple.py --server.port 8501 --server.headless false
