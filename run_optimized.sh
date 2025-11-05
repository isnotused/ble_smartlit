#!/bin/bash
# 运行优化版Streamlit应用
echo "🚀 启动优化版低功耗蓝牙信号优化系统..."
echo "📡 访问地址: http://localhost:8503"
echo ""

cd "$(dirname "$0")"
streamlit run streamlit_app_optimized.py --server.port=8503 --server.headless=true
