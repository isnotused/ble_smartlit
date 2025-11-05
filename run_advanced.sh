#!/bin/bash

# 低功耗蓝牙信号优化系统 - 高级版启动脚本

echo "🚀 启动低功耗蓝牙信号优化系统（高级版）..."
echo "================================"
echo ""
echo "📋 系统特点："
echo "  ✓ 完整展示8步优化流程"
echo "  ✓ 实时性能对比分析"
echo "  ✓ 注意力机制可视化"
echo "  ✓ 优化前后效果对比"
echo ""
echo "================================"

# 检查是否在项目根目录
if [ ! -f "streamlit_app_advanced.py" ]; then
    echo "❌ 错误：请在项目根目录下运行此脚本"
    echo "   当前目录: $(pwd)"
    echo "   应在: /Users/fuwei/ble_smartlit"
    exit 1
fi

# 使用 uv 运行
echo "🌐 启动 Web 界面..."
echo "访问地址: http://localhost:8502"
echo "按 Ctrl+C 停止服务"
echo "================================"
echo ""

# 运行应用
uv run streamlit run streamlit_app_advanced.py \
    --server.port 8502 \
    --server.headless true \
    --browser.gatherUsageStats false

