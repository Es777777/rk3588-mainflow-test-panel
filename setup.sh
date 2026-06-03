#!/usr/bin/env bash
set -e

echo "======================================"
echo "  Debug Panel - 安装依赖"
echo "======================================"
echo ""

# 检测 Python
if ! command -v python3 &>/dev/null; then
    echo "❌ 未找到 python3，请先安装 Python 3.8+"
    exit 1
fi
echo "✅ Python: $(python3 --version)"

# 安装 pip 依赖
echo ""
echo "📦 安装 Python 依赖..."
pip3 install -r requirements.txt --quiet 2>/dev/null || pip install -r requirements.txt --quiet

# 可选依赖（语音/视觉模块）
echo ""
echo "📦 安装可选依赖（语音识别、目标检测）..."
pip3 install openai-whisper ultralytics transformers 2>/dev/null && echo "✅ 可选依赖已安装" || echo "⚠️  可选依赖安装失败（不影响基础功能）"

# 系统依赖检测
echo ""
echo "🔍 检测系统依赖..."
MISSING=""
for cmd in python3; do
    command -v $cmd &>/dev/null && echo "  ✅ $cmd" || { echo "  ❌ $cmd"; MISSING+=" $cmd"; }
done

echo ""
if [ -n "$MISSING" ]; then
    echo "⚠️  部分系统依赖缺失:${MISSING}"
    echo "  请使用系统包管理器安装"
else
    echo "✅ 所有依赖检查通过"
fi

echo ""
echo "======================================"
echo "  安装完成！运行方式："
echo "  python3 main.py"
echo "======================================"
