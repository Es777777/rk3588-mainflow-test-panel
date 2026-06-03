#!/usr/bin/env bash
cd "$(dirname "$0")"

kill $(lsof -t -i:5000 2>/dev/null) 2>/dev/null

echo "======================================"
echo "  Debug Panel - 调试面板"
echo "======================================"
echo ""
echo "启动中..."
python3 main.py
