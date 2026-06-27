#!/usr/bin/env bash
# Debug Panel 安装脚本
# 用法: ./install.sh [目标路径]
# 默认安装到当前用户目录

set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
DEST="${1:-$DIR}"

echo "======================================"
echo "  Debug Panel - 安装"
echo "======================================"
echo ""
echo "目标路径: $DEST"

# 创建桌面快捷方式
DESKTOP_FILE="$HOME/桌面/debug-panel.desktop"
if [ ! -d "$HOME/桌面" ]; then
    DESKTOP_FILE="$HOME/Desktop/debug-panel.desktop"
fi

cat > "$DESKTOP_FILE" << EOF
[Desktop Entry]
Type=Application
Name=Debug Panel
Comment=S1/S2/S3 调试面板
Exec=$DEST/run.sh
Icon=$DEST/icon.png
Terminal=true
StartupNotify=true
Categories=Development;Utility;
EOF

chmod +x "$DESKTOP_FILE"
gio set "$DESKTOP_FILE" metadata::trusted true 2>/dev/null || true

# 注册到系统菜单
mkdir -p "$HOME/.local/share/applications"
cp "$DESKTOP_FILE" "$HOME/.local/share/applications/debug-panel.desktop"
update-desktop-database "$HOME/.local/share/applications" 2>/dev/null || true

# 安装桌面会话自启动
mkdir -p "$HOME/.config/autostart"
cp "$DESKTOP_FILE" "$HOME/.config/autostart/debug-panel.desktop"

echo ""
echo "✅ 安装完成！"
echo ""
echo "双击桌面图标 Debug Panel 启动"
echo "或终端运行: $DEST/run.sh"
echo "已安装到开机自启: $HOME/.config/autostart/debug-panel.desktop"
echo ""
