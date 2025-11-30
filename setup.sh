#!/bin/bash

# 設定虛擬環境資料夾名稱
VENV_DIR="venv"

# 檢查 venv 是否存在
if [ ! -d "$VENV_DIR" ]; then
    echo "🔧 虛擬環境不存在，正在建立..."
    /usr/local/bin/python3.12 -m venv "$VENV_DIR"
fi

# 啟動 venv
echo "🚀 啟動虛擬環境..."
source "$VENV_DIR/bin/activate"

# 設定使用 gcc10 (如果可用)
if command -v gcc10-gcc &> /dev/null; then
    echo "🔧 設定使用 GCC 10..."
    export CC=gcc10-gcc
    export CXX=gcc10-c++
fi

# 升級 pip 和安裝基本工具
echo "🔧 升級 pip 和安裝基本工具..."
pip install --upgrade pip
pip install --upgrade setuptools wheel

# 檢查 requirements.txt 是否存在
if [ -f "requirements.txt" ]; then
    echo "📦 安裝 requirements.txt 內的套件..."
    pip install -r requirements.txt
else
    echo "⚠️ 找不到 requirements.txt，跳過安裝"
fi

echo "✅ 設定完成！虛擬環境已啟動 🎉"
echo "💡 如需使用 TA-Lib，記得先執行: source venv/bin/activate"

