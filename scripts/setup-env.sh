#!/bin/bash
# AGCH-Impl 開發環境安裝腳本
# 完全隔離環境（包含 CUDA）- 適用於 RTX 5080 (Blackwell)
# 
# 此腳本會建立完全隔離的環境，包含：
# - Python 3.11
# - CUDA 12.8 (cudatoolkit + cudnn)
# - PyTorch 2.6+
# - PyTorch Geometric
# - 所有開發工具

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "🚀 AGCH-Impl 開發環境安裝"
echo "=========================="
echo ""
echo "📁 專案目錄: $PROJECT_DIR"
echo ""
echo "⚠️  此安裝會建立完全隔離的環境（包含 CUDA）"
echo "   預計需要 10-15GB 磁碟空間"
echo ""

# 檢測套件管理器
check_mamba() {
    command -v mamba &> /dev/null
}

check_conda() {
    command -v conda &> /dev/null
}

# 安裝 Miniforge (mamba)
install_miniforge() {
    echo "� 下載並安裝 Miniforge (mamba)..."
    echo ""
    
    MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
    INSTALLER="/tmp/Miniforge3-Linux-x86_64.sh"
    
    curl -L -o "$INSTALLER" "$MINIFORGE_URL"
    bash "$INSTALLER" -b -p "$HOME/miniforge3"
    
    # 初始化 shell
    eval "$("$HOME/miniforge3/bin/conda" shell.bash hook)"
    conda init bash
    
    echo ""
    echo "✅ Miniforge 安裝完成"
    echo ""
    echo "⚠️  請重新啟動終端機或執行："
    echo "   source ~/.bashrc"
    echo ""
    echo "然後重新執行此腳本"
    exit 0
}

# 使用 mamba 建立環境
create_environment() {
    echo "📦 建立 AGCH 環境..."
    echo ""
    echo "這可能需要 5-10 分鐘，取決於網路速度..."
    echo ""
    
    cd "$PROJECT_DIR"
    
    # 使用 mamba 或 conda
    if check_mamba; then
        PKG_MGR="mamba"
    else
        PKG_MGR="conda"
    fi
    
    # 移除舊環境（如果存在）
    $PKG_MGR env remove -n agch -y 2>/dev/null || true
    
    # 建立新環境
    $PKG_MGR env create -f environment.yml
    
    echo ""
    echo "✅ 環境建立完成！"
}

# 驗證安裝
verify_installation() {
    echo ""
    echo "🔍 驗證安裝..."
    echo ""
    
    # 啟動環境並驗證
    eval "$(conda shell.bash hook)"
    conda activate agch
    
    python -c "
import sys
print(f'Python: {sys.version}')

import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')

import torch_geometric
print(f'PyTorch Geometric: {torch_geometric.__version__}')

print()
print('✅ 所有套件安裝成功！')
"
}

# 主邏輯
main() {
    # 檢查是否有 mamba 或 conda
    if ! check_mamba && ! check_conda; then
        echo "❌ 未發現 mamba 或 conda"
        echo ""
        read -p "是否自動安裝 Miniforge (mamba)? [Y/n]: " install_choice
        
        if [[ "${install_choice:-Y}" =~ ^[Yy]$ ]]; then
            install_miniforge
        else
            echo ""
            echo "請手動安裝 Miniforge:"
            echo "  curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
            echo "  bash Miniforge3-Linux-x86_64.sh"
            exit 1
        fi
    fi
    
    # 確認安裝
    echo "即將建立完全隔離的開發環境，包含："
    echo "  - Python 3.11"
    echo "  - CUDA 12.8 (cudatoolkit + cudnn)"
    echo "  - PyTorch 2.6+"
    echo "  - PyTorch Geometric"
    echo "  - 開發工具 (pytest, black, jupyter...)"
    echo ""
    read -p "繼續安裝? [Y/n]: " confirm
    
    if [[ "${confirm:-Y}" =~ ^[Yy]$ ]]; then
        create_environment
        verify_installation
        
        echo ""
        echo "=========================================="
        echo "🎉 安裝完成！"
        echo "=========================================="
        echo ""
        echo "使用方式:"
        echo "  conda activate agch"
        echo ""
        echo "驗證 GPU:"
        echo "  python -c \"import torch; print(torch.cuda.is_available())\""
        echo ""
    else
        echo "取消安裝"
        exit 0
    fi
}

main "$@"
