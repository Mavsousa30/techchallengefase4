#!/bin/bash

# Script para iniciar a interface web do Tech Challenge Fase 4

echo "🎬 Tech Challenge Fase 4 - Interface Web"
echo "========================================"
echo ""

# Verificar se streamlit está instalado
if ! python3 -c "import streamlit" &> /dev/null; then
    echo "📦 Streamlit não encontrado. Instalando..."
    pip3 install streamlit plotly
    echo ""
fi

# Verificar se as dependências estão instaladas
if ! python3 -c "import cv2" &> /dev/null; then
    echo "📦 Instalando dependências..."
    pip3 install -r requirements.txt
    echo ""
fi

# Iniciar aplicação
echo "🚀 Iniciando aplicação..."
echo "📍 A interface será aberta em: http://localhost:8501"
echo ""
echo "💡 Dica: Use Ctrl+C para encerrar"
echo ""

streamlit run app.py
