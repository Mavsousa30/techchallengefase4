#!/usr/bin/env python3
"""
Script simplificado para testar o pipeline localmente
Usa backends mais leves para evitar travamentos
"""

import sys
import os

# Adicionar diretório src ao path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.pipeline.inference import InferencePipeline
from src.metrics.reporter import Reporter


def test_with_lightweight_backends(video_path: str):
    """
    Testa o pipeline com backends leves (sem DeepFace inicialmente)
    
    Args:
        video_path: Caminho do vídeo
    """
    print("=" * 70)
    print("🧪 TESTE RÁPIDO - Pipeline com Backends Leves")
    print("=" * 70)
    print()
    print("ℹ️  Usando OpenCV puro (sem DeepFace) para evitar travamentos")
    print("ℹ️  Ideal para testar se o pipeline básico funciona")
    print()
    
    try:
        # Pipeline com backend mais leve
        print("🔧 Inicializando pipeline...")
        pipeline = InferencePipeline(
            video_path=video_path,
            output_video_path="outputs/test_output.mp4",
            save_preview=True,
            face_backend="opencv",      # OpenCV puro - mais rápido
            emotion_backend="fallback"  # Fallback simples - evita DeepFace
        )
        
        print("✅ Pipeline inicializado!")
        print()
        
        # Executar
        print("🎬 Processando vídeo...")
        summary = pipeline.run()
        
        print()
        print("=" * 70)
        print("✅ TESTE CONCLUÍDO COM SUCESSO!")
        print("=" * 70)
        print()
        
        # Mostrar resumo
        print(f"📊 Frames processados: {summary['frames_total']}")
        print(f"⏱️  Duração: {summary['duration_seconds']:.2f}s")
        print(f"👤 Faces detectadas: {summary['faces_stats']['total']}")
        print(f"⚠️  Anomalias: {summary['anomalies_total']}")
        print()
        
        # Gerar relatórios
        print("📝 Gerando relatórios...")
        reporter = Reporter()
        reporter.save_metrics(summary, "outputs/test_metrics.json")
        reporter.generate_markdown_report(summary, "outputs/test_report.md")
        
        print("✅ Relatórios salvos em outputs/")
        print()
        print("📦 Arquivos gerados:")
        print("  - outputs/test_output.mp4")
        print("  - outputs/test_metrics.json")
        print("  - outputs/test_report.md")
        print()
        print("=" * 70)
        print("💡 PRÓXIMO PASSO: Se funcionou, teste com DeepFace no Colab")
        print("   para ter análise de emoções completa e GPU acelerada!")
        print("=" * 70)
        
    except Exception as e:
        print()
        print("=" * 70)
        print("❌ ERRO DURANTE O TESTE")
        print("=" * 70)
        print()
        print(f"Erro: {str(e)}")
        print()
        print("💡 SOLUÇÕES:")
        print("1. Use o Google Colab (10-20x mais rápido)")
        print("2. Certifique-se que o vídeo existe")
        print("3. Verifique se tem espaço em disco")
        print("4. Tente com um vídeo menor (< 30 segundos)")
        print()
        raise


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Uso: python test_pipeline.py <caminho_do_video>")
        print()
        print("Exemplo:")
        print("  python test_pipeline.py video/video.mp4")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    if not os.path.exists(video_path):
        print(f"❌ Erro: Vídeo não encontrado: {video_path}")
        print()
        print("📂 Vídeos disponíveis:")
        for root, dirs, files in os.walk("."):
            for file in files:
                if file.endswith((".mp4", ".avi", ".mov", ".mkv")):
                    print(f"  - {os.path.join(root, file)}")
        sys.exit(1)
    
    test_with_lightweight_backends(video_path)
