"""
Main Script - Tech Challenge Fase 4

Script principal para executar análise de vídeo com IA.
Detecta faces, classifica emoções, reconhece atividades e gera relatórios.
"""

import argparse
import os
import sys
from pathlib import Path

from src.pipeline.inference import InferencePipeline
from src.metrics.reporter import Reporter


def parse_args():
    """Parse argumentos da linha de comando."""
    parser = argparse.ArgumentParser(
        description="Análise de Vídeo com IA - Tech Challenge Fase 4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  # Processar vídeo com configurações padrão
  python -m src.main --video data/input_video/video.mp4
  
  # Processar e salvar vídeo anotado
  python -m src.main --video input.mp4 --save-preview --output output.mp4
  
  # Especificar diretório de saída customizado
  python -m src.main --video input.mp4 --output-dir custom_outputs/
  
  # Usar backends específicos
  python -m src.main --video input.mp4 --face-backend opencv --emotion-backend deepface
        """
    )
    
    # Argumentos obrigatórios
    parser.add_argument(
        '--video',
        type=str,
        required=True,
        help='Caminho do vídeo de entrada'
    )
    
    # Argumentos opcionais
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Caminho do vídeo de saída com anotações (default: outputs/annotated_video.mp4)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs',
        help='Diretório para salvar resultados (default: outputs/)'
    )
    
    parser.add_argument(
        '--save-preview',
        action='store_true',
        help='Salvar vídeo anotado com detecções'
    )
    
    parser.add_argument(
        '--face-backend',
        type=str,
        default='auto',
        choices=['auto', 'opencv', 'face_recognition', 'deepface'],
        help='Backend para detecção de faces (default: auto)'
    )
    
    parser.add_argument(
        '--emotion-backend',
        type=str,
        default='auto',
        choices=['auto', 'deepface'],
        help='Backend para classificação de emoções (default: auto)'
    )
    
    parser.add_argument(
        '--no-report',
        action='store_true',
        help='Não gerar relatórios (apenas processar)'
    )
    
    return parser.parse_args()


def validate_video_path(video_path: str) -> bool:
    """
    Valida se o caminho do vídeo é válido.
    
    Args:
        video_path: Caminho do vídeo
        
    Returns:
        True se válido, False caso contrário
    """
    if not os.path.exists(video_path):
        print(f"❌ Erro: Vídeo não encontrado: {video_path}")
        return False
    
    if not os.path.isfile(video_path):
        print(f"❌ Erro: Caminho não é um arquivo: {video_path}")
        return False
    
    # Verificar extensão
    valid_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    ext = os.path.splitext(video_path)[1].lower()
    
    if ext not in valid_extensions:
        print(f"⚠️  Aviso: Extensão '{ext}' pode não ser suportada")
        print(f"    Extensões recomendadas: {', '.join(valid_extensions)}")
    
    return True


def setup_output_directory(output_dir: str):
    """
    Cria estrutura de diretórios de saída.
    
    Args:
        output_dir: Diretório raiz de saída
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'frames'), exist_ok=True)


def print_header():
    """Imprime cabeçalho do programa."""
    print("=" * 70)
    print("🎬 Tech Challenge Fase 4 - Análise de Vídeo com IA")
    print("=" * 70)
    print()


def print_summary_stats(summary: dict):
    """
    Imprime estatísticas resumidas do processamento.
    
    Args:
        summary: Dicionário com resumo do processamento
    """
    print()
    print("=" * 70)
    print("📊 RESUMO DO PROCESSAMENTO")
    print("=" * 70)
    print()
    
    print(f"📹 Vídeo: {summary.get('video_path', 'N/A')}")
    print(f"🎞️  Frames processados: {summary.get('frames_total', 0):,}")
    print(f"⏱️  Duração: {summary.get('duration_seconds', 0):.2f}s")
    print(f"📈 FPS: {summary.get('fps', 0):.2f}")
    print()
    
    # Faces
    faces_stats = summary.get('faces_stats', {})
    print(f"👤 Faces detectadas: {faces_stats.get('total_detections', 0):,}")
    print(f"   Média por frame: {faces_stats.get('avg_faces_per_frame', 0):.2f}")
    print()
    
    # Emoções
    emotions = summary.get('emotions_distribution', {})
    if emotions:
        print("😊 Emoções:")
        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
        for emotion, count in sorted_emotions[:5]:  # Top 5
            print(f"   {emotion}: {count}")
        print()
    
    # Atividades
    activities = summary.get('activities_timeline', [])
    print(f"🏃 Atividades detectadas: {len(activities)}")
    print()
    
    # Anomalias
    anomalies_total = summary.get('anomalies_total', 0)
    anomalies_by_sev = summary.get('anomalies_by_severity', {})
    print(f"⚠️  Anomalias detectadas: {anomalies_total}")
    if anomalies_by_sev:
        print(f"   Alta: {anomalies_by_sev.get('high', 0)}")
        print(f"   Média: {anomalies_by_sev.get('medium', 0)}")
        print(f"   Baixa: {anomalies_by_sev.get('low', 0)}")
    print()


def main():
    """Função principal."""
    # Parse argumentos
    args = parse_args()
    
    # Print header
    print_header()
    
    # Validar vídeo
    if not validate_video_path(args.video):
        sys.exit(1)
    
    # Setup diretório de saída
    setup_output_directory(args.output_dir)
    
    # Determinar caminho do vídeo de saída
    output_video_path = args.output
    if args.save_preview and not output_video_path:
        output_video_path = os.path.join(args.output_dir, 'annotated_video.mp4')
    
    try:
        # Criar e executar pipeline
        print("🚀 Inicializando pipeline...")
        print()
        
        pipeline = InferencePipeline(
            video_path=args.video,
            output_video_path=output_video_path,
            save_preview=args.save_preview,
            face_backend=args.face_backend,
            emotion_backend=args.emotion_backend
        )
        
        # Executar processamento
        summary = pipeline.run()
        
        # Gerar relatórios
        if not args.no_report:
            print()
            print("📝 Gerando relatórios...")
            
            reporter = Reporter()
            report_files = reporter.save_report_bundle(
                summary=summary,
                output_dir=args.output_dir
            )
            
            print(f"✅ Métricas salvas em: {report_files['metrics_json']}")
            print(f"✅ Relatório salvo em: {report_files['report_markdown']}")
        
        # Print resumo
        print_summary_stats(summary)
        
        print("=" * 70)
        print("✨ Processamento concluído com sucesso!")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print()
        print("⚠️  Processamento interrompido pelo usuário")
        sys.exit(1)
        
    except Exception as e:
        print()
        print(f"❌ Erro durante processamento: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
