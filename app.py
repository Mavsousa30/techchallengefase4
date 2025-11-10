"""
Streamlit Web Interface - Tech Challenge Fase 4

Interface visual para análise de vídeo com IA.
"""

import streamlit as st
import os
import json
from pathlib import Path
import time
from datetime import datetime

from src.pipeline.inference import InferencePipeline
from src.metrics.reporter import Reporter


# Configuração da página
st.set_page_config(
    page_title="Análise de Vídeo com IA",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos customizados
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Função principal da aplicação."""
    
    # Header
    st.markdown('<div class="main-header">🎬 Análise de Vídeo com IA</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar - Configurações
    with st.sidebar:
        st.header("⚙️ Configurações")
        
        # Upload de vídeo
        uploaded_file = st.file_uploader(
            "Carregar Vídeo",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Selecione um arquivo de vídeo para análise"
        )
        
        st.markdown("---")
        
        # Configurações de processamento
        st.subheader("🔧 Opções de Processamento")
        
        save_preview = st.checkbox(
            "Salvar vídeo anotado",
            value=True,
            help="Gera vídeo com anotações visuais"
        )
        
        face_backend = st.selectbox(
            "Backend de Detecção Facial",
            options=["auto", "opencv", "face_recognition", "deepface"],
            index=0,
            help="Escolha o algoritmo de detecção de faces"
        )
        
        emotion_backend = st.selectbox(
            "Backend de Emoções",
            options=["auto", "deepface"],
            index=0,
            help="Escolha o algoritmo de classificação de emoções"
        )
        
        st.markdown("---")
        
        # Botão de processamento
        process_button = st.button(
            "🚀 Processar Vídeo",
            type="primary",
            disabled=uploaded_file is None,
            use_container_width=True
        )
        
        st.markdown("---")
        
        # Informações
        with st.expander("ℹ️ Sobre"):
            st.markdown("""
            **Tech Challenge Fase 4**
            
            Esta aplicação realiza:
            - 👤 Detecção de faces
            - 😊 Classificação de emoções
            - 🏃 Reconhecimento de atividades
            - ⚠️ Detecção de anomalias
            - 📊 Geração de relatórios
            
            **Desenvolvido por:** Marco Aurélio
            
            **Tecnologias:**
            - OpenCV
            - DeepFace
            - MediaPipe
            - Streamlit
            """)
    
    # Main content
    if uploaded_file is None:
        show_welcome_screen()
    elif process_button:
        process_video(uploaded_file, save_preview, face_backend, emotion_backend)
    else:
        st.info("👆 Configure as opções na barra lateral e clique em 'Processar Vídeo'")


def show_welcome_screen():
    """Mostra tela de boas-vindas."""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        ## 👋 Bem-vindo!
        
        Esta aplicação utiliza **Inteligência Artificial** para analisar vídeos e extrair insights valiosos.
        
        ### 🎯 Funcionalidades:
        
        1. **👤 Detecção de Faces**
           - Identifica rostos em cada frame
           - Rastreia múltiplas faces simultaneamente
        
        2. **😊 Análise de Emoções**
           - Classifica 7 emoções básicas
           - Calcula distribuição ao longo do vídeo
        
        3. **🏃 Reconhecimento de Atividades**
           - Detecta atividades: caminhando, sentado, gesticulando
           - Gera timeline completa
        
        4. **⚠️ Detecção de Anomalias**
           - Identifica padrões incomuns
           - Classifica por severidade
        
        5. **📊 Relatórios Detalhados**
           - Métricas completas em JSON
           - Relatório formatado em Markdown
           - Vídeo anotado com visualizações
        
        ### 🚀 Como usar:
        
        1. Carregue um vídeo na barra lateral
        2. Configure as opções de processamento
        3. Clique em "Processar Vídeo"
        4. Aguarde a análise
        5. Visualize os resultados!
        
        ---
        
        📌 **Dica:** Para melhores resultados, use vídeos com boa iluminação e pessoas visíveis.
        """)


def process_video(uploaded_file, save_preview, face_backend, emotion_backend):
    """
    Processa o vídeo carregado.
    
    Args:
        uploaded_file: Arquivo de vídeo carregado
        save_preview: Se deve salvar vídeo anotado
        face_backend: Backend de detecção facial
        emotion_backend: Backend de emoções
    """
    # Criar diretórios
    os.makedirs("temp", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    
    # Salvar arquivo temporariamente
    temp_video_path = f"temp/{uploaded_file.name}"
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Caminho do vídeo de saída
    output_video_path = None
    if save_preview:
        output_video_path = f"outputs/annotated_{uploaded_file.name}"
    
    # Progress
    st.markdown("## 🎬 Processando Vídeo...")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Criar pipeline
        status_text.text("Inicializando pipeline...")
        pipeline = InferencePipeline(
            video_path=temp_video_path,
            output_video_path=output_video_path,
            save_preview=save_preview,
            face_backend=face_backend,
            emotion_backend=emotion_backend
        )
        
        progress_bar.progress(10)
        
        # Processar vídeo
        status_text.text("Processando frames... Isso pode levar alguns minutos.")
        
        # Criar container para logs
        with st.expander("📋 Log de Processamento", expanded=False):
            log_placeholder = st.empty()
            
        summary = pipeline.run()
        
        progress_bar.progress(90)
        status_text.text("Gerando relatórios...")
        
        # Gerar relatórios
        reporter = Reporter()
        report_files = reporter.save_report_bundle(summary, output_dir="outputs")
        
        progress_bar.progress(100)
        status_text.text("✅ Processamento concluído!")
        
        # Mostrar resultados
        time.sleep(0.5)
        show_results(summary, report_files, output_video_path, save_preview)
        
    except Exception as e:
        st.error(f"❌ Erro durante processamento: {str(e)}")
        st.exception(e)
    
    finally:
        # Limpar arquivo temporário
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)


def show_results(summary, report_files, output_video_path, save_preview):
    """
    Mostra os resultados da análise.
    
    Args:
        summary: Dicionário com resumo do processamento
        report_files: Caminhos dos arquivos de relatório
        output_video_path: Caminho do vídeo anotado
        save_preview: Se o vídeo anotado foi salvo
    """
    st.markdown("---")
    st.markdown("## 📊 Resultados da Análise")
    
    # Métricas principais
    st.markdown("### 🎯 Métricas Principais")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📹 Total de Frames",
            value=f"{summary['frames_total']:,}",
            help="Número total de frames processados"
        )
    
    with col2:
        st.metric(
            label="⏱️ Duração",
            value=f"{summary['duration_seconds']:.1f}s",
            help="Duração do vídeo em segundos"
        )
    
    with col3:
        faces_stats = summary.get('faces_stats', {})
        st.metric(
            label="👤 Faces Detectadas",
            value=f"{faces_stats.get('total_detections', 0):,}",
            help="Total de detecções de faces"
        )
    
    with col4:
        st.metric(
            label="⚠️ Anomalias",
            value=f"{summary['anomalies_total']}",
            help="Total de anomalias detectadas",
            delta="Atenção" if summary['anomalies_total'] > 0 else "Normal",
            delta_color="inverse" if summary['anomalies_total'] > 0 else "normal"
        )
    
    st.markdown("---")
    
    # Tabs para diferentes visualizações
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "👤 Faces", 
        "😊 Emoções", 
        "🏃 Atividades", 
        "⚠️ Anomalias", 
        "📄 Relatórios"
    ])
    
    # Tab 1: Faces
    with tab1:
        show_faces_stats(summary)
    
    # Tab 2: Emoções
    with tab2:
        show_emotions_stats(summary)
    
    # Tab 3: Atividades
    with tab3:
        show_activities_stats(summary)
    
    # Tab 4: Anomalias
    with tab4:
        show_anomalies_stats(summary)
    
    # Tab 5: Relatórios
    with tab5:
        show_reports(report_files, output_video_path, save_preview)


def show_faces_stats(summary):
    """Mostra estatísticas de detecção de faces."""
    st.markdown("### 📊 Estatísticas de Detecção de Faces")
    
    faces_stats = summary.get('faces_stats', {})
    
    if not faces_stats:
        st.warning("Nenhuma estatística de faces disponível.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 Métricas")
        st.metric("Total de Detecções", f"{faces_stats.get('total_detections', 0):,}")
        st.metric("Média por Frame", f"{faces_stats.get('avg_faces_per_frame', 0):.2f}")
        st.metric("Máximo em um Frame", f"{faces_stats.get('max_faces_in_frame', 0)}")
    
    with col2:
        st.markdown("#### 📊 Distribuição")
        st.metric("Frames com Faces", f"{faces_stats.get('frames_with_faces', 0):,}")
        st.metric("Frames sem Faces", f"{faces_stats.get('frames_without_faces', 0):,}")
        
        # Calcular percentual
        total_frames = faces_stats.get('frames_with_faces', 0) + faces_stats.get('frames_without_faces', 0)
        if total_frames > 0:
            percent_with_faces = (faces_stats.get('frames_with_faces', 0) / total_frames) * 100
            st.metric("% com Faces", f"{percent_with_faces:.1f}%")


def show_emotions_stats(summary):
    """Mostra estatísticas de emoções."""
    st.markdown("### 😊 Distribuição de Emoções")
    
    emotions = summary.get('emotions_distribution', {})
    
    if not emotions:
        st.warning("Nenhuma emoção detectada.")
        return
    
    # Ordenar emoções por contagem
    sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
    total_emotions = sum(emotions.values())
    
    # Criar visualização
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Chart
        import pandas as pd
        
        df = pd.DataFrame(sorted_emotions, columns=['Emoção', 'Contagem'])
        df['Percentual'] = (df['Contagem'] / total_emotions * 100).round(1)
        
        st.bar_chart(df.set_index('Emoção')['Contagem'])
    
    with col2:
        st.markdown("#### 📊 Detalhes")
        
        emoji_map = {
            'happy': '😊',
            'sad': '😢',
            'angry': '😠',
            'fear': '😨',
            'surprise': '😲',
            'disgust': '🤢',
            'neutral': '😐'
        }
        
        for emotion, count in sorted_emotions:
            percentage = (count / total_emotions * 100)
            emoji = emoji_map.get(emotion, '😐')
            st.markdown(f"**{emoji} {emotion.capitalize()}**: {count} ({percentage:.1f}%)")


def show_activities_stats(summary):
    """Mostra estatísticas de atividades."""
    st.markdown("### 🏃 Timeline de Atividades")
    
    activities = summary.get('activities_timeline', [])
    
    if not activities:
        st.info("Nenhuma atividade detectada.")
        return
    
    # Contar por tipo
    activity_counts = {}
    for activity in activities:
        label = activity.get('label', 'unknown')
        activity_counts[label] = activity_counts.get(label, 0) + 1
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### 📊 Resumo")
        st.metric("Total de Eventos", len(activities))
        
        emoji_map = {
            'walking': '🚶',
            'sitting': '🪑',
            'gesturing': '👋'
        }
        
        for activity_type, count in activity_counts.items():
            emoji = emoji_map.get(activity_type, '🤷')
            st.markdown(f"**{emoji} {activity_type.capitalize()}**: {count} eventos")
    
    with col2:
        st.markdown("#### 📋 Primeiros Eventos")
        
        # Mostrar primeiros 10 eventos
        for i, activity in enumerate(activities[:10]):
            label = activity.get('label', 'unknown')
            start = activity.get('start', 0)
            end = activity.get('end', 0)
            score = activity.get('score', 0)
            
            emoji = emoji_map.get(label, '🤷')
            
            st.markdown(
                f"{i+1}. {emoji} **{label.capitalize()}** "
                f"(frames {start}-{end}, score: {score:.2f})"
            )
        
        if len(activities) > 10:
            st.markdown(f"_... e mais {len(activities) - 10} eventos_")


def show_anomalies_stats(summary):
    """Mostra estatísticas de anomalias."""
    st.markdown("### ⚠️ Anomalias Detectadas")
    
    anomalies_total = summary.get('anomalies_total', 0)
    anomalies_by_severity = summary.get('anomalies_by_severity', {})
    
    if anomalies_total == 0:
        st.success("✅ Nenhuma anomalia detectada! O vídeo apresenta padrões normais.")
        return
    
    # Métricas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total", anomalies_total)
    
    with col2:
        high = anomalies_by_severity.get('high', 0)
        st.metric("🔴 Alta", high, delta="Crítico" if high > 0 else None)
    
    with col3:
        medium = anomalies_by_severity.get('medium', 0)
        st.metric("🟡 Média", medium)
    
    with col4:
        low = anomalies_by_severity.get('low', 0)
        st.metric("🟢 Baixa", low)
    
    # Alertas
    if anomalies_by_severity.get('high', 0) > 0:
        st.error(f"⚠️ **Atenção**: {anomalies_by_severity.get('high', 0)} anomalia(s) de alta severidade detectada(s)!")
    elif anomalies_by_severity.get('medium', 0) > 0:
        st.warning(f"⚠️ {anomalies_by_severity.get('medium', 0)} anomalia(s) de média severidade detectada(s).")
    else:
        st.info("ℹ️ Apenas anomalias de baixa severidade detectadas.")


def show_reports(report_files, output_video_path, save_preview):
    """Mostra links para downloads de relatórios."""
    st.markdown("### 📄 Downloads")
    
    col1, col2, col3 = st.columns(3)
    
    # JSON
    with col1:
        st.markdown("#### 📊 Métricas JSON")
        json_path = report_files.get('metrics_json')
        if json_path and os.path.exists(json_path):
            with open(json_path, 'r') as f:
                json_data = f.read()
            
            st.download_button(
                label="⬇️ Download JSON",
                data=json_data,
                file_name="metrics.json",
                mime="application/json"
            )
            
            # Preview
            with st.expander("👁️ Preview"):
                st.json(json.loads(json_data))
    
    # Markdown
    with col2:
        st.markdown("#### 📝 Relatório MD")
        md_path = report_files.get('report_markdown')
        if md_path and os.path.exists(md_path):
            with open(md_path, 'r', encoding='utf-8') as f:
                md_data = f.read()
            
            st.download_button(
                label="⬇️ Download Markdown",
                data=md_data,
                file_name="report.md",
                mime="text/markdown"
            )
            
            # Preview
            with st.expander("👁️ Preview"):
                st.markdown(md_data)
    
    # Vídeo
    with col3:
        st.markdown("#### 🎬 Vídeo Anotado")
        if save_preview and output_video_path and os.path.exists(output_video_path):
            with open(output_video_path, 'rb') as f:
                video_data = f.read()
            
            st.download_button(
                label="⬇️ Download Vídeo",
                data=video_data,
                file_name=os.path.basename(output_video_path),
                mime="video/mp4"
            )
            
            # Preview
            with st.expander("👁️ Preview"):
                st.video(output_video_path)
        else:
            st.info("Vídeo anotado não foi gerado")


if __name__ == '__main__':
    main()
