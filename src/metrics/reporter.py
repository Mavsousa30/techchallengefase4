"""
Reporter Module

Salva métricas e gera relatórios em múltiplos formatos (JSON, Markdown).
"""

import json
import os
from pathlib import Path
from typing import Dict, Any
from datetime import datetime


class Reporter:
    """
    Classe para salvar métricas e gerar relatórios formatados.
    
    Suporta exportação em JSON e geração de relatórios Markdown.
    
    Example:
        >>> reporter = Reporter()
        >>> reporter.save_metrics(metrics_dict, "outputs/metrics.json")
        >>> reporter.generate_markdown_report(summary, "outputs/report.md")
    """
    
    def __init__(self):
        """Inicializa o Reporter."""
        pass
    
    def save_metrics(
        self,
        metrics: Dict[str, Any],
        output_path: str
    ) -> None:
        """
        Salva métricas em arquivo JSON.
        
        Args:
            metrics: Dicionário com métricas a salvar
            output_path: Caminho do arquivo JSON de saída
        """
        # Criar diretório se não existir
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Adicionar timestamp
        metrics_with_timestamp = {
            'generated_at': datetime.now().isoformat(),
            **metrics
        }
        
        # Salvar JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_with_timestamp, f, indent=2, ensure_ascii=False)
    
    def generate_markdown_report(
        self,
        summary: Dict[str, Any],
        output_path: str
    ) -> None:
        """
        Gera relatório em formato Markdown.
        
        Args:
            summary: Dicionário com resumo do vídeo (VideoSummary.to_dict())
            output_path: Caminho do arquivo Markdown de saída
        """
        # Criar diretório se não existir
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Gerar conteúdo Markdown
        md_content = self._format_markdown_content(summary)
        
        # Salvar arquivo
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
    
    def _format_markdown_content(self, summary: Dict[str, Any]) -> str:
        """
        Formata o conteúdo do relatório em Markdown.
        
        Args:
            summary: Dicionário com dados do resumo
            
        Returns:
            String com conteúdo Markdown formatado
        """
        lines = []
        
        # Cabeçalho
        lines.append("# Relatório de Análise de Vídeo")
        lines.append("")
        lines.append(f"**Data de Geração:** {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
        lines.append("")
        
        # Informações do Vídeo
        lines.append("## 📹 Informações do Vídeo")
        lines.append("")
        lines.append(f"- **Arquivo:** `{summary.get('video_path', 'N/A')}`")
        lines.append(f"- **Total de Frames:** {summary.get('frames_total', 0):,}")
        lines.append(f"- **Duração:** {summary.get('duration_seconds', 0):.2f} segundos")
        lines.append(f"- **FPS:** {summary.get('fps', 0):.2f}")
        lines.append("")
        
        # Métricas Obrigatórias
        lines.append("## 📊 Métricas Obrigatórias")
        lines.append("")
        lines.append(f"- **frames_total:** {summary.get('frames_total', 0):,}")
        lines.append(f"- **anomalies_total:** {summary.get('anomalies_total', 0):,}")
        lines.append("")
        
        # Detecção de Faces
        faces_stats = summary.get('faces_stats', {})
        if faces_stats:
            lines.append("## 👤 Detecção de Faces")
            lines.append("")
            lines.append(f"- **Total de Detecções:** {faces_stats.get('total_detections', 0):,}")
            lines.append(f"- **Média de Faces por Frame:** {faces_stats.get('avg_faces_per_frame', 0):.2f}")
            lines.append(f"- **Máximo de Faces em um Frame:** {faces_stats.get('max_faces_in_frame', 0)}")
            lines.append(f"- **Frames com Faces:** {faces_stats.get('frames_with_faces', 0):,}")
            lines.append(f"- **Frames sem Faces:** {faces_stats.get('frames_without_faces', 0):,}")
            lines.append("")
        
        # Distribuição de Emoções
        emotions_dist = summary.get('emotions_distribution', {})
        if emotions_dist:
            lines.append("## 😊 Distribuição de Emoções")
            lines.append("")
            
            # Ordenar emoções por contagem
            sorted_emotions = sorted(
                emotions_dist.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            total_emotions = sum(emotions_dist.values())
            
            for emotion, count in sorted_emotions:
                percentage = (count / total_emotions * 100) if total_emotions > 0 else 0
                lines.append(f"- **{emotion.capitalize()}:** {count} ({percentage:.1f}%)")
            
            lines.append("")
        
        # Timeline de Atividades
        activities = summary.get('activities_timeline', [])
        if activities:
            lines.append("## 🏃 Timeline de Atividades")
            lines.append("")
            lines.append(f"**Total de Eventos:** {len(activities)}")
            lines.append("")
            
            # Agrupar por tipo de atividade
            activity_types = {}
            for activity in activities:
                label = activity.get('label', 'unknown')
                activity_types[label] = activity_types.get(label, 0) + 1
            
            for activity_type, count in activity_types.items():
                lines.append(f"- **{activity_type.capitalize()}:** {count} eventos")
            
            lines.append("")
            
            # Listar primeiros eventos (máximo 10)
            lines.append("### Primeiros Eventos Detectados")
            lines.append("")
            for i, activity in enumerate(activities[:10]):
                label = activity.get('label', 'unknown')
                start = activity.get('start', 0)
                end = activity.get('end', 0)
                score = activity.get('score', 0)
                lines.append(
                    f"{i+1}. **{label.capitalize()}** "
                    f"(frames {start}-{end}, score: {score:.2f})"
                )
            
            if len(activities) > 10:
                lines.append(f"\n_... e mais {len(activities) - 10} eventos_")
            
            lines.append("")
        
        # Anomalias
        anomalies_by_severity = summary.get('anomalies_by_severity', {})
        lines.append("## ⚠️ Anomalias Detectadas")
        lines.append("")
        lines.append(f"**Total:** {summary.get('anomalies_total', 0)}")
        lines.append("")
        
        if anomalies_by_severity:
            lines.append("### Por Severidade")
            lines.append("")
            lines.append(f"- 🔴 **Alta:** {anomalies_by_severity.get('high', 0)}")
            lines.append(f"- 🟡 **Média:** {anomalies_by_severity.get('medium', 0)}")
            lines.append(f"- 🟢 **Baixa:** {anomalies_by_severity.get('low', 0)}")
            lines.append("")
        
        # Rodapé
        lines.append("---")
        lines.append("")
        lines.append("_Relatório gerado automaticamente pelo Tech Challenge Fase 4_")
        lines.append("")
        
        return "\n".join(lines)
    
    def save_report_bundle(
        self,
        summary: Dict[str, Any],
        output_dir: str = "outputs"
    ) -> Dict[str, str]:
        """
        Salva bundle completo de relatórios (JSON + Markdown).
        
        Args:
            summary: Dicionário com resumo do vídeo
            output_dir: Diretório de saída
            
        Returns:
            Dicionário com caminhos dos arquivos gerados
        """
        # Caminhos dos arquivos
        metrics_path = os.path.join(output_dir, "metrics.json")
        report_path = os.path.join(output_dir, "report.md")
        
        # Salvar métricas JSON
        self.save_metrics(summary, metrics_path)
        
        # Salvar relatório Markdown
        self.generate_markdown_report(summary, report_path)
        
        return {
            'metrics_json': metrics_path,
            'report_markdown': report_path
        }
    
    def load_metrics(self, input_path: str) -> Dict[str, Any]:
        """
        Carrega métricas de um arquivo JSON.
        
        Args:
            input_path: Caminho do arquivo JSON
            
        Returns:
            Dicionário com métricas carregadas
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def __repr__(self) -> str:
        """Representação em string do Reporter."""
        return "Reporter(formats=['json', 'markdown'])"
