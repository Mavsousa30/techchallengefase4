"""
Summarizer Module

Agrega resultados do processamento de vídeo e gera resumos estruturados.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from collections import Counter, defaultdict

import numpy as np


@dataclass
class VideoSummary:
    """
    Representa o resumo completo do processamento de um vídeo.
    
    Attributes:
        video_path: Caminho do vídeo processado
        frames_total: Total de frames processados
        duration_seconds: Duração do vídeo em segundos
        fps: Taxa de frames por segundo
        anomalies_total: Total de anomalias detectadas
        faces_stats: Estatísticas de detecção de faces
        emotions_distribution: Distribuição de emoções detectadas
        activities_timeline: Timeline de atividades detectadas
        anomalies_by_severity: Anomalias agrupadas por severidade
    """
    video_path: str
    frames_total: int
    duration_seconds: float
    fps: float
    anomalies_total: int
    faces_stats: Dict
    emotions_distribution: Dict[str, int]
    activities_timeline: List[Dict]
    anomalies_by_severity: Dict[str, int]
    
    def to_dict(self) -> dict:
        """Converte para dicionário."""
        return {
            'video_path': self.video_path,
            'frames_total': self.frames_total,
            'duration_seconds': self.duration_seconds,
            'fps': self.fps,
            'anomalies_total': self.anomalies_total,
            'faces_stats': self.faces_stats,
            'emotions_distribution': self.emotions_distribution,
            'activities_timeline': self.activities_timeline,
            'anomalies_by_severity': self.anomalies_by_severity
        }


class Summarizer:
    """
    Classe para agregar e sumarizar resultados do processamento de vídeo.
    
    Coleta dados de faces, emoções, atividades e anomalias e gera
    estatísticas agregadas e resumos estruturados.
    
    Example:
        >>> summarizer = Summarizer("video.mp4")
        >>> summarizer.add_frame_data(frame_idx=0, faces=[...], emotions=[...])
        >>> summarizer.add_activities([...])
        >>> summarizer.add_anomalies([...])
        >>> summary = summarizer.generate_summary(fps=30.0, total_frames=300)
    """
    
    def __init__(self, video_path: str):
        """
        Inicializa o Summarizer.
        
        Args:
            video_path: Caminho do vídeo sendo processado
        """
        self.video_path = video_path
        
        # Dados agregados
        self.frames_processed = 0
        self.faces_per_frame: List[int] = []
        self.emotions_list: List[str] = []
        self.activities_list: List[Dict] = []
        self.anomalies_list: List[Dict] = []
    
    def add_frame_data(
        self,
        frame_idx: int,
        faces: List,
        emotions: List
    ) -> None:
        """
        Adiciona dados de um frame processado.
        
        Args:
            frame_idx: Índice do frame
            faces: Lista de faces detectadas no frame
            emotions: Lista de emoções classificadas no frame
        """
        self.frames_processed += 1
        self.faces_per_frame.append(len(faces))
        
        # Adicionar emoções
        for emotion in emotions:
            if hasattr(emotion, 'label'):
                self.emotions_list.append(emotion.label)
            elif isinstance(emotion, dict) and 'label' in emotion:
                self.emotions_list.append(emotion['label'])
    
    def add_activities(self, activities: List[Dict]) -> None:
        """
        Adiciona atividades detectadas.
        
        Args:
            activities: Lista de eventos de atividade
        """
        self.activities_list.extend(activities)
    
    def add_anomalies(self, anomalies: List) -> None:
        """
        Adiciona anomalias detectadas.
        
        Args:
            anomalies: Lista de anomalias
        """
        for anomaly in anomalies:
            if hasattr(anomaly, 'to_dict'):
                self.anomalies_list.append(anomaly.to_dict())
            elif isinstance(anomaly, dict):
                self.anomalies_list.append(anomaly)
    
    def generate_summary(
        self,
        fps: float,
        total_frames: int
    ) -> VideoSummary:
        """
        Gera resumo completo do processamento.
        
        Args:
            fps: Taxa de frames por segundo do vídeo
            total_frames: Total de frames no vídeo
            
        Returns:
            Objeto VideoSummary com estatísticas agregadas
        """
        duration_seconds = total_frames / fps if fps > 0 else 0.0
        
        # Estatísticas de faces
        faces_stats = self._compute_faces_stats()
        
        # Distribuição de emoções
        emotions_distribution = self._compute_emotions_distribution()
        
        # Timeline de atividades
        activities_timeline = self._process_activities_timeline()
        
        # Anomalias por severidade
        anomalies_by_severity = self._compute_anomalies_by_severity()
        
        return VideoSummary(
            video_path=self.video_path,
            frames_total=total_frames,
            duration_seconds=duration_seconds,
            fps=fps,
            anomalies_total=len(self.anomalies_list),
            faces_stats=faces_stats,
            emotions_distribution=emotions_distribution,
            activities_timeline=activities_timeline,
            anomalies_by_severity=anomalies_by_severity
        )
    
    def _compute_faces_stats(self) -> Dict:
        """Calcula estatísticas de detecção de faces."""
        if not self.faces_per_frame:
            return {
                'total_detections': 0,
                'avg_faces_per_frame': 0.0,
                'max_faces_in_frame': 0,
                'frames_with_faces': 0,
                'frames_without_faces': 0
            }
        
        total_detections = sum(self.faces_per_frame)
        avg_faces = np.mean(self.faces_per_frame)
        max_faces = max(self.faces_per_frame)
        frames_with_faces = sum(1 for count in self.faces_per_frame if count > 0)
        frames_without_faces = sum(1 for count in self.faces_per_frame if count == 0)
        
        return {
            'total_detections': total_detections,
            'avg_faces_per_frame': float(avg_faces),
            'max_faces_in_frame': max_faces,
            'frames_with_faces': frames_with_faces,
            'frames_without_faces': frames_without_faces
        }
    
    def _compute_emotions_distribution(self) -> Dict[str, int]:
        """Calcula distribuição de emoções detectadas."""
        if not self.emotions_list:
            return {}
        
        return dict(Counter(self.emotions_list))
    
    def _process_activities_timeline(self) -> List[Dict]:
        """Processa e organiza timeline de atividades."""
        # Atividades já estão em formato de dicionário
        return self.activities_list.copy()
    
    def _compute_anomalies_by_severity(self) -> Dict[str, int]:
        """Agrupa anomalias por severidade."""
        severity_counter = Counter(
            anomaly.get('severity', 'medium') 
            for anomaly in self.anomalies_list
        )
        
        return {
            'low': severity_counter.get('low', 0),
            'medium': severity_counter.get('medium', 0),
            'high': severity_counter.get('high', 0)
        }
    
    def get_metrics_dict(self, fps: float, total_frames: int) -> Dict:
        """
        Retorna métricas em formato de dicionário simples.
        
        Args:
            fps: Taxa de frames por segundo
            total_frames: Total de frames
            
        Returns:
            Dicionário com métricas principais
        """
        summary = self.generate_summary(fps, total_frames)
        return summary.to_dict()
    
    def get_top_emotions(self, n: int = 3) -> List[tuple[str, int]]:
        """
        Retorna as top N emoções mais frequentes.
        
        Args:
            n: Número de emoções a retornar
            
        Returns:
            Lista de tuplas (emoção, contagem)
        """
        if not self.emotions_list:
            return []
        
        counter = Counter(self.emotions_list)
        return counter.most_common(n)
    
    def get_activity_summary(self) -> Dict[str, int]:
        """
        Retorna resumo de atividades (contagem por tipo).
        
        Returns:
            Dicionário com contagem de cada tipo de atividade
        """
        activity_counts = Counter(
            activity.get('label', 'unknown')
            for activity in self.activities_list
        )
        return dict(activity_counts)
    
    def reset(self):
        """Reseta o summarizer, limpando todos os dados."""
        self.frames_processed = 0
        self.faces_per_frame.clear()
        self.emotions_list.clear()
        self.activities_list.clear()
        self.anomalies_list.clear()
    
    def __repr__(self) -> str:
        """Representação em string do Summarizer."""
        return (
            f"Summarizer(video='{self.video_path}', "
            f"frames_processed={self.frames_processed}, "
            f"anomalies={len(self.anomalies_list)})"
        )
