"""
Inference Pipeline Module

Orquestra todos os componentes para processar vídeos end-to-end.
"""

import os
from typing import Optional, Dict, Any
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.io.video_reader import VideoReader
from src.io.writer import VideoWriter
from src.face.detector import FaceDetector
from src.emotion.classifier import EmotionClassifier
from src.activity.recognizer import ActivityRecognizer
from src.pipeline.anomaly_detector import AnomalyDetector
from src.pipeline.summarizer import Summarizer
from src.utils.viz import draw_box_and_label, put_hud, COLORS


class InferencePipeline:
    """
    Pipeline de inferência completo para análise de vídeo.
    
    Orquestra todos os componentes:
    - VideoReader para leitura
    - FaceDetector para detecção de faces
    - EmotionClassifier para classificação de emoções
    - ActivityRecognizer para reconhecimento de atividades
    - AnomalyDetector para detecção de anomalias
    - Summarizer para agregação de resultados
    
    Example:
        >>> pipeline = InferencePipeline(
        ...     video_path="input.mp4",
        ...     output_video_path="output.mp4"
        ... )
        >>> summary = pipeline.run()
        >>> print(f"Processed {summary['frames_total']} frames")
    """
    
    def __init__(
        self,
        video_path: str,
        output_video_path: Optional[str] = None,
        save_preview: bool = True,
        face_backend: str = "auto",
        emotion_backend: str = "auto"
    ):
        """
        Inicializa o pipeline de inferência.
        
        Args:
            video_path: Caminho do vídeo de entrada
            output_video_path: Caminho do vídeo de saída (opcional)
            save_preview: Se deve salvar vídeo com anotações
            face_backend: Backend para detecção de faces
            emotion_backend: Backend para classificação de emoções
        """
        self.video_path = video_path
        self.output_video_path = output_video_path
        self.save_preview = save_preview
        
        # Inicializar componentes
        self.video_reader = VideoReader(video_path)
        self.face_detector = FaceDetector(backend=face_backend)
        self.emotion_classifier = EmotionClassifier(backend=emotion_backend)
        self.activity_recognizer = ActivityRecognizer(window_size=30, stride=15)
        self.anomaly_detector = AnomalyDetector(window_size=50, z_threshold=2.5)
        self.summarizer = Summarizer(video_path)
        
        # Video writer (inicializado depois)
        self.video_writer: Optional[VideoWriter] = None
    
    def run(self) -> Dict[str, Any]:
        """
        Executa o pipeline completo de processamento.
        
        Returns:
            Dicionário com resumo completo do processamento
        """
        print(f"🎬 Iniciando processamento de: {self.video_path}")
        print(f"📊 FPS: {self.video_reader.fps():.2f}")
        print(f"🎞️  Total de frames: {self.video_reader.frame_count()}")
        print(f"⏱️  Duração: {self.video_reader.duration():.2f}s")
        print()
        
        # Configurar video writer se necessário
        if self.save_preview and self.output_video_path:
            self._setup_video_writer()
        
        # Processar frames
        try:
            self._process_frames()
        finally:
            # Garantir limpeza de recursos
            self.video_reader.release()
            if self.video_writer:
                self.video_writer.release()
        
        # Gerar resumo final
        summary = self.summarizer.generate_summary(
            fps=self.video_reader.fps(),
            total_frames=self.video_reader.frame_count()
        )
        
        print()
        print("✅ Processamento concluído!")
        print(f"📈 Frames processados: {summary.frames_total}")
        print(f"⚠️  Anomalias detectadas: {summary.anomalies_total}")
        
        return summary.to_dict()
    
    def _setup_video_writer(self):
        """Configura o video writer para salvar o vídeo anotado."""
        # Ler um frame para obter dimensões
        for idx, frame, ts in self.video_reader:
            frame_height, frame_width = frame.shape[:2]
            break
        
        # Reset do video reader
        self.video_reader.release()
        self.video_reader = VideoReader(self.video_path)
        
        # Criar video writer
        self.video_writer = VideoWriter(
            path=self.output_video_path,
            fps=self.video_reader.fps(),
            frame_size=(frame_width, frame_height),
            codec="mp4v"
        )
        
        print(f"💾 Vídeo anotado será salvo em: {self.output_video_path}")
        print()
    
    def _process_frames(self):
        """Processa todos os frames do vídeo."""
        total_frames = self.video_reader.frame_count()
        
        # Barra de progresso
        with tqdm(total=total_frames, desc="Processando frames", unit="frame") as pbar:
            for idx, frame, timestamp in self.video_reader:
                # Processar frame
                annotated_frame = self._process_single_frame(idx, frame, timestamp)
                
                # Salvar frame anotado se configurado
                if self.video_writer and annotated_frame is not None:
                    self.video_writer.write(annotated_frame)
                
                # Atualizar barra
                pbar.update(1)
    
    def _process_single_frame(
        self,
        idx: int,
        frame: np.ndarray,
        timestamp: float
    ) -> Optional[np.ndarray]:
        """
        Processa um único frame através de todo o pipeline.
        
        Args:
            idx: Índice do frame
            frame: Frame a processar
            timestamp: Timestamp em segundos
            
        Returns:
            Frame anotado ou None se não deve salvar
        """
        # 1. Detectar faces
        faces = self.face_detector.detect(frame)
        
        # 2. Classificar emoções
        emotions = self.emotion_classifier.predict(frame, faces)
        
        # 3. Reconhecer atividades (sliding window)
        activities = self.activity_recognizer.update(idx, frame)
        
        # 4. Detectar anomalias
        metrics = {
            'faces_count': len(faces),
            'avg_emotion_score': self._compute_avg_emotion_score(emotions)
        }
        anomalies = self.anomaly_detector.update(idx, metrics)
        
        # 5. Adicionar ao summarizer
        self.summarizer.add_frame_data(idx, faces, emotions)
        if activities:
            self.summarizer.add_activities(activities)
        if anomalies:
            self.summarizer.add_anomalies(anomalies)
        
        # 6. Anotar frame para visualização
        if self.save_preview:
            return self._annotate_frame(
                frame, idx, timestamp, faces, emotions, activities, anomalies
            )
        
        return None
    
    def _compute_avg_emotion_score(self, emotions: list) -> float:
        """Calcula score médio de emoções."""
        if not emotions:
            return 0.0
        
        scores = [e.normalized_score for e in emotions if hasattr(e, 'normalized_score')]
        return sum(scores) / len(scores) if scores else 0.0
    
    def _annotate_frame(
        self,
        frame: np.ndarray,
        idx: int,
        timestamp: float,
        faces: list,
        emotions: list,
        activities: list,
        anomalies: list
    ) -> np.ndarray:
        """
        Anota frame com detecções e informações.
        
        Args:
            frame: Frame original
            idx: Índice do frame
            timestamp: Timestamp em segundos
            faces: Lista de faces detectadas
            emotions: Lista de emoções classificadas
            activities: Lista de atividades detectadas
            anomalies: Lista de anomalias detectadas
            
        Returns:
            Frame anotado
        """
        output = frame.copy()
        
        # Desenhar faces e emoções
        for i, face in enumerate(faces):
            # Determinar label
            label = f"Face {i+1}"
            color = COLORS['green']
            
            # Adicionar emoção se disponível
            if i < len(emotions):
                emotion = emotions[i]
                label += f" - {emotion.label}"
                score = emotion.normalized_score
                label += f" ({score:.2f})"
                
                # Cor baseada em emoção
                emotion_colors = {
                    'happy': COLORS['green'],
                    'sad': COLORS['blue'],
                    'angry': COLORS['red'],
                    'fear': COLORS['purple'],
                    'surprise': COLORS['yellow'],
                    'disgust': COLORS['orange'],
                    'neutral': COLORS['gray']
                }
                color = emotion_colors.get(emotion.label, COLORS['green'])
            
            # Desenhar box e label
            output = draw_box_and_label(output, face.box, label, color=color)
        
        # Criar HUD com estatísticas
        hud_stats = {
            'Frame': idx,
            'Time': f"{timestamp:.2f}s",
            'Faces': len(faces),
            'FPS': self.video_reader.fps()
        }
        
        # Adicionar atividade atual se houver
        if activities:
            last_activity = activities[-1]
            hud_stats['Activity'] = last_activity['label']
        
        # Adicionar indicador de anomalia
        if anomalies:
            hud_stats['⚠️ Anomalies'] = len(anomalies)
        
        # Desenhar HUD
        output = put_hud(output, hud_stats, position="top-left")
        
        return output
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Retorna resumo atual do processamento.
        
        Returns:
            Dicionário com resumo
        """
        summary = self.summarizer.generate_summary(
            fps=self.video_reader.fps(),
            total_frames=self.video_reader.frame_count()
        )
        return summary.to_dict()
    
    def __repr__(self) -> str:
        """Representação em string do pipeline."""
        return f"InferencePipeline(video='{self.video_path}')"
