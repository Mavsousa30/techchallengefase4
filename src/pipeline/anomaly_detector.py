"""
Anomaly Detection Module

Implementa detector de anomalias em sequências temporais de métricas.
Usa z-score para identificar valores atípicos.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from collections import deque
import statistics

import numpy as np


@dataclass
class Anomaly:
    """
    Representa uma anomalia detectada.
    
    Attributes:
        frame_idx: Índice do frame onde ocorreu a anomalia
        metric_name: Nome da métrica que apresentou anomalia
        value: Valor observado
        expected_range: Tupla (min, max) do range esperado
        z_score: Score z da anomalia
        severity: Gravidade ('low', 'medium', 'high')
    """
    frame_idx: int
    metric_name: str
    value: float
    expected_range: tuple[float, float]
    z_score: float
    severity: str = 'medium'
    
    def __post_init__(self):
        """Valida os valores após inicialização."""
        if self.severity not in ('low', 'medium', 'high'):
            raise ValueError("Severity must be 'low', 'medium', or 'high'")
    
    def to_dict(self) -> dict:
        """Converte para dicionário."""
        return {
            'frame_idx': self.frame_idx,
            'metric_name': self.metric_name,
            'value': self.value,
            'expected_range': self.expected_range,
            'z_score': self.z_score,
            'severity': self.severity
        }


class AnomalyDetector:
    """
    Detector de anomalias usando z-score em séries temporais.
    
    Mantém buffer de valores históricos para cada métrica e detecta
    valores que desviam significativamente da média.
    
    Attributes:
        window_size: Tamanho da janela para cálculo de estatísticas
        z_threshold: Threshold do z-score para considerar anomalia
        min_samples: Mínimo de amostras antes de começar detecção
        
    Example:
        >>> detector = AnomalyDetector(window_size=50, z_threshold=2.5)
        >>> anomalies = detector.update(frame_idx=10, metrics={'faces': 5})
        >>> for anomaly in anomalies:
        ...     print(f"Anomaly at frame {anomaly.frame_idx}: {anomaly.metric_name}")
    """
    
    def __init__(
        self,
        window_size: int = 50,
        z_threshold: float = 2.5,
        min_samples: int = 10
    ):
        """
        Inicializa o detector de anomalias.
        
        Args:
            window_size: Tamanho da janela deslizante para estatísticas
            z_threshold: Threshold z-score para detectar anomalias
            min_samples: Mínimo de amostras antes de começar a detectar
        """
        self.window_size = window_size
        self.z_threshold = z_threshold
        self.min_samples = min_samples
        
        # Buffers para cada métrica
        self.metrics_buffers: Dict[str, deque] = {}
        
        # Lista de todas as anomalias detectadas
        self.anomalies: List[Anomaly] = []
    
    def update(
        self,
        frame_idx: int,
        metrics: Dict[str, float]
    ) -> List[Anomaly]:
        """
        Atualiza o detector com novas métricas e detecta anomalias.
        
        Args:
            frame_idx: Índice do frame atual
            metrics: Dicionário com métricas do frame
                    Ex: {'faces_count': 3, 'avg_emotion_score': 0.85}
                    
        Returns:
            Lista de anomalias detectadas neste frame
        """
        frame_anomalies = []
        
        for metric_name, value in metrics.items():
            # Garantir que temos buffer para esta métrica
            if metric_name not in self.metrics_buffers:
                self.metrics_buffers[metric_name] = deque(maxlen=self.window_size)
            
            buffer = self.metrics_buffers[metric_name]
            
            # Detectar anomalia se tivermos amostras suficientes
            if len(buffer) >= self.min_samples:
                anomaly = self._check_anomaly(
                    frame_idx, metric_name, value, buffer
                )
                if anomaly:
                    frame_anomalies.append(anomaly)
                    self.anomalies.append(anomaly)
            
            # Adicionar valor ao buffer
            buffer.append(value)
        
        return frame_anomalies
    
    def _check_anomaly(
        self,
        frame_idx: int,
        metric_name: str,
        value: float,
        buffer: deque
    ) -> Optional[Anomaly]:
        """
        Verifica se um valor é anômalo baseado no buffer histórico.
        
        Args:
            frame_idx: Índice do frame
            metric_name: Nome da métrica
            value: Valor atual
            buffer: Buffer de valores históricos
            
        Returns:
            Objeto Anomaly se detectada, None caso contrário
        """
        # Calcular estatísticas do buffer
        try:
            mean = statistics.mean(buffer)
            
            # Se não há variação, não pode haver anomalia
            if len(buffer) < 2:
                return None
            
            stdev = statistics.stdev(buffer)
            
            # Se desvio padrão é zero, não há variação
            if stdev == 0:
                return None
            
            # Calcular z-score
            z_score = abs((value - mean) / stdev)
            
            # Verificar se é anomalia
            if z_score >= self.z_threshold:
                # Determinar severidade
                if z_score >= 4.0:
                    severity = 'high'
                elif z_score >= 3.0:
                    severity = 'medium'
                else:
                    severity = 'low'
                
                # Calcular range esperado (±2 desvios padrão)
                expected_min = mean - 2 * stdev
                expected_max = mean + 2 * stdev
                
                return Anomaly(
                    frame_idx=frame_idx,
                    metric_name=metric_name,
                    value=value,
                    expected_range=(expected_min, expected_max),
                    z_score=z_score,
                    severity=severity
                )
            
            return None
            
        except (statistics.StatisticsError, ZeroDivisionError):
            # Não há dados suficientes ou erro no cálculo
            return None
    
    def get_all_anomalies(self) -> List[Anomaly]:
        """
        Retorna todas as anomalias detectadas até o momento.
        
        Returns:
            Lista de todas as anomalias
        """
        return self.anomalies.copy()
    
    def get_anomalies_by_severity(self, severity: str) -> List[Anomaly]:
        """
        Retorna anomalias filtradas por severidade.
        
        Args:
            severity: 'low', 'medium' ou 'high'
            
        Returns:
            Lista de anomalias com a severidade especificada
        """
        return [a for a in self.anomalies if a.severity == severity]
    
    def get_anomalies_by_metric(self, metric_name: str) -> List[Anomaly]:
        """
        Retorna anomalias filtradas por métrica.
        
        Args:
            metric_name: Nome da métrica
            
        Returns:
            Lista de anomalias para a métrica especificada
        """
        return [a for a in self.anomalies if a.metric_name == metric_name]
    
    def get_statistics(self) -> Dict[str, Dict]:
        """
        Retorna estatísticas sobre as anomalias detectadas.
        
        Returns:
            Dicionário com estatísticas por métrica e severidade
        """
        stats = {
            'total_anomalies': len(self.anomalies),
            'by_severity': {
                'low': len(self.get_anomalies_by_severity('low')),
                'medium': len(self.get_anomalies_by_severity('medium')),
                'high': len(self.get_anomalies_by_severity('high'))
            },
            'by_metric': {}
        }
        
        # Estatísticas por métrica
        for metric_name in self.metrics_buffers.keys():
            metric_anomalies = self.get_anomalies_by_metric(metric_name)
            stats['by_metric'][metric_name] = len(metric_anomalies)
        
        return stats
    
    def reset(self):
        """Reseta o detector, limpando todos os buffers e anomalias."""
        self.metrics_buffers.clear()
        self.anomalies.clear()
    
    def __repr__(self) -> str:
        """Representação em string do detector."""
        return (
            f"AnomalyDetector(window_size={self.window_size}, "
            f"z_threshold={self.z_threshold}, "
            f"anomalies_detected={len(self.anomalies)})"
        )
