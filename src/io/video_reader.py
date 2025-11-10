"""
Video Reader Module

Implementa a classe VideoReader para leitura eficiente de vídeos usando OpenCV.
"""

import os
from pathlib import Path
from typing import Iterator, Tuple

import cv2
import numpy as np


class VideoReaderError(Exception):
    """Exceção base para erros do VideoReader."""
    pass


class VideoNotFoundError(VideoReaderError):
    """Exceção lançada quando o arquivo de vídeo não é encontrado."""
    pass


class VideoOpenError(VideoReaderError):
    """Exceção lançada quando o vídeo não pode ser aberto."""
    pass


class VideoReader:
    """
    Classe para leitura de vídeos frame a frame.
    
    Permite iteração sobre frames de vídeo com informações de índice e timestamp.
    Valida a existência e abertura do arquivo de vídeo.
    
    Attributes:
        path (str): Caminho para o arquivo de vídeo
        _cap (cv2.VideoCapture): Objeto de captura do OpenCV
        _fps (float): Frames por segundo do vídeo
        _frame_count (int): Número total de frames
        
    Example:
        >>> reader = VideoReader("video.mp4")
        >>> for idx, frame, timestamp in reader:
        ...     print(f"Frame {idx} at {timestamp:.2f}s")
        ...     # Processar frame
    """
    
    def __init__(self, path: str) -> None:
        """
        Inicializa o VideoReader e valida o arquivo de vídeo.
        
        Args:
            path: Caminho para o arquivo de vídeo
            
        Raises:
            VideoNotFoundError: Se o arquivo não existir
            VideoOpenError: Se o vídeo não puder ser aberto pelo OpenCV
        """
        self.path = path
        self._validate_file()
        self._cap = self._open_video()
        self._fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
    def _validate_file(self) -> None:
        """
        Valida se o arquivo existe e é acessível.
        
        Raises:
            VideoNotFoundError: Se o arquivo não existir ou não for acessível
        """
        if not os.path.exists(self.path):
            raise VideoNotFoundError(
                f"Video file not found: {self.path}"
            )
        
        if not os.path.isfile(self.path):
            raise VideoNotFoundError(
                f"Path is not a file: {self.path}"
            )
        
        if not os.access(self.path, os.R_OK):
            raise VideoNotFoundError(
                f"Video file is not readable: {self.path}"
            )
    
    def _open_video(self) -> cv2.VideoCapture:
        """
        Abre o vídeo usando OpenCV.
        
        Returns:
            Objeto VideoCapture configurado
            
        Raises:
            VideoOpenError: Se o vídeo não puder ser aberto
        """
        cap = cv2.VideoCapture(self.path)
        
        if not cap.isOpened():
            raise VideoOpenError(
                f"Failed to open video file: {self.path}. "
                "The file may be corrupted or in an unsupported format."
            )
        
        # Validar que conseguimos ler propriedades básicas
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        
        if fps <= 0:
            cap.release()
            raise VideoOpenError(
                f"Invalid FPS value ({fps}) for video: {self.path}"
            )
        
        if frame_count <= 0:
            cap.release()
            raise VideoOpenError(
                f"Invalid frame count ({frame_count}) for video: {self.path}"
            )
        
        return cap
    
    def __iter__(self) -> Iterator[Tuple[int, np.ndarray, float]]:
        """
        Itera sobre os frames do vídeo.
        
        Yields:
            Tuple contendo:
                - idx (int): Índice do frame (começando em 0)
                - frame (np.ndarray): Frame como array NumPy (BGR)
                - ts_sec (float): Timestamp em segundos
                
        Example:
            >>> for idx, frame, ts in reader:
            ...     print(f"Frame {idx} at {ts:.2f}s, shape: {frame.shape}")
        """
        # Reset para o início do vídeo
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        idx = 0
        while True:
            ret, frame = self._cap.read()
            
            if not ret:
                break
            
            # Calcular timestamp baseado no índice e FPS
            ts_sec = idx / self._fps if self._fps > 0 else 0.0
            
            yield idx, frame, ts_sec
            idx += 1
    
    def fps(self) -> float:
        """
        Retorna a taxa de frames por segundo do vídeo.
        
        Returns:
            Taxa de FPS do vídeo
        """
        return self._fps
    
    def frame_count(self) -> int:
        """
        Retorna o número total de frames no vídeo.
        
        Returns:
            Número total de frames
        """
        return self._frame_count
    
    def duration(self) -> float:
        """
        Retorna a duração total do vídeo em segundos.
        
        Returns:
            Duração em segundos
        """
        return self._frame_count / self._fps if self._fps > 0 else 0.0
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - libera recursos."""
        self.release()
    
    def release(self) -> None:
        """
        Libera os recursos do VideoCapture.
        
        Deve ser chamado quando terminar de usar o VideoReader.
        """
        if hasattr(self, '_cap') and self._cap is not None:
            self._cap.release()
    
    def __del__(self):
        """Destrutor - garante que os recursos sejam liberados."""
        self.release()
    
    def __repr__(self) -> str:
        """Representação em string do VideoReader."""
        return (
            f"VideoReader(path='{self.path}', "
            f"fps={self._fps:.2f}, "
            f"frames={self._frame_count}, "
            f"duration={self.duration():.2f}s)"
        )
