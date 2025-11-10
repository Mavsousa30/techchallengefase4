"""
Video Writer Module

Implementa a classe VideoWriter para salvar vídeos anotados usando OpenCV.
"""

import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


class VideoWriterError(Exception):
    """Exceção base para erros do VideoWriter."""
    pass


class VideoWriter:
    """
    Classe para salvar vídeos frame a frame.
    
    Permite escrever frames processados em um arquivo de vídeo,
    com configuração de codec, fps e resolução.
    
    Attributes:
        path (str): Caminho para o arquivo de vídeo de saída
        fps (float): Frames por segundo
        frame_size (tuple): Dimensões (width, height)
        codec (str): Codec de vídeo (4 caracteres)
        _writer (cv2.VideoWriter): Objeto VideoWriter do OpenCV
        
    Example:
        >>> writer = VideoWriter("output.mp4", fps=30.0, frame_size=(640, 480))
        >>> writer.write(frame)
        >>> writer.release()
        
        # Ou com context manager:
        >>> with VideoWriter("output.mp4", fps=30.0, frame_size=(640, 480)) as writer:
        ...     writer.write(frame)
    """
    
    def __init__(
        self,
        path: str,
        fps: float,
        frame_size: tuple[int, int],
        codec: str = "mp4v"
    ) -> None:
        """
        Inicializa o VideoWriter.
        
        Args:
            path: Caminho para o arquivo de vídeo de saída
            fps: Taxa de frames por segundo
            frame_size: Dimensões do vídeo (width, height)
            codec: Codec de vídeo (4 caracteres). Opções comuns:
                   - "mp4v" (MPEG-4, padrão)
                   - "XVID" (XVID MPEG-4)
                   - "avc1" (H.264)
                   - "MJPG" (Motion JPEG)
                   
        Raises:
            VideoWriterError: Se não conseguir criar o writer
        """
        self.path = path
        self.fps = fps
        self.frame_size = frame_size
        self.codec = codec
        
        # Criar diretório se não existir
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        # Inicializar VideoWriter
        self._writer = self._create_writer()
        
        if not self._writer.isOpened():
            raise VideoWriterError(
                f"Failed to create VideoWriter for {path}. "
                f"Check codec '{codec}' and path permissions."
            )
    
    def _create_writer(self) -> cv2.VideoWriter:
        """
        Cria o objeto VideoWriter do OpenCV.
        
        Returns:
            Objeto VideoWriter configurado
        """
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        writer = cv2.VideoWriter(
            self.path,
            fourcc,
            self.fps,
            self.frame_size
        )
        return writer
    
    def write(self, frame: np.ndarray) -> None:
        """
        Escreve um frame no vídeo.
        
        Args:
            frame: Frame a ser escrito (numpy array BGR)
            
        Raises:
            ValueError: Se o frame for inválido
            VideoWriterError: Se falhar ao escrever
        """
        if frame is None or frame.size == 0:
            raise ValueError("Frame is None or empty")
        
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Frame must be BGR (H, W, 3), got shape {frame.shape}")
        
        # Verificar/redimensionar se necessário
        frame_height, frame_width = frame.shape[:2]
        expected_width, expected_height = self.frame_size
        
        if (frame_width, frame_height) != self.frame_size:
            # Redimensionar para o tamanho correto
            frame = cv2.resize(frame, self.frame_size)
        
        # Escrever frame
        success = self._writer.write(frame)
        
        if not success:
            raise VideoWriterError(f"Failed to write frame to {self.path}")
    
    def release(self) -> None:
        """
        Libera os recursos do VideoWriter.
        
        Deve ser chamado quando terminar de escrever o vídeo.
        """
        if hasattr(self, '_writer') and self._writer is not None:
            self._writer.release()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - libera recursos."""
        self.release()
    
    def __del__(self):
        """Destrutor - garante que os recursos sejam liberados."""
        self.release()
    
    def is_opened(self) -> bool:
        """
        Verifica se o writer está aberto e pronto para escrever.
        
        Returns:
            True se aberto, False caso contrário
        """
        return self._writer is not None and self._writer.isOpened()
    
    def __repr__(self) -> str:
        """Representação em string do VideoWriter."""
        return (
            f"VideoWriter(path='{self.path}', fps={self.fps}, "
            f"frame_size={self.frame_size}, codec='{self.codec}')"
        )
