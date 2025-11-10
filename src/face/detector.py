"""
Face Detector Module

Implementa detecção de faces usando múltiplos backends (face_recognition, deepface, opencv).
"""

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


@dataclass
class Face:
    """
    Representa uma face detectada.
    
    Attributes:
        box: Bounding box da face no formato (x, y, width, height)
        score: Confiança da detecção (0.0 a 1.0)
        landmarks: Pontos faciais chave, dict com nomes dos pontos e coordenadas (x, y)
                  Ex: {'left_eye': (x, y), 'right_eye': (x, y), 'nose': (x, y), ...}
    """
    box: tuple[int, int, int, int]
    score: float
    landmarks: Optional[dict[str, tuple[int, int]]] = None
    
    def __post_init__(self):
        """Valida os valores após inicialização."""
        if len(self.box) != 4:
            raise ValueError("Box must have 4 values: (x, y, width, height)")
        
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("Score must be between 0.0 and 1.0")
    
    @property
    def center(self) -> tuple[int, int]:
        """Retorna o centro da face."""
        x, y, w, h = self.box
        return (x + w // 2, y + h // 2)
    
    @property
    def area(self) -> int:
        """Retorna a área da bounding box."""
        _, _, w, h = self.box
        return w * h


class FaceDetector:
    """
    Detector de faces com suporte a múltiplos backends.
    
    Tenta usar backends na seguinte ordem de prioridade:
    1. face_recognition (HOG/CNN)
    2. deepface (múltiplos backends)
    3. opencv (Haar Cascade como fallback)
    
    Example:
        >>> detector = FaceDetector()
        >>> faces = detector.detect(frame)
        >>> for face in faces:
        ...     x, y, w, h = face.box
        ...     print(f"Face at ({x}, {y}) with confidence {face.score:.2f}")
    """
    
    def __init__(self, backend: str = "auto", model: str = "hog"):
        """
        Inicializa o detector de faces.
        
        Args:
            backend: Backend a usar ('auto', 'face_recognition', 'deepface', 'opencv')
            model: Modelo a usar ('hog', 'cnn' para face_recognition)
        """
        self.backend = backend
        self.model = model
        self._detector = None
        
        # Tentar inicializar o backend
        self._initialize_backend()
    
    def _initialize_backend(self):
        """Inicializa o backend de detecção."""
        if self.backend == "auto":
            # Tentar backends em ordem de prioridade
            if self._try_face_recognition():
                self.backend = "face_recognition"
            elif self._try_deepface():
                self.backend = "deepface"
            else:
                self.backend = "opencv"
                self._initialize_opencv()
        elif self.backend == "face_recognition":
            if not self._try_face_recognition():
                raise RuntimeError("face_recognition backend not available")
        elif self.backend == "deepface":
            if not self._try_deepface():
                raise RuntimeError("deepface backend not available")
        elif self.backend == "opencv":
            self._initialize_opencv()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    def _try_face_recognition(self) -> bool:
        """Tenta importar e inicializar face_recognition."""
        try:
            import face_recognition
            self._face_recognition = face_recognition
            return True
        except ImportError:
            return False
    
    def _try_deepface(self) -> bool:
        """Tenta importar e inicializar deepface."""
        try:
            from deepface import DeepFace
            self._deepface = DeepFace
            return True
        except (ImportError, ValueError, Exception):
            # ValueError pode ocorrer se tensorflow não tiver tf-keras
            # Capturar Exception genérico para outros problemas de dependência
            return False
    
    def _initialize_opencv(self):
        """Inicializa o detector OpenCV Haar Cascade."""
        # Usar o classificador Haar Cascade pré-treinado do OpenCV
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self._detector = cv2.CascadeClassifier(cascade_path)
        
        if self._detector.empty():
            raise RuntimeError("Failed to load OpenCV Haar Cascade classifier")
    
    def detect(self, frame: np.ndarray) -> list[Face]:
        """
        Detecta faces em um frame.
        
        Args:
            frame: Frame de imagem (numpy array BGR)
            
        Returns:
            Lista de faces detectadas
            
        Raises:
            ValueError: Se o frame for inválido
        """
        if frame is None or frame.size == 0:
            raise ValueError("Invalid frame: empty or None")
        
        if frame.ndim not in (2, 3):
            raise ValueError(f"Invalid frame dimensions: {frame.ndim}")
        
        # Delegar para o backend apropriado
        if self.backend == "face_recognition":
            return self._detect_face_recognition(frame)
        elif self.backend == "deepface":
            return self._detect_deepface(frame)
        elif self.backend == "opencv":
            return self._detect_opencv(frame)
        else:
            raise RuntimeError(f"Backend not initialized: {self.backend}")
    
    def _detect_face_recognition(self, frame: np.ndarray) -> list[Face]:
        """Detecta faces usando face_recognition."""
        # Converter BGR para RGB (face_recognition usa RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detectar localizações de faces
        face_locations = self._face_recognition.face_locations(
            rgb_frame,
            model=self.model
        )
        
        # Detectar landmarks (opcional)
        face_landmarks_list = []
        if face_locations:
            try:
                face_landmarks_list = self._face_recognition.face_landmarks(
                    rgb_frame,
                    face_locations=face_locations
                )
            except Exception:
                # Se falhar, continuar sem landmarks
                face_landmarks_list = [None] * len(face_locations)
        
        # Converter para objetos Face
        faces = []
        for i, location in enumerate(face_locations):
            # face_locations retorna (top, right, bottom, left)
            top, right, bottom, left = location
            
            # Converter para (x, y, width, height)
            x, y = left, top
            w, h = right - left, bottom - top
            
            # face_recognition não retorna score diretamente, usar 0.99 como padrão
            score = 0.99
            
            # Processar landmarks se disponíveis
            landmarks = None
            if i < len(face_landmarks_list) and face_landmarks_list[i]:
                landmarks = self._convert_face_recognition_landmarks(
                    face_landmarks_list[i]
                )
            
            faces.append(Face(box=(x, y, w, h), score=score, landmarks=landmarks))
        
        return faces
    
    def _convert_face_recognition_landmarks(
        self,
        face_landmarks: dict
    ) -> dict[str, tuple[int, int]]:
        """
        Converte landmarks do face_recognition para formato simplificado.
        
        face_recognition retorna múltiplos pontos por feature. Aqui pegamos
        pontos representativos.
        """
        landmarks = {}
        
        # Pegar pontos médios de cada feature
        if 'left_eye' in face_landmarks and face_landmarks['left_eye']:
            points = face_landmarks['left_eye']
            avg_x = sum(p[0] for p in points) // len(points)
            avg_y = sum(p[1] for p in points) // len(points)
            landmarks['left_eye'] = (avg_x, avg_y)
        
        if 'right_eye' in face_landmarks and face_landmarks['right_eye']:
            points = face_landmarks['right_eye']
            avg_x = sum(p[0] for p in points) // len(points)
            avg_y = sum(p[1] for p in points) // len(points)
            landmarks['right_eye'] = (avg_x, avg_y)
        
        if 'nose_tip' in face_landmarks and face_landmarks['nose_tip']:
            points = face_landmarks['nose_tip']
            # Pegar o ponto do meio
            landmarks['nose'] = points[len(points) // 2]
        
        if 'top_lip' in face_landmarks and face_landmarks['top_lip']:
            points = face_landmarks['top_lip']
            landmarks['mouth_top'] = points[len(points) // 2]
        
        if 'bottom_lip' in face_landmarks and face_landmarks['bottom_lip']:
            points = face_landmarks['bottom_lip']
            landmarks['mouth_bottom'] = points[len(points) // 2]
        
        return landmarks if landmarks else None
    
    def _detect_deepface(self, frame: np.ndarray) -> list[Face]:
        """Detecta faces usando deepface."""
        try:
            # DeepFace.extract_faces retorna lista de dicts com info das faces
            results = self._deepface.extract_faces(
                img_path=frame,
                detector_backend='opencv',  # Usar opencv como backend do deepface
                enforce_detection=False,
                align=False
            )
            
            faces = []
            for result in results:
                # Extrair região facial
                facial_area = result.get('facial_area', {})
                x = facial_area.get('x', 0)
                y = facial_area.get('y', 0)
                w = facial_area.get('w', 0)
                h = facial_area.get('h', 0)
                
                # Confiança
                confidence = result.get('confidence', 0.0)
                
                # DeepFace não retorna landmarks facilmente neste método
                faces.append(Face(box=(x, y, w, h), score=confidence, landmarks=None))
            
            return faces
        
        except Exception as e:
            # Se falhar, retornar lista vazia
            return []
    
    def _detect_opencv(self, frame: np.ndarray) -> list[Face]:
        """Detecta faces usando OpenCV Haar Cascade."""
        # Converter para grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detectar faces
        detections = self._detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Converter para objetos Face
        faces = []
        for (x, y, w, h) in detections:
            # OpenCV Haar Cascade não retorna score/confidence
            # Usar 0.8 como padrão
            score = 0.8
            
            # OpenCV Haar Cascade não retorna landmarks
            faces.append(Face(box=(x, y, w, h), score=score, landmarks=None))
        
        return faces
    
    def __repr__(self) -> str:
        """Representação em string do detector."""
        return f"FaceDetector(backend='{self.backend}', model='{self.model}')"
