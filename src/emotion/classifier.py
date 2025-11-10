"""
Emotion Classifier Module

Implementa classificação de emoções faciais usando DeepFace.
"""

from dataclasses import dataclass
from typing import Optional, List

import cv2
import numpy as np


@dataclass
class EmotionResult:
    """
    Representa o resultado da classificação de emoção.
    
    Attributes:
        label: Emoção detectada (ex: 'happy', 'sad', 'angry', 'neutral', etc.)
        score: Confiança da predição (0.0 a 1.0)
        box: Bounding box da face no formato (x, y, width, height)
    """
    label: str
    score: float
    box: tuple[int, int, int, int]
    
    def __post_init__(self):
        """Valida os valores após inicialização."""
        if not self.label:
            raise ValueError("Label cannot be empty")
        
        if not 0.0 <= self.score <= 100.0:  # DeepFace retorna 0-100
            raise ValueError("Score must be between 0.0 and 100.0")
        
        if len(self.box) != 4:
            raise ValueError("Box must have 4 values: (x, y, width, height)")
    
    @property
    def normalized_score(self) -> float:
        """Retorna score normalizado entre 0.0 e 1.0."""
        return self.score / 100.0


class EmotionClassifier:
    """
    Classificador de emoções faciais usando DeepFace.
    
    Detecta emoções básicas como:
    - angry (raiva)
    - disgust (nojo)
    - fear (medo)
    - happy (felicidade)
    - sad (tristeza)
    - surprise (surpresa)
    - neutral (neutro)
    
    Example:
        >>> classifier = EmotionClassifier()
        >>> from src.face.detector import Face
        >>> faces = [Face(box=(100, 100, 200, 200), score=0.95)]
        >>> results = classifier.predict(frame, faces)
        >>> for result in results:
        ...     print(f"{result.label}: {result.normalized_score:.2f}")
    """
    
    def __init__(self, backend: str = "auto"):
        """
        Inicializa o classificador de emoções.
        
        Args:
            backend: Backend a usar ('auto', 'deepface', 'opencv')
        """
        self.backend = backend
        self._model_loaded = False
        self._deepface = None
        
        # Tentar carregar o modelo
        self._initialize_model()
    
    def _initialize_model(self):
        """Inicializa o modelo de emoções."""
        if self.backend == "auto" or self.backend == "deepface":
            if self._try_load_deepface():
                self.backend = "deepface"
                self._model_loaded = True
            else:
                self.backend = "fallback"
                self._model_loaded = False
        else:
            self.backend = "fallback"
            self._model_loaded = False
    
    def _try_load_deepface(self) -> bool:
        """Tenta carregar o DeepFace."""
        try:
            from deepface import DeepFace
            self._deepface = DeepFace
            
            # Fazer uma predição dummy para carregar o modelo
            # Isso garante que o modelo está carregado na memória
            dummy_img = np.zeros((48, 48, 3), dtype=np.uint8)
            try:
                _ = self._deepface.analyze(
                    img_path=dummy_img,
                    actions=['emotion'],
                    enforce_detection=False,
                    silent=True
                )
            except Exception:
                # Se falhar no dummy, ainda podemos tentar usar depois
                pass
            
            return True
        except (ImportError, ValueError, Exception) as e:
            # ValueError pode ocorrer se tensorflow não tiver tf-keras
            return False
    
    def predict(
        self,
        frame: np.ndarray,
        faces: List
    ) -> List[EmotionResult]:
        """
        Prediz emoções para as faces detectadas no frame.
        
        Args:
            frame: Frame de imagem (numpy array BGR)
            faces: Lista de objetos Face detectados
            
        Returns:
            Lista de EmotionResult com emoções detectadas
            
        Raises:
            ValueError: Se o frame for inválido
        """
        if frame is None or frame.size == 0:
            raise ValueError("Frame is None or empty")
        
        if not faces:
            return []
        
        results = []
        
        # Processar cada face individualmente
        for face in faces:
            try:
                emotion_result = self._predict_single_face(frame, face)
                if emotion_result:
                    results.append(emotion_result)
            except Exception as e:
                # Se falhar em uma face, continuar com as outras
                # Log do erro poderia ser adicionado aqui
                continue
        
        return results
    
    def _predict_single_face(
        self,
        frame: np.ndarray,
        face
    ) -> Optional[EmotionResult]:
        """
        Prediz emoção para uma única face.
        
        Args:
            frame: Frame completo
            face: Objeto Face com informações da detecção
            
        Returns:
            EmotionResult ou None se falhar
        """
        # Extrair região da face
        x, y, w, h = face.box
        
        # Garantir que coordenadas estão dentro dos limites
        frame_h, frame_w = frame.shape[:2]
        x = max(0, min(x, frame_w - 1))
        y = max(0, min(y, frame_h - 1))
        w = max(1, min(w, frame_w - x))
        h = max(1, min(h, frame_h - y))
        
        # Recortar face
        face_roi = frame[y:y+h, x:x+w]
        
        # Verificar se ROI é válido
        if face_roi.size == 0 or face_roi.shape[0] < 10 or face_roi.shape[1] < 10:
            return None
        
        # Classificar emoção baseado no backend
        if self.backend == "deepface" and self._model_loaded:
            return self._predict_with_deepface(face_roi, face.box)
        else:
            return self._predict_fallback(face.box)
    
    def _predict_with_deepface(
        self,
        face_roi: np.ndarray,
        box: tuple[int, int, int, int]
    ) -> Optional[EmotionResult]:
        """
        Prediz emoção usando DeepFace.
        
        Args:
            face_roi: Região da face recortada
            box: Bounding box original
            
        Returns:
            EmotionResult ou None se falhar
        """
        try:
            # Analisar emoções
            result = self._deepface.analyze(
                img_path=face_roi,
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )
            
            # DeepFace pode retornar lista ou dict
            if isinstance(result, list):
                result = result[0]
            
            # Extrair emoção dominante
            emotions = result.get('emotion', {})
            
            if not emotions:
                return None
            
            # Encontrar emoção com maior score
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])
            label, score = dominant_emotion
            
            return EmotionResult(
                label=label,
                score=float(score),
                box=box
            )
            
        except Exception as e:
            # Se falhar, retornar None
            return None
    
    def _predict_fallback(
        self,
        box: tuple[int, int, int, int]
    ) -> EmotionResult:
        """
        Fallback quando modelo não está disponível.
        Retorna resultado neutro.
        
        Args:
            box: Bounding box da face
            
        Returns:
            EmotionResult com emoção neutral
        """
        return EmotionResult(
            label="neutral",
            score=50.0,  # Score médio
            box=box
        )
    
    def predict_batch(
        self,
        frame: np.ndarray,
        faces: List
    ) -> List[EmotionResult]:
        """
        Prediz emoções em batch (otimizado para múltiplas faces).
        
        Atualmente delega para predict() individual.
        Implementação de batch real poderia ser adicionada no futuro.
        
        Args:
            frame: Frame de imagem
            faces: Lista de faces detectadas
            
        Returns:
            Lista de EmotionResult
        """
        # Por enquanto, usar processamento individual
        # Batch processing real requereria modificações no DeepFace
        return self.predict(frame, faces)
    
    def get_emotion_labels(self) -> List[str]:
        """
        Retorna lista de labels de emoções suportados.
        
        Returns:
            Lista de strings com nomes das emoções
        """
        return [
            'angry',
            'disgust',
            'fear',
            'happy',
            'sad',
            'surprise',
            'neutral'
        ]
    
    def is_model_loaded(self) -> bool:
        """
        Verifica se o modelo está carregado.
        
        Returns:
            True se modelo está carregado, False caso contrário
        """
        return self._model_loaded
    
    def __repr__(self) -> str:
        """Representação em string do classificador."""
        return (
            f"EmotionClassifier(backend='{self.backend}', "
            f"loaded={self._model_loaded})"
        )
