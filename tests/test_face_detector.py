"""
Tests for Face Detector
"""

import numpy as np
import pytest
import cv2

from src.face.detector import Face, FaceDetector


def _is_face_recognition_available() -> bool:
    """Check if face_recognition is available."""
    try:
        import face_recognition
        return True
    except ImportError:
        return False


def _is_deepface_available() -> bool:
    """Check if deepface is available."""
    try:
        from deepface import DeepFace
        return True
    except (ImportError, ValueError, Exception):
        # ValueError pode ocorrer se tensorflow não tiver tf-keras
        return False


class TestFaceDataclass:
    """Tests for Face dataclass."""
    
    def test_face_initialization(self):
        """Test Face initialization with valid data."""
        face = Face(box=(10, 20, 100, 150), score=0.95)
        
        assert face.box == (10, 20, 100, 150)
        assert face.score == 0.95
        assert face.landmarks is None
    
    def test_face_with_landmarks(self):
        """Test Face with landmarks."""
        landmarks = {
            'left_eye': (30, 40),
            'right_eye': (70, 40),
            'nose': (50, 70)
        }
        face = Face(box=(10, 20, 100, 150), score=0.95, landmarks=landmarks)
        
        assert face.landmarks == landmarks
        assert face.landmarks['left_eye'] == (30, 40)
    
    def test_face_invalid_box(self):
        """Test Face raises error with invalid box."""
        with pytest.raises(ValueError, match="Box must have 4 values"):
            Face(box=(10, 20), score=0.95)
    
    def test_face_invalid_score_high(self):
        """Test Face raises error with score > 1.0."""
        with pytest.raises(ValueError, match="Score must be between"):
            Face(box=(10, 20, 100, 150), score=1.5)
    
    def test_face_invalid_score_low(self):
        """Test Face raises error with score < 0.0."""
        with pytest.raises(ValueError, match="Score must be between"):
            Face(box=(10, 20, 100, 150), score=-0.1)
    
    def test_face_center_property(self):
        """Test Face center property."""
        face = Face(box=(0, 0, 100, 100), score=0.9)
        
        assert face.center == (50, 50)
    
    def test_face_area_property(self):
        """Test Face area property."""
        face = Face(box=(0, 0, 100, 150), score=0.9)
        
        assert face.area == 15000


class TestFaceDetector:
    """Tests for FaceDetector class."""
    
    @pytest.fixture
    def dummy_frame(self):
        """Create a dummy frame for testing."""
        # Create a simple 640x480 BGR image
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add some content (white rectangle simulating a face)
        cv2.rectangle(frame, (200, 150), (400, 350), (255, 255, 255), -1)
        return frame
    
    @pytest.fixture
    def frame_with_face(self):
        """Create a frame with a clear face-like pattern."""
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        
        # Draw a simple face-like structure
        # Face oval
        cv2.ellipse(frame, (320, 240), (80, 100), 0, 0, 360, (255, 200, 180), -1)
        
        # Eyes
        cv2.circle(frame, (290, 220), 10, (50, 50, 50), -1)
        cv2.circle(frame, (350, 220), 10, (50, 50, 50), -1)
        
        # Nose
        cv2.line(frame, (320, 230), (320, 270), (150, 100, 100), 3)
        
        # Mouth
        cv2.ellipse(frame, (320, 290), (30, 15), 0, 0, 180, (100, 50, 50), 2)
        
        return frame
    
    def test_detector_initialization_auto(self):
        """Test FaceDetector initialization with auto backend."""
        detector = FaceDetector(backend="auto")
        
        # Should select one of the available backends
        assert detector.backend in ["face_recognition", "deepface", "opencv"]
    
    def test_detector_initialization_opencv(self):
        """Test FaceDetector initialization with OpenCV backend."""
        detector = FaceDetector(backend="opencv")
        
        assert detector.backend == "opencv"
        assert detector._detector is not None
    
    def test_detect_invalid_frame_none(self):
        """Test detect raises error with None frame."""
        detector = FaceDetector(backend="opencv")
        
        with pytest.raises(ValueError, match="Invalid frame"):
            detector.detect(None)
    
    def test_detect_invalid_frame_empty(self):
        """Test detect raises error with empty frame."""
        detector = FaceDetector(backend="opencv")
        empty_frame = np.array([])
        
        with pytest.raises(ValueError, match="Invalid frame"):
            detector.detect(empty_frame)
    
    def test_detect_invalid_frame_dimensions(self):
        """Test detect raises error with invalid dimensions."""
        detector = FaceDetector(backend="opencv")
        invalid_frame = np.zeros((10, 10, 10, 10))  # 4D array
        
        with pytest.raises(ValueError, match="Invalid frame dimensions"):
            detector.detect(invalid_frame)
    
    def test_detect_returns_list(self, dummy_frame):
        """Test detect returns a list."""
        detector = FaceDetector(backend="opencv")
        
        result = detector.detect(dummy_frame)
        
        assert isinstance(result, list)
    
    def test_detect_returns_face_objects(self, frame_with_face):
        """Test detect returns Face objects."""
        detector = FaceDetector(backend="opencv")
        
        faces = detector.detect(frame_with_face)
        
        # May or may not detect the simple pattern, but should return Face objects
        for face in faces:
            assert isinstance(face, Face)
            assert isinstance(face.box, tuple)
            assert len(face.box) == 4
            assert isinstance(face.score, float)
            assert 0.0 <= face.score <= 1.0
    
    def test_detect_opencv_on_real_pattern(self, frame_with_face):
        """Test OpenCV detector on frame with face pattern."""
        detector = FaceDetector(backend="opencv")
        
        faces = detector.detect(frame_with_face)
        
        # OpenCV might detect the pattern, verify structure is correct
        assert isinstance(faces, list)
    
    def test_detector_repr(self):
        """Test detector string representation."""
        detector = FaceDetector(backend="opencv", model="hog")
        
        repr_str = repr(detector)
        assert "FaceDetector" in repr_str
        assert "opencv" in repr_str
    
    @pytest.mark.skipif(
        not _is_face_recognition_available(),
        reason="face_recognition not available"
    )
    def test_face_recognition_backend(self, dummy_frame):
        """Test face_recognition backend if available."""
        detector = FaceDetector(backend="face_recognition", model="hog")
        
        assert detector.backend == "face_recognition"
        
        faces = detector.detect(dummy_frame)
        assert isinstance(faces, list)
    
    @pytest.mark.skipif(
        not _is_deepface_available(),
        reason="deepface not available"
    )
    def test_deepface_backend(self, dummy_frame):
        """Test deepface backend if available."""
        detector = FaceDetector(backend="deepface")
        
        assert detector.backend == "deepface"
        
        faces = detector.detect(dummy_frame)
        assert isinstance(faces, list)
    
    def test_multiple_detections(self, dummy_frame):
        """Test detector can be called multiple times."""
        detector = FaceDetector(backend="opencv")
        
        faces1 = detector.detect(dummy_frame)
        faces2 = detector.detect(dummy_frame)
        
        # Both should succeed
        assert isinstance(faces1, list)
        assert isinstance(faces2, list)
