"""
Tests for Emotion Classifier
"""

import numpy as np
import pytest
import cv2

from src.emotion.classifier import EmotionResult, EmotionClassifier
from src.face.detector import Face


class TestEmotionResult:
    """Tests for EmotionResult dataclass."""
    
    def test_emotion_result_initialization(self):
        """Test EmotionResult initialization with valid data."""
        result = EmotionResult(
            label="happy",
            score=85.5,
            box=(100, 100, 200, 200)
        )
        
        assert result.label == "happy"
        assert result.score == 85.5
        assert result.box == (100, 100, 200, 200)
    
    def test_emotion_result_normalized_score(self):
        """Test normalized score property."""
        result = EmotionResult(
            label="sad",
            score=75.0,
            box=(0, 0, 100, 100)
        )
        
        assert result.normalized_score == 0.75
    
    def test_emotion_result_invalid_label(self):
        """Test error with empty label."""
        with pytest.raises(ValueError, match="Label cannot be empty"):
            EmotionResult(label="", score=50.0, box=(0, 0, 100, 100))
    
    def test_emotion_result_invalid_score_high(self):
        """Test error with score > 100."""
        with pytest.raises(ValueError, match="Score must be between"):
            EmotionResult(label="happy", score=150.0, box=(0, 0, 100, 100))
    
    def test_emotion_result_invalid_score_low(self):
        """Test error with score < 0."""
        with pytest.raises(ValueError, match="Score must be between"):
            EmotionResult(label="happy", score=-10.0, box=(0, 0, 100, 100))
    
    def test_emotion_result_invalid_box(self):
        """Test error with invalid box format."""
        with pytest.raises(ValueError, match="Box must have 4 values"):
            EmotionResult(label="happy", score=50.0, box=(0, 0))


class TestEmotionClassifier:
    """Tests for EmotionClassifier class."""
    
    @pytest.fixture
    def classifier(self):
        """Create an emotion classifier."""
        return EmotionClassifier()
    
    @pytest.fixture
    def frame_with_face(self):
        """Create a frame with a face-like pattern."""
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        
        # Draw simple face
        cv2.ellipse(frame, (320, 240), (80, 100), 0, 0, 360, (255, 200, 180), -1)
        cv2.circle(frame, (290, 220), 10, (50, 50, 50), -1)
        cv2.circle(frame, (350, 220), 10, (50, 50, 50), -1)
        cv2.ellipse(frame, (320, 290), (30, 15), 0, 0, 180, (100, 50, 50), 2)
        
        return frame
    
    @pytest.fixture
    def sample_faces(self):
        """Create sample face detections."""
        return [
            Face(box=(100, 100, 200, 200), score=0.95),
            Face(box=(400, 150, 180, 180), score=0.88)
        ]
    
    def test_classifier_initialization(self, classifier):
        """Test classifier initialization."""
        assert classifier is not None
        assert classifier.backend in ["deepface", "fallback"]
    
    def test_predict_with_empty_faces(self, classifier, frame_with_face):
        """Test predict with no faces."""
        results = classifier.predict(frame_with_face, [])
        
        assert isinstance(results, list)
        assert len(results) == 0
    
    def test_predict_invalid_frame_none(self, classifier, sample_faces):
        """Test error with None frame."""
        with pytest.raises(ValueError, match="Frame is None or empty"):
            classifier.predict(None, sample_faces)
    
    def test_predict_invalid_frame_empty(self, classifier, sample_faces):
        """Test error with empty frame."""
        empty_frame = np.array([])
        with pytest.raises(ValueError, match="Frame is None or empty"):
            classifier.predict(empty_frame, sample_faces)
    
    def test_predict_returns_list(self, classifier, frame_with_face, sample_faces):
        """Test predict returns a list."""
        results = classifier.predict(frame_with_face, sample_faces)
        
        assert isinstance(results, list)
    
    def test_predict_returns_emotion_results(self, classifier, frame_with_face):
        """Test predict returns EmotionResult objects."""
        # Create a face in the center of the frame
        faces = [Face(box=(220, 140, 200, 200), score=0.95)]
        
        results = classifier.predict(frame_with_face, faces)
        
        # Should return at least one result (even if fallback)
        assert len(results) >= 1
        
        for result in results:
            assert isinstance(result, EmotionResult)
            assert isinstance(result.label, str)
            assert len(result.label) > 0
            assert isinstance(result.score, float)
            assert 0.0 <= result.score <= 100.0
            assert isinstance(result.box, tuple)
            assert len(result.box) == 4
    
    def test_predict_with_single_face(self, classifier, frame_with_face):
        """Test predict with single face."""
        faces = [Face(box=(220, 140, 200, 200), score=0.95)]
        
        results = classifier.predict(frame_with_face, faces)
        
        # Should return exactly one result
        assert len(results) == 1
        assert results[0].label in classifier.get_emotion_labels()
    
    def test_predict_handles_invalid_face_region(self, classifier, frame_with_face):
        """Test predict handles face outside frame bounds gracefully."""
        # Face outside frame bounds
        faces = [Face(box=(1000, 1000, 100, 100), score=0.95)]
        
        # Should not crash, might return empty or handle gracefully
        results = classifier.predict(frame_with_face, faces)
        
        assert isinstance(results, list)
    
    def test_predict_batch(self, classifier, frame_with_face, sample_faces):
        """Test batch prediction."""
        results = classifier.predict_batch(frame_with_face, sample_faces)
        
        assert isinstance(results, list)
    
    def test_get_emotion_labels(self, classifier):
        """Test getting emotion labels."""
        labels = classifier.get_emotion_labels()
        
        assert isinstance(labels, list)
        assert len(labels) > 0
        assert 'happy' in labels
        assert 'sad' in labels
        assert 'neutral' in labels
    
    def test_is_model_loaded(self, classifier):
        """Test model loaded check."""
        loaded = classifier.is_model_loaded()
        
        assert isinstance(loaded, bool)
    
    def test_repr(self, classifier):
        """Test string representation."""
        repr_str = repr(classifier)
        
        assert "EmotionClassifier" in repr_str
        assert "backend" in repr_str
    
    def test_predict_with_small_face(self, classifier):
        """Test predict with very small face region."""
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
        
        # Very small face
        faces = [Face(box=(40, 40, 20, 20), score=0.9)]
        
        results = classifier.predict(frame, faces)
        
        # Should handle gracefully
        assert isinstance(results, list)
    
    def test_predict_consistency(self, classifier, frame_with_face):
        """Test prediction consistency."""
        faces = [Face(box=(220, 140, 200, 200), score=0.95)]
        
        results1 = classifier.predict(frame_with_face, faces)
        results2 = classifier.predict(frame_with_face, faces)
        
        # Should return consistent results
        assert len(results1) == len(results2)


class TestEmotionClassifierIntegration:
    """Integration tests with real-world scenarios."""
    
    def test_full_pipeline_with_face_detector(self):
        """Test emotion classification with face detector."""
        from src.face.detector import FaceDetector
        
        # Create frame with face pattern
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        cv2.ellipse(frame, (320, 240), (80, 100), 0, 0, 360, (255, 200, 180), -1)
        
        # Detect faces
        detector = FaceDetector(backend="opencv")
        faces = detector.detect(frame)
        
        # Classify emotions
        classifier = EmotionClassifier()
        results = classifier.predict(frame, faces)
        
        # Should return results for detected faces
        assert isinstance(results, list)
        assert len(results) >= 0
    
    def test_multiple_frames_processing(self):
        """Test processing multiple frames sequentially."""
        classifier = EmotionClassifier()
        
        frames = [
            np.ones((100, 100, 3), dtype=np.uint8) * i
            for i in range(50, 200, 50)
        ]
        
        faces = [Face(box=(20, 20, 60, 60), score=0.9)]
        
        all_results = []
        for frame in frames:
            results = classifier.predict(frame, faces)
            all_results.append(results)
        
        # Should process all frames without error
        assert len(all_results) == len(frames)
        assert all(isinstance(r, list) for r in all_results)
    
    def test_empty_roi_handling(self):
        """Test handling of edge cases with ROI extraction."""
        classifier = EmotionClassifier()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Face at edge
        faces = [Face(box=(90, 90, 50, 50), score=0.9)]
        
        results = classifier.predict(frame, faces)
        
        # Should handle gracefully
        assert isinstance(results, list)
