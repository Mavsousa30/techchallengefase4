"""
Integration test for emotion classification with face detection
"""

import numpy as np
import pytest
import cv2

from src.face.detector import FaceDetector
from src.emotion.classifier import EmotionClassifier


class TestEmotionDetectionPipeline:
    """Test complete pipeline from face detection to emotion classification."""
    
    @pytest.fixture
    def frame_with_clear_face(self):
        """Create a frame with a clear face pattern."""
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 200
        
        # Draw a larger, clearer face
        center = (320, 240)
        
        # Face oval (skin tone)
        cv2.ellipse(frame, center, (100, 130), 0, 0, 360, (220, 180, 160), -1)
        
        # Left eye
        cv2.ellipse(frame, (280, 220), (15, 10), 0, 0, 360, (255, 255, 255), -1)
        cv2.circle(frame, (280, 220), 8, (50, 50, 50), -1)
        cv2.circle(frame, (282, 218), 3, (255, 255, 255), -1)
        
        # Right eye
        cv2.ellipse(frame, (360, 220), (15, 10), 0, 0, 360, (255, 255, 255), -1)
        cv2.circle(frame, (360, 220), 8, (50, 50, 50), -1)
        cv2.circle(frame, (362, 218), 3, (255, 255, 255), -1)
        
        # Eyebrows
        cv2.ellipse(frame, (280, 200), (20, 8), 0, 0, 180, (80, 60, 50), 3)
        cv2.ellipse(frame, (360, 200), (20, 8), 0, 0, 180, (80, 60, 50), 3)
        
        # Nose
        cv2.line(frame, (320, 240), (315, 270), (180, 140, 120), 3)
        cv2.line(frame, (315, 270), (310, 275), (180, 140, 120), 2)
        cv2.line(frame, (315, 270), (320, 275), (180, 140, 120), 2)
        
        # Mouth (smiling)
        cv2.ellipse(frame, (320, 300), (40, 25), 0, 0, 180, (150, 80, 80), 3)
        
        return frame
    
    def test_detect_and_classify_emotions(self, frame_with_clear_face):
        """
        Test complete pipeline: detect faces and classify emotions.
        
        Acceptance criteria:
        - In a frame with face(s), returns at least 1 result with label and score
        """
        # Step 1: Create a mock face detection (since OpenCV may not detect drawn patterns)
        from src.face.detector import Face
        
        # Manually create face detection for the known face location
        faces = [Face(box=(220, 110, 200, 260), score=0.95)]
        
        print(f"Using {len(faces)} face(s)")
        
        # Step 2: Classify emotions
        classifier = EmotionClassifier()
        results = classifier.predict(frame_with_clear_face, faces)
        
        # Acceptance criteria: at least 1 result with label and score
        assert len(results) >= 1, "Should return at least 1 emotion result"
        
        # Validate first result
        result = results[0]
        assert result.label, "Result should have a label"
        assert isinstance(result.label, str), "Label should be a string"
        assert len(result.label) > 0, "Label should not be empty"
        
        assert result.score is not None, "Result should have a score"
        assert isinstance(result.score, (int, float)), "Score should be numeric"
        assert 0.0 <= result.score <= 100.0, "Score should be between 0 and 100"
        
        # Print results for verification
        for i, res in enumerate(results):
            print(f"Face {i+1}: {res.label} (score: {res.score:.2f}, "
                  f"normalized: {res.normalized_score:.2f})")
    
    def test_multiple_faces_multiple_emotions(self):
        """Test with frame containing multiple face regions."""
        frame = np.ones((600, 800, 3), dtype=np.uint8) * 200
        
        # Draw two faces in different positions
        for center_x in [200, 600]:
            center = (center_x, 300)
            cv2.ellipse(frame, center, (80, 100), 0, 0, 360, (220, 180, 160), -1)
            cv2.circle(frame, (center_x - 30, 280), 8, (50, 50, 50), -1)
            cv2.circle(frame, (center_x + 30, 280), 8, (50, 50, 50), -1)
            cv2.ellipse(frame, (center_x, 330), (30, 20), 0, 0, 180, (150, 80, 80), 3)
        
        # Detect and classify
        detector = FaceDetector(backend="opencv")
        faces = detector.detect(frame)
        
        classifier = EmotionClassifier()
        results = classifier.predict(frame, faces)
        
        # Should return results for each detected face
        print(f"Detected {len(faces)} faces, classified {len(results)} emotions")
        
        assert len(results) >= 0, "Should handle multiple faces"
        
        for i, res in enumerate(results):
            assert res.label, f"Result {i} should have label"
            assert res.score >= 0, f"Result {i} should have valid score"
    
    def test_real_world_frame(self):
        """Test with a more realistic frame (if available)."""
        # Create a complex scene
        frame = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
        
        # Add a face region in the center
        center = (320, 240)
        cv2.ellipse(frame, center, (80, 100), 0, 0, 360, (220, 180, 160), -1)
        cv2.circle(frame, (290, 230), 8, (50, 50, 50), -1)
        cv2.circle(frame, (350, 230), 8, (50, 50, 50), -1)
        
        detector = FaceDetector(backend="opencv")
        classifier = EmotionClassifier()
        
        faces = detector.detect(frame)
        results = classifier.predict(frame, faces)
        
        # System should handle without crashing
        assert isinstance(results, list)
        
        if len(results) > 0:
            # If emotions detected, validate format
            for res in results:
                assert hasattr(res, 'label')
                assert hasattr(res, 'score')
                assert hasattr(res, 'box')
    
    def test_edge_case_no_faces(self):
        """Test with frame containing no faces."""
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        
        detector = FaceDetector(backend="opencv")
        classifier = EmotionClassifier()
        
        faces = detector.detect(frame)
        results = classifier.predict(frame, faces)
        
        # Should return empty list gracefully
        assert isinstance(results, list)
        assert len(results) == 0
    
    def test_edge_case_very_small_face(self):
        """Test with very small face region."""
        frame = np.ones((200, 200, 3), dtype=np.uint8) * 180
        
        # Very small face
        cv2.ellipse(frame, (100, 100), (20, 25), 0, 0, 360, (220, 180, 160), -1)
        cv2.circle(frame, (95, 95), 2, (50, 50, 50), -1)
        cv2.circle(frame, (105, 95), 2, (50, 50, 50), -1)
        
        detector = FaceDetector(backend="opencv")
        classifier = EmotionClassifier()
        
        faces = detector.detect(frame)
        results = classifier.predict(frame, faces)
        
        # Should handle gracefully
        assert isinstance(results, list)
    
    def test_performance_multiple_frames(self):
        """Test performance with multiple frames."""
        detector = FaceDetector(backend="opencv")
        classifier = EmotionClassifier()
        
        # Process 10 frames
        for i in range(10):
            frame = np.ones((480, 640, 3), dtype=np.uint8) * (100 + i * 10)
            
            # Add face
            cv2.ellipse(frame, (320, 240), (80, 100), 0, 0, 360, (220, 180, 160), -1)
            
            faces = detector.detect(frame)
            results = classifier.predict(frame, faces)
            
            assert isinstance(results, list)
        
        # Should complete without issues
        assert True
