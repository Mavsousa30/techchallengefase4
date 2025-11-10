"""
Integration tests for face detection and visualization
"""

import numpy as np
import pytest

from src.face.detector import FaceDetector
from src.utils.viz import draw_box_and_label, put_hud


class TestFaceDetectionIntegration:
    """Integration tests for face detection pipeline."""
    
    @pytest.fixture
    def detector(self):
        """Create a face detector."""
        return FaceDetector(backend="opencv")
    
    @pytest.fixture
    def blank_frame(self):
        """Create a blank frame."""
        return np.zeros((480, 640, 3), dtype=np.uint8)
    
    @pytest.fixture
    def colored_frame(self):
        """Create a colored frame without faces."""
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        return frame
    
    def test_detect_returns_list_on_empty_frame(self, detector, blank_frame):
        """Test that detect returns consistent list on empty frame."""
        faces = detector.detect(blank_frame)
        
        assert isinstance(faces, list)
        # Empty frame should not crash and return empty or small list
        assert len(faces) >= 0
    
    def test_detect_returns_list_on_colored_frame(self, detector, colored_frame):
        """Test that detect returns consistent list on colored frame."""
        faces = detector.detect(colored_frame)
        
        assert isinstance(faces, list)
        assert len(faces) >= 0
    
    def test_detect_no_crash_on_various_frames(self, detector):
        """Test that detector doesn't crash on various frame types."""
        # Black frame
        black = np.zeros((480, 640, 3), dtype=np.uint8)
        faces1 = detector.detect(black)
        assert isinstance(faces1, list)
        
        # White frame
        white = np.ones((480, 640, 3), dtype=np.uint8) * 255
        faces2 = detector.detect(white)
        assert isinstance(faces2, list)
        
        # Random noise
        noise = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        faces3 = detector.detect(noise)
        assert isinstance(faces3, list)
        
        # Gradient
        gradient = np.tile(np.arange(0, 256, 256/640, dtype=np.uint8), (480, 1))
        gradient = np.stack([gradient] * 3, axis=-1)
        faces4 = detector.detect(gradient)
        assert isinstance(faces4, list)
    
    def test_visualization_with_detection_results(self, detector, colored_frame):
        """Test visualization works with detection results."""
        # Detect faces
        faces = detector.detect(colored_frame)
        
        # Visualize results
        output = colored_frame.copy()
        
        for i, face in enumerate(faces):
            label = f"Face {i+1}: {face.score:.2f}"
            output = draw_box_and_label(output, face.box, label)
        
        # Add HUD
        stats = {
            'Faces': len(faces),
            'Frame': 1,
            'FPS': 30.0
        }
        output = put_hud(output, stats)
        
        assert output is not None
        assert output.shape == colored_frame.shape
    
    def test_detection_consistency(self, detector, blank_frame):
        """Test that detection is consistent across multiple calls."""
        faces1 = detector.detect(blank_frame)
        faces2 = detector.detect(blank_frame)
        faces3 = detector.detect(blank_frame)
        
        # All should return lists
        assert isinstance(faces1, list)
        assert isinstance(faces2, list)
        assert isinstance(faces3, list)
        
        # Results should be consistent (same number of faces)
        assert len(faces1) == len(faces2) == len(faces3)
    
    def test_empty_detection_visualization(self, blank_frame):
        """Test visualization when no faces are detected."""
        # Simulate empty detection
        faces = []
        
        output = blank_frame.copy()
        
        # Should not crash even with no faces
        for face in faces:
            output = draw_box_and_label(output, face.box, "Face")
        
        stats = {'Faces': len(faces), 'Frame': 1}
        output = put_hud(output, stats)
        
        assert output is not None
    
    def test_multiple_frames_processing(self, detector):
        """Test processing multiple frames in sequence."""
        frames = [
            np.zeros((480, 640, 3), dtype=np.uint8),
            np.ones((480, 640, 3), dtype=np.uint8) * 128,
            np.ones((480, 640, 3), dtype=np.uint8) * 255,
        ]
        
        results = []
        for frame in frames:
            faces = detector.detect(frame)
            assert isinstance(faces, list)
            results.append(len(faces))
        
        # All should return valid results
        assert len(results) == 3
        assert all(r >= 0 for r in results)
