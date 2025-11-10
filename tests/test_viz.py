"""
Tests for Visualization Utilities
"""

import numpy as np
import pytest
import cv2

from src.utils.viz import (
    draw_box_and_label,
    put_hud,
    draw_landmarks,
    format_timestamp,
    create_color_palette,
    COLORS
)


class TestDrawBoxAndLabel:
    """Tests for draw_box_and_label function."""
    
    @pytest.fixture
    def blank_frame(self):
        """Create a blank frame for testing."""
        return np.zeros((480, 640, 3), dtype=np.uint8)
    
    def test_draw_box_and_label_basic(self, blank_frame):
        """Test basic box and label drawing."""
        box = (100, 100, 200, 150)
        label = "Test Label"
        
        result = draw_box_and_label(blank_frame, box, label)
        
        assert result is not None
        assert result.shape == blank_frame.shape
        assert not np.array_equal(result, blank_frame)  # Should have changed
    
    def test_draw_box_without_label(self, blank_frame):
        """Test drawing box without label."""
        box = (50, 50, 100, 100)
        
        result = draw_box_and_label(blank_frame, box, "")
        
        assert result is not None
        assert result.shape == blank_frame.shape
    
    def test_draw_box_invalid_frame_none(self):
        """Test error with None frame."""
        with pytest.raises(ValueError, match="Frame is None or empty"):
            draw_box_and_label(None, (0, 0, 10, 10), "label")
    
    def test_draw_box_invalid_frame_empty(self):
        """Test error with empty frame."""
        empty_frame = np.array([])
        with pytest.raises(ValueError, match="Frame is None or empty"):
            draw_box_and_label(empty_frame, (0, 0, 10, 10), "label")
    
    def test_draw_box_invalid_box_format(self, blank_frame):
        """Test error with invalid box format."""
        with pytest.raises(ValueError, match="Box must have 4 values"):
            draw_box_and_label(blank_frame, (10, 20), "label")
    
    def test_draw_box_with_custom_color(self, blank_frame):
        """Test drawing with custom color."""
        box = (100, 100, 200, 150)
        label = "Red Box"
        
        result = draw_box_and_label(
            blank_frame, box, label, color=COLORS['red']
        )
        
        assert result is not None
        assert not np.array_equal(result, blank_frame)
    
    def test_draw_box_edge_cases(self, blank_frame):
        """Test drawing box at frame edges."""
        # Box at top-left corner
        result1 = draw_box_and_label(blank_frame, (0, 0, 50, 50), "Corner")
        assert result1 is not None
        
        # Box at bottom-right corner
        h, w = blank_frame.shape[:2]
        result2 = draw_box_and_label(
            blank_frame, (w - 60, h - 60, 50, 50), "Corner"
        )
        assert result2 is not None
    
    def test_draw_box_does_not_modify_original(self, blank_frame):
        """Test that original frame is not modified."""
        original = blank_frame.copy()
        box = (100, 100, 200, 150)
        
        result = draw_box_and_label(blank_frame, box, "Label")
        
        assert np.array_equal(blank_frame, original)


class TestPutHud:
    """Tests for put_hud function."""
    
    @pytest.fixture
    def blank_frame(self):
        """Create a blank frame for testing."""
        return np.zeros((480, 640, 3), dtype=np.uint8)
    
    @pytest.fixture
    def sample_stats(self):
        """Sample statistics dictionary."""
        return {
            'FPS': 30.5,
            'Frame': 120,
            'Faces': 2,
            'Timestamp': '02:00'
        }
    
    def test_put_hud_basic(self, blank_frame, sample_stats):
        """Test basic HUD drawing."""
        result = put_hud(blank_frame, sample_stats)
        
        assert result is not None
        assert result.shape == blank_frame.shape
        assert not np.array_equal(result, blank_frame)
    
    def test_put_hud_invalid_frame_none(self, sample_stats):
        """Test error with None frame."""
        with pytest.raises(ValueError, match="Frame is None or empty"):
            put_hud(None, sample_stats)
    
    def test_put_hud_invalid_frame_empty(self, sample_stats):
        """Test error with empty frame."""
        empty_frame = np.array([])
        with pytest.raises(ValueError, match="Frame is None or empty"):
            put_hud(empty_frame, sample_stats)
    
    def test_put_hud_empty_stats(self, blank_frame):
        """Test HUD with empty stats dict."""
        result = put_hud(blank_frame, {})
        
        # Should return unchanged frame
        assert np.array_equal(result, blank_frame)
    
    def test_put_hud_different_positions(self, blank_frame, sample_stats):
        """Test HUD at different positions."""
        positions = ['top-left', 'top-right', 'bottom-left', 'bottom-right']
        
        for position in positions:
            result = put_hud(blank_frame, sample_stats, position=position)
            assert result is not None
            assert result.shape == blank_frame.shape
    
    def test_put_hud_invalid_position(self, blank_frame, sample_stats):
        """Test error with invalid position."""
        with pytest.raises(ValueError, match="Invalid position"):
            put_hud(blank_frame, sample_stats, position="invalid")
    
    def test_put_hud_various_data_types(self, blank_frame):
        """Test HUD with various data types."""
        stats = {
            'Float': 3.14159,
            'Int': 42,
            'String': 'test',
            'Bool': True
        }
        
        result = put_hud(blank_frame, stats)
        
        assert result is not None
        assert not np.array_equal(result, blank_frame)
    
    def test_put_hud_does_not_modify_original(self, blank_frame, sample_stats):
        """Test that original frame is not modified."""
        original = blank_frame.copy()
        
        result = put_hud(blank_frame, sample_stats)
        
        assert np.array_equal(blank_frame, original)


class TestDrawLandmarks:
    """Tests for draw_landmarks function."""
    
    @pytest.fixture
    def blank_frame(self):
        """Create a blank frame for testing."""
        return np.zeros((480, 640, 3), dtype=np.uint8)
    
    @pytest.fixture
    def sample_landmarks(self):
        """Sample landmarks dictionary."""
        return {
            'left_eye': (200, 150),
            'right_eye': (250, 150),
            'nose': (225, 180),
            'mouth': (225, 210)
        }
    
    def test_draw_landmarks_basic(self, blank_frame, sample_landmarks):
        """Test basic landmarks drawing."""
        result = draw_landmarks(blank_frame, sample_landmarks)
        
        assert result is not None
        assert result.shape == blank_frame.shape
        assert not np.array_equal(result, blank_frame)
    
    def test_draw_landmarks_invalid_frame_none(self, sample_landmarks):
        """Test error with None frame."""
        with pytest.raises(ValueError, match="Frame is None or empty"):
            draw_landmarks(None, sample_landmarks)
    
    def test_draw_landmarks_empty_landmarks(self, blank_frame):
        """Test landmarks with empty dict."""
        result = draw_landmarks(blank_frame, {})
        
        # Should return unchanged frame
        assert np.array_equal(result, blank_frame)
    
    def test_draw_landmarks_custom_style(self, blank_frame, sample_landmarks):
        """Test landmarks with custom style."""
        result = draw_landmarks(
            blank_frame,
            sample_landmarks,
            color=COLORS['red'],
            radius=5,
            thickness=2
        )
        
        assert result is not None
        assert not np.array_equal(result, blank_frame)


class TestFormatTimestamp:
    """Tests for format_timestamp function."""
    
    def test_format_timestamp_zero(self):
        """Test formatting zero seconds."""
        assert format_timestamp(0) == "00:00"
    
    def test_format_timestamp_seconds(self):
        """Test formatting seconds only."""
        assert format_timestamp(45) == "00:45"
    
    def test_format_timestamp_minutes(self):
        """Test formatting with minutes."""
        assert format_timestamp(125) == "02:05"
    
    def test_format_timestamp_large_value(self):
        """Test formatting large values."""
        assert format_timestamp(3665) == "61:05"
    
    def test_format_timestamp_decimal(self):
        """Test formatting with decimal seconds."""
        assert format_timestamp(125.7) == "02:05"


class TestCreateColorPalette:
    """Tests for create_color_palette function."""
    
    def test_create_color_palette_basic(self):
        """Test creating basic color palette."""
        n_colors = 5
        colors = create_color_palette(n_colors)
        
        assert len(colors) == n_colors
        assert all(isinstance(c, tuple) for c in colors)
        assert all(len(c) == 3 for c in colors)
    
    def test_create_color_palette_single(self):
        """Test creating single color."""
        colors = create_color_palette(1)
        
        assert len(colors) == 1
        assert isinstance(colors[0], tuple)
    
    def test_create_color_palette_many(self):
        """Test creating many colors."""
        colors = create_color_palette(20)
        
        assert len(colors) == 20
        # Colors should be different
        assert len(set(colors)) > 1


class TestIntegration:
    """Integration tests combining multiple visualization functions."""
    
    def test_combined_visualization(self):
        """Test combining multiple visualization functions."""
        # Create frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Draw box
        frame = draw_box_and_label(frame, (100, 100, 200, 150), "Face: 0.95")
        
        # Draw landmarks
        landmarks = {
            'left_eye': (150, 150),
            'right_eye': (250, 150)
        }
        frame = draw_landmarks(frame, landmarks)
        
        # Add HUD
        stats = {'FPS': 30.0, 'Frame': 1, 'Faces': 1}
        frame = put_hud(frame, stats)
        
        assert frame is not None
        assert frame.shape == (480, 640, 3)
    
    def test_multiple_boxes(self):
        """Test drawing multiple boxes on same frame."""
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        
        boxes = [
            (50, 50, 100, 100),
            (200, 200, 150, 150),
            (400, 100, 100, 120)
        ]
        
        for i, box in enumerate(boxes):
            frame = draw_box_and_label(frame, box, f"Object {i+1}")
        
        assert frame is not None
