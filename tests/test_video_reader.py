"""
Tests for VideoReader class
"""

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.io.video_reader import VideoReader, VideoNotFoundError, VideoOpenError


class TestVideoReader:
    """Test suite for VideoReader class."""
    
    @pytest.fixture
    def dummy_video_path(self, tmp_path):
        """
        Create a dummy video file for testing.
        
        Returns:
            Path to the created dummy video file
        """
        video_path = tmp_path / "test_video.mp4"
        
        # Create a simple dummy video with OpenCV
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30.0
        frame_size = (640, 480)
        
        out = cv2.VideoWriter(str(video_path), fourcc, fps, frame_size)
        
        # Create 10 frames with different colors
        for i in range(10):
            # Create a frame with a gradient based on frame number
            frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
            frame[:, :] = (i * 25, 100, 255 - i * 25)  # BGR color changing per frame
            out.write(frame)
        
        out.release()
        
        return video_path
    
    def test_video_reader_initialization(self, dummy_video_path):
        """Test that VideoReader initializes correctly with valid video."""
        reader = VideoReader(str(dummy_video_path))
        
        assert reader.path == str(dummy_video_path)
        assert reader.fps() > 0
        assert reader.frame_count() >= 0
        
        reader.release()
    
    def test_video_not_found_error(self):
        """Test that VideoNotFoundError is raised for non-existent file."""
        with pytest.raises(VideoNotFoundError):
            VideoReader("/path/to/nonexistent/video.mp4")
    
    def test_video_fps_positive(self, dummy_video_path):
        """Test that FPS is positive."""
        reader = VideoReader(str(dummy_video_path))
        
        assert reader.fps() > 0
        assert isinstance(reader.fps(), float)
        
        reader.release()
    
    def test_video_frame_count_non_negative(self, dummy_video_path):
        """Test that frame count is non-negative."""
        reader = VideoReader(str(dummy_video_path))
        
        assert reader.frame_count() >= 0
        assert isinstance(reader.frame_count(), int)
        
        reader.release()
    
    def test_iterate_first_five_frames(self, dummy_video_path):
        """Test iterating through the first 5 frames without error."""
        reader = VideoReader(str(dummy_video_path))
        
        frame_count = 0
        for idx, frame, ts_sec in reader:
            # Verify frame properties
            assert isinstance(idx, int)
            assert isinstance(frame, np.ndarray)
            assert isinstance(ts_sec, float)
            
            # Verify frame has correct shape (height, width, channels)
            assert frame.ndim == 3
            assert frame.shape[2] == 3  # BGR channels
            
            # Verify timestamp is non-negative
            assert ts_sec >= 0
            
            # Verify index matches iteration count
            assert idx == frame_count
            
            frame_count += 1
            
            # Only check first 5 frames
            if frame_count >= 5:
                break
        
        assert frame_count == 5, "Should iterate through 5 frames"
        
        reader.release()
    
    def test_context_manager(self, dummy_video_path):
        """Test that VideoReader works as a context manager."""
        with VideoReader(str(dummy_video_path)) as reader:
            assert reader.fps() > 0
            assert reader.frame_count() >= 0
    
    def test_duration_calculation(self, dummy_video_path):
        """Test that duration is calculated correctly."""
        reader = VideoReader(str(dummy_video_path))
        
        duration = reader.duration()
        expected_duration = reader.frame_count() / reader.fps()
        
        assert duration > 0
        assert abs(duration - expected_duration) < 0.01  # Allow small floating point error
        
        reader.release()
    
    def test_repr(self, dummy_video_path):
        """Test string representation of VideoReader."""
        reader = VideoReader(str(dummy_video_path))
        
        repr_str = repr(reader)
        assert "VideoReader" in repr_str
        assert str(dummy_video_path) in repr_str
        assert "fps=" in repr_str
        assert "frames=" in repr_str
        
        reader.release()
    
    def test_invalid_file_type(self, tmp_path):
        """Test that proper error is raised for invalid file types."""
        # Create a text file instead of video
        invalid_path = tmp_path / "not_a_video.txt"
        invalid_path.write_text("This is not a video file")
        
        with pytest.raises(VideoOpenError):
            VideoReader(str(invalid_path))
    
    def test_directory_instead_of_file(self, tmp_path):
        """Test that error is raised when path is a directory."""
        with pytest.raises(VideoNotFoundError):
            VideoReader(str(tmp_path))
    
    def test_multiple_iterations(self, dummy_video_path):
        """Test that we can iterate multiple times over the same video."""
        reader = VideoReader(str(dummy_video_path))
        
        # First iteration
        first_iteration_count = sum(1 for _ in reader)
        
        # Second iteration
        second_iteration_count = sum(1 for _ in reader)
        
        assert first_iteration_count == second_iteration_count
        assert first_iteration_count == reader.frame_count()
        
        reader.release()


class TestVideoReaderWithRealVideo:
    """Tests that require a real video file."""
    
    @pytest.mark.skipif(
        not os.path.exists("data/input_video/video.mp4"),
        reason="Real video file not available"
    )
    def test_real_video_file(self):
        """Test with actual video file if available."""
        video_path = "data/input_video/video.mp4"
        reader = VideoReader(video_path)
        
        assert reader.fps() > 0
        assert reader.frame_count() >= 0
        
        # Iterate first 5 frames
        frame_count = 0
        for idx, frame, ts_sec in reader:
            assert frame is not None
            assert frame.size > 0
            frame_count += 1
            if frame_count >= 5:
                break
        
        reader.release()
