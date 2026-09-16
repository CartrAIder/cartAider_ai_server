import numpy as np
import pytest

from cartgate.server.runtime import CapturedFrame, validate_two_camera_frames
from cartgate.server.service import CameraUnavailableError, InvalidImageDataError


def test_two_camera_validation_rejects_an_empty_uploaded_stream():
    """Catches an upload with no images for one fixed camera entering normal inference."""
    frames = {"cam_left": [CapturedFrame(np.zeros((10, 10, 3), dtype=np.uint8))], "cam_right": []}

    with pytest.raises(CameraUnavailableError, match="cam_right"):
        validate_two_camera_frames(frames, min_frames=1)


def test_two_camera_validation_rejects_non_bgr_image_data():
    """Catches a decoded grayscale or corrupt frame being passed to OpenCV/YOLO."""
    frames = {
        "cam_left": [CapturedFrame(np.zeros((10, 10), dtype=np.uint8))],
        "cam_right": [CapturedFrame(np.zeros((10, 10, 3), dtype=np.uint8))],
    }

    with pytest.raises(InvalidImageDataError, match="BGR"):
        validate_two_camera_frames(frames, min_frames=1)
