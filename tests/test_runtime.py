import numpy as np
import pytest

from cartgate.server import runtime as runtime_module
from cartgate.server.runtime import CapturedFrame, validate_two_camera_frames
from cartgate.server.service import CameraUnavailableError, InvalidImageDataError


def test_vision_runtime_loads_the_portable_npz_gallery(tmp_path, monkeypatch):
    """Catches production startup selecting the incompatible pickle gallery."""
    requested_paths = []

    def record_gallery_path(path):
        requested_paths.append(path)
        return {"sku-1": {"vectors": np.ones((1, 2)), "views": []}}

    monkeypatch.setattr(runtime_module, "load_gallery", record_gallery_path)

    runtime_module.VisionRuntime(
        tmp_path,
        detector=object(),
        embedder=object(),
        fusion=object(),
    )

    assert requested_paths == [str(tmp_path / "out/gallery.npz")]


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
