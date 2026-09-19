import json

import cv2
import numpy as np
import pytest

from cartgate import gallery as gallery_module


def test_load_gallery_falls_back_to_npz_when_requested_pickle_is_missing(tmp_path):
    """Catches production startup requiring a pickle absent from the model bundle."""
    np.savez_compressed(
        tmp_path / "gallery.npz",
        sku_ids=np.array(["SKU-001", "SKU-002"]),
        counts=np.array([2, 1], dtype=np.int32),
        vectors=np.array(
            [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]],
            dtype=np.float32,
        ),
        views_json=np.array(json.dumps({
            "SKU-001": [{"src": "front.jpg"}, {"src": "side.jpg"}],
            "SKU-002": [{"src": "top.jpg"}],
        })),
    )

    loaded = gallery_module.load_gallery(str(tmp_path / "gallery.pkl"))

    assert list(loaded) == ["SKU-001", "SKU-002"]
    np.testing.assert_array_equal(
        loaded["SKU-001"]["vectors"],
        np.array([[1.0, 0.0], [0.5, 0.5]], dtype=np.float32),
    )
    assert loaded["SKU-002"]["views"] == [{"src": "top.jpg"}]


def test_load_gallery_does_not_fall_back_to_pickle_when_npz_is_requested(tmp_path):
    """Catches production silently reopening an incompatible pickle gallery."""
    (tmp_path / "gallery.pkl").write_bytes(b"not a portable gallery")

    with pytest.raises(FileNotFoundError):
        gallery_module.load_gallery(str(tmp_path / "gallery.npz"))


def test_build_gallery_writes_a_portable_npz_copy(tmp_path):
    """Catches a rebuilt model bundle containing only a NumPy-version-specific pickle."""
    dataset = tmp_path / "dataset" / "SKU-001"
    dataset.mkdir(parents=True)
    assert cv2.imwrite(str(dataset / "front.png"), np.full((8, 6, 3), 127, dtype=np.uint8))

    class Embedder:
        model_id = "test-model"

        @staticmethod
        def embed(image, mask):
            return np.array([float(image.mean()), float(mask is None)], dtype=np.float32)

    out_dir = tmp_path / "out"
    built = gallery_module.build_gallery(str(dataset.parent), Embedder(), out_dir=str(out_dir))

    assert (out_dir / "gallery.pkl").is_file()
    assert (out_dir / "gallery.npz").is_file()
    loaded = gallery_module.load_gallery(str(out_dir / "gallery.npz"))
    np.testing.assert_array_equal(loaded["SKU-001"]["vectors"], built["SKU-001"]["vectors"])
    assert loaded["SKU-001"]["views"] == built["SKU-001"]["views"]


def test_build_gallery_backfills_npz_for_a_pickle_only_cache_hit(tmp_path):
    """Catches an unchanged legacy pickle cache remaining non-portable forever."""
    dataset = tmp_path / "dataset" / "SKU-001"
    dataset.mkdir(parents=True)
    assert cv2.imwrite(str(dataset / "front.png"), np.full((8, 6, 3), 127, dtype=np.uint8))

    class Embedder:
        model_id = "test-model"

        @staticmethod
        def embed(image, mask):
            return np.array([float(image.mean()), float(mask is None)], dtype=np.float32)

    out_dir = tmp_path / "out"
    expected = gallery_module.build_gallery(str(dataset.parent), Embedder(), out_dir=str(out_dir))
    (out_dir / "gallery.npz").unlink()

    cached = gallery_module.build_gallery(str(dataset.parent), Embedder(), out_dir=str(out_dir))

    assert (out_dir / "gallery.npz").is_file()
    np.testing.assert_array_equal(cached["SKU-001"]["vectors"], expected["SKU-001"]["vectors"])
