"""Appearance-embedding backends: ClassicalEmbedder (HSV+HOG, no downloads) and
OnnxEmbedder (a CNN/ViT backbone via onnxruntime). Both expose
embed(bgr, mask) -> 1-D L2-normalized float32 vector."""
import os

import cv2
import numpy as np


def _l2n(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


class ClassicalEmbedder:
    name = "classical(hsv+hog)"
    color_weight = 0.65
    hog_weight = 0.35

    def embed(self, bgr: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        if mask is None:
            mask = np.full(bgr.shape[:2], 255, np.uint8)

        # --- masked HSV histogram (Hellinger kernel via sqrt) ---
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], mask, [12, 8, 4], [0, 180, 0, 256, 0, 256])
        hist = hist.flatten()
        s = hist.sum()
        if s > 0:
            hist /= s
        color = np.sqrt(hist)

        # --- HOG on the masked, square-padded object ---
        from skimage.feature import hog
        obj = cv2.bitwise_and(bgr, bgr, mask=mask)
        ys, xs = np.where(mask > 0)
        if len(ys) > 0:
            obj = obj[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        h, w = obj.shape[:2]
        side = max(h, w)
        sq = np.zeros((side, side, 3), np.uint8)
        sq[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = obj
        gray = cv2.cvtColor(cv2.resize(sq, (96, 96), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        hg = hog(gray, orientations=9, pixels_per_cell=(12, 12), cells_per_block=(2, 2),
                 block_norm="L2-Hys", feature_vector=True).astype(np.float32)

        return _l2n(np.concatenate([
            self.color_weight * _l2n(color.astype(np.float32)),
            self.hog_weight * _l2n(hg),
        ]))


def _preload_cuda_libs() -> None:
    """onnxruntime-gpu's CUDA EP needs libcudnn.so.9 / libcublas on the loader path.
    Importing torch first pulls its bundled CUDA libraries into the process, which
    satisfies that; without it ORT silently falls back to CPU (~5x slower here)."""
    try:
        import torch  # noqa: F401
    except Exception:
        pass


class OnnxEmbedder:
    """A CNN/ViT backbone via onnxruntime; the global-pool output is the embedding.
    Set pad=True for aspect-preserving letterbox (needed by ViT/DINOv2)."""
    def __init__(self, onnx_path: str, input_size: int = 224, pad: bool = False):
        import onnxruntime as ort
        _preload_cuda_libs()
        prov = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.sess = ort.InferenceSession(onnx_path, providers=prov)
        self.providers = self.sess.get_providers()   # what actually loaded, not what we asked for
        self.input_name = self.sess.get_inputs()[0].name
        st = os.stat(onnx_path)                      # identity of the WEIGHTS, not just the path:
        self.model_id = f"{onnx_path}|{st.st_mtime_ns}|{st.st_size}"   # swapping the file must
        #                                             invalidate anything derived from it
        self.size = input_size
        self.pad = pad          # letterbox to square instead of stretch
        self.name = f"onnx({onnx_path})"
        self.mean = np.array([0.485, 0.456, 0.406], np.float32)
        self.std = np.array([0.229, 0.224, 0.225], np.float32)

    def _fit(self, img: np.ndarray) -> np.ndarray:
        if not self.pad:
            return cv2.resize(img, (self.size, self.size))
        h, w = img.shape[:2]
        s = self.size / max(h, w)
        r = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
        canvas = np.full((self.size, self.size, 3), 128, np.uint8)
        y0, x0 = (self.size - r.shape[0]) // 2, (self.size - r.shape[1]) // 2
        canvas[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
        return canvas

    def embed(self, bgr: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        img = bgr
        if mask is not None:  # neutral-gray fill outside the object
            img = bgr.copy()
            img[mask == 0] = 128
        rgb = cv2.cvtColor(self._fit(img), cv2.COLOR_BGR2RGB)
        x = (rgb.astype(np.float32) / 255.0 - self.mean) / self.std
        x = x.transpose(2, 0, 1)[None]
        out = self.sess.run(None, {self.input_name: x})[0].reshape(-1).astype(np.float32)
        return _l2n(out)

    def _pre(self, bgr: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
        img = bgr
        if mask is not None:
            img = bgr.copy()
            img[mask == 0] = 128
        rgb = cv2.cvtColor(self._fit(img), cv2.COLOR_BGR2RGB)
        return ((rgb.astype(np.float32) / 255.0 - self.mean) / self.std).transpose(2, 0, 1)

    def embed_batch(self, crops: list, masks: list | None = None) -> np.ndarray:
        """Embed several crops in ONE ORT call -> [N, D] L2-normalized.

        The exported graph has a dynamic batch axis, so batching amortizes the
        per-call overhead; on GPU this is several times faster than looping embed().
        """
        if not crops:
            return np.zeros((0, 0), np.float32)
        x = np.stack([self._pre(c, None if masks is None else masks[i])
                      for i, c in enumerate(crops)])
        out = self.sess.run(None, {self.input_name: x})[0].astype(np.float32)
        return out / np.clip(np.linalg.norm(out, axis=1, keepdims=True), 1e-12, None)


def get_embedder(onnx_path: str | None = None, pad: bool = False):
    if onnx_path:
        return OnnxEmbedder(onnx_path, pad=pad)
    return ClassicalEmbedder()
