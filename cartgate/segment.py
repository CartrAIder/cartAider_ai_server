"""Background removal for single-object studio shots. Backend is chosen by the
CARTGATE_SEG env var: rembg (U^2-Net matting), grabcut (border-prior), or auto
(rembg with a grabcut fallback). Returns (rgba_cutout_BGRA, mask{0,255}) at input
size. The rembg model is set by CARTGATE_REMBG_MODEL (default "u2net")."""
import os

import cv2
import numpy as np

_SESSION = None


def _rembg_session():
    global _SESSION
    if _SESSION is None:
        from rembg import new_session
        _SESSION = new_session(os.environ.get("CARTGATE_REMBG_MODEL", "u2net"))
    return _SESSION


def _clean_mask(fg: np.ndarray, w: int, h: int) -> np.ndarray:
    """Keep the largest connected component, close small holes, resize to (w,h)."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    if n > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        fg = np.where(labels == largest, 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)
    if fg.shape[1] != w or fg.shape[0] != h:
        fg = cv2.resize(fg, (w, h), interpolation=cv2.INTER_NEAREST)
    return fg


def _grabcut_fg(bgr: np.ndarray, downscale: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    scale = min(1.0, downscale / max(h, w))
    small = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else bgr.copy()
    sh, sw = small.shape[:2]

    # Border pixels are assumed background; center block assumed probable-foreground.
    mask = np.full((sh, sw), cv2.GC_PR_BGD, np.uint8)
    b = max(4, int(0.03 * min(sh, sw)))
    mask[:b, :] = cv2.GC_BGD
    mask[-b:, :] = cv2.GC_BGD
    mask[:, :b] = cv2.GC_BGD
    mask[:, -b:] = cv2.GC_BGD
    cy0, cy1 = int(sh * 0.30), int(sh * 0.70)
    cx0, cx1 = int(sw * 0.30), int(sw * 0.70)
    mask[cy0:cy1, cx0:cx1] = cv2.GC_PR_FGD

    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(small, mask, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    return _clean_mask(fg, w, h)


def _rembg_fg(bgr: np.ndarray, thresh: int = 128) -> np.ndarray:
    from PIL import Image
    from rembg import remove
    h, w = bgr.shape[:2]
    pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    out = np.array(remove(pil, session=_rembg_session()))  # RGBA
    alpha = out[:, :, 3] if out.ndim == 3 and out.shape[2] == 4 else np.full((h, w), 255, np.uint8)
    fg = np.where(alpha >= thresh, 255, 0).astype(np.uint8)
    return _clean_mask(fg, w, h)


def remove_background(bgr: np.ndarray, downscale: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Return (rgba_cutout, mask) where mask is uint8 {0,255} at input size.

    Backend from CARTGATE_SEG: auto (default) | rembg | grabcut.
    """
    backend = os.environ.get("CARTGATE_SEG", "auto").lower()
    if backend in ("auto", "rembg"):
        try:
            fg = _rembg_fg(bgr)
        except Exception:
            if backend == "rembg":
                raise
            fg = _grabcut_fg(bgr, downscale)
    else:
        fg = _grabcut_fg(bgr, downscale)

    rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = fg
    return rgba, fg


def crop_to_object(rgba: np.ndarray, pad: float = 0.03) -> np.ndarray:
    """Tight crop around the alpha mask with small padding."""
    ys, xs = np.where(rgba[:, :, 3] > 0)
    if len(ys) == 0:
        return rgba
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    ph = int((y1 - y0) * pad)
    pw = int((x1 - x0) * pad)
    y0, y1 = max(0, y0 - ph), min(rgba.shape[0], y1 + ph)
    x0, x1 = max(0, x0 - pw), min(rgba.shape[1], x1 + pw)
    return rgba[y0:y1, x0:x1]
