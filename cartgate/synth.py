"""Synthesize cart scenes from single-object cutouts for detector training.
Objects are piled with overlap, perspective/rotation/lighting jitter, motion blur
and quality degradation. GT boxes come from the composition (occlusion-aware: an
object mostly covered by items on top of it is dropped), so no manual boxing."""
from dataclasses import dataclass, field

import cv2
import numpy as np

from cartgate.gallery import rotate_rgba

# Real-world longest dimension per SKU (cm), so objects keep correct RELATIVE sizes
# in a scene (a monitor dwarfs a lip balm). Rough but consistent.
SIZE_CM = {
    "S0001": 15, "S0002": 8, "S0003": 45, "S0004": 10, "S0005": 10, "S0006": 30,
    "S0007": 23, "S0008": 6, "S0009": 10, "S0010": 20, "S0011": 17, "S0012": 40,
    "S0013": 33, "S0014": 16, "S0015": 13, "S0016": 100, "S0017": 15, "S0018": 44,
    "S0019": 7, "S0020": 54, "S0021": 11, "S0022": 40, "S0023": 24, "S0024": 14,
    "S0025": 20, "S0026": 24, "S0027": 19, "S0028": 40, "S0029": 6, "S0030": 11,
    "S0031": 50, "S0032": 20, "S0033": 12, "S0034": 30, "S0035": 20, "S0036": 20,
    "S0037": 25, "S0038": 12, "S0039": 10, "S0040": 8, "S0041": 11, "S0042": 60,
    "S0043": 28, "S0044": 28, "S0045": 8, "S0046": 30, "S0047": 7, "S0048": 28,
    "S0049": 20, "S0050": 10, "S0051": 11,
}


@dataclass
class PlacedObject:
    sku: str
    track_id: int
    box: tuple  # x0, y0, x1, y1


@dataclass
class Frame:
    image: np.ndarray
    objects: list = field(default_factory=list)


def random_degrade(img: np.ndarray, rng: np.random.Generator, p: float = 0.65) -> np.ndarray:
    """Degrade like a cheap/far/low-light camera (low-res, noise, defocus,
    brightness, JPEG). Geometry is preserved so GT boxes stay valid."""
    if rng.random() > p:
        return img
    h, w = img.shape[:2]
    if rng.random() < 0.6:                                   # low resolution
        s = rng.uniform(0.35, 0.7)
        img = cv2.resize(cv2.resize(img, (max(1, int(w*s)), max(1, int(h*s)))), (w, h))
    if rng.random() < 0.4:                                   # extra defocus
        img = cv2.GaussianBlur(img, (int(rng.choice([3, 5, 7])),) * 2, 0)
    if rng.random() < 0.5:                                   # sensor noise
        img = np.clip(img.astype(np.float32) + rng.normal(0, rng.uniform(5, 20), img.shape), 0, 255).astype(np.uint8)
    if rng.random() < 0.45:                                  # brightness / low light
        img = np.clip(img.astype(np.float32) * rng.uniform(0.5, 1.2), 0, 255).astype(np.uint8)
    if rng.random() < 0.4:                                   # JPEG blocking
        _, e = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(25, 60))])
        img = cv2.imdecode(e, 1)
    return img


def make_cart_background(w: int, h: int, rng: np.random.Generator) -> np.ndarray:
    """A shopping-cart interior: metallic wire mesh over a darker floor, with a
    slight perspective warp and edge vignette so it reads as a 3D basket."""
    top, bot = int(rng.integers(115, 170)), int(rng.integers(70, 105))
    col = rng.uniform(0.9, 1.1, 3)
    grad = np.linspace(top, bot, h, dtype=np.float32)[:, None, None]
    bg = np.repeat(np.clip(grad * col, 0, 255), w, axis=1)
    bg = np.clip(bg + rng.normal(0, 7, (h, w, 3)), 0, 255).astype(np.uint8)

    step = int(rng.integers(48, 72))                      # metallic wires: dark tube + highlight
    dark, hi = (48, 50, 54), (188, 193, 199)
    for x in range(step // 2, w, step):
        cv2.line(bg, (x, 0), (x, h), dark, 3)
        cv2.line(bg, (x - 1, 0), (x - 1, h), hi, 1)
    for y in range(step // 2, h, step):
        cv2.line(bg, (0, y), (w, y), dark, 3)
        cv2.line(bg, (0, y - 1), (w, y - 1), hi, 1)

    d = 0.10 * min(w, h)                                  # mild perspective
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[rng.uniform(0, d), rng.uniform(0, d)], [w - rng.uniform(0, d), rng.uniform(0, d)],
                      [w - rng.uniform(0, d), h - rng.uniform(0, d)], [rng.uniform(0, d), h - rng.uniform(0, d)]])
    bg = cv2.warpPerspective(bg, cv2.getPerspectiveTransform(src, dst), (w, h), borderMode=cv2.BORDER_REFLECT)

    yy, xx = np.mgrid[0:h, 0:w]                           # vignette (basket depth)
    v = np.clip(1 - 0.35 * (((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2), 0.5, 1)
    bg = np.clip(bg.astype(np.float32) * v[..., None], 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(bg, (3, 3), 0)


def _feather(rgba: np.ndarray, px: float = 1.6) -> np.ndarray:
    """Soften the hard cutout edge so it doesn't read as a pasted sticker."""
    a = cv2.erode(rgba[:, :, 3], np.ones((3, 3), np.uint8), iterations=1)
    rgba[:, :, 3] = cv2.GaussianBlur(a, (0, 0), px)
    return rgba


def _harmonize(rgba: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Lightly match a studio cutout to the cart interior — a mild brightness/colour
    shift so it doesn't read as a bright pasted sticker (no heavy darkening, so dark
    objects don't turn into black blobs)."""
    rgb = rgba[:, :, :3].astype(np.float32)
    gray = rgb.mean(2, keepdims=True)
    rgb = 0.9 * rgb + 0.1 * gray                          # slight desaturate
    rgb = rgb * rng.uniform(0.82, 1.02) * rng.uniform(0.94, 1.06, 3)
    rgba[:, :, :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgba


def _motion_blur(img: np.ndarray, rng: np.random.Generator, p: float = 0.2) -> np.ndarray:
    """Mild directional blur -> the cart is nearly stationary while being read at
    the gate, so this is rare and gentle."""
    if rng.random() >= p:
        return img
    L = int(rng.integers(5, 12))
    k = np.zeros((L, L), np.float32)
    k[L // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((L / 2, L / 2), float(rng.uniform(0, 180)), 1.0)
    k = cv2.warpAffine(k, M, (L, L))
    s = k.sum()
    return cv2.filter2D(img, -1, k / s) if s > 0 else img


def _perspective_jitter(rgba: np.ndarray, rng: np.random.Generator, amt: float = 0.13) -> np.ndarray:
    """Random perspective warp -> same product seen from top vs side vantage."""
    h, w = rgba.shape[:2]
    d = amt * min(h, w)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = src + rng.uniform(-d, d, src.shape).astype(np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(rgba, M, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))


def _paste_alpha(canvas: np.ndarray, rgba: np.ndarray, cx: int, cy: int):
    """Alpha-composite rgba centred at (cx,cy); return this object's alpha over the
    FULL canvas (for occlusion bookkeeping), or None if it landed off-frame."""
    h, w = rgba.shape[:2]
    x0, y0 = cx - w // 2, cy - h // 2
    x1, y1 = x0 + w, y0 + h
    H, W = canvas.shape[:2]
    sx0, sy0 = max(0, -x0), max(0, -y0)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x1), min(H, y1)
    if x1 <= x0 or y1 <= y0:
        return None
    patch = rgba[sy0:sy0 + (y1 - y0), sx0:sx0 + (x1 - x0)]
    alpha = patch[:, :, 3:4].astype(np.float32) / 255.0
    region = canvas[y0:y1, x0:x1].astype(np.float32)
    canvas[y0:y1, x0:x1] = (alpha * patch[:, :, :3] + (1 - alpha) * region).astype(np.uint8)
    full = np.zeros(canvas.shape[:2], np.uint8)
    full[y0:y1, x0:x1] = patch[:, :, 3]
    return full


CART_PLANE_CM = (90.0, 60.0)     # the cart opening the canvas stands for


def camera_params(direction: float, rng: np.random.Generator | None = None,
                  W: int = 640, jitter: bool = False) -> tuple[float, float]:
    """A camera's viewpoint: (pinch, shift) in pixels.

    Deployment geometry is FIXED — the gate cameras are bolted in place, which is
    the only reason a single homography per camera can exist. So this is
    deterministic by default; jitter=True re-samples it for training-time
    augmentation, where varied viewpoints help the detector generalize.
    """
    if jitter and rng is not None:
        return (rng.uniform(0.14, 0.22) * W, direction * rng.uniform(0.06, 0.12) * W)
    return (0.18 * W, direction * 0.09 * W)


def camera_matrix(direction: float, W: int, H: int,
                  params: tuple[float, float] | None = None) -> np.ndarray:
    """Perspective transform from the top-down cart canvas to this camera's view.
    Exposed so a calibration can be derived from known plane points."""
    pinch, shift = params if params is not None else camera_params(direction, W=W)
    src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    dst = np.float32([[pinch + shift, 0], [W - pinch + shift, 0], [W, H], [0, H]])
    return cv2.getPerspectiveTransform(src, dst)


def _apply_camera(frame: "Frame", direction: float, rng: np.random.Generator,
                  params: tuple[float, float] | None = None) -> "Frame":
    """Warp a top-down composite to an upper-diagonal viewpoint (the deployment
    setup: two cameras angled down from opposite sides). direction -1/+1 selects
    the side. Boxes are re-derived by warping their corners."""
    H, W = frame.image.shape[:2]
    M = camera_matrix(direction, W, H, params)
    frame.image = cv2.warpPerspective(frame.image, M, (W, H), borderMode=cv2.BORDER_REFLECT)
    kept = []
    for o in frame.objects:
        x0, y0, x1, y1 = o.box
        pts = np.float32([[[x0, y0]], [[x1, y0]], [[x1, y1]], [[x0, y1]]])
        w = cv2.perspectiveTransform(pts, M).reshape(-1, 2)
        nx0, ny0 = max(0, w[:, 0].min()), max(0, w[:, 1].min())
        nx1, ny1 = min(W, w[:, 0].max()), min(H, w[:, 1].max())
        if nx1 - nx0 > 2 and ny1 - ny0 > 2:
            kept.append(PlacedObject(o.sku, o.track_id, (int(nx0), int(ny0), int(nx1), int(ny1))))
    frame.objects = kept
    return frame


def _build_layout(cart_contents: list[str], cutouts: dict[str, list[np.ndarray]],
                  rng: np.random.Generator, W: int, H: int) -> list[dict]:
    """Sample ONE physical arrangement of the cart: appearance, size and place
    per item, with a fixed stacking order. Every camera films this same pile and
    every frame re-renders it, which is what makes cross-camera merging and
    frame-to-frame tracking meaningful."""
    ccx, ccy = rng.uniform(0.35, 0.65) * W, rng.uniform(0.35, 0.65) * H  # pile centre
    layout = []
    for track_id in rng.permutation(len(cart_contents)):
        sku = cart_contents[track_id]
        rgba = cutouts[sku][rng.integers(len(cutouts[sku]))].copy()
        rel = (SIZE_CM.get(sku, 15) / 15.0) ** 0.4            # mild size hint (compressed)
        depth = rng.uniform(0.65, 1.5)                        # pile depth / near-far dominates
        target = float(np.clip(0.28 * min(W, H) * rel * depth, 0.10 * min(W, H), 0.60 * min(W, H)))
        s = target / max(rgba.shape[:2])
        rgba = cv2.resize(rgba, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        if rng.random() < 0.6:
            rgba = _perspective_jitter(rgba, rng)             # multi-view
        rgba = rotate_rgba(rgba, float(rng.uniform(0, 360)))
        rgba = _harmonize(rgba, rng)                          # match dim cart lighting
        rgba = _feather(rgba)                                 # soft edges
        if rng.random() < 0.85:                               # dense central pile (loaded cart)
            cx = float(np.clip(rng.normal(ccx, 0.12 * W), 0, W))
            cy = float(np.clip(rng.normal(ccy, 0.12 * H), 0, H))
        else:                                                 # near edge (partial view)
            cx, cy = rng.uniform(0.1, 0.9) * W, rng.uniform(0.1, 0.9) * H
        layout.append({"sku": sku, "track_id": int(track_id), "rgba": rgba, "cx": cx, "cy": cy})
    return layout


def _render(layout: list[dict], bg: np.ndarray, offsets: list[tuple],
            vis_thresh: float) -> Frame:
    """Composite the layout onto a copy of bg with per-item offsets, and derive
    occlusion-aware boxes (an item buried by the items stacked on top of it gets
    no label)."""
    canvas = bg.copy()
    H, W = canvas.shape[:2]
    placed = []  # (track_id, sku, full-canvas alpha)
    for item, (dx, dy) in zip(layout, offsets):
        alpha = _paste_alpha(canvas, item["rgba"], int(item["cx"] + dx), int(item["cy"] + dy))
        if alpha is not None and alpha.any():
            placed.append((item["track_id"], item["sku"], alpha))

    frame = Frame(image=canvas)
    for idx, (tid, sku, alpha) in enumerate(placed):
        occ = np.zeros((H, W), bool)
        for _, _, a2 in placed[idx + 1:]:
            occ |= a2 > 0
        vis = (alpha > 0) & ~occ
        area = int((alpha > 0).sum())
        if area == 0 or vis.sum() / area < vis_thresh:        # mostly buried -> no label
            continue
        ys, xs = np.where(vis)
        frame.objects.append(PlacedObject(
            sku=sku, track_id=tid,
            box=(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))))
    return frame


def synth_cart_views(cart_contents: list[str], cutouts: dict[str, list[np.ndarray]],
                     rng: np.random.Generator, cameras=(-1.0, 1.0), n_frames: int = 4,
                     size: tuple = (640, 640), vis_thresh: float = 0.18,
                     drift_px: float = 6.0,
                     jitter_camera: bool = False) -> dict[float, list[Frame]]:
    """One cart, filmed by fixed cameras over consecutive frames.

    * the pile is sampled once, so both cameras see the SAME physical objects
    * camera viewpoints are fixed (jitter_camera=True re-samples them for
      detector-training variety), so one homography per camera exists — a
      per-frame or per-cart random warp would make calibration meaningless
    * between frames the whole cart translates a few pixels (it is rolling
      through the gate) plus a small per-item settle -> a tracker has continuous
      motion to follow instead of an unrelated pile every frame

    Returns {camera direction: [Frame, ...]}, oldest frame first.
    """
    W, H = size
    layout = _build_layout(cart_contents, cutouts, rng, W, H)
    bg = make_cart_background(W, H, rng)                      # same cart interior for all views
    vx, vy = rng.normal(0, drift_px), rng.normal(0, drift_px)  # constant cart velocity
    jitter = [[(rng.normal(0, 1.2), rng.normal(0, 1.2)) for _ in layout]
              for _ in range(n_frames)]

    views = {}
    for d in cameras:
        d = float(d)
        params = camera_params(d, rng, W, jitter=jitter_camera)   # fixed per camera, not per frame
        frames = []
        for fi in range(n_frames):
            offsets = [(vx * fi + jx, vy * fi + jy) for jx, jy in jitter[fi]]
            frame = _render(layout, bg, offsets, vis_thresh)
            frame = _apply_camera(frame, d, rng, params=params)    # upper-diagonal viewpoint
            frame.image = _motion_blur(frame.image, rng)          # rolling cart
            k = 3 if rng.random() < 0.5 else 5
            frame.image = cv2.GaussianBlur(frame.image, (k, k), 0)
            frames.append(frame)
        views[d] = frames
    return views


def synth_cart_frames(cart_contents: list[str], cutouts: dict[str, list[np.ndarray]],
                      rng: np.random.Generator, n_frames: int = 2,
                      size: tuple = (960, 720), vis_thresh: float = 0.18,
                      cam_dirs: list | None = None) -> list[Frame]:
    """Single-camera convenience wrapper over synth_cart_views: n_frames
    consecutive frames from ONE camera. cam_dirs picks the side (a list is
    accepted for compatibility; its first entry is used, since one camera has
    one viewpoint)."""
    d = float(cam_dirs[0]) if cam_dirs else float(rng.choice([-1.0, 1.0]))
    return synth_cart_views(cart_contents, cutouts, rng, cameras=(d,), n_frames=n_frames,
                            size=size, vis_thresh=vis_thresh)[d]
