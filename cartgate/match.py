"""Crop-vs-gallery similarity — the only piece of receipt reasoning that lives
on the VISION side.

Everything that used to compare observations against a receipt (decide_cart,
verify_cart, match_frame_crops, aggregate_tracks) moved across the boundary:
cross-camera fusion is now cartgate/vision_fusion.py, and receipt
reconciliation is the decision layer's (docs/CONTRACT_v1.1.md §5, reference
implementation in cartgate/verification/reference_verify.py).
"""
import cv2
import numpy as np


def best_sim_against_sku(vec: np.ndarray, gallery_entry: dict) -> float:
    """Max cosine similarity against all gallery variants of one SKU."""
    return float(gallery_entry["vectors"] @ vec)  # vectors are L2-normalized


def sku_similarity(vec: np.ndarray, gallery: dict, sku: str) -> float:
    return float(np.max(gallery[sku]["vectors"] @ vec))


def sift_inliers(query_bgr: np.ndarray, ref_bgr: np.ndarray, max_side: int = 320) -> int:
    """Geometric verification: SIFT matches consistent with a homography.

    Currently unused — kept for a future geometric re-check of ambiguous crops
    (and as the SIFT dependency's only consumer; opencv-contrib is pinned for it).
    """
    def prep(img):
        s = max_side / max(img.shape[:2])
        if s < 1:
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=800)
    k1, d1 = sift.detectAndCompute(prep(query_bgr), None)
    k2, d2 = sift.detectAndCompute(prep(ref_bgr), None)
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        return 0
    bf = cv2.BFMatcher()
    good = [m for m, n in bf.knnMatch(d1, d2, k=2) if m.distance < 0.75 * n.distance]
    if len(good) < 8:
        return len(good)
    src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    _, inl = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return int(inl.sum()) if inl is not None else 0
