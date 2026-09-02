"""One-time cart-plane calibration for the AI exit gate.

Because the QR reader fixes WHERE the cart stands when the shutter fires, a
single homography per camera stays valid for the life of the gate. This maps
image pixels -> a shared cart-plane coordinate system in centimetres, which is
what lets us tell "one cola seen by two cameras" from "two colas".

Two ways to collect correspondences:

  A. ArUco board (recommended, repeatable)
     Print a sheet with 4 ArUco markers at known cm positions, lay it across
     the cart opening at the trigger position, capture one frame per camera.
     -> calibrate_from_aruco()

  B. Manual 4-point (no printer needed)
     Put tape marks at 4 known positions on the cart rim, note their cm
     coordinates, click them in the image.
     -> calibrate_from_points()

Validation matters more than the method: check_calibration() reports the
reprojection error and the cross-camera agreement error. If cross-camera
agreement is worse than ~5 cm, the merge radius must be widened or the
calibration redone.
"""
from __future__ import annotations

import json

import cv2
import numpy as np


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------

def calibrate_from_points(image_pts: list[tuple[float, float]],
                          plane_pts_cm: list[tuple[float, float]]) -> np.ndarray:
    """Homography from >=4 correspondences. image px -> cart-plane cm."""
    if len(image_pts) < 4 or len(image_pts) != len(plane_pts_cm):
        raise ValueError("need >=4 matched point pairs")
    src = np.array(image_pts, np.float32).reshape(-1, 1, 2)
    dst = np.array(plane_pts_cm, np.float32).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if H is None:
        raise RuntimeError("homography estimation failed - check correspondences")
    return H


def calibrate_from_aruco(image_bgr: np.ndarray,
                         marker_plane_cm: dict[int, tuple[float, float]],
                         dictionary: int = cv2.aruco.DICT_4X4_50) -> np.ndarray:
    """Detect ArUco markers and build the homography from their centers.

    marker_plane_cm: {marker_id: (x_cm, y_cm)} - the known plane position of
    each marker's CENTER on the calibration sheet.
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(image_bgr)
    if ids is None:
        raise RuntimeError("no ArUco markers detected")

    img_pts, plane_pts = [], []
    for c, i in zip(corners, ids.flatten()):
        if int(i) in marker_plane_cm:
            img_pts.append(tuple(c.reshape(4, 2).mean(axis=0)))
            plane_pts.append(marker_plane_cm[int(i)])
    if len(img_pts) < 4:
        raise RuntimeError(f"only {len(img_pts)} known markers found, need >=4")
    return calibrate_from_points(img_pts, plane_pts)


# --------------------------------------------------------------------------
# projection + validation
# --------------------------------------------------------------------------

def project(H: np.ndarray, pt_px: tuple[float, float]) -> tuple[float, float]:
    v = H @ np.array([pt_px[0], pt_px[1], 1.0], np.float64)
    if abs(v[2]) < 1e-9:
        return (float("inf"), float("inf"))
    return (float(v[0] / v[2]), float(v[1] / v[2]))


def check_calibration(H: np.ndarray,
                      image_pts: list[tuple[float, float]],
                      plane_pts_cm: list[tuple[float, float]]) -> dict:
    """Reprojection error in cm."""
    errs = [float(np.hypot(*(np.array(project(H, p)) - np.array(q))))
            for p, q in zip(image_pts, plane_pts_cm)]
    return {"mean_cm": float(np.mean(errs)), "max_cm": float(np.max(errs)),
            "per_point_cm": [round(e, 2) for e in errs]}


def check_cross_camera(HA: np.ndarray, HB: np.ndarray,
                       pts_a_px: list[tuple[float, float]],
                       pts_b_px: list[tuple[float, float]]) -> dict:
    """Agreement error: the SAME physical points seen by both cameras should
    land on the same plane coordinate. This is the number that decides the
    merge radius."""
    errs = [float(np.hypot(*(np.array(project(HA, a)) - np.array(project(HB, b)))))
            for a, b in zip(pts_a_px, pts_b_px)]
    mean, mx = float(np.mean(errs)), float(np.max(errs))
    return {"mean_cm": mean, "max_cm": mx,
            "suggested_merge_radius_cm": round(max(8.0, mx * 2.0), 1)}


# --------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------

def save(path: str, homographies: dict[str, np.ndarray], meta: dict | None = None):
    json.dump({"schema_version": "1.0",
               "homographies": {k: v.tolist() for k, v in homographies.items()},
               "meta": meta or {}},
              open(path, "w"), indent=2)


def load(path: str) -> dict[str, np.ndarray]:
    d = json.load(open(path))
    return {k: np.array(v, np.float64) for k, v in d["homographies"].items()}


# --------------------------------------------------------------------------
# self-test with a synthetic gate geometry
# --------------------------------------------------------------------------

if __name__ == "__main__":
    # Simulate two cameras viewing a 90x60 cm cart opening from opposite
    # upper diagonals, then verify we recover the plane coordinates.
    plane = [(0, 0), (90, 0), (90, 60), (0, 60)]

    def synth_camera(shift, tilt):
        """Fake a perspective view of the plane."""
        src = np.array(plane, np.float32)
        dst = np.array([[100 + shift, 80 + tilt], [820 + shift, 60 - tilt],
                        [760 + shift, 520], [160 + shift, 540]], np.float32)
        return cv2.getPerspectiveTransform(src, dst)

    M_L, M_R = synth_camera(0, 0), synth_camera(120, 25)

    px_L = [tuple(project(M_L, p)) for p in plane]
    px_R = [tuple(project(M_R, p)) for p in plane]

    H_L = calibrate_from_points(px_L, plane)
    H_R = calibrate_from_points(px_R, plane)

    print("cam_left  reprojection :", check_calibration(H_L, px_L, plane))
    print("cam_right reprojection :", check_calibration(H_R, px_R, plane))

    # a physical object at (45, 30) cm, seen by both cameras
    obj_L = project(M_L, (45, 30))
    obj_R = project(M_R, (45, 30))
    print("object px in cam_left  :", tuple(round(v, 1) for v in obj_L))
    print("object px in cam_right :", tuple(round(v, 1) for v in obj_R))
    print("-> plane from cam_left :", tuple(round(v, 2) for v in project(H_L, obj_L)))
    print("-> plane from cam_right:", tuple(round(v, 2) for v in project(H_R, obj_R)))

    print("cross-camera agreement :",
          check_cross_camera(H_L, H_R,
                             [obj_L] + px_L, [obj_R] + px_R))

    save("/tmp/gate_calib.json", {"cam_left": H_L, "cam_right": H_R},
         meta={"plane_cm": "90x60 cart opening", "trigger": "qr_scan"})
    print("saved -> /tmp/gate_calib.json ; reload ok:", list(load("/tmp/gate_calib.json")))
