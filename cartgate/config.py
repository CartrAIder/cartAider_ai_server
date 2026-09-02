"""Runtime knobs, split by which side of the contract owns them.

docs/CONTRACT_v1.1.md §1 draws the line: vision answers "what is there, how
many"; the decision layer answers "does it match the receipt". Thresholds
follow that same split, so recalibrating similarity on real footage touches the
decision layer only and leaves the vision pipeline untouched.
"""
from cartgate.vision_fusion import MERGE_RADIUS_CM, MIN_FRAMES   # noqa: F401  (re-export)

# ---------------------------------------------------------------------------
# VISION-owned — observation quality and geometry
# ---------------------------------------------------------------------------
DET_CONF = 0.25          # YOLO confidence threshold
TRACK_IOU = 0.4          # IoU needed to associate a detection with a tracked object
# MIN_FRAMES / MERGE_RADIUS_CM are defined in cartgate/vision_fusion.py (the
# module that acts on them) and re-exported here so there is one definition.

GATE_ID = "demo-gate-1"  # per-deployment identity stamped into VisionObservation

# ---------------------------------------------------------------------------
# DECISION-owned — "how similar is similar enough" is a judgement, not a measurement
# ---------------------------------------------------------------------------
# The authoritative copy lives in cartgate/verification/reference_verify.py as
# SIM_STRONG / SIM_WEAK; recalibrate there. These are kept only for demo and
# visualization tooling (scripts/viz_recognition.py colours boxes by band) and
# must NOT be used inside the vision pipeline — it reports similarities, it
# does not interpret them.
PASS_SIM = 0.55          # == reference_verify.SIM_STRONG
REVIEW_SIM = 0.42        # == reference_verify.SIM_WEAK


def band(sim: float) -> str:
    """strong / weak / none. DECISION-side interpretation; demo tooling only."""
    return "strong" if sim >= PASS_SIM else ("weak" if sim >= REVIEW_SIM else "none")
