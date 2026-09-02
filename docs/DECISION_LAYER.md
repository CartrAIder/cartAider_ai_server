# Decision layer — spec & recommended approach

This is the part **you** own: given what the vision models see, and the payment
DB receipt, decide whether a cart is fine or should be stopped. The vision side
(detection + recognition) is handled separately; treat its output as your input.

The core idea to keep in mind: this is **receipt-conditioned anomaly detection**.
We can never see every item in a loaded cart, so the job is not "list everything
in the cart". It is "is there *positive evidence* that something unpaid or in
excess is present?" Absence of evidence (a paid item we can't see) is normal.

---

## 1. Inputs / outputs (data contract)

**Input per cart:**

- `receipt`: `{sku_id: paid_quantity}` from the payment DB.
- `observations`: per camera, a list of tracks. Each track is one physical object
  followed across frames, with:
  - `track_id`
  - `sku` — best receipt-SKU match (or `None` = matches nothing on the receipt)
  - `sim` — cosine similarity of the match (confidence proxy, 0-1)
  - `n_frames` — how many frames it was seen in (stability)
  - optionally: box trajectory, timestamps, camera id.

The recognition step already restricts each object to the cart's receipt SKUs, so
`sku` is either a receipt item or `None`.

**Output:**

- `verdict`: `PASS` | `FLAG` | `REVIEW`
- `reasons`: human-readable explanation (for the admin console)
- structured detail: which items triggered it, confidences, counts.

---

## 2. Start from what exists

> **`docs/CONTRACT_v1.1.md` is the authoritative interface.** Section 1 below
> predates it; where they disagree, the contract wins. The old `match.py`
> helpers (`decide_cart`, `verify_cart`, `match_frame_crops`,
> `aggregate_tracks`) have been deleted — they mixed vision and decision
> concerns, which is exactly what the contract separates.

- **Input**: `VisionObservation` JSON (contract §3). Vision hands you instances
  with the full `candidates` similarity vector — no verdict, no band, no
  threshold interpretation. Similarity thresholds are *yours* (contract §1).
- **Reference implementation**: `cartgate/verification/reference_verify.py`
  (`verify(observation, receipt) -> GateVerdict`). It implements contract §5:
  global Hungarian assignment when `cross_camera_resolved` is true, and a
  conservative per-camera max/min count when it is false. This is the starting
  point for `VerificationService` in PR #1.
- **Boundary test**: `tests/test_boundary.py` — 6 scenarios × 2 fusion modes,
  crossing a real `json.dumps`/`loads` so no Python objects are shared.

These are rule-based and calibrated on synthetic data. Your job is to harden this
into something that holds up on real footage. Read `reference_verify.verify()`
first; the rest of this doc explains the design and where to take it.

---

## 3. Sub-problems and recommended approach

### 3a. Aggregate over time and cameras
A single frame is noisy. Aggregate before deciding.

- **Tracking**: use ByteTrack (built into `ultralytics`) on each camera's 30 fps
  stream so each physical item is one track. Vote its SKU across frames (majority
  or sim-weighted) instead of trusting any single frame.
- **Per-SKU count per camera** = number of distinct stable tracks resolved to that
  SKU.
- **Fuse cameras**: take the **max count per SKU across cameras**, not the sum — a
  top camera and a side camera see the *same* physical apple, so summing double-
  counts. Max de-duplicates the common case while still recovering an item one
  camera missed to occlusion.
  - Caveat: max cannot tell "same item in 2 views" from "two identical items in
    different spots". Truly counting identical duplicates needs camera geometry
    (calibrate the 2 cameras once, triangulate box centers). Until then, max is the
    safe, calibratable stand-in — document it as a known limitation.

### 3b. Match visible items to the receipt
Given fused per-SKU observations vs `receipt`, decide what is explained.

- Model it as **bipartite assignment**: expand the receipt into one slot per paid
  unit (`{apple:2}` → two apple slots), and assign confident visible items to slots
  with the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`), cost =
  `1 - sim` for a matching SKU, ∞ otherwise. `match.py` already does this.
- After assignment:
  - **Visible item matched to no slot** → unpaid item or over-quantity → evidence for FLAG.
  - **Item resolving to `None`** (matches nothing on the receipt) → strong evidence
    of an unpaid item → FLAG.
  - **Slots left empty** (paid item never seen) → *expected occlusion, ignore* (see 3c).

### 3c. Occlusion / invisible items — the key principle
You cannot see everything in a cart, so:

- **Never** flag a cart because a *paid* item wasn't seen. Missing-and-paid is the
  normal case (buried at the bottom). Only accuse on positive evidence: a **visible**
  item that is unpaid, or **more** confident instances of an item than were paid for.
- The two cameras view from opposite upper-diagonal angles, so each covers the
  other's blind spots — an item occluded in one view is often visible in the other.
- Optional sanity signal: a coarse **cart-fullness / volume estimate** (from the
  top view). It shouldn't accuse on its own, but a gross mismatch — receipt says 2
  items, cart is visibly full — is a good reason to escalate to REVIEW rather than PASS.

### 3d. Decision policy — 3-way, cost-sensitive
Wrongly stopping an honest shopper is worse than missing one item, so don't force a
binary call.

- **PASS** — no positive evidence of a problem.
- **FLAG** — stable, confident evidence of an unpaid or excess item (act on it).
- **REVIEW** — ambiguous: mid-confidence match, a single unstable extra track, or a
  fullness mismatch → send to a human, don't auto-stop.
- Bands come from the recognition similarity (`pass_sim` / `review_sim`) and track
  stability (`min_frames`). Tune for **high precision on FLAG**; push uncertainty
  into REVIEW.

---

## 4. Rules first, learning later

Recommended order — don't jump to a heavy model:

1. **Rule-based + calibrated thresholds** (what's here now). Fast, debuggable,
   explainable to a store operator. Get this solid on real footage first.
2. **Calibrate on real data**: collect gate videos with ground-truth receipts,
   then set `pass_sim / review_sim / min_frames` and the count-fusion rule from an
   actual PASS/FLAG/REVIEW confusion analysis. Optimize an operating point, not
   accuracy — the two error types have very different costs.
3. **Learned decision (optional, once labeled data exists)**: features per cart
   (per-SKU observed vs paid counts, similarity distribution, track stability,
   fullness, #None-tracks) → a small gradient-boosted classifier or logistic model
   outputting PASS/FLAG/REVIEW with a calibrated probability. Keep it interpretable;
   the rules become the features. This mainly buys you better thresholds and item
   interactions, not a different paradigm.
4. A probabilistic framing (estimate `P(unpaid item present | observations, receipt)`
   and threshold on expected cost) is a clean way to unify 2 and 3 if you want it.

---

## 5. Evaluation

- Build a labeled set of real carts: {videos, receipt, true verdict}.
- Report a **PASS/FLAG/REVIEW confusion matrix**, plus the two costs separately:
  false-stop rate (honest shopper flagged) and miss rate (theft passed).
- Track "matched-recall": at a fixed catch rate, how often do we wrongly stop
  someone. That trade-off, not raw accuracy, is the number that decides deployment.

## 6. Open limitations to design around

- Identical-duplicate counting across cameras needs geometry/calibration.
- Everything is calibrated on synthetic carts today; thresholds **will** move on
  real footage.
- The receipt is assumed correct and available at decision time.
