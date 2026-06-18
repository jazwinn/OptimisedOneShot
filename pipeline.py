"""
OptimisedOneShot — GPU Pipeline Process
========================================
Three-stage producer-consumer pipeline running entirely inside a spawned
child process so the GPU has its own CUDA context, isolated from the Qt
main process.

Architecture
------------
GPUPipelineProcess (mp.Process, daemon=True)
  │
  ├─ DecodeThread  (threading.Thread)
  │    raw_frame_queue (mp.Queue) ──► decode_to_infer_q (threading.Queue, size=3)
  │    Converts incoming BGR numpy frames to contiguous CUDA float32 tensors.
  │
  ├─ InferThread   (threading.Thread)
  │    decode_to_infer_q ──► infer_to_blend_q (threading.Queue, size=3)
  │    Tracker predict → FastSAM ROI / full-frame → Re-ID batch embed
  │    → EMA gallery cosine match → best mask selection.
  │
  └─ BlendThread   (threading.Thread)
       infer_to_blend_q ──► display_queue (mp.Queue, size=3) → Main Process
       GPU alpha-composite, BGR→RGB channel swap, pull to CPU numpy, tobytes().
       Optionally writes frames to VideoWriter for batch export.

CUDA contract
-------------
* torch is imported ONLY inside run() / thread run() methods, never at module
  level or in __init__. This ensures the spawned process owns a clean CUDA
  context.
* CUDA tensors travel through threading.Queue (intra-process, no serialisation).
* Only numpy bytes cross the mp.Queue boundary to the main process.

Bbox coordinate convention
--------------------------
* All bounding boxes inside this module use **xyxy** (x1,y1,x2,y2) format,
  expressed in full-frame pixel coordinates.
* TrackerWrapper (models.py) accepts/returns xyxy and converts to cv2 xywh
  internally.
* The SAM2 registration result bbox arrives as xywh and is converted once at
  process startup.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import threading
import time
import traceback
from typing import Any, NamedTuple, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public command type (mirrors _PipelineCmd in gui.py)
# ---------------------------------------------------------------------------

class PipelineCmd(NamedTuple):
    """Command token routed through pipeline_cmd_queue."""
    type:    str       # START | STOP | UPDATE_THRESHOLD | UPDATE_ALPHA | UPDATE_STRIDE
    payload: Any = None


# ---------------------------------------------------------------------------
# Module-level helper functions (pure, no CUDA, usable in all threads)
# ---------------------------------------------------------------------------

def _xywh_to_xyxy(bbox_xywh: tuple) -> tuple:
    """Convert (x, y, w, h) → (x1, y1, x2, y2)."""
    x, y, w, h = bbox_xywh
    return (x, y, x + w, y + h)


def _pad_bbox_xyxy(
    bbox_xywh: tuple,
    pad_frac: float,
    frame_h: int,
    frame_w: int,
) -> tuple:
    """
    Expand a tracker bbox (xywh) by pad_frac on each side, clamp to frame,
    and return as xyxy for use as an ROI crop region.
    """
    x, y, bw, bh = bbox_xywh
    px = bw * pad_frac
    py = bh * pad_frac
    x1 = max(0, int(x - px))
    y1 = max(0, int(y - py))
    x2 = min(frame_w, int(x + bw + px))
    y2 = min(frame_h, int(y + bh + py))
    return (x1, y1, x2, y2)


def _union_bboxes_padded(
    bboxes_xyxy: list,
    pad_frac: float,
    frame_h: int,
    frame_w: int,
) -> tuple:
    """
    Return a single xyxy ROI that is the union of all input bboxes (xyxy),
    padded by pad_frac of the union's dimensions.
    """
    x1 = min(b[0] for b in bboxes_xyxy)
    y1 = min(b[1] for b in bboxes_xyxy)
    x2 = max(b[2] for b in bboxes_xyxy)
    y2 = max(b[3] for b in bboxes_xyxy)
    pw = (x2 - x1) * pad_frac
    ph = (y2 - y1) * pad_frac
    return (
        max(0, int(x1 - pw)),
        max(0, int(y1 - ph)),
        min(frame_w, int(x2 + pw)),
        min(frame_h, int(y2 + ph)),
    )


def _crop_bgr(
    frame_bgr: np.ndarray,
    bbox_xyxy: tuple,
    mask_cuda=None,
) -> Optional[np.ndarray]:
    """
    Crop frame_bgr to bbox_xyxy.  If mask_cuda is given, pixels outside the
    mask are filled with the mean colour of the masked pixels so the Re-ID
    embedding sees the object shape without being anchored to the background
    colour (black backgrounds bias cosine similarity toward dark embeddings).
    Returns None when the crop has zero area.
    """
    x1, y1, x2, y2 = (int(v) for v in bbox_xyxy)
    fh, fw = frame_bgr.shape[:2]
    x1 = max(0, x1);  y1 = max(0, y1)
    x2 = min(fw, x2); y2 = min(fh, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame_bgr[y1:y2, x1:x2].copy()
    if mask_cuda is not None:
        import torch
        m = (mask_cuda[y1:y2, x1:x2] > 0.5).cpu().numpy()
        if m.any():
            mean_col = crop[m].mean(axis=0).astype(np.uint8)
            crop[~m] = mean_col
    return crop


def _bbox_to_mask_cuda(bbox_xyxy: tuple, H: int, W: int):
    """
    Create a float32 CUDA mask tensor filled with 1.0 inside the given xyxy
    bounding box.  Used as a fallback when no segmentation mask is available.
    Requires torch to be already imported in the calling thread.
    """
    import torch
    x1, y1, x2, y2 = (int(v) for v in bbox_xyxy)
    mask = torch.zeros(H, W, dtype=torch.float32, device="cuda")
    x1 = max(0, x1);  y1 = max(0, y1)
    x2 = min(W, x2);  y2 = min(H, y2)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = 1.0
    return mask


def _projection_valley_split(m_np: np.ndarray, min_area: int, max_ratio: float = 0.35) -> list | None:
    """
    Split a merged mask at any column or row whose projection depth is a valley.

    All positions where proj / avg <= max_ratio are treated as candidates.
    They are tried deepest-first so the clearest separator wins. Each candidate
    is validated by checking that both resulting halves exceed min_area —
    this naturally rejects false valleys at the mask boundary without any
    positional exclusion zone.
    """

    def _try_axis(proj: np.ndarray, idx_on: np.ndarray, make_halves) -> list | None:
        """
        Find the best valid split along this axis.
        make_halves(pos) → (left_np, right_np)
        """
        if len(idx_on) < 8:
            return None
        x0, x1 = int(idx_on[0]), int(idx_on[-1])
        if x1 - x0 < 8:
            return None

        avg = float(proj[idx_on].mean())
        if avg <= 0:
            return None

        # All interior positions (excluding the outermost active cols) where
        # the valley is deep enough. "Interior" = not the very first/last pixel
        # of the mask, since those would produce a zero-pixel half.
        inner = np.arange(x0 + 1, x1, dtype=np.int32)
        if len(inner) == 0:
            return None
        ratios = proj[inner] / avg
        valid  = inner[ratios <= max_ratio]
        if len(valid) == 0:
            return None

        # Sort candidates by depth (smallest ratio = deepest valley first).
        order      = np.argsort(ratios[ratios <= max_ratio])
        candidates = valid[order]

        # For each candidate, fast-check min_area from the projection sums
        # before allocating mask copies — projection sum ≈ pixel count.
        cumsum = np.cumsum(proj)  # full-frame cumulative column/row sum
        total  = float(proj[idx_on].sum())

        for pos in candidates:
            pos = int(pos)
            left_sum  = float(cumsum[pos - 1] - (cumsum[x0 - 1] if x0 > 0 else 0))
            right_sum = total - left_sum
            if left_sum < min_area or right_sum < min_area:
                continue
            left_np, right_np = make_halves(pos)
            parts = [s for s in (left_np, right_np) if int(s.sum()) >= min_area]
            if len(parts) >= 2:
                return parts

        return None

    proj_x  = m_np.sum(axis=0).astype(np.float32)
    proj_y  = m_np.sum(axis=1).astype(np.float32)
    cols_on = (proj_x > 0).nonzero()[0]
    rows_on = (proj_y > 0).nonzero()[0]

    # --- Try left-right split ---
    def _halves_x(x):
        left  = m_np.copy(); left[:, x:] = 0
        right = m_np.copy(); right[:, :x] = 0
        return left, right

    result = _try_axis(proj_x, cols_on, _halves_x)
    if result:
        return result

    # --- Try top-bottom split ---
    def _halves_y(y):
        top    = m_np.copy(); top[y:, :] = 0
        bottom = m_np.copy(); bottom[:y, :] = 0
        return top, bottom

    return _try_axis(proj_y, rows_on, _halves_y)


def _erosion_split(m_np: np.ndarray, min_area: int) -> list | None:
    """
    Split a single binary mask by eroding until multiple components appear,
    then Voronoi-expand each eroded seed back into the original mask pixels.

    Returns a list of uint8 sub-masks, or None if the mask cannot be split
    into ≥2 components each meeting min_area.
    """
    # Try increasing erosion sizes until we get multiple seeds.
    for erode_px in (8, 14, 20, 28):
        k = np.ones((erode_px, erode_px), np.uint8)
        eroded = cv2.erode(m_np, k, iterations=1)
        n, labels_e, stats_e, _ = cv2.connectedComponentsWithStats(eroded, connectivity=4)
        seed_ids = [i for i in range(1, n) if stats_e[i, cv2.CC_STAT_AREA] >= max(16, min_area // 8)]
        if len(seed_ids) >= 2:
            break
    else:
        return None

    # Voronoi assignment: each original mask pixel → nearest eroded seed centroid.
    # Build distance maps by treating each seed as the "foreground" and computing
    # distance-to-nearest-seed-pixel for every point in the frame.
    dist_maps = []
    for sid in seed_ids:
        seed_bin = (labels_e == sid).astype(np.uint8)
        dist = cv2.distanceTransform((1 - seed_bin).astype(np.uint8), cv2.DIST_L2, 3)
        dist_maps.append(dist)

    dist_stack = np.stack(dist_maps, axis=0)          # (n_seeds, H, W)
    assignment  = np.argmin(dist_stack, axis=0)       # H×W → seed index [0..n-1]

    result = []
    for idx in range(len(seed_ids)):
        sub = ((m_np > 0) & (assignment == idx)).astype(np.uint8)
        if int(sub.sum()) >= min_area:
            result.append(sub)

    return result if len(result) >= 2 else None



def split_masks_by_separators(
    frame_bgr:          np.ndarray,
    masks_cuda:         list,
    bboxes_xyxy:        list,
    *,
    sep_thresh:         int,
    min_area:           int,
    dilate:             int  = 2,
    use_col_cut:        bool = True,
    use_proj_valley:    bool = True,
    proj_valley_thresh: int  = 35,
    use_erosion:        bool = True,
) -> tuple[list, list]:
    """
    Split merged FastSAM masks along near-black separator lines.

    Two-stage approach:
    1. Column/row mean separator detection — far more robust than per-pixel
       thresholding because a single bright compression artifact cannot break
       an otherwise-dark column's average.  Combined with per-pixel dark pixels
       via union so we never lose existing coverage.
    2. Erosion-based Voronoi fallback — when separator cutting still produces
       only one component (e.g. the mask routes above/below the separator), we
       progressively erode the mask until the bridge between two objects breaks,
       use the resulting seeds, and Voronoi-expand each seed back into the
       original mask.  This works purely on mask shape and needs no separator.
    """
    import torch

    if not masks_cuda:
        return masks_cuda, bboxes_xyxy

    gray   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    H, W   = gray.shape
    kernel = np.ones((3, 3), np.uint8)

    # --- Stage 1: Dark Column Cut (optional) ---------------------------------
    if use_col_cut:
        sep = np.zeros((H, W), dtype=np.uint8)
        col_means = gray.mean(axis=0)
        sep[:, col_means <= sep_thresh] = 1
        row_means = gray.mean(axis=1)
        sep[row_means <= sep_thresh, :] = 1
        sep = np.maximum(sep, (gray <= sep_thresh).astype(np.uint8))
        sep = cv2.morphologyEx(sep, cv2.MORPH_CLOSE, kernel)
        if dilate > 0:
            sep = cv2.dilate(sep, kernel, iterations=int(dilate))
    else:
        sep = None

    out_masks:  list = []
    out_bboxes: list = []

    for m_cuda, box in zip(masks_cuda, bboxes_xyxy):
        m_np_orig = (m_cuda > 0.5).to("cpu", torch.uint8).numpy()

        # --- Global separator cut (full-frame column/row means) ---
        if use_col_cut and sep is not None:
            m_np = m_np_orig.copy()
            m_np[sep == 1] = 0
        else:
            m_np = m_np_orig.copy()

        # --- Local-bbox separator cut ------------------------------------
        # Compute column/row means WITHIN this mask's bounding box only.
        # A thin separator that's washed out in the full-frame average is
        # dominant within the bbox where it spans the entire local height.
        if use_col_cut:
            ys, xs = np.where(m_np_orig > 0)
            if len(xs) > 0:
                bx0, bx1 = int(xs.min()), int(xs.max()) + 1
                by0, by1 = int(ys.min()), int(ys.max()) + 1
                if bx1 > bx0 and by1 > by0:
                    roi = gray[by0:by1, bx0:bx1]
                    local_col = roi.mean(axis=0)    # (roi_w,)
                    local_row = roi.mean(axis=1)    # (roi_h,)
                    local_sep = np.zeros((H, W), dtype=np.uint8)
                    dark_c = (local_col <= sep_thresh).nonzero()[0] + bx0
                    dark_r = (local_row <= sep_thresh).nonzero()[0] + by0
                    if len(dark_c):
                        local_sep[:, dark_c] = 1
                    if len(dark_r):
                        local_sep[dark_r, :] = 1
                    if local_sep.any():
                        local_sep = cv2.dilate(local_sep, kernel, iterations=int(dilate))
                        m_np[local_sep == 1] = 0

        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m_np, connectivity=4)
        # Use a smaller floor when the separator cut produces multiple pieces.
        # The standard min_area (≈30% of registered area) would discard a valid
        # small component from an off-centre separator; halving it keeps edge splits.
        comp_min = max(64, min_area // 2) if n_labels > 2 else min_area
        comps = [i for i in range(1, n_labels) if stats[i, cv2.CC_STAT_AREA] >= comp_min]

        if len(comps) <= 1:
            # --- Stage 2a: projection-valley split ---------------------------
            sub_masks = _projection_valley_split(
                m_np_orig, min_area,
                max_ratio=proj_valley_thresh / 100.0,
            ) if use_proj_valley else None

            # --- Stage 2b: erosion-based Voronoi fallback --------------------
            if not sub_masks and use_erosion:
                sub_masks = _erosion_split(m_np_orig, min_area)
            if sub_masks:
                for sub in sub_masks:
                    sub_t = torch.from_numpy(sub).float().to("cuda")
                    x, y, bw, bh = cv2.boundingRect(sub)
                    out_masks.append(sub_t)
                    out_bboxes.append((float(x), float(y), float(x + bw), float(y + bh)))
                continue

            out_masks.append(m_cuda)
            out_bboxes.append(box)
            continue

        labels_cuda = torch.from_numpy(labels).to("cuda")
        for i in comps:
            comp_mask = (labels_cuda == i).float()
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])
            out_masks.append(comp_mask)
            out_bboxes.append((float(x), float(y), float(x + w), float(y + h)))

    return out_masks, out_bboxes


def _alpha_blend(frame_cuda, mask_cuda, alpha: float):
    """
    GPU alpha-composite a cyan overlay onto frame_cuda where mask_cuda > 0.

    Parameters
    ----------
    frame_cuda : torch.Tensor  float32 3×H×W  BGR channel order [0, 1]
    mask_cuda  : torch.Tensor  float32 H×W    values in [0, 1]
    alpha      : float         overlay opacity

    Returns
    -------
    torch.Tensor  float32 3×H×W BGR [0, 1]
    """
    import torch
    # Cyan in BGR: (B=1, G=1, R=0)
    cyan = torch.tensor([1.0, 1.0, 0.0], device=frame_cuda.device).view(3, 1, 1)

    if mask_cuda.dtype == torch.bool:
        mask_f = mask_cuda.float()
    else:
        mask_f = mask_cuda.clamp(0.0, 1.0)

    mask_3 = mask_f.unsqueeze(0)  # 1×H×W → broadcasts over channels
    out = frame_cuda * (1.0 - alpha * mask_3) + cyan * (alpha * mask_3)
    return out.clamp_(0.0, 1.0)


def _draw_quad_overlays(rgb_np: "np.ndarray", mask_np: "np.ndarray") -> None:
    """
    Fit a 4-corner polygon to each distinct blob in mask_np and draw it
    in-place on rgb_np (H×W×3 uint8 RGB).

    For each blob:
      • semi-transparent cyan fill
      • solid cyan outline
      • cyan corner circles
    """
    import cv2
    import numpy as np

    MIN_AREA = 200   # ignore tiny noise blobs

    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < MIN_AREA:
            continue

        # Fit quad: relax approxPolyDP until ≤4 corners.
        hull = cv2.convexHull(contour)
        arc  = cv2.arcLength(hull, True)
        approx = hull
        for eps in (0.02, 0.04, 0.06, 0.10, 0.15, 0.20, 0.30):
            cand = cv2.approxPolyDP(hull, eps * arc, True)
            approx = cand
            if len(cand) <= 4:
                break
        if len(approx) == 4:
            quad = approx.reshape(4, 2).astype(np.int32)
        else:
            rect = cv2.minAreaRect(contour)
            quad = np.int32(np.round(cv2.boxPoints(rect)))

        # Semi-transparent fill (blend 22% cyan over current pixels).
        overlay = rgb_np.copy()
        cv2.fillPoly(overlay, [quad], (0, 230, 230))   # RGB cyan
        cv2.addWeighted(overlay, 0.22, rgb_np, 0.78, 0, rgb_np)

        # Solid outline.
        cv2.polylines(rgb_np, [quad], isClosed=True, color=(0, 230, 230), thickness=2)

        # Corner circles.
        for pt in quad:
            cv2.circle(rgb_np, tuple(pt), 5, (0, 230, 230), -1)
            cv2.circle(rgb_np, tuple(pt), 5, (255, 255, 255), 1)


# ---------------------------------------------------------------------------
# DecodeThread
# ---------------------------------------------------------------------------

class DecodeThread(threading.Thread):
    """
    Stage A — Decode.

    Consumes (frame_idx, frame_bgr: np.ndarray uint8 H×W×3) from
    raw_frame_queue and produces (frame_idx, frame_bgr, frame_cuda) tuples
    on decode_to_infer_q.

    frame_cuda is a contiguous float32 3×H×W CUDA tensor in BGR channel order,
    values normalised to [0, 1].  Both the numpy and CUDA representations are
    forwarded so InferThread can use numpy for cv2 tracker operations without
    an extra round-trip.
    """

    def __init__(
        self,
        raw_frame_queue: mp.Queue,
        decode_to_infer_q: "queue.Queue",
        stop_event: threading.Event,
    ) -> None:
        super().__init__(daemon=True, name="DecodeThread")
        self._raw_q  = raw_frame_queue
        self._out_q  = decode_to_infer_q
        self._stop   = stop_event

    def run(self) -> None:
        import torch

        while not self._stop.is_set():
            try:
                item = self._raw_q.get(timeout=0.1)
            except Exception:
                continue

            if item is None:
                # Sentinel from VideoReaderThread — propagate to next stage.
                self._out_q.put(None)
                return

            idx, frame_bgr_np = item

            try:
                # non_blocking=True lets the H2D transfer overlap with CPU work.
                frame_cuda = (
                    torch.from_numpy(frame_bgr_np)
                    .to("cuda", non_blocking=True)
                    .float()
                    .div_(255.0)
                    .permute(2, 0, 1)
                    .contiguous()
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                continue  # drop this frame and keep going

            try:
                self._out_q.put((idx, frame_bgr_np, frame_cuda), timeout=0.15)
            except queue.Full:
                pass  # inference is behind; drop frame silently

        # stop_event was set — send sentinel downstream.
        try:
            self._out_q.put_nowait(None)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# InferThread
# ---------------------------------------------------------------------------

class InferThread(threading.Thread):
    """
    Stage B — Tracker-gated inference.

    Per-frame logic:
      1. TrackerWrapper.update() → predicted xyxy bbox + confidence.
      2a. conf ≥ 0.4 AND frame falls on a stride boundary:
            Pad bbox 20% → ROI crop → FastSAMTracker.predict_roi().
      2b. conf < 0.4 AND stride boundary:
            Full-frame FastSAMTracker.predict_full() for re-acquisition.
      2c. Intermediate (between stride boundaries):
            Propagate the last accepted mask without re-running FastSAM/ReID.
      3. Batch embed all candidate crops via ReIDEmbedder.embed_batch().
      4. EMAGallery.best_match() → cosine similarity gate.
      5. EMAGallery.update() if similarity strictly exceeds ema_threshold.
      6. Re-initialise tracker from the matched bbox (prevents drift).

    Output tuple (pushed to infer_to_blend_q):
        (idx, frame_cuda, mask_cuda, bbox_xyxy, accepted, sim, mode, fps_gpu)
        Where mode ∈ {'roi', 'full', 'propagate', 'OOM', 'error', 'no_candidates'}
    """

    def __init__(
        self,
        decode_to_infer_q: "queue.Queue",
        infer_to_blend_q:  "queue.Queue",
        stop_event:        threading.Event,
        fast_sam,          # FastSAMTracker instance
        embedder,          # ReIDEmbedder instance
        ema_gallery,       # EMAGallery instance
        tracker,           # TrackerWrapper instance
        live_cfg:          dict,
        vid_h:             int,
        vid_w:             int,
        expected_area:     int = 0,
        min_area_frac:     float = 0.30,
    ) -> None:
        super().__init__(daemon=True, name="InferThread")
        self._in_q     = decode_to_infer_q
        self._out_q    = infer_to_blend_q
        self._stop     = stop_event
        self._fsam     = fast_sam
        self._embedder = embedder
        self._gallery  = ema_gallery
        self._tracker  = tracker
        self._cfg      = live_cfg
        self._H        = vid_h
        self._W        = vid_w

        # Minimum connected-component area kept when splitting merged masks.
        # Derived from the registered object's area (similar-size prior) with a
        # small absolute floor so noise specks are never treated as objects.
        self._min_component_area = max(64, int(expected_area * min_area_frac))

        # Mutable inter-frame state
        self._frame_count:    int            = 0
        self._last_mask_cuda               = None
        self._last_bbox_xyxy: Optional[tuple] = None
        self._last_sim:       float          = 0.0
        self._no_match_streak: int           = 0
        # All accepted bboxes (xyxy) from the last full Re-ID scan.
        # Used to widen the ROI so every matched object stays in frame.
        self._last_accepted_bboxes: list     = []

    def run(self) -> None:
        import torch

        while not self._stop.is_set():
            try:
                item = self._in_q.get(timeout=0.1)
            except queue.Empty:
                continue

            if item is None:
                self._out_q.put(None)
                return

            idx, frame_bgr_np, frame_cuda = item
            t0 = time.monotonic()

            try:
                result_7 = self._process_frame(idx, frame_bgr_np, frame_cuda)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                fallback_bbox = self._last_bbox_xyxy or (0, 0, self._W, self._H)
                result_7 = (
                    idx, frame_cuda, self._last_mask_cuda,
                    fallback_bbox, False, 0.0, "OOM",
                )
            except Exception:
                traceback.print_exc()
                fallback_bbox = self._last_bbox_xyxy or (0, 0, self._W, self._H)
                result_7 = (
                    idx, frame_cuda, self._last_mask_cuda,
                    fallback_bbox, False, 0.0, "error",
                )

            fps_gpu = 1.0 / max(time.monotonic() - t0, 1e-9)

            try:
                self._out_q.put((*result_7, fps_gpu), timeout=0.15)
            except queue.Full:
                pass

            self._frame_count += 1

        try:
            self._out_q.put_nowait(None)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Core per-frame logic
    # ------------------------------------------------------------------

    def _process_frame(
        self,
        idx:          int,
        frame_bgr_np: np.ndarray,
        frame_cuda,
    ) -> tuple:
        """
        Returns 7-tuple:
            (idx, frame_cuda, mask_cuda, bbox_xyxy, accepted, sim, mode)
        """
        import torch

        base_stride    = max(1, int(self._cfg.get("stride", 1)))
        detection_mode = bool(self._cfg.get("detection_mode", False))
        H, W           = self._H, self._W

        # ── 1. Tracker predict ────────────────────────────────────────
        if detection_mode:
            # No tracker state — bbox falls back to full frame.
            ok, bbox_xywh, conf = False, (0, 0, W, H), 0.0
            bbox_xyxy = (0, 0, W, H)
        else:
            ok, bbox_xywh, conf = self._tracker.update(frame_bgr_np)
            bbox_xyxy = _xywh_to_xyxy(bbox_xywh) if ok else self._last_bbox_xyxy or (0, 0, W, H)

        # ── 1b. Adaptive stride (P1) ──────────────────────────────────
        # Auto-throttle FastSAM during stable high-confidence tracking to save
        # GPU time.  Revert to base_stride when tracker or Re-ID confidence drops.
        if bool(self._cfg.get("adaptive_stride", False)) and not detection_mode and ok:
            if conf > 0.7 and self._last_sim > 0.80:
                effective_stride = max(base_stride, 2)
            elif conf < 0.5 or self._last_sim < 0.70:
                effective_stride = 1
            else:
                effective_stride = base_stride
        else:
            effective_stride = base_stride

        run_full = detection_mode or (self._frame_count % effective_stride == 0)

        # ── 1c. Temporal consistency gate (A5) ───────────────────────
        # If the tracker bbox center jumps more than 50% of the object width in
        # one frame, the tracker likely snapped to the wrong object.  Force a
        # full-frame re-acquisition pass so Re-ID can correct it.
        if ok and self._last_bbox_xyxy is not None and not detection_mode:
            lx1, ly1, lx2, ly2 = self._last_bbox_xyxy
            cx1, cy1, cx2, cy2 = bbox_xyxy
            obj_w = max(lx2 - lx1, 1)
            dc = ((cx1 + cx2) / 2 - (lx1 + lx2) / 2) ** 2 + \
                 ((cy1 + cy2) / 2 - (ly1 + ly2) / 2) ** 2
            if dc > (0.5 * obj_w) ** 2:
                run_full = True
                conf     = 0.0   # treat as low-confidence → full-frame mode

        # ── 2. FastSAM (ROI or full) ──────────────────────────────────
        if not run_full:
            return self._propagate(idx, frame_cuda, bbox_xyxy, ok)

        fs_conf = float(self._cfg.get("fastsam_conf", 0.35))
        fs_iou  = float(self._cfg.get("fastsam_iou",  0.90))
        if detection_mode or conf < 0.4:
            masks_cuda, bboxes_xyxy = self._fsam.predict_full(frame_bgr_np, conf=fs_conf, iou=fs_iou)
            mode = "detect" if detection_mode else "full"
        else:
            # Widen ROI to cover every object matched last frame, not just
            # the single tracker-predicted object.
            tracker_xyxy = _xywh_to_xyxy(bbox_xywh)
            if self._last_accepted_bboxes:
                roi_xyxy = _union_bboxes_padded(
                    self._last_accepted_bboxes + [tracker_xyxy], 0.20, H, W
                )
            else:
                roi_xyxy = _pad_bbox_xyxy(bbox_xywh, 0.20, H, W)
            masks_cuda, bboxes_xyxy = self._fsam.predict_roi(frame_bgr_np, roi_xyxy, conf=fs_conf, iou=fs_iou)
            mode = "roi"

        # ── 2b. Split merged masks at black separator lines ───────────
        any_split_on = (
            self._cfg.get("separator_split",   True)
            or self._cfg.get("proj_valley_split", True)
            or self._cfg.get("erosion_split",     True)
        )
        if any_split_on and masks_cuda:
            # P3: skip expensive erosion split when tracker is already confident
            erosion_ok = bool(self._cfg.get("erosion_split", True))
            if erosion_ok and conf > 0.6 and self._last_sim > 0.85:
                erosion_ok = False
            masks_cuda, bboxes_xyxy = split_masks_by_separators(
                frame_bgr_np, masks_cuda, bboxes_xyxy,
                sep_thresh        = int(self._cfg.get("separator_thresh",   40)),
                min_area          = self._min_component_area,
                use_col_cut       = bool(self._cfg.get("separator_split",   True)),
                use_proj_valley   = bool(self._cfg.get("proj_valley_split", True)),
                proj_valley_thresh= int(self._cfg.get("proj_valley_thresh", 35)),
                use_erosion       = erosion_ok,
            )

        # ── 2c. Reject background-scale masks ────────────────────────
        area_ceil_pct   = float(self._cfg.get("area_ceiling_pct", 55)) / 100.0
        max_mask_pixels = int(H * W * area_ceil_pct)
        if masks_cuda:
            filtered = [
                (m, b) for m, b in zip(masks_cuda, bboxes_xyxy)
                if int((m > 0.5).sum().item()) <= max_mask_pixels
            ]
            if filtered:
                masks_cuda, bboxes_xyxy = zip(*filtered)
                masks_cuda  = list(masks_cuda)
                bboxes_xyxy = list(bboxes_xyxy)
            else:
                masks_cuda, bboxes_xyxy = [], []

        # ── 3. Batch Re-ID embed ──────────────────────────────────────
        if not bboxes_xyxy:
            self._no_match_streak += 1
            return (idx, frame_cuda, self._last_mask_cuda, bbox_xyxy, False, 0.0, "no_candidates")

        # Pass the corresponding mask so non-object pixels in each crop are
        # filled with the object's mean colour instead of the background colour.
        crops = [
            _crop_bgr(frame_bgr_np, b, m)
            for b, m in zip(bboxes_xyxy, masks_cuda)
        ]
        crops = [c for c in crops if c is not None and c.size > 0]
        if not crops:
            self._no_match_streak += 1
            return (idx, frame_cuda, self._last_mask_cuda, bbox_xyxy, False, 0.0, "empty_crops")

        emb_batch = self._embedder.embed_batch(crops)  # np.float32 [N, D]

        # ── 4. Gallery cosine match (all candidates above threshold) ──
        accepted_idxs, accepted_sims, best_idx, sim = \
            self._gallery.all_matches_above_threshold(emb_batch)
        accepted = len(accepted_idxs) > 0

        # ── 5. Accept / reject ───────────────────────────────────────
        if accepted and masks_cuda is not None and best_idx < len(masks_cuda):
            # Combine every accepted mask into one overlay so all matching
            # objects are highlighted simultaneously.
            valid_idxs = [i for i in accepted_idxs if i < len(masks_cuda)]
            if len(valid_idxs) > 1:
                best_mask = torch.stack(
                    [masks_cuda[i] for i in valid_idxs]
                ).max(dim=0).values   # logical OR across accepted masks
            else:
                best_mask = masks_cuda[valid_idxs[0]]

            best_bbox  = bboxes_xyxy[best_idx]
            self._gallery.update(emb_batch[best_idx], sim)
            # Re-seed tracker on the single highest-scoring object only,
            # so the ROI crop stays focused on the primary target.
            x1b, y1b, x2b, y2b = (int(v) for v in best_bbox)
            self._tracker.init(frame_bgr_np, (x1b, y1b, x2b - x1b, y2b - y1b))
            self._last_mask_cuda      = best_mask
            self._last_bbox_xyxy      = best_bbox
            self._last_sim            = sim
            self._no_match_streak     = 0
            # Remember all accepted bboxes so the next ROI covers every object.
            self._last_accepted_bboxes = [bboxes_xyxy[i] for i in valid_idxs]
        else:
            self._no_match_streak += 1
            best_mask = self._last_mask_cuda
            best_bbox = bbox_xyxy
            accepted  = False
            if self._no_match_streak > 5:
                self._last_accepted_bboxes = []

        # Emit re-acquiring status if target has been lost for a while
        if self._no_match_streak >= 30:
            mode = f"re-acquiring ({self._no_match_streak} frames)"

        return (idx, frame_cuda, best_mask, best_bbox, accepted, sim, mode)

    def _propagate(
        self,
        idx:       int,
        frame_cuda,
        bbox_xyxy: tuple,
        tracker_ok: bool,
    ) -> tuple:
        """
        Intermediate frame (between stride boundaries).
        Propagate the last accepted mask; no FastSAM / Re-ID call.
        """
        mask = self._last_mask_cuda
        if mask is None:
            mask = _bbox_to_mask_cuda(bbox_xyxy, self._H, self._W)
        return (idx, frame_cuda, mask, bbox_xyxy, tracker_ok, self._last_sim, "propagate")


# ---------------------------------------------------------------------------
# BlendThread
# ---------------------------------------------------------------------------

class BlendThread(threading.Thread):
    """
    Stage C — Compositing and output.

    For each frame:
      • Runs GPU alpha-blend (green overlay, 50% opacity default).
      • Swaps BGR → RGB for Qt display.
      • Pulls the composited tensor to CPU, converts to bytes.
      • Puts (raw_bytes, W, H, frame_idx, meta_dict) into display_queue.
      • If batch_render is active, also writes the BGR frame to VideoWriter.

    Uses a dedicated torch.cuda.Stream to overlap composite with CPU work of
    the previous frame (double-buffering effect).  stream.synchronize() is
    called before the CPU→host transfer to guarantee data integrity.
    """

    def __init__(
        self,
        infer_to_blend_q: "queue.Queue",
        display_queue:    mp.Queue,
        stop_event:       threading.Event,
        live_cfg:         dict,
        vid_h:            int,
        vid_w:            int,
        video_writer      = None,
        total_frames:     int = 0,
        batch_render:     bool = False,
        native_h:         int = 0,
        native_w:         int = 0,
        video_path:       str = "",
    ) -> None:
        super().__init__(daemon=True, name="BlendThread")
        self._in_q        = infer_to_blend_q
        self._disp_q      = display_queue
        self._stop        = stop_event
        self._cfg         = live_cfg
        self._H           = vid_h
        self._W           = vid_w
        self._writer      = video_writer
        self._total       = total_frames
        self._batch       = batch_render
        self._frame_count = 0
        self._stream      = None   # created in run() after CUDA context exists
        # Native-resolution export: read raw frames here and composite at full res.
        self._native_h    = native_h if native_h > 0 else vid_h
        self._native_w    = native_w if native_w > 0 else vid_w
        self._video_path  = video_path
        self._native_cap  = None   # cv2.VideoCapture, opened in run() if exporting

    def run(self) -> None:
        import torch
        self._stream = torch.cuda.Stream()

        # For native-resolution export: open a separate capture that reads at
        # full resolution while inference runs on the downscaled stream.
        needs_native = (
            self._batch
            and self._video_path
            and (self._native_h != self._H or self._native_w != self._W)
        )
        if needs_native:
            self._native_cap = cv2.VideoCapture(self._video_path, cv2.CAP_MSMF)
            if not self._native_cap.isOpened():
                self._native_cap = cv2.VideoCapture(self._video_path)

        while not self._stop.is_set():
            try:
                item = self._in_q.get(timeout=0.1)
            except queue.Empty:
                continue

            if item is None:
                # Pipeline drained — notify display and exit.
                if self._native_cap is not None:
                    self._native_cap.release()
                    self._native_cap = None
                self._send_sentinel()
                return

            # Unpack 8-element tuple from InferThread
            idx, frame_cuda, mask_cuda, bbox_xyxy, accepted, sim, mode, fps_gpu = item

            try:
                self._process_and_dispatch(
                    idx, frame_cuda, mask_cuda, bbox_xyxy,
                    accepted, sim, mode, fps_gpu,
                )
            except Exception:
                traceback.print_exc()

            self._frame_count += 1

        # stop_event was set externally.
        if self._native_cap is not None:
            self._native_cap.release()
            self._native_cap = None
        self._send_sentinel()

    def _send_sentinel(self) -> None:
        if self._batch:
            # Batch render complete — report 100% so the UI closes out the
            # export (independent of whether total_frames was known).
            try:
                self._disp_q.put(
                    (b"", 0, 0, -1, {"export_pct": 100}), timeout=1.0
                )
            except Exception:
                pass
        else:
            try:
                self._disp_q.put_nowait(None)
            except Exception:
                pass

    def _process_and_dispatch(
        self,
        idx:        int,
        frame_cuda,
        mask_cuda,
        bbox_xyxy:  tuple,
        accepted:   bool,
        sim:        float,
        mode:       str,
        fps_gpu:    float,
    ) -> None:
        import torch

        alpha = float(self._cfg.get("overlay_alpha", 0.5))
        H, W  = self._H, self._W

        # ── GPU composite ─────────────────────────────────────────────
        with torch.cuda.stream(self._stream):
            if accepted and mask_cuda is not None:
                out_bgr_cuda = _alpha_blend(frame_cuda, mask_cuda, alpha)
            else:
                out_bgr_cuda = frame_cuda

            # BGR → RGB for Qt display
            out_rgb_cuda = out_bgr_cuda[[2, 1, 0], :, :]

        self._stream.synchronize()

        # ── Pull to CPU ────────────────────────────────────────────────
        # .copy() detaches from the torch tensor's storage so OpenCV can use
        # this array as a read-write dst argument without layout complaints.
        out_rgb_np = (
            out_rgb_cuda
            .mul(255.0)
            .byte()
            .permute(1, 2, 0)   # 3×H×W → H×W×3
            .contiguous()
            .cpu()
            .numpy()
            .copy()
        )   # H×W×3 uint8 RGB, fully owned writable C-contiguous array

        # ── Polygon overlay on each segmented blob ─────────────────────
        if accepted and mask_cuda is not None:
            mask_np = (
                (mask_cuda > 0.5).byte().contiguous().cpu().numpy().copy()
                .astype(np.uint8) * 255
            )
            _draw_quad_overlays(out_rgb_np, mask_np)

        raw_bytes = out_rgb_np.tobytes()

        # ── Build metadata ────────────────────────────────────────────
        pct = None
        if self._total > 0 and self._frame_count > 0:
            pct = min(99, int(self._frame_count / self._total * 100))

        meta: dict = {
            "bbox":      bbox_xyxy,
            "sim_score": sim,
            "accepted":  accepted,
            "mode":      mode,
            "fps_gpu":   fps_gpu,
        }
        if pct is not None:
            meta["export_pct"] = pct

        # ── Push to display_queue (drop frame on Full — never stall) ──
        try:
            self._disp_q.put((raw_bytes, W, H, idx, meta), timeout=0.1)
        except queue.Full:
            pass

        # ── Batch render: write frame to VideoWriter ─────────────────
        if self._writer is not None:
            if self._native_cap is not None:
                # Seek to the exact frame index so native and inference streams
                # stay in sync even when frames are dropped or export starts
                # mid-video.
                self._native_cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret_n, native_bgr = self._native_cap.read()
                if ret_n and native_bgr is not None:
                    NH, NW = self._native_h, self._native_w
                    if accepted and mask_cuda is not None:
                        import torch.nn.functional as F
                        # Upscale binary mask to native resolution (nearest so
                        # hard edges stay sharp, no interpolation artefacts).
                        mask_up = F.interpolate(
                            (mask_cuda > 0.5).float().unsqueeze(0).unsqueeze(0),
                            size=(NH, NW),
                            mode="nearest",
                        ).squeeze()  # NH×NW float32 CUDA
                        # Build native CUDA frame for blending.
                        native_rgb = cv2.cvtColor(native_bgr, cv2.COLOR_BGR2RGB)
                        native_t = (
                            torch.from_numpy(native_rgb)
                            .to(mask_cuda.device)
                            .float()
                            .div(255.0)
                            .permute(2, 0, 1)   # H×W×3 → 3×H×W
                        )
                        native_blended = _alpha_blend(native_t, mask_up, alpha)
                        export_bgr_np = (
                            native_blended[[2, 1, 0], :, :]   # RGB → BGR
                            .mul(255.0)
                            .byte()
                            .permute(1, 2, 0)
                            .contiguous()
                            .cpu()
                            .numpy()
                            .copy()
                        )
                        # Draw polygon overlays on native frame too.
                        mask_up_np = (mask_up > 0.5).byte().cpu().numpy().astype(np.uint8) * 255
                        _draw_quad_overlays(export_bgr_np, mask_up_np)
                        # _draw_quad_overlays works on RGB; we have BGR — swap channels.
                        export_bgr_np = cv2.cvtColor(export_bgr_np, cv2.COLOR_RGB2BGR)
                    else:
                        export_bgr_np = native_bgr
                    self._writer.write_frame(export_bgr_np)
            else:
                # Inference res == native res: write the already-composited frame.
                out_bgr_np = (
                    out_bgr_cuda
                    .mul(255.0)
                    .byte()
                    .permute(1, 2, 0)
                    .contiguous()
                    .cpu()
                    .numpy()
                    .copy()
                )
                self._writer.write_frame(out_bgr_np)


# ---------------------------------------------------------------------------
# GPUPipelineProcess
# ---------------------------------------------------------------------------

class GPUPipelineProcess(mp.Process):
    """
    Spawned child process that owns the entire GPU inference pipeline.

    IMPORTANT: Do NOT import torch or call any CUDA API in __init__.
    All GPU-related imports and initialisation happen inside run() so that
    this process owns a clean, isolated CUDA context.

    Parameters
    ----------
    queues : dict
        Cross-process mp.Queue instances created in main.build_queues().
        Keys: raw_frame_queue, reg_result_queue (unused here — result
        passed directly), pipeline_cmd_queue, display_queue.
    reg_result : dict
        Serialised registration output: {mask, bbox, reid_emb, score,
        frame_bgr, frame_idx}.  All values are CPU-resident numpy arrays
        or Python primitives — no CUDA tensors.
    config : dict
        Runtime configuration; see _CONFIG_DEFAULTS below for valid keys.
    """

    _CONFIG_DEFAULTS: dict = {
        "match_threshold": 0.85,
        "ema_threshold":   0.92,
        "ema_alpha":       0.90,
        "overlay_alpha":   0.50,
        "stride":          1,
        "fastsam_weights": "FastSAM-x.pt",
        "reid_weights":    None,
        "batch_render":    False,
        "output_path":     "",
        "video_path":      "",
        "vid_w":           1280,
        "vid_h":           720,
        "video_fps":       30.0,
        "total_frames":    0,
        "separator_split":         True,
        "separator_thresh":        40,
        "proj_valley_split":       True,
        "proj_valley_thresh":      35,
        "erosion_split":           True,
        "min_component_area_frac": 0.30,
        "detection_mode":          False,
        "native_w":                0,
        "native_h":                0,
        "fastsam_conf":            0.35,
        "fastsam_iou":             0.90,
        "area_ceiling_pct":        55,
        "adaptive_stride":         False,
    }

    def __init__(
        self,
        queues:     dict,
        reg_result: dict,
        config:     dict,
    ) -> None:
        super().__init__(daemon=True)
        self._queues     = queues
        self._reg_result = reg_result
        self._pipeline_cfg = {**self._CONFIG_DEFAULTS, **config}

    # ------------------------------------------------------------------
    # Process entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Spawned process entry point.
        All imports and CUDA initialisations happen here.
        """
        import os, torch

        # Mark this as the GPU pipeline process so _compile_safe() in models.py
        # can reliably skip torch.compile (the daemon-flag check is unreliable on
        # Windows/spawn because _config['daemon'] is not always propagated).
        os.environ['_OPTSHOT_GPU_PROC'] = '1'

        torch.cuda.set_device(0)

        try:
            self._run_pipeline()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            traceback.print_exc()
            self._signal_error()
        except Exception:
            traceback.print_exc()
            self._signal_error()

    def _signal_error(self) -> None:
        """Push a None sentinel so FrameDisplayWorker unblocks cleanly."""
        try:
            self._queues["display_queue"].put_nowait(None)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Pipeline orchestration
    # ------------------------------------------------------------------

    def _run_pipeline(self) -> None:
        import torch
        from models import FastSAMTracker, ReIDEmbedder, EMAGallery, TrackerWrapper

        cfg = self._pipeline_cfg
        reg = self._reg_result

        # ── Phase-2 model loading ─────────────────────────────────────
        fast_sam = FastSAMTracker(
            weights=cfg["fastsam_weights"],
            device="cuda",
        )
        fast_sam.load(progress_cb=lambda msg: logger.info("[FastSAM] %s", msg))
        fast_sam.warmup(cfg["vid_h"], cfg["vid_w"])

        embedder = ReIDEmbedder(
            weights_path=cfg["reid_weights"],
            device="cuda",
        )
        embedder.load()
        embedder.warmup()

        ema_gallery = EMAGallery(
            initial_emb   = reg["reid_emb"],
            ema_alpha     = cfg["ema_alpha"],
            ema_threshold = cfg["ema_threshold"],
            match_threshold = cfg["match_threshold"],
        )

        tracker = TrackerWrapper()
        # Seed tracker with SAM2 bbox.  SAM2 returns xywh; TrackerWrapper
        # also accepts xywh in init().
        init_bbox_xywh = tuple(int(v) for v in reg["bbox"])
        if reg.get("frame_bgr") is not None:
            tracker.init(reg["frame_bgr"], init_bbox_xywh)

        # Registered object area drives the min-component filter when splitting
        # merged masks (similar-size prior). Fall back to the bbox area if the
        # mask is unavailable.
        reg_mask = reg.get("mask")
        if reg_mask is not None:
            registered_area = int((np.asarray(reg_mask) > 0).sum())
        else:
            bx, by, bw, bh = init_bbox_xywh
            registered_area = int(bw * bh)

        # ── VideoWriter (batch render only) ───────────────────────────
        video_writer = None
        total_frames = 0
        if cfg["batch_render"] and cfg["output_path"]:
            from video_io import VideoWriter
            export_w = cfg.get("native_w") or cfg["vid_w"]
            export_h = cfg.get("native_h") or cfg["vid_h"]
            video_writer = VideoWriter(
                output_path = cfg["output_path"],
                width       = export_w,
                height      = export_h,
                fps         = cfg["video_fps"],
            )
            video_writer.open()

            # Frame count for progress reporting. Prefer the value the main
            # process already measured (reliable); fall back to a cv2 probe
            # only if it was not supplied.
            total_frames = int(cfg.get("total_frames", 0) or 0)
            if total_frames <= 0:
                probe = cv2.VideoCapture(cfg["video_path"])
                total_frames = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
                probe.release()

        # ── Internal queues ───────────────────────────────────────────
        decode_to_infer: queue.Queue = queue.Queue(maxsize=6)
        infer_to_blend:  queue.Queue = queue.Queue(maxsize=6)
        stop_event = threading.Event()

        # Shared live config — mutated by the command loop below.
        live_cfg: dict = {
            "match_threshold":    cfg["match_threshold"],
            "overlay_alpha":      cfg["overlay_alpha"],
            "stride":             cfg["stride"],
            "separator_split":    cfg["separator_split"],
            "separator_thresh":   cfg["separator_thresh"],
            "proj_valley_split":  cfg.get("proj_valley_split",  True),
            "proj_valley_thresh": cfg.get("proj_valley_thresh", 35),
            "erosion_split":      cfg.get("erosion_split",      True),
            "detection_mode":     cfg.get("detection_mode",     False),
            "fastsam_conf":       cfg.get("fastsam_conf",       0.35),
            "fastsam_iou":        cfg.get("fastsam_iou",        0.90),
            "area_ceiling_pct":   cfg.get("area_ceiling_pct",   55),
            "adaptive_stride":    cfg.get("adaptive_stride",    False),
        }

        # ── Spawn stage threads ───────────────────────────────────────
        dec_t = DecodeThread(
            raw_frame_queue   = self._queues["raw_frame_queue"],
            decode_to_infer_q = decode_to_infer,
            stop_event        = stop_event,
        )
        inf_t = InferThread(
            decode_to_infer_q = decode_to_infer,
            infer_to_blend_q  = infer_to_blend,
            stop_event        = stop_event,
            fast_sam          = fast_sam,
            embedder          = embedder,
            ema_gallery       = ema_gallery,
            tracker           = tracker,
            live_cfg          = live_cfg,
            vid_h             = cfg["vid_h"],
            vid_w             = cfg["vid_w"],
            expected_area     = registered_area,
            min_area_frac     = float(cfg["min_component_area_frac"]),
        )
        bld_t = BlendThread(
            infer_to_blend_q  = infer_to_blend,
            display_queue     = self._queues["display_queue"],
            stop_event        = stop_event,
            live_cfg          = live_cfg,
            vid_h             = cfg["vid_h"],
            vid_w             = cfg["vid_w"],
            video_writer      = video_writer,
            total_frames      = total_frames,
            batch_render      = bool(cfg["batch_render"] and cfg["output_path"]),
            native_h          = cfg.get("native_h", 0),
            native_w          = cfg.get("native_w", 0),
            video_path        = cfg.get("video_path", ""),
        )

        dec_t.start()
        inf_t.start()
        bld_t.start()

        # ── Command loop (main thread of GPU process) ─────────────────
        cmd_q = self._queues["pipeline_cmd_queue"]
        while True:
            # Break if all stage threads have exited (end of video / error)
            if (not dec_t.is_alive()
                    and not inf_t.is_alive()
                    and not bld_t.is_alive()):
                break

            try:
                cmd = cmd_q.get(timeout=0.1)
            except Exception:
                continue

            if cmd.type == "STOP":
                stop_event.set()
                break
            elif cmd.type == "UPDATE_THRESHOLD":
                live_cfg["match_threshold"] = float(cmd.payload)
                ema_gallery.set_match_threshold(float(cmd.payload))
            elif cmd.type == "UPDATE_ALPHA":
                live_cfg["overlay_alpha"] = float(cmd.payload)
            elif cmd.type == "UPDATE_STRIDE":
                live_cfg["stride"] = max(1, int(cmd.payload))
            elif cmd.type == "UPDATE_SEP_SPLIT":
                live_cfg["separator_split"] = bool(cmd.payload)
            elif cmd.type == "UPDATE_SEP_THRESH":
                live_cfg["separator_thresh"] = max(0, int(cmd.payload))
            elif cmd.type == "UPDATE_PROJ_VALLEY":
                live_cfg["proj_valley_split"] = bool(cmd.payload)
            elif cmd.type == "UPDATE_PROJ_VALLEY_THRESH":
                live_cfg["proj_valley_thresh"] = max(5, min(80, int(cmd.payload)))
            elif cmd.type == "UPDATE_EROSION_SPLIT":
                live_cfg["erosion_split"] = bool(cmd.payload)
            elif cmd.type == "UPDATE_DETECTION_MODE":
                live_cfg["detection_mode"] = bool(cmd.payload)
            elif cmd.type == "UPDATE_FASTSAM_CONF":
                live_cfg["fastsam_conf"] = float(cmd.payload)
            elif cmd.type == "UPDATE_FASTSAM_IOU":
                live_cfg["fastsam_iou"] = float(cmd.payload)
            elif cmd.type == "UPDATE_AREA_CEILING":
                live_cfg["area_ceiling_pct"] = float(cmd.payload)
            elif cmd.type == "UPDATE_ADAPTIVE_STRIDE":
                live_cfg["adaptive_stride"] = bool(cmd.payload)
            elif cmd.type == "UPDATE_EMA_THRESHOLD":
                live_cfg["ema_threshold"] = float(cmd.payload)
                ema_gallery.set_ema_threshold(float(cmd.payload))
            elif cmd.type == "UPDATE_EMA_ALPHA":
                live_cfg["ema_alpha"] = float(cmd.payload)
                ema_gallery.set_ema_alpha(float(cmd.payload))

        # ── Graceful shutdown ─────────────────────────────────────────
        stop_event.set()
        dec_t.join(timeout=4.0)
        inf_t.join(timeout=4.0)
        bld_t.join(timeout=4.0)

        if video_writer is not None:
            video_writer.close()

        torch.cuda.empty_cache()
