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


def _crop_bgr(frame_bgr: np.ndarray, bbox_xyxy: tuple) -> Optional[np.ndarray]:
    """
    Crop frame_bgr to the given xyxy bounding box, clamped to frame
    dimensions.  Returns None if the resulting region has zero area.
    """
    x1, y1, x2, y2 = (int(v) for v in bbox_xyxy)
    fh, fw = frame_bgr.shape[:2]
    x1 = max(0, x1);  y1 = max(0, y1)
    x2 = min(fw, x2); y2 = min(fh, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame_bgr[y1:y2, x1:x2]


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


def _alpha_blend(frame_cuda, mask_cuda, alpha: float):
    """
    GPU alpha-composite a green overlay onto frame_cuda where mask_cuda > 0.

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
    # Green in BGR: (B=0, G=1, R=0)
    green = torch.tensor([0.0, 1.0, 0.0], device=frame_cuda.device).view(3, 1, 1)

    if mask_cuda.dtype == torch.bool:
        mask_f = mask_cuda.float()
    else:
        mask_f = mask_cuda.clamp(0.0, 1.0)

    mask_3 = mask_f.unsqueeze(0)  # 1×H×W → broadcasts over channels
    out = frame_cuda * (1.0 - alpha * mask_3) + green * (alpha * mask_3)
    return out.clamp_(0.0, 1.0)


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

        # Mutable inter-frame state
        self._frame_count:    int            = 0
        self._last_mask_cuda               = None
        self._last_bbox_xyxy: Optional[tuple] = None
        self._last_sim:       float          = 0.0
        self._no_match_streak: int           = 0

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

        stride   = max(1, int(self._cfg.get("stride", 1)))
        H, W     = self._H, self._W
        run_full = (self._frame_count % stride == 0)

        # ── 1. Tracker predict ────────────────────────────────────────
        ok, bbox_xywh, conf = self._tracker.update(frame_bgr_np)
        bbox_xyxy = _xywh_to_xyxy(bbox_xywh) if ok else self._last_bbox_xyxy or (0, 0, W, H)

        # ── 2. FastSAM (ROI or full) ──────────────────────────────────
        if not run_full:
            return self._propagate(idx, frame_cuda, bbox_xyxy, ok)

        if conf >= 0.4:
            roi_xyxy = _pad_bbox_xyxy(bbox_xywh, 0.20, H, W)
            masks_cuda, bboxes_xyxy = self._fsam.predict_roi(frame_bgr_np, roi_xyxy)
            mode = "roi"
        else:
            masks_cuda, bboxes_xyxy = self._fsam.predict_full(frame_bgr_np)
            mode = "full"

        # ── 3. Batch Re-ID embed ──────────────────────────────────────
        if not bboxes_xyxy:
            self._no_match_streak += 1
            return (idx, frame_cuda, self._last_mask_cuda, bbox_xyxy, False, 0.0, "no_candidates")

        crops = [_crop_bgr(frame_bgr_np, b) for b in bboxes_xyxy]
        crops = [c for c in crops if c is not None and c.size > 0]
        if not crops:
            self._no_match_streak += 1
            return (idx, frame_cuda, self._last_mask_cuda, bbox_xyxy, False, 0.0, "empty_crops")

        emb_batch = self._embedder.embed_batch(crops)  # np.float32 [N, D]

        # ── 4. Gallery cosine match ───────────────────────────────────
        best_idx, sim = self._gallery.best_match(emb_batch)
        sim = float(sim)
        threshold = float(self._cfg.get("match_threshold", 0.85))
        accepted = sim >= threshold

        # ── 5. Accept / reject ───────────────────────────────────────
        if accepted and masks_cuda is not None and best_idx < len(masks_cuda):
            best_mask  = masks_cuda[best_idx]   # float32 H×W CUDA
            best_bbox  = bboxes_xyxy[best_idx]
            self._gallery.update(emb_batch[best_idx], sim)
            # Convert xyxy → xywh for tracker re-init
            x1b, y1b, x2b, y2b = (int(v) for v in best_bbox)
            self._tracker.init(frame_bgr_np, (x1b, y1b, x2b - x1b, y2b - y1b))
            self._last_mask_cuda  = best_mask
            self._last_bbox_xyxy  = best_bbox
            self._last_sim        = sim
            self._no_match_streak = 0
        else:
            self._no_match_streak += 1
            best_mask = self._last_mask_cuda
            best_bbox = bbox_xyxy
            accepted  = False

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
        self._frame_count = 0
        self._stream      = None   # created in run() after CUDA context exists

    def run(self) -> None:
        import torch
        self._stream = torch.cuda.Stream()

        while not self._stop.is_set():
            try:
                item = self._in_q.get(timeout=0.1)
            except queue.Empty:
                continue

            if item is None:
                # Pipeline drained — notify display and exit.
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
        self._send_sentinel()

    def _send_sentinel(self) -> None:
        if self._total > 0:
            # Batch render complete — report 100% and exit.
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
        out_rgb_np = (
            out_rgb_cuda
            .mul(255.0)
            .byte()
            .permute(1, 2, 0)   # 3×H×W → H×W×3
            .cpu()
            .numpy()
        )   # H×W×3 uint8 RGB

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

        # ── Batch render: write BGR frame to VideoWriter ──────────────
        if self._writer is not None:
            out_bgr_np = (
                out_bgr_cuda
                .mul(255.0)
                .byte()
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )   # H×W×3 uint8 BGR
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
        "fastsam_weights": "FastSAM-s.pt",
        "reid_weights":    None,
        "batch_render":    False,
        "output_path":     "",
        "video_path":      "",
        "vid_w":           1280,
        "vid_h":           720,
        "video_fps":       30.0,
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
        self._config     = {**self._CONFIG_DEFAULTS, **config}

    # ------------------------------------------------------------------
    # Process entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Spawned process entry point.
        All imports and CUDA initialisations happen here.
        """
        import torch

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

        cfg = self._config
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

        # ── VideoWriter (batch render only) ───────────────────────────
        video_writer = None
        total_frames = 0
        if cfg["batch_render"] and cfg["output_path"]:
            from video_io import VideoWriter
            video_writer = VideoWriter(
                output_path = cfg["output_path"],
                width       = cfg["vid_w"],
                height      = cfg["vid_h"],
                fps         = cfg["video_fps"],
            )
            video_writer.open()

            # Count frames for progress reporting
            probe = cv2.VideoCapture(cfg["video_path"])
            total_frames = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
            probe.release()

        # ── Internal queues ───────────────────────────────────────────
        decode_to_infer: queue.Queue = queue.Queue(maxsize=3)
        infer_to_blend:  queue.Queue = queue.Queue(maxsize=3)
        stop_event = threading.Event()

        # Shared live config — mutated by the command loop below.
        live_cfg: dict = {
            "match_threshold": cfg["match_threshold"],
            "overlay_alpha":   cfg["overlay_alpha"],
            "stride":          cfg["stride"],
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

        # ── Graceful shutdown ─────────────────────────────────────────
        stop_event.set()
        dec_t.join(timeout=4.0)
        inf_t.join(timeout=4.0)
        bld_t.join(timeout=4.0)

        if video_writer is not None:
            video_writer.close()

        torch.cuda.empty_cache()
