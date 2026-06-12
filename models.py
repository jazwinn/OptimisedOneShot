"""
OptimisedOneShot — Models Module
==================================
All model wrappers, gallery logic, and the tracker:

  HeavySAMRegistrar  — SAM2 Large image predictor; loads/unloads on demand.
  FastSAMTracker     — Ultralytics FastSAM with torch.compile; ROI and full-
                       frame segmentation modes.
  ReIDEmbedder       — OSNet-x0.25 (torchreid) appearance embedder with
                       torchvision ResNet18 fallback; batched inference.
  EMAGallery         — Thread-safe reference embedding with EMA update gate.
  TrackerWrapper     — cv2.TrackerMIL (CSRT removed in OpenCV 4.13); returns
                       xyxy bbox + heuristic confidence.

VRAM budget on RTX 3070 (8.59 GB total)
  SAM2 Large (fp16 autocast) : ~3.5 GB peak during set_image + predict
  FastSAM-s                  : ~0.4 GB
  OSNet-x0.25                : ~0.3 GB
  ResNet18 fallback          : ~0.2 GB
  → SAM2 always unloads before Phase-2 models load.

torch.compile notes
  Applied to FastSAM's inner backbone and the Re-ID model with
  fullgraph=False so dynamic control-flow graph breaks are allowed.
  Falls back to eager silently on compilation failure.
"""

from __future__ import annotations

import logging
import os
import threading
import traceback
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# VRAM threshold (GB free) below which the heavy model is evicted.
VRAM_SWAP_THRESHOLD_GB: float = 8.0


def _triton_available() -> bool:
    """Return True only if the Triton compiler is installed (required by torch.compile on CUDA)."""
    try:
        import triton  # noqa: F401
        return True
    except ImportError:
        return False


def _compile_safe() -> bool:
    """
    Return True only when torch.compile / inductor can actually be used.

    Two conditions must hold:
    1. Triton is installed (torch.compile requires it on CUDA).
    2. The calling process is NOT the GPU pipeline child process.

    Condition 2 is needed because torch.compile's inductor backend uses tqdm,
    which tries to create an mp.RLock() the first time it renders a progress
    bar.  On Windows, creating a named semaphore requires
    `multiprocessing.current_process()._config['semprefix']`, which is absent
    in the spawned GPUPipelineProcess.  This causes a `KeyError: 'semprefix'`
    that propagates as BackendCompilerFailed and crashes the GPU pipeline.
    GPUPipelineProcess.run() sets _OPTSHOT_GPU_PROC=1 as a reliable marker;
    the mp.daemon flag check is unreliable on Windows/spawn (MS Store Python
    3.11 does not always propagate _config['daemon'] into the child).
    """
    import os, multiprocessing as mp
    if not _triton_available():
        return False
    # Primary guard: env var set by GPUPipelineProcess.run() before model loading.
    # GPUPipelineProcess used to shadow BaseProcess._config with its own pipeline
    # config dict (which has no 'daemon' key), causing mp.current_process().daemon
    # to return False even though the process was spawned with daemon=True.
    # The env-var approach is independent of that attribute and always reliable.
    if os.environ.get('_OPTSHOT_GPU_PROC'):
        return False
    # Belt-and-suspenders: also skip if mp correctly reports daemon=True.
    if mp.current_process().daemon:
        return False
    return True

# Auto-download URLs for SAM2.1 checkpoints (Meta CDN).
SAM2_CHECKPOINT_URLS: dict[str, str] = {
    "sam2.1_hiera_large.pt":     "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
    "sam2.1_hiera_base_plus.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
    "sam2.1_hiera_small.pt":     "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
    "sam2.1_hiera_tiny.pt":      "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
}

# ImageNet normalisation constants used by both OSNet and ResNet18.
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Re-ID model input size (width × height) as expected by OSNet person-ReID.
_REID_W, _REID_H = 128, 256


# ---------------------------------------------------------------------------
# HeavySAMRegistrar
# ---------------------------------------------------------------------------

class HeavySAMRegistrar:
    """
    Wraps SAM2 Large as a one-shot image predictor.

    Lifecycle
    ---------
    1. load()     — builds SAM2 graph, moves to GPU.
    2. register() — set_image + predict with user point prompts.
    3. unload()   — deletes predictor, calls torch.cuda.empty_cache().

    Mixed precision
    ---------------
    Uses torch.autocast(device_type='cuda', dtype=torch.float16) during
    inference instead of model.half().  This keeps numerically sensitive
    operations (sigmoid, softmax) in fp32 while running the bulk of the
    vision encoder in fp16 — avoiding NaN on certain hardware.
    """

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        device: str = "cuda",
    ) -> None:
        self._checkpoint = checkpoint or "sam2.1_hiera_large.pt"
        self._device     = device
        self._predictor  = None

    # ------------------------------------------------------------------
    # Load / unload
    # ------------------------------------------------------------------

    def load(self, progress_cb=None) -> None:
        """
        Build SAM2 Large and instantiate SAM2ImagePredictor.
        Auto-downloads the checkpoint if it is missing and a known URL exists.
        Raises ImportError if the sam2 package is not installed.
        """
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        config = self._resolve_config()
        ckpt   = self._checkpoint

        if not os.path.exists(ckpt):
            self._download_checkpoint(progress_cb=progress_cb)

        logger.info("Building SAM2 from config=%s  ckpt=%s", config, ckpt)
        model = build_sam2(config, ckpt, device=self._device)
        self._predictor = SAM2ImagePredictor(model)
        logger.info("SAM2 loaded.  Free VRAM: %.2f GB", self._free_vram_gb())

    def _download_checkpoint(self, progress_cb=None) -> None:
        """Download the SAM2 checkpoint from Meta's CDN with progress reporting."""
        import urllib.request

        fname = os.path.basename(self._checkpoint)
        url = SAM2_CHECKPOINT_URLS.get(fname)
        if url is None:
            raise FileNotFoundError(
                f"SAM2 checkpoint not found: '{self._checkpoint}'\n"
                f"No auto-download URL is registered for '{fname}'. "
                "Download it manually from Meta's SAM2 releases and place it "
                "in the project directory, or set the correct path in the UI."
            )

        dest = self._checkpoint
        tmp  = dest + ".download"

        def _reporthook(block_num: int, block_size: int, total_size: int) -> None:
            if progress_cb is None or total_size <= 0:
                return
            downloaded = min(block_num * block_size, total_size)
            pct        = downloaded * 100 // total_size
            mb_done    = downloaded / 1_048_576
            mb_total   = total_size / 1_048_576
            progress_cb(
                f"Downloading {fname}… {pct}%  "
                f"({mb_done:.0f} / {mb_total:.0f} MB)"
            )

        if progress_cb:
            progress_cb(f"Downloading {fname} from Meta (~1.9 GB)…")
        logger.info("Auto-downloading SAM2 checkpoint: %s → %s", url, dest)

        try:
            urllib.request.urlretrieve(url, tmp, reporthook=_reporthook)
            os.replace(tmp, dest)
            if progress_cb:
                progress_cb(f"Download complete → {dest}")
            logger.info("SAM2 checkpoint saved to %s", dest)
        except Exception:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            raise

    def unload(self) -> None:
        """Delete the predictor and free GPU memory."""
        import torch
        if self._predictor is not None:
            del self._predictor
            self._predictor = None
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("SAM2 unloaded.  Free VRAM: %.2f GB", self._free_vram_gb())

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def register(
        self,
        frame_bgr: np.ndarray,
        points: list[tuple[float, float, int]],
        separator_thresh: int = 40,
    ) -> dict:
        """
        Run SAM2 one-shot segmentation.

        Parameters
        ----------
        frame_bgr : np.ndarray  uint8 H×W×3 BGR
        points    : list of (x, y, label) where label 1=positive, 0=negative

        Returns
        -------
        dict
            mask  : np.ndarray uint8 H×W (0 or 255)
            bbox  : tuple (x, y, w, h) bounding box of the mask in pixels
            score : float  SAM2 confidence for the selected mask
        """
        import torch

        if self._predictor is None:
            raise RuntimeError("Call load() before register().")

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        coords = np.array([[x, y] for x, y, _ in points], dtype=np.float32)
        labels = np.array([l          for _, _, l in points], dtype=np.int32)

        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                self._predictor.set_image(frame_rgb)
                masks, scores, _ = self._predictor.predict(
                    point_coords     = coords,
                    point_labels     = labels,
                    multimask_output = True,
                )

        # Retry with single mask on empty output (rare OOM-adjacent case)
        if masks is None or len(masks) == 0:
            with torch.inference_mode():
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    masks, scores, _ = self._predictor.predict(
                        point_coords     = coords,
                        point_labels     = labels,
                        multimask_output = False,
                    )

        if masks is None or len(masks) == 0:
            raise RuntimeError("SAM2 returned no masks. Check your point prompts.")

        best_idx  = int(np.argmax(scores))
        best_mask = masks[best_idx].astype(bool)   # H×W bool

        # ── Split at black separator lines ────────────────────────────
        # If the SAM2 mask spans two touching objects, cut it along the
        # near-black separators and keep only the component under the user's
        # positive click — so the reference embedding is a single object.
        best_mask = self._isolate_clicked_component(
            best_mask, frame_bgr, points, separator_thresh
        )

        # Bounding box from mask
        rows = np.where(best_mask.any(axis=1))[0]
        cols = np.where(best_mask.any(axis=0))[0]
        if len(rows) == 0 or len(cols) == 0:
            raise RuntimeError("SAM2 mask is empty after threshold.")
        bbox_xywh = (
            int(cols[0]),
            int(rows[0]),
            int(cols[-1] - cols[0]),
            int(rows[-1] - rows[0]),
        )

        return {
            "mask":  (best_mask.astype(np.uint8) * 255),
            "bbox":  bbox_xywh,
            "score": float(scores[best_idx]),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _isolate_clicked_component(
        mask_bool:        np.ndarray,
        frame_bgr:        np.ndarray,
        points:           list[tuple[float, float, int]],
        separator_thresh: int,
    ) -> np.ndarray:
        """
        Cut a SAM2 mask along near-black separator lines and return only the
        connected component containing the user's positive click.

        Returns the input unchanged when there is no positive point or the cut
        yields a single component (so non-line-separated videos are unaffected).
        """
        pos = [(x, y) for x, y, l in points if l == 1]
        if not pos:
            return mask_bool

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        sep  = (gray <= int(separator_thresh)).astype(np.uint8)
        sep  = cv2.dilate(sep, np.ones((3, 3), np.uint8), iterations=1)

        cut = mask_bool.astype(np.uint8)
        cut[sep == 1] = 0

        n_labels, labels = cv2.connectedComponents(cut, connectivity=8)
        if n_labels <= 2:
            # 0 = background, 1 = single object → nothing merged to split.
            return mask_bool

        H, W = mask_bool.shape[:2]
        # Prefer the label directly under the first positive click.
        px, py = pos[0]
        cx = min(max(0, int(round(px))), W - 1)
        cy = min(max(0, int(round(py))), H - 1)
        target_label = int(labels[cy, cx])

        if target_label == 0:
            # Click landed on a separator/cut pixel — choose the largest
            # foreground component as the best guess.
            counts = np.bincount(labels.ravel())
            counts[0] = 0
            target_label = int(np.argmax(counts))

        return labels == target_label

    @staticmethod
    def _resolve_config() -> str:
        """
        Locate the SAM2 Large config YAML relative to the installed package.
        Returns a relative path string accepted by build_sam2 / Hydra.
        """
        try:
            import sam2
            pkg = os.path.dirname(sam2.__file__)
        except ImportError:
            return "configs/sam2.1/sam2.1_hiera_l.yaml"

        candidates = [
            ("configs/sam2.1/sam2.1_hiera_l.yaml",
             os.path.join(pkg, "configs", "sam2.1", "sam2.1_hiera_l.yaml")),
            ("configs/sam2/sam2_hiera_l.yaml",
             os.path.join(pkg, "configs", "sam2", "sam2_hiera_l.yaml")),
        ]
        for rel, abs_path in candidates:
            if os.path.exists(abs_path):
                return rel
        logger.warning(
            "SAM2 config YAML not found on disk. Relying on Hydra search path."
        )
        return "configs/sam2.1/sam2.1_hiera_l.yaml"

    @staticmethod
    def _free_vram_gb() -> float:
        try:
            import torch
            free, _ = torch.cuda.mem_get_info()
            return free / 1e9
        except Exception:
            return -1.0

    @staticmethod
    def should_unload() -> bool:
        """Return True if free VRAM is below VRAM_SWAP_THRESHOLD_GB."""
        free = HeavySAMRegistrar._free_vram_gb()
        return free > 0 and free < VRAM_SWAP_THRESHOLD_GB


# ---------------------------------------------------------------------------
# FastSAMTracker
# ---------------------------------------------------------------------------

class FastSAMTracker:
    """
    Ultralytics FastSAM wrapper for per-frame mask prediction.

    Two inference modes are exposed:
    - predict_roi(frame_bgr, roi_xyxy)  — FastSAM on a padded crop; masks are
      mapped back to full-frame coordinates.
    - predict_full(frame_bgr)           — full-frame 'everything' mode for
      target re-acquisition after occlusion.

    torch.compile() is applied to the inner backbone network with
    fullgraph=False (graph breaks allowed). A warmup pass is mandatory before
    the tracking loop to avoid the first-frame compilation stall.
    """

    def __init__(
        self,
        weights: str = "FastSAM-s.pt",
        device:  str = "cuda",
    ) -> None:
        self._weights = weights
        self._device  = device
        self._model   = None

    # ------------------------------------------------------------------
    # Load / warmup
    # ------------------------------------------------------------------

    def load(self, progress_cb=None, skip_compile: bool = False) -> None:
        """Load FastSAM-s and attempt torch.compile on the backbone.
        If the weights file is absent, Ultralytics auto-downloads it; a status
        message is emitted via progress_cb beforehand so the caller can update
        the UI.

        skip_compile=True prevents torch.compile regardless of _compile_safe().
        Use this for one-shot registration calls where compile overhead is wasted
        and where the compiled OptimizedModule causes Ultralytics' setup_model()
        to crash on `model or self.args.model` (triggers __len__ which is
        unsupported on OptimizedModule).
        """
        import torch
        from ultralytics import FastSAM

        fname = os.path.basename(self._weights)
        if not os.path.exists(self._weights):
            if progress_cb:
                progress_cb(f"Downloading {fname} (~23 MB via Ultralytics)…")
            logger.info(
                "FastSAM weights '%s' not found locally — "
                "Ultralytics will auto-download on first use.",
                self._weights,
            )

        self._model = FastSAM(self._weights)
        self._model.to(self._device)

        if progress_cb:
            progress_cb(f"{fname} loaded.")

        # Apply torch.compile to the inner nn.Module, not the Ultralytics
        # wrapper.  fullgraph=False is required because Ultralytics uses
        # dynamic control flow that would break full-graph tracing.
        # Skipped when Triton is absent — torch.compile defers the TritonMissing
        # error to the first forward pass, making it uncatchable here.
        # Also skipped when skip_compile=True (e.g. one-shot registration).
        if _compile_safe() and not skip_compile:
            try:
                self._model.model = torch.compile(
                    self._model.model,
                    mode="reduce-overhead",
                    fullgraph=False,
                )
                logger.info("FastSAM backbone compiled with torch.compile.")
            except Exception as exc:
                logger.warning("torch.compile failed for FastSAM (%s). Eager mode.", exc)
        else:
            logger.info("Skipping torch.compile for FastSAM — not safe or not needed in this context.")

    def unload(self) -> None:
        """Delete the model and free GPU memory."""
        import torch, gc
        if self._model is not None:
            del self._model
            self._model = None
        gc.collect()
        torch.cuda.empty_cache()

    def warmup(self, vid_h: int, vid_w: int) -> None:
        """
        Run two dummy inferences to trigger torch.compile graph capture.
        Uses a small dummy frame (faster than full-res).
        """
        import torch

        dummy_h = min(vid_h, 640)
        dummy_w = min(vid_w, 640)
        dummy   = np.zeros((dummy_h, dummy_w, 3), dtype=np.uint8)

        logger.info("Warming up FastSAM (%d×%d)…", dummy_w, dummy_h)
        for _ in range(2):
            try:
                with torch.inference_mode():
                    self._model.predict(
                        dummy,
                        device=self._device,
                        verbose=False,
                        conf=0.4,
                    )
            except Exception as exc:
                logger.warning("FastSAM warmup pass failed: %s", exc)
        logger.info("FastSAM warmup complete.")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_roi(
        self,
        frame_bgr: np.ndarray,
        roi_xyxy:  tuple,
    ) -> tuple[list, list]:
        """
        Segment inside a padded ROI.  Faster than full-frame; yields 2–5
        candidates instead of ~50.

        Parameters
        ----------
        frame_bgr : np.ndarray  uint8 H×W×3
        roi_xyxy  : (x1, y1, x2, y2) crop coordinates in full-frame pixels

        Returns
        -------
        (masks_cuda, bboxes_xyxy)
        masks_cuda  : list of float32 CUDA tensors, each H×W (full-frame size)
        bboxes_xyxy : list of (x1,y1,x2,y2) tuples in full-frame pixel coords
        """
        import torch
        import torch.nn.functional as F

        H_full, W_full = frame_bgr.shape[:2]
        x1, y1, x2, y2 = (int(v) for v in roi_xyxy)
        x1 = max(0, x1);  y1 = max(0, y1)
        x2 = min(W_full, x2); y2 = min(H_full, y2)

        if x2 <= x1 or y2 <= y1:
            logger.debug("ROI collapsed; falling back to full-frame.")
            return self.predict_full(frame_bgr)

        crop    = frame_bgr[y1:y2, x1:x2]
        crop_h  = y2 - y1
        crop_w  = x2 - x1

        try:
            with torch.inference_mode():
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    results = self._model.predict(
                        source       = crop,
                        device       = self._device,
                        conf         = 0.40,
                        iou          = 0.90,
                        retina_masks = True,
                        verbose      = False,
                    )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return [], []

        if not results or results[0].masks is None:
            return [], []

        return self._extract_masks(
            result   = results[0],
            H_full   = H_full,
            W_full   = W_full,
            x_off    = x1,
            y_off    = y1,
            crop_h   = crop_h,
            crop_w   = crop_w,
        )

    def predict_full(
        self,
        frame_bgr: np.ndarray,
    ) -> tuple[list, list]:
        """
        Full-frame FastSAM 'everything' mode for re-acquisition.
        Returns all detected masks sorted by area (largest first).

        Returns
        -------
        (masks_cuda, bboxes_xyxy) — same format as predict_roi
        """
        import torch

        H, W = frame_bgr.shape[:2]

        try:
            with torch.inference_mode():
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    results = self._model.predict(
                        source       = frame_bgr,
                        device       = self._device,
                        conf         = 0.35,
                        iou          = 0.90,
                        retina_masks = True,
                        verbose      = False,
                    )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return [], []

        if not results or results[0].masks is None:
            return [], []

        masks_out, bboxes_out = self._extract_masks(
            result  = results[0],
            H_full  = H,
            W_full  = W,
            x_off   = 0,
            y_off   = 0,
            crop_h  = H,
            crop_w  = W,
        )

        # Sort by mask area descending so the largest candidate is first
        if masks_out:
            areas  = [m.sum().item() for m in masks_out]
            order  = sorted(range(len(areas)), key=lambda i: -areas[i])
            masks_out  = [masks_out[i]  for i in order]
            bboxes_out = [bboxes_out[i] for i in order]

        return masks_out, bboxes_out

    # ------------------------------------------------------------------
    # Internal mask extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_masks(
        result,
        H_full: int,
        W_full: int,
        x_off:  int,
        y_off:  int,
        crop_h: int,
        crop_w: int,
    ) -> tuple[list, list]:
        """
        Convert Ultralytics Results masks to full-frame CUDA float32 tensors.

        Parameters
        ----------
        result       : Ultralytics Results object
        H_full/W_full: dimensions of the complete original frame
        x_off/y_off  : top-left corner of the crop region in full-frame coords
                       (0, 0 for full-frame mode)
        crop_h/crop_w: expected mask spatial dimensions inside the crop region
        """
        import torch
        import torch.nn.functional as F

        masks_data: "torch.Tensor" = result.masks.data   # [N, mh, mw]
        boxes = result.boxes
        N = masks_data.shape[0]
        mh, mw = masks_data.shape[1], masks_data.shape[2]

        masks_out:  list = []
        bboxes_out: list = []

        for i in range(N):
            mask_t = masks_data[i].float().unsqueeze(0).unsqueeze(0)   # 1×1×mh×mw

            # Resize from model output resolution to the crop region size
            if (mh, mw) != (crop_h, crop_w):
                mask_t = F.interpolate(
                    mask_t,
                    size            = (crop_h, crop_w),
                    mode            = "bilinear",
                    align_corners   = False,
                )
            mask_crop = (mask_t[0, 0] > 0.5)   # bool, crop_h × crop_w

            # Embed into a full-frame zero tensor
            full_mask = torch.zeros(H_full, W_full, dtype=torch.float32, device="cuda")
            full_mask[y_off: y_off + crop_h, x_off: x_off + crop_w] = mask_crop.float()
            masks_out.append(full_mask)

            # Bbox: offset from crop coords to full-frame coords
            if boxes is not None and i < len(boxes.xyxy):
                bx1, by1, bx2, by2 = (float(v) for v in boxes.xyxy[i].tolist())
                bboxes_out.append((
                    bx1 + x_off,
                    by1 + y_off,
                    bx2 + x_off,
                    by2 + y_off,
                ))
            else:
                # Derive bbox from the mask itself
                rows = full_mask.bool().any(dim=1)
                cols = full_mask.bool().any(dim=0)
                if rows.any() and cols.any():
                    ry = rows.nonzero(as_tuple=False)[:, 0]
                    cx = cols.nonzero(as_tuple=False)[:, 0]
                    bboxes_out.append((
                        float(cx[0].item()),
                        float(ry[0].item()),
                        float(cx[-1].item()),
                        float(ry[-1].item()),
                    ))
                else:
                    bboxes_out.append((0.0, 0.0, float(W_full), float(H_full)))

        return masks_out, bboxes_out


# ---------------------------------------------------------------------------
# ReIDEmbedder
# ---------------------------------------------------------------------------

class ReIDEmbedder:
    """
    Appearance embedding network for instance discrimination.

    Primary backend  : OSNet-x0.25 via torchreid (512-d, trained for person ReID).
    Fallback backend : torchvision ResNet18 with fc replaced by nn.Identity()
                       (512-d, ImageNet features — less discriminative but
                       sufficient for single-target tracking in most scenes).

    The active backend is logged at load time so the user can see which path
    is active (via the status bar that reads application logs).

    All embeddings are L2-normalised before return so that cosine similarity
    equals the dot product.
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        device:       str           = "cuda",
    ) -> None:
        self._weights  = weights_path
        self._device   = device
        self._model    = None
        self._backend  = "none"

    # ------------------------------------------------------------------
    # Load / warmup
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load the Re-ID model; falls back to ResNet18 if torchreid absent."""
        import torch
        import torch.nn as nn

        try:
            self._model   = self._load_osnet()
            self._backend = "osnet_x0_25"
            logger.info("Re-ID backend: OSNet-x0.25")
        except ImportError:
            self._model   = self._load_resnet18()
            self._backend = "resnet18_fallback"
            logger.warning(
                "torchreid not installed — using ResNet18 as Re-ID fallback. "
                "Install torchreid for higher identity discrimination accuracy."
            )
        except Exception as exc:
            logger.warning("OSNet load failed (%s) — falling back to ResNet18.", exc)
            self._model   = self._load_resnet18()
            self._backend = "resnet18_fallback"

        # Compile for faster batched inference.
        # Skipped when Triton is absent — torch.compile defers the TritonMissing
        # error to the first forward pass, making it uncatchable at load time.
        if _compile_safe():
            try:
                self._model = torch.compile(
                    self._model,
                    mode       = "reduce-overhead",
                    fullgraph  = False,
                )
                logger.info("Re-ID model compiled with torch.compile.")
            except Exception as exc:
                logger.warning("torch.compile failed for Re-ID model (%s). Eager mode.", exc)
        else:
            logger.info("Skipping torch.compile for Re-ID — not safe in this process context.")

    def warmup(self) -> None:
        """Trigger compilation graph capture with two dummy passes."""
        import torch

        dummy = torch.zeros(1, 3, _REID_H, _REID_W, device=self._device)
        with torch.inference_mode():
            self._model(dummy)
            self._model(dummy)
        logger.info("Re-ID embedder warmed up.")

    def unload(self) -> None:
        import torch, gc
        del self._model
        self._model = None
        gc.collect()
        torch.cuda.empty_cache()

    @property
    def backend(self) -> str:
        return self._backend

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def embed(self, crop_bgr: np.ndarray) -> np.ndarray:
        """
        Embed a single BGR crop.

        Returns
        -------
        np.ndarray  float32 (D,)  L2-normalised
        """
        import torch
        import torch.nn.functional as F

        t = self._preprocess_crop(crop_bgr).unsqueeze(0).to(self._device)   # 1×3×H×W
        with torch.inference_mode():
            emb = self._model(t)               # [1, D]
        emb = F.normalize(emb, dim=1)
        return emb.squeeze(0).cpu().numpy()

    def embed_batch(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        """
        Embed a list of BGR crops in a single forward pass.

        Parameters
        ----------
        crops_bgr : list of np.ndarray  (each H×W×3 uint8 BGR)

        Returns
        -------
        np.ndarray  float32 [N, D]  each row L2-normalised
        """
        import torch
        import torch.nn.functional as F

        if not crops_bgr:
            return np.zeros((0, 512), dtype=np.float32)

        tensors = [self._preprocess_crop(c) for c in crops_bgr]
        batch   = torch.stack(tensors).to(self._device)    # [N, 3, H, W]

        with torch.inference_mode():
            embs = self._model(batch)                      # [N, D]
        embs = F.normalize(embs, dim=1)
        return embs.cpu().numpy()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _preprocess_crop(crop_bgr: np.ndarray) -> "torch.Tensor":
        """
        Resize to (REID_W, REID_H), convert BGR→RGB, normalise to ImageNet
        stats, and return a float32 (3, H, W) CPU tensor.
        """
        import torch

        resized = cv2.resize(crop_bgr, (_REID_W, _REID_H),
                             interpolation=cv2.INTER_LINEAR)
        rgb     = resized[..., ::-1].copy().astype(np.float32) / 255.0
        rgb     = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
        return torch.from_numpy(rgb).permute(2, 0, 1).float()

    def _load_osnet(self):
        """
        Load OSNet-x0.25 via torchreid.
        Raises ImportError if torchreid is not installed.
        """
        import torchreid
        import torch

        model = torchreid.models.build_model(
            name       = "osnet_x0_25",
            num_classes = 1000,
            pretrained  = True,   # downloads ImageNet weights if absent
        )
        model = model.to(self._device).eval()

        if self._weights and os.path.isfile(self._weights):
            torchreid.utils.load_pretrained_weights(model, self._weights)
            logger.info("Loaded OSNet weights from %s", self._weights)

        return model

    def _load_resnet18(self):
        """
        Load torchvision ResNet18; replace the FC head with nn.Identity()
        to expose the 512-d avgpool feature vector.
        """
        import torch
        import torch.nn as nn
        from torchvision.models import resnet18, ResNet18_Weights

        model      = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model.fc   = nn.Identity()   # 512-d output
        return model.to(self._device).eval()


# ---------------------------------------------------------------------------
# EMAGallery
# ---------------------------------------------------------------------------

class EMAGallery:
    """
    Thread-safe reference embedding with Exponential Moving Average update.

    Gallery holds a single reference vector v_ref (L2-normalised float32 (D,)).
    Candidate embeddings are matched via dot-product (= cosine similarity
    when both vectors are unit-norm).

    EMA update rule
    ---------------
    v_new = alpha * v_ref + (1 - alpha) * v_match
    v_ref = v_new / ||v_new||                       ← re-normalise

    The update fires ONLY when the similarity exceeds ema_threshold (default
    0.92), which is stricter than the match_threshold (default 0.85).  This
    prevents the reference from drifting towards an occluder if the tracker
    briefly snaps to a foreground object with marginal similarity.
    """

    def __init__(
        self,
        initial_emb:     np.ndarray,
        ema_alpha:       float = 0.90,
        ema_threshold:   float = 0.92,
        match_threshold: float = 0.85,
    ) -> None:
        v = initial_emb.copy().astype(np.float32)
        norm = np.linalg.norm(v)
        self._v_ref          = v / norm if norm > 0 else v
        self._ema_alpha      = float(ema_alpha)
        self._ema_threshold  = float(ema_threshold)
        self._match_threshold = float(match_threshold)
        self._lock           = threading.Lock()

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def cosine_sim(self, emb: np.ndarray) -> float:
        """
        Cosine similarity between emb and the current reference.
        Thread-safe.
        """
        with self._lock:
            return float(np.dot(self._v_ref, emb))

    def best_match(self, emb_batch: np.ndarray) -> tuple[int, float]:
        """
        Find the best-matching candidate in a batch of embeddings.

        Parameters
        ----------
        emb_batch : np.ndarray  float32 [N, D]  (rows must be L2-normalised)

        Returns
        -------
        (best_idx: int, similarity: float)
        """
        with self._lock:
            sims = emb_batch @ self._v_ref   # [N]  dot products
        best_idx = int(np.argmax(sims))
        return best_idx, float(sims[best_idx])

    def is_match(self, emb: np.ndarray) -> tuple[bool, float]:
        """
        Single-embedding match against the match_threshold.

        Returns
        -------
        (accepted: bool, similarity: float)
        """
        sim = self.cosine_sim(emb)
        return sim >= self._match_threshold, sim

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, emb: np.ndarray, sim: float) -> None:
        """
        Apply EMA update if sim strictly exceeds ema_threshold.
        Thread-safe.

        Parameters
        ----------
        emb : np.ndarray  float32 (D,)  L2-normalised match embedding
        sim : float  cosine similarity of emb against v_ref
        """
        if sim <= self._ema_threshold:
            return
        with self._lock:
            v    = self._ema_alpha * self._v_ref + (1.0 - self._ema_alpha) * emb
            norm = np.linalg.norm(v)
            if norm > 0:
                self._v_ref = v / norm

    def reset(self, emb: np.ndarray) -> None:
        """Hard-reset reference to a new embedding (e.g., after re-registration)."""
        v = emb.copy().astype(np.float32)
        norm = np.linalg.norm(v)
        with self._lock:
            self._v_ref = v / norm if norm > 0 else v

    def get_reference(self) -> np.ndarray:
        """Return a copy of the current reference embedding."""
        with self._lock:
            return self._v_ref.copy()

    # ------------------------------------------------------------------
    # Live config update (called from GPUPipelineProcess command loop)
    # ------------------------------------------------------------------

    def set_match_threshold(self, value: float) -> None:
        self._match_threshold = float(value)


# ---------------------------------------------------------------------------
# TrackerWrapper
# ---------------------------------------------------------------------------

class TrackerWrapper:
    """
    Thin wrapper around cv2.TrackerMIL.

    TrackerCSRT was removed in OpenCV 4.10+.  TrackerMIL is the closest
    available alternative and performs adequately for single-target sports
    tracking.

    Coordinate convention
    ---------------------
    init() accepts xywh (x, y, width, height) to match cv2's convention.
    update() returns **xywh** (same as cv2) so InferThread can convert to
    xyxy using _xywh_to_xyxy() in pipeline.py.

    Confidence estimation
    ---------------------
    cv2 trackers do not expose a native confidence score.  We estimate
    confidence from the IoU between the last accepted bbox and the current
    prediction.  Consecutive failures decay the estimate toward 0.
    """

    def __init__(self) -> None:
        self._tracker:             Optional[cv2.Tracker] = None
        self._last_bbox_xywh:      Optional[tuple]       = None
        self._consecutive_failures: int                  = 0

    def init(self, frame_bgr: np.ndarray, bbox_xywh: tuple) -> None:
        """
        Initialise (or re-initialise) the tracker.

        Parameters
        ----------
        frame_bgr  : np.ndarray  uint8 H×W×3
        bbox_xywh  : (x, y, w, h) in pixel coordinates
        """
        x, y, w, h = self._sanitize_bbox(bbox_xywh, frame_bgr.shape)

        self._tracker = cv2.TrackerMIL_create()
        try:
            self._tracker.init(frame_bgr, (x, y, w, h))
        except cv2.error:
            # MIL needs a background margin to sample negatives; a degenerate
            # or full-frame box leaves none. Drop the tracker so update() falls
            # back to the last good bbox instead of crashing the pipeline.
            self._tracker = None
            logger.warning("TrackerMIL init failed for bbox=%s; tracking disabled "
                           "until next re-acquire", (x, y, w, h))
        self._last_bbox_xywh       = (x, y, w, h)
        self._consecutive_failures = 0

    @staticmethod
    def _sanitize_bbox(bbox_xywh: tuple, frame_shape: tuple) -> tuple:
        """
        Clamp a bbox inside the frame and leave a >=1px margin on every side so
        the MIL tracker always has room to draw negative samples. Returns
        (x, y, w, h) ints guaranteed to satisfy cv2's init assertions.
        """
        H, W = int(frame_shape[0]), int(frame_shape[1])
        x, y, w, h = (int(v) for v in bbox_xywh)
        w = max(1, w); h = max(1, h)

        # Inset the box by 1px from each frame edge so negatives exist.
        x = min(max(1, x), max(1, W - 2))
        y = min(max(1, y), max(1, H - 2))
        w = max(1, min(w, W - 1 - x))
        h = max(1, min(h, H - 1 - y))
        return (x, y, w, h)

    def update(self, frame_bgr: np.ndarray) -> tuple[bool, tuple, float]:
        """
        Predict the target location in the next frame.

        Returns
        -------
        ok         : bool   True if tracker reports a successful update
        bbox_xywh  : tuple  (x, y, w, h)
        confidence : float  in [0.0, 1.0]
        """
        if self._tracker is None:
            fallback = self._last_bbox_xywh or (0, 0, 64, 64)
            return False, fallback, 0.0

        try:
            ok, raw_bbox = self._tracker.update(frame_bgr)
        except cv2.error:
            # OpenCV can throw an opaque C++ exception when the tracked box
            # drifts to the frame edge. Treat as a failed update and coast on
            # the last known bbox rather than tearing down the GPU process.
            self._tracker = None
            ok, raw_bbox = False, self._last_bbox_xywh or (0, 0, 64, 64)

        if ok:
            bbox_xywh = tuple(int(v) for v in raw_bbox)
            conf = self._estimate_confidence(bbox_xywh)
            self._last_bbox_xywh       = bbox_xywh
            self._consecutive_failures = 0
        else:
            bbox_xywh = self._last_bbox_xywh or (0, 0, 64, 64)
            self._consecutive_failures += 1
            conf = max(0.0, 0.8 - self._consecutive_failures * 0.12)

        return ok, bbox_xywh, float(conf)

    # ------------------------------------------------------------------
    # Confidence heuristic
    # ------------------------------------------------------------------

    def _estimate_confidence(self, new_bbox_xywh: tuple) -> float:
        """
        Estimate tracking confidence as a rescaled IoU between the previous
        and current bounding boxes.

        An IoU of 1.0 → confidence 0.90.
        An IoU of 0.0 → confidence 0.15 (bbox teleported; likely lost).
        """
        if self._last_bbox_xywh is None:
            return 0.90   # first frame

        x1, y1, w1, h1 = self._last_bbox_xywh
        x2, y2, w2, h2 = new_bbox_xywh

        ax1, ay1, ax2, ay2 = x1, y1, x1 + w1, y1 + h1
        bx1, by1, bx2, by2 = x2, y2, x2 + w2, y2 + h2

        ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2); iy2 = min(ay2, by2)

        if ix2 <= ix1 or iy2 <= iy1:
            return 0.15   # no overlap

        inter = (ix2 - ix1) * (iy2 - iy1)
        union = w1 * h1 + w2 * h2 - inter
        iou   = inter / union if union > 0 else 0.0

        return 0.15 + 0.75 * iou


# ---------------------------------------------------------------------------
# FastSAM registration helper
# ---------------------------------------------------------------------------

def _pick_best_fastsam_mask(
    masks_cuda: list,
    user_bbox_xyxy: tuple,
    pos_points: list,
    neg_points: list,
):
    """
    Select the best FastSAM candidate mask given a user-drawn bbox and optional
    positive / negative point hints.

    Priority
    --------
    1. If positive points: pick the mask containing the most positive clicks.
    2. Reject masks that cover any negative click.
    3. Fallback: highest IoU with the user-drawn bbox region.
    4. Final fallback: largest area.
    """
    import torch

    if len(masks_cuda) == 1:
        return masks_cuda[0]

    m0 = masks_cuda[0]
    H = m0.shape[-2] if m0.dim() >= 2 else int(m0.shape[0])
    W = m0.shape[-1] if m0.dim() >= 2 else int(m0.shape[1])

    x1, y1, x2, y2 = user_bbox_xyxy
    bbox_mask = torch.zeros(H, W, device=m0.device)
    r_x1 = max(0, int(x1)); r_y1 = max(0, int(y1))
    r_x2 = min(W, int(x2)); r_y2 = min(H, int(y2))
    bbox_mask[r_y1:r_y2, r_x1:r_x2] = 1.0

    scores = []
    for m in masks_cuda:
        m_bin = (m > 0.5).float()

        # Reject if a negative click falls inside this mask
        neg_hit = any(
            m_bin[min(H - 1, max(0, int(py))), min(W - 1, max(0, int(px)))] > 0.5
            for px, py in neg_points
        )
        if neg_hit:
            scores.append(-1.0)
            continue

        if pos_points:
            pos_hits = sum(
                float(m_bin[min(H - 1, max(0, int(py))), min(W - 1, max(0, int(px)))] > 0.5)
                for px, py in pos_points
            )
            scores.append(pos_hits)
        else:
            inter = (m_bin * bbox_mask).sum()
            union = (m_bin + bbox_mask).clamp(0, 1).sum()
            scores.append(float(inter / union.clamp(min=1)))

    best_idx = int(max(range(len(scores)), key=lambda i: scores[i]))
    if scores[best_idx] < 0:
        best_idx = int(max(range(len(masks_cuda)), key=lambda i: masks_cuda[i].sum()))
    return masks_cuda[best_idx]
