# OptimisedOneShot — One-Shot Video Object Segmentation

Desktop application for real-time, one-shot video object segmentation on a single consumer GPU. The user clicks a target object on a paused frame; SAM2 generates a pixel-perfect mask in one shot; the heavy model is evicted; a lightweight FastSAM + OSNet pipeline tracks the target across all remaining frames.

---

## Confirmed Environment

| Package | Version |
|---|---|
| PyTorch | 2.12.0+cu132 |
| CUDA | 13.2 |
| torchvision | 0.27.0 |
| PyQt5 | 5.15.11 |
| qtpy | 2.4.3 |
| OpenCV | 4.13.0 |
| SAM2 | 1.0 |
| Ultralytics | 8.4.14 |
| ffmpeg-python | 0.2.0 |

**Absent packages (fallbacks activate automatically):**
- `decord` → video decode falls back to `cv2.CAP_MSMF` (Windows Media Foundation)
- `torchaudio` → same fallback
- `torchreid` → Re-ID falls back to ResNet18 with `fc=nn.Identity()` (512-d)
- `triton` → `torch.compile` is skipped entirely; models run in eager mode

**GPU:** RTX 3070, 8.59 GB VRAM total

---

## Project Structure

```
OptimisedOneShot/
├── main.py          # Entry point, queue creation, multiprocessing bootstrap
├── gui.py           # MainWindow, VideoCanvas, ControlPanel, TimelinePanel,
│                    # RegistrationThread, VideoReaderThread, FrameDisplayWorker
├── pipeline.py      # GPUPipelineProcess, DecodeThread, InferThread, BlendThread
├── models.py        # HeavySAMRegistrar, FastSAMTracker, ReIDEmbedder,
│                    # EMAGallery, TrackerWrapper
├── video_io.py      # NVDECReader (multi-backend), VideoWriter (NVENC fallback)
├── requirements.txt # All dependencies with version pins
└── .gitignore       # Excludes *.pt, *.pth, __pycache__, output videos
```

---

## Architecture Overview

### Two-Process Topology

```
┌─────────────────────────────────────────────────────────────────┐
│  MAIN PROCESS  ·  Qt Event Loop  ·  NO CUDA                    │
│                                                                 │
│  RegistrationThread (QThread)                                   │
│    HeavySAMRegistrar → SAM2 Large fp16 → mask + bbox           │
│    ReIDEmbedder → reference embedding                           │
│    Unloads both models before GPU process starts                │
│                                                                 │
│  VideoReaderThread (QThread)                                    │
│    cv2.VideoCapture (CAP_MSMF) → raw_frame_queue               │
│    Modes: SEEK (scrub), PLAY (preview), START_TRACKING (GPU)   │
│                                                                 │
│  FrameDisplayWorker (QThread)                                   │
│    Drains display_queue → QPixmap → VideoCanvas                │
└──────────────────────┬──────────────────────────────────────────┘
                       │  mp.Queue (CPU numpy bytes only)
┌──────────────────────▼──────────────────────────────────────────┐
│  GPU PIPELINE PROCESS  ·  torch.multiprocessing 'spawn'        │
│  CUDA initialised in run(), never in __init__                   │
│                                                                 │
│  DecodeThread  → raw_frame_queue → CUDA tensor                 │
│  InferThread   → TrackerMIL → FastSAM ROI → OSNet embed        │
│                → EMA gallery cosine match → gate mask          │
│  BlendThread   → GPU alpha-composite → CPU bytes               │
│                → display_queue + metadata_queue                │
└─────────────────────────────────────────────────────────────────┘
```

### Cross-Process Queues (all CPU data — no CUDA tensors)

| Queue | Size | Payload |
|---|---|---|
| `raw_frame_queue` | 4 | `(frame_idx: int, frame_bgr: np.uint8 H×W×3)` |
| `reg_result_queue` | 1 | `{mask, bbox, reid_emb, score, frame_bgr, frame_idx}` |
| `pipeline_cmd_queue` | 8 | `Cmd(type, payload)` — STOP / UPDATE_THRESHOLD / UPDATE_ALPHA / UPDATE_STRIDE |
| `display_queue` | 3 | `(raw_bytes, W, H, frame_idx, meta: dict)` |

---

## Critical Constraints

- **CUDA tensors never cross `mp.Queue` boundaries.** Convert to CPU numpy bytes before enqueue; reconstruct on the GPU side.
- **CUDA is never initialised in the main process.** No `import torch` at module level in `main.py`, `gui.py`, or `video_io.py`. All GPU work happens inside `GPUPipelineProcess.run()` or `RegistrationThread.run()`.
- **`cv2.TrackerCSRT_create()` was removed in OpenCV 4.10.** Use `cv2.TrackerMIL_create()` unconditionally throughout.
- **SAM2 and FastSAM never coexist in VRAM.** `HeavySAMRegistrar.unload()` (del + `empty_cache()`) fires before `FastSAMTracker.load()`. `VRAM_SWAP_THRESHOLD_GB = 8.0` always triggers on the RTX 3070.
- **`torch.compile` requires Triton.** Triton is not installed in this environment. `_triton_available()` in `models.py` guards all `torch.compile` calls — models run in eager mode.
- **Mixed precision:** `torch.autocast(device_type='cuda', dtype=torch.float16)` is used (not `.half()`) to keep sigmoid/softmax in fp32.

---

## Model Weights

| Model | File | How obtained |
|---|---|---|
| SAM2 Large | `sam2.1_hiera_large.pt` | **Auto-downloaded** on first "Register Target" click (~1.9 GB from Meta CDN). Progress shown in status bar. |
| FastSAM-s | `FastSAM-s.pt` | **Auto-downloaded** via Ultralytics on first tracking launch (~23 MB). |
| OSNet x0.25 | `osnet_x0_25_imagenet.pth` | Optional. If absent, ResNet18 fallback activates automatically. |

All `*.pt` and `*.pth` files are gitignored.

---

## Key Classes

### `gui.py`

| Class | Role |
|---|---|
| `VideoCanvas` | QGraphicsView — letterboxed video, left/right-click point prompts, mask overlay |
| `TimelinePanel` | Frame slider (seeks on release), Play/Pause button, FPS label, export progress |
| `ControlPanel` | Sidebar: model paths, prompt mode, register/track/export controls |
| `RegistrationThread` | QThread — SAM2 load → register → ReID embed → unload; emits `registration_done` |
| `VideoReaderThread` | QThread — cv2.VideoCapture; modes: SEEK / PLAY / START_TRACKING / STOP_TRACKING |
| `FrameDisplayWorker` | QThread — drains `display_queue`; emits `frame_ready(bytes, W, H, idx)` |
| `MainWindow` | Orchestrator — state machine, wires all signals, spawns GPU process |

### `models.py`

| Class | Role |
|---|---|
| `HeavySAMRegistrar` | SAM2 Large fp16; `load(progress_cb)` auto-downloads checkpoint if missing |
| `FastSAMTracker` | Ultralytics FastSAM-s; `predict_roi` / `predict_full`; auto-downloads weights |
| `ReIDEmbedder` | OSNet-x0.25 (torchreid) with ResNet18 fallback; L2-normalised embeddings |
| `EMAGallery` | Thread-safe reference embedding; EMA update only when `sim > 0.92` |
| `TrackerWrapper` | cv2.TrackerMIL; xywh↔xyxy conversion; heuristic IoU-based confidence |

### `pipeline.py`

| Class | Role |
|---|---|
| `GPUPipelineProcess` | `mp.Process(daemon=True)`; imports torch only inside `run()` |
| `DecodeThread` | `raw_frame_queue` → `.cuda()` → `decode_to_infer_q` |
| `InferThread` | TrackerMIL → ROI crop → FastSAM → ReID → EMA gallery → gate |
| `BlendThread` | GPU alpha-composite → `.cpu().numpy().tobytes()` → `display_queue` |

### `video_io.py`

| Class | Role |
|---|---|
| `NVDECReader` | Backend cascade: torchaudio → decord → CAP_MSMF (active) → CAP_FFMPEG → default |
| `VideoWriter` | ffmpeg h264_nvenc → libx264 → cv2.VideoWriter fallback |

---

## User Workflow

1. **Load Video** — opens `cv2.VideoCapture`; first frame displayed; scrubbing enabled.
2. **Add Point Prompts** — left-click positive (green dot), right-click negative (red dot).
3. **Register Target** — `RegistrationThread` runs SAM2 → mask overlay shown; models unloaded.
4. **Start Live Preview** — spawns `GPUPipelineProcess`; green mask tracks target at video FPS.
5. **Export** — batch-renders entire video with mask composite to `.mp4`.

---

## Known Bugs Fixed in This Codebase

- **Slider seek on drag release** — `_pending_seek_value` was stale during drag; fixed by capturing `self.slider.value()` in `_on_slider_released`.
- **Play button in preview mode** — was calling `start_tracking()` which pushes to `raw_frame_queue` with no consumer; fixed by adding a PLAY/PAUSE mode to `VideoReaderThread` that emits `preview_frame` directly.
- **TritonMissing on first inference** — `torch.compile` defers its backend error to the first forward pass; fixed by checking `_triton_available()` before compiling.
- **Process not closing on exit** — `mp.Queue` feeder threads blocked Python exit; fixed by calling `q.cancel_join_thread()` on all queues in `closeEvent`, plus `join()` after `terminate()` on the GPU process.
- **TrackerCSRT removed in OpenCV 4.13** — all code uses `cv2.TrackerMIL_create()`.

---

## Running

```bash
pip install -r requirements.txt
# SAM2 also requires:
pip install git+https://github.com/facebookresearch/segment-anything-2.git

python main.py
```

ffmpeg must be on `PATH` for video export (`https://ffmpeg.org/download.html`).
