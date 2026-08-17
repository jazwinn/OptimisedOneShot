"""
OptimisedOneShot — Video I/O Module
=====================================
NVDECReader   — Multi-backend video reader with automatic fallback chain.
VideoWriter   — Multi-backend video writer (NVENC → libx264 → cv2 mp4v),
                or cv2.imwrite when the output path is a still image.
ImageCapture  — A single still image behind the cv2.VideoCapture API.
open_capture  — Opens either a video or a still image as a frame source.

Still images
------------
The app treats a still image as a one-frame video: open_capture() returns
an ImageCapture for image extensions and a real cv2.VideoCapture otherwise,
so every caller that scrubs, previews, registers, tracks or exports works
unchanged for both.  Exporting a still source writes an image file rather
than a one-frame MP4 — VideoWriter switches to its cv2_imwrite backend when
output_path has an image extension.

NVDECReader backend priority
-----------------------------
1. torchaudio.io.StreamReader  — NVDEC hardware decode; frames delivered as
                                  GPU tensors (zero host round-trip).
2. decord.VideoReader(gpu(0))  — NVDEC hardware decode with direct indexing.
3. cv2.VideoCapture(CAP_MSMF)  — Windows Media Foundation; hardware-accelerated
                                  decode via DXVA2/D3D11 on Windows 11.
4. cv2.VideoCapture(CAP_FFMPEG)— CPU decode via the bundled FFMPEG.
5. cv2.VideoCapture()          — System default (last resort).

On the target machine (Windows 11, torchaudio and decord absent) backend 3
(CAP_MSMF) activates automatically.  All backends return BGR uint8 numpy
arrays from read_frame() so callers are backend-agnostic.

VideoWriter backend priority
-----------------------------
1. ffmpeg subprocess with h264_nvenc (GPU encode via NVIDIA NVENC).
2. ffmpeg subprocess with libx264   (CPU encode, high quality).
3. cv2.VideoWriter with avc1/mp4v   (CPU encode, universal fallback).

Threading
---------
Both classes protect their internal state with threading.Lock.
NVDECReader is held exclusively by VideoReaderThread; VideoWriter is used
by BlendThread (GPU process) during batch render.  Both are safe for
single-producer, single-consumer access patterns.
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Still-image sources
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = (
    ".png", ".jpg", ".jpeg", ".bmp", ".webp",
    ".tif", ".tiff", ".ppm", ".pgm", ".jp2",
)
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm")


def _glob_filter(label: str, extensions) -> str:
    """Build one Qt file-dialog filter clause, e.g. 'Images (*.png *.jpg)'."""
    return f"{label} (" + " ".join(f"*{e}" for e in extensions) + ")"


# Qt file-dialog filter strings, shared by every "open source" dialog.
VIDEO_FILTER = _glob_filter("Video Files", VIDEO_EXTENSIONS)
IMAGE_FILTER = _glob_filter("Image Files", IMAGE_EXTENSIONS)
MEDIA_FILTER = _glob_filter("Video & Image Files", VIDEO_EXTENSIONS + IMAGE_EXTENSIONS)


def is_image_path(path: str) -> bool:
    """True if `path` looks like a still image rather than a video container."""
    return bool(path) and os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS


class ImageCapture:
    """
    A single still image wearing a cv2.VideoCapture costume.

    Implements the subset of the VideoCapture API the app actually uses
    (isOpened / get / set / read / release / grab) over a one-frame
    "video", so every caller that scrubs, previews, registers, tracks or
    exports works unchanged when handed a JPEG instead of an MP4.

    Frame 0 is the image; reading past it returns (False, None) exactly
    like the end of a video stream.
    """

    #: Reported FPS.  Arbitrary but non-zero — downstream code divides by it.
    DEFAULT_FPS = 30.0

    def __init__(self, path: str) -> None:
        self.path   = path
        self._frame = cv2.imread(path, cv2.IMREAD_COLOR)
        self._pos   = 0

    # -- VideoCapture-compatible API ------------------------------------

    def isOpened(self) -> bool:                     # noqa: N802 — cv2 naming
        return self._frame is not None

    def get(self, prop: int) -> float:              # noqa: D102
        if self._frame is None:
            return 0.0
        h, w = self._frame.shape[:2]
        return {
            cv2.CAP_PROP_FRAME_WIDTH:  float(w),
            cv2.CAP_PROP_FRAME_HEIGHT: float(h),
            cv2.CAP_PROP_FPS:          self.DEFAULT_FPS,
            cv2.CAP_PROP_FRAME_COUNT:  1.0,
            cv2.CAP_PROP_POS_FRAMES:   float(self._pos),
        }.get(prop, 0.0)

    def set(self, prop: int, value: float) -> bool:  # noqa: D102
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self._pos = max(0, int(value))
            return True
        return False

    def read(self):
        """Return (True, frame_copy) for frame 0, (False, None) afterwards."""
        if self._frame is None or self._pos != 0:
            self._pos += 1
            return False, None
        self._pos = 1
        return True, self._frame.copy()

    def grab(self) -> bool:                          # noqa: D102
        ok, _ = self.read()
        return ok

    def release(self) -> None:
        self._frame = None

    def __enter__(self) -> "ImageCapture":
        return self

    def __exit__(self, *_) -> None:
        self.release()


def open_capture(path: str):
    """
    Open `path` as a frame source and return a cv2.VideoCapture-compatible
    object.

    Still images yield an :class:`ImageCapture`; videos yield a real
    cv2.VideoCapture opened with hardware-accelerated MSMF where possible,
    falling back to OpenCV's default backend.

    The result may be closed (``isOpened() == False``) — callers check, as
    they already do for cv2.VideoCapture.
    """
    if is_image_path(path):
        return ImageCapture(path)

    cap = cv2.VideoCapture(path, cv2.CAP_MSMF)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(path)
    return cap


# ---------------------------------------------------------------------------
# NVDECReader
# ---------------------------------------------------------------------------

class NVDECReader:
    """
    Hardware-accelerated video reader with graceful five-level backend cascade.

    Usage (basic)
    -------------
    reader = NVDECReader(path)
    reader.open()
    meta   = reader.get_metadata()      # {width, height, fps, total_frames, …}
    frame  = reader.read_frame()        # sequential BGR uint8 H×W×3
    frame  = reader.read_frame(idx=42)  # seek + read (thread-safe)
    reader.close()

    Usage (context manager — preferred)
    ------------------------------------
    with NVDECReader(path) as reader:
        for i in range(total_frames):
            frame = reader.read_frame()
    """

    def __init__(self, path: str, device: str = "cuda") -> None:
        self.path    = path
        self.device  = device

        self._backend:      str = "none"
        self._cap           = None        # cv2.VideoCapture
        self._vr            = None        # decord.VideoReader
        self._sr            = None        # torchaudio StreamReader
        self._lock          = threading.Lock()
        self._metadata: dict = {}
        self._current_pos: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> None:
        """
        Open the video file by trying each backend in priority order.
        Raises RuntimeError if all backends fail.
        """
        attempts = [
            (self._try_torchaudio,          "torchaudio NVDEC"),
            (self._try_decord,              "decord GPU"),
            (lambda: self._try_cv2(cv2.CAP_MSMF,   "cv2_msmf"),   "cv2 CAP_MSMF"),
            (lambda: self._try_cv2(cv2.CAP_FFMPEG,  "cv2_ffmpeg"), "cv2 CAP_FFMPEG"),
            (lambda: self._try_cv2(0,               "cv2_default"),"cv2 default"),
        ]

        for fn, label in attempts:
            try:
                if fn():
                    logger.info("Video decode backend: %s  [%s]", label, self.path)
                    return
            except Exception as exc:
                logger.debug("Backend '%s' raised: %s", label, exc)

        raise RuntimeError(
            f"No video backend could open '{self.path}'.\n"
            "Supported formats: mp4, mov, avi, mkv, webm.\n"
            "Install decord or torchaudio for GPU-accelerated decode."
        )

    def close(self) -> None:
        """Release all backend resources."""
        with self._lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            self._vr = None   # decord: reference-counted
            self._sr = None   # torchaudio: GC handles it
        self._backend = "none"

    def __enter__(self) -> "NVDECReader":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    def read_frame(self, frame_idx: Optional[int] = None) -> np.ndarray:
        """
        Read one frame and return it as BGR uint8 H×W×3.

        Parameters
        ----------
        frame_idx : int or None
            If given, seek to that position before reading.
            If None, read the next sequential frame.

        Raises
        ------
        EOFError      — stream is exhausted (past last frame).
        RuntimeError  — backend not open or read failed.
        """
        if self._backend == "none":
            raise RuntimeError("NVDECReader.open() must be called before read_frame().")

        with self._lock:
            if frame_idx is not None and frame_idx != self._current_pos:
                self._seek(frame_idx)
            frame = self._read_next()
            self._current_pos += 1
            return frame

    def get_metadata(self) -> dict:
        """
        Return a copy of video metadata dict.

        Keys: width, height, fps, total_frames, duration_s, backend.
        """
        return dict(self._metadata)

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def current_position(self) -> int:
        return self._current_pos

    # ------------------------------------------------------------------
    # Backend: torchaudio StreamReader (NVDEC)
    # ------------------------------------------------------------------

    def _try_torchaudio(self) -> bool:
        try:
            import torchaudio

            # Quick probe — raises if file unreadable
            sr_probe = torchaudio.io.StreamReader(self.path)
            sr_probe.add_basic_video_stream(frames_per_chunk=1, format="rgb24")
            chunks = list(sr_probe.stream())
            if not chunks or chunks[0][0] is None:
                return False

            # Build production reader with NVDEC
            sr = torchaudio.io.StreamReader(self.path)
            sr.add_basic_video_stream(
                frames_per_chunk = 1,
                format           = "rgb24",
                hw_accel         = "cuda:0",
            )

            self._sr      = sr
            self._backend = "torchaudio_nvdec"
            self._metadata = self._probe_metadata_cv2("torchaudio_nvdec")
            self._current_pos = 0
            return True

        except Exception as exc:
            logger.debug("torchaudio backend unavailable: %s", exc)
            return False

    def _seek_torchaudio(self, frame_idx: int) -> None:
        fps = self._metadata.get("fps", 30.0)
        ts  = max(0.0, (frame_idx - 0.5) / fps)   # half-frame back to land on idx
        self._sr.seek(ts)
        self._current_pos = frame_idx

    def _read_next_torchaudio(self) -> np.ndarray:
        for chunk in self._sr.stream():
            t = chunk[0]
            if t is None:
                raise EOFError("torchaudio StreamReader exhausted.")
            # t: [1, H, W, 3] uint8 RGB (on CPU or CUDA depending on hw_accel)
            frame_rgb = t[0].cpu().numpy()
            return frame_rgb[..., ::-1].copy()   # RGB → BGR
        raise EOFError("torchaudio StreamReader exhausted.")

    # ------------------------------------------------------------------
    # Backend: decord VideoReader (GPU)
    # ------------------------------------------------------------------

    def _try_decord(self) -> bool:
        try:
            import decord
            decord.bridge.set_bridge("numpy")

            vr = decord.VideoReader(self.path, ctx=decord.gpu(0))
            # Test random access
            _ = vr[0].asnumpy()

            self._vr      = vr
            self._backend = "decord_gpu"
            self._metadata = self._probe_metadata_cv2("decord_gpu")
            self._current_pos = 0
            return True

        except Exception as exc:
            logger.debug("decord backend unavailable: %s", exc)
            return False

    def _seek_decord(self, frame_idx: int) -> None:
        # decord supports direct indexing; track position manually
        self._current_pos = frame_idx

    def _read_next_decord(self) -> np.ndarray:
        total = len(self._vr)
        if self._current_pos >= total:
            raise EOFError("decord: past last frame.")
        frame = self._vr[self._current_pos]   # decord NDArray RGB uint8
        if hasattr(frame, "asnumpy"):
            frame_rgb = frame.asnumpy()
        elif hasattr(frame, "numpy"):
            frame_rgb = frame.numpy()
        else:
            frame_rgb = np.asarray(frame)
        return frame_rgb[..., ::-1].copy()    # RGB → BGR

    # ------------------------------------------------------------------
    # Backend: cv2.VideoCapture
    # ------------------------------------------------------------------

    def _try_cv2(self, flag: int, name: str) -> bool:
        try:
            cap = cv2.VideoCapture(self.path, flag) if flag != 0 \
                  else cv2.VideoCapture(self.path)

            if not cap.isOpened():
                cap.release()
                return False

            ret, frame = cap.read()
            if not ret or frame is None:
                cap.release()
                return False

            # Rewind for actual use
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0.0)

            self._cap         = cap
            self._backend     = name
            self._metadata    = self._cv2_metadata(cap, name)
            self._current_pos = 0
            return True

        except Exception as exc:
            logger.debug("cv2 backend '%s' failed: %s", name, exc)
            return False

    def _seek_cv2(self, frame_idx: int) -> None:
        ok = self._cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
        if not ok:
            logger.warning(
                "cv2 seek to frame %d may have failed (set() returned False).",
                frame_idx,
            )
        self._current_pos = frame_idx

    def _read_next_cv2(self) -> np.ndarray:
        ret, frame = self._cap.read()
        if not ret or frame is None:
            raise EOFError("cv2: end of video stream.")
        return frame   # already BGR uint8

    # ------------------------------------------------------------------
    # Dispatch (called inside the lock)
    # ------------------------------------------------------------------

    def _seek(self, frame_idx: int) -> None:
        if "cv2" in self._backend:
            self._seek_cv2(frame_idx)
        elif self._backend == "decord_gpu":
            self._seek_decord(frame_idx)
        elif self._backend == "torchaudio_nvdec":
            self._seek_torchaudio(frame_idx)

    def _read_next(self) -> np.ndarray:
        if "cv2" in self._backend:
            return self._read_next_cv2()
        if self._backend == "decord_gpu":
            return self._read_next_decord()
        if self._backend == "torchaudio_nvdec":
            return self._read_next_torchaudio()
        raise RuntimeError(f"Unknown backend: '{self._backend}'.")

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    def _probe_metadata_cv2(self, backend_name: str) -> dict:
        """Open a throwaway cv2 capture just to read metadata."""
        cap = cv2.VideoCapture(self.path, cv2.CAP_MSMF)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.path)
        meta = self._cv2_metadata(cap, backend_name)
        cap.release()
        return meta

    @staticmethod
    def _cv2_metadata(cap: cv2.VideoCapture, backend: str) -> dict:
        fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return {
            "width":        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height":       int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps":          fps,
            "total_frames": total,
            "duration_s":   total / fps if fps > 0 else 0.0,
            "backend":      backend,
        }


# ---------------------------------------------------------------------------
# VideoWriter
# ---------------------------------------------------------------------------

class VideoWriter:
    """
    Video output writer with NVENC → libx264 → cv2 fallback.

    Frames are accepted as BGR uint8 numpy arrays and written to disk.
    All backends produce a playable MP4 file.  NVENC provides the fastest
    encode and smallest file size for GPU-equipped machines.

    Usage (context manager — preferred)
    ------------------------------------
    with VideoWriter(output_path, width, height, fps) as writer:
        for frame_bgr in frames:
            writer.write_frame(frame_bgr)
    # file is flushed and closed on __exit__

    Usage (manual)
    --------------
    writer = VideoWriter(output_path, width, height, fps)
    writer.open()
    writer.write_frame(frame_bgr)
    writer.close()
    """

    def __init__(
        self,
        output_path: str,
        width:       int,
        height:      int,
        fps:         float,
    ) -> None:
        self.output_path = output_path
        self.width       = width
        self.height      = height
        self.fps         = float(fps)

        self._backend:         str  = "none"
        self._ffmpeg_proc            = None    # subprocess.Popen
        self._cv2_writer             = None    # cv2.VideoWriter
        self._lock               = threading.Lock()
        self._frame_count: int   = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> None:
        """
        Open the output file.  Tries NVENC → libx264 → cv2 in order.
        Raises RuntimeError if the output file cannot be created.
        """
        parent = os.path.dirname(os.path.abspath(self.output_path))
        if parent:
            os.makedirs(parent, exist_ok=True)

        # Still-image output — no encoder involved; each frame is written
        # straight to disk by write_frame().
        if is_image_path(self.output_path):
            self._backend = "cv2_imwrite"
            logger.info("VideoWriter: cv2.imwrite (still image output)")
            return

        if self._probe_nvenc():
            if self._try_ffmpeg("h264_nvenc", extra_opts={"preset": "p4", "rc": "vbr", "cq": "23"}):
                logger.info("VideoWriter: ffmpeg h264_nvenc (NVENC GPU encode)")
                return

        if self._try_ffmpeg("libx264", extra_opts={"preset": "fast", "crf": "22"}):
            logger.info("VideoWriter: ffmpeg libx264 (CPU encode)")
            return

        if self._try_cv2():
            logger.info("VideoWriter: cv2.VideoWriter (CPU encode, mp4v)")
            return

        raise RuntimeError(
            f"Could not open output file '{self.output_path}' with any "
            "available encoder.  Ensure ffmpeg is on PATH or OpenCV has "
            "mp4v codec support."
        )

    def write_frame(self, frame_bgr: np.ndarray) -> None:
        """
        Write a single BGR uint8 frame.
        Thread-safe.  Resizes frame if dimensions do not match configuration.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return

        h, w = frame_bgr.shape[:2]
        if w != self.width or h != self.height:
            frame_bgr = cv2.resize(frame_bgr, (self.width, self.height),
                                   interpolation=cv2.INTER_LINEAR)

        with self._lock:
            if self._backend == "none":
                raise RuntimeError("Call open() before write_frame().")
            if "ffmpeg" in self._backend:
                self._write_ffmpeg(frame_bgr)
            elif self._backend == "cv2_mp4v":
                self._cv2_writer.write(frame_bgr)
            elif self._backend == "cv2_imwrite":
                self._write_image(frame_bgr)
            self._frame_count += 1

    def close(self) -> None:
        """Flush encoder, wait for subprocess to finish, release resources."""
        with self._lock:
            if self._ffmpeg_proc is not None:
                try:
                    self._ffmpeg_proc.stdin.close()
                    self._ffmpeg_proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    logger.warning(
                        "ffmpeg did not exit within 60 s after %d frames; "
                        "terminating.", self._frame_count
                    )
                    self._ffmpeg_proc.terminate()
                    self._ffmpeg_proc.wait()
                except Exception as exc:
                    logger.warning("ffmpeg close error: %s", exc)
                finally:
                    self._ffmpeg_proc = None

            if self._cv2_writer is not None:
                self._cv2_writer.release()
                self._cv2_writer = None

        logger.info(
            "VideoWriter closed: %d frames → '%s' (%s).",
            self._frame_count, self.output_path, self._backend,
        )
        self._backend     = "none"
        self._frame_count = 0

    def __enter__(self) -> "VideoWriter":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def backend(self) -> str:
        return self._backend

    # ------------------------------------------------------------------
    # Backend: ffmpeg subprocess
    # ------------------------------------------------------------------

    def _try_ffmpeg(self, vcodec: str, extra_opts: dict) -> bool:
        """
        Launch an ffmpeg subprocess with the given video codec.
        Frames are piped as raw BGR24 via stdin.
        """
        try:
            import ffmpeg as _ffmpeg

            out_kwargs = {
                "vcodec":  vcodec,
                "pix_fmt": "yuv420p",
                **extra_opts,
            }

            process = (
                _ffmpeg
                .input(
                    "pipe:",
                    format    = "rawvideo",
                    pix_fmt   = "bgr24",
                    s         = f"{self.width}x{self.height}",
                    framerate = self.fps,
                )
                .output(self.output_path, **out_kwargs)
                .overwrite_output()
                .run_async(pipe_stdin=True, quiet=True)
            )

            self._ffmpeg_proc = process
            self._backend     = f"ffmpeg_{vcodec}"
            return True

        except Exception as exc:
            logger.debug("ffmpeg %s failed: %s", vcodec, exc)
            self._ffmpeg_proc = None
            return False

    def _write_ffmpeg(self, frame_bgr: np.ndarray) -> None:
        """
        Write raw frame bytes to the ffmpeg stdin pipe.
        BrokenPipeError is caught so a downstream ffmpeg crash does not
        propagate to the GPU pipeline thread.
        """
        try:
            self._ffmpeg_proc.stdin.write(frame_bgr.tobytes())
        except BrokenPipeError:
            logger.error(
                "ffmpeg pipe closed unexpectedly after %d frames. "
                "Output file may be truncated.",
                self._frame_count,
            )
        except Exception as exc:
            logger.error("ffmpeg write error: %s", exc)

    # ------------------------------------------------------------------
    # Backend: still image (cv2.imwrite)
    # ------------------------------------------------------------------

    def _write_image(self, frame_bgr: np.ndarray) -> None:
        """
        Write one composited frame to disk as a still image.

        The first frame takes the requested output path.  A still-image
        source yields exactly one frame, so subsequent frames only appear
        if this writer was pointed at an image path by mistake — those get
        a numeric suffix rather than silently overwriting the first.
        """
        if self._frame_count == 0:
            path = self.output_path
        else:
            stem, ext = os.path.splitext(self.output_path)
            path = f"{stem}_{self._frame_count:05d}{ext}"

        if not cv2.imwrite(path, frame_bgr):
            logger.error("cv2.imwrite failed for '%s'.", path)

    # ------------------------------------------------------------------
    # Backend: cv2.VideoWriter
    # ------------------------------------------------------------------

    def _try_cv2(self) -> bool:
        """
        Try several FourCC codecs in order of preference.

        'mp4v' (MPEG-4 Part 2) is first because it is encoded by OpenCV's
        bundled FFmpeg with no external DLL dependency, so it always works.
        'avc1' (H.264) is tried only as an upgrade — but ONLY if it both opens
        AND survives a probe write.  On Windows without the openh264 runtime DLL,
        cv2.VideoWriter('avc1') reports isOpened()==True yet silently encodes
        nothing, producing an empty/corrupt file; the probe-write detects that.
        """
        for fourcc_str in ("mp4v", "avc1", "XVID"):
            try:
                fourcc  = cv2.VideoWriter_fourcc(*fourcc_str)
                writer  = cv2.VideoWriter(
                    self.output_path,
                    fourcc,
                    self.fps,
                    (self.width, self.height),
                )
                if writer.isOpened():
                    self._cv2_writer = writer
                    self._backend    = "cv2_mp4v"
                    logger.info("cv2.VideoWriter using fourcc '%s'", fourcc_str)
                    return True
                writer.release()
            except Exception as exc:
                logger.debug("cv2 VideoWriter fourcc '%s' failed: %s", fourcc_str, exc)

        return False

    # ------------------------------------------------------------------
    # NVENC availability probe
    # ------------------------------------------------------------------

    @staticmethod
    def _probe_nvenc() -> bool:
        """
        Query the system ffmpeg for h264_nvenc encoder support.
        Returns False if ffmpeg is not on PATH or NVENC is unavailable.
        Result is not cached — call once at open() time.
        """
        try:
            result = subprocess.run(
                ["ffmpeg", "-hide_banner", "-encoders"],
                capture_output = True,
                text           = True,
                timeout        = 8,
            )
            return "h264_nvenc" in result.stdout
        except FileNotFoundError:
            logger.debug("ffmpeg not found on PATH; NVENC unavailable.")
            return False
        except Exception as exc:
            logger.debug("NVENC probe failed: %s", exc)
            return False
