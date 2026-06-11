"""
OptimisedOneShot — GUI Module
=============================
Contains every Qt class needed by the application:

  VideoCanvas          — QGraphicsView with letterboxed video, point-prompt
                         interaction, and mask overlay rendering.
  TimelinePanel        — Frame scrubber, play/pause, progress bar, FPS label.
  ControlPanel         — Sidebar: model paths, prompt mode, registration,
                         threshold/alpha/stride controls, export.
  RegistrationThread   — QThread that runs heavy SAM2 + ReID embedding in
                         one shot, then unloads models from VRAM.
  VideoReaderThread    — QThread driving cv2.VideoCapture; dual-mode: preview
                         (single-frame seek) and tracking (continuous feed into
                         raw_frame_queue).
  FrameDisplayWorker   — QThread that drains display_queue from the GPU
                         pipeline process and emits composited frames as Qt
                         signals for VideoCanvas to render.
  MainWindow           — Top-level orchestrator: wires all threads/processes,
                         manages state machine, routes signals.

Threading contract
------------------
* RegistrationThread, VideoReaderThread, FrameDisplayWorker all live in
  non-main threads and emit signals that are automatically queued to the main
  thread (Qt.AutoConnection → Qt.QueuedConnection cross-thread).
* No torch or CUDA calls occur in the main Qt thread. All GPU work is
  delegated to RegistrationThread (Phase 1, once) or the spawned
  GPUPipelineProcess (Phase 2, continuous).
* cv2.VideoCapture is owned exclusively by VideoReaderThread; no other thread
  touches it.
"""

from __future__ import annotations

import os
import queue
import threading
import time
import traceback
from typing import Any, NamedTuple, Optional

import cv2
import numpy as np

from qtpy.QtCore import (
    QPointF,
    QRectF,
    QSize,
    Qt,
    QTimer,
    Signal,
    QThread,
)
from qtpy.QtGui import (
    QBrush,
    QColor,
    QCursor,
    QFont,
    QImage,
    QPainter,
    QPen,
    QPixmap,
)
from qtpy.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QStyle,
    QVBoxLayout,
    QWidget,
)


# ---------------------------------------------------------------------------
# Shared command type
# ---------------------------------------------------------------------------

class Cmd(NamedTuple):
    """Lightweight command message routed through in-process threading.Queues."""
    type: str       # SEEK | PLAY | PAUSE | START_TRACKING | STOP_TRACKING | STOP
    payload: Any = None


# ---------------------------------------------------------------------------
# VideoCanvas
# ---------------------------------------------------------------------------

class VideoCanvas(QGraphicsView):
    """
    Interactive video viewport rendered inside a QGraphicsScene.

    The scene coordinate system is set to exactly (0, 0, video_w, video_h) so
    that mapToScene(click_pos) returns original video pixel coordinates
    directly — no manual scale/offset arithmetic needed.

    Left-click  → positive point prompt (green, label=1).
    Right-click → negative point prompt (red,   label=0).

    Signals
    -------
    point_added(x_vid, y_vid, label)
        Emitted on each click when mode == 'registration'.
    """

    point_added = Signal(float, float, int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)

        self._vid_w: int = 1280
        self._vid_h: int = 720
        self._mode: str = "idle"          # 'idle' | 'registration' | 'tracking'
        self._prompt_points: list[tuple[float, float, int]] = []
        self._current_pixmap: Optional[QPixmap] = None
        self._mask_overlay: Optional[np.ndarray] = None   # H×W uint8

        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QBrush(QColor(15, 15, 15)))
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_video_size(self, width: int, height: int) -> None:
        """Configure scene rect for the loaded video resolution."""
        self._vid_w = width
        self._vid_h = height
        self._scene.setSceneRect(0.0, 0.0, float(width), float(height))
        self.fitInView(QRectF(0.0, 0.0, float(width), float(height)),
                       Qt.KeepAspectRatio)

    def set_mode(self, mode: str) -> None:
        """Switch interaction mode: 'idle' | 'registration' | 'tracking'."""
        self._mode = mode
        self.setCursor(Qt.CrossCursor if mode == "registration" else Qt.ArrowCursor)

    def load_frame(self, frame_bgr: np.ndarray) -> None:
        """
        Display a raw BGR numpy frame (H×W×3 uint8) with the current overlay.
        Called from the main thread (preview / seek path).
        """
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        qimg = QImage(frame_rgb.data.tobytes(), w, h, 3 * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self._current_pixmap = pixmap
        self._render_with_overlay(pixmap)

    def load_composited_frame(self, raw_bytes: bytes, w: int, h: int) -> None:
        """
        Display a pre-composited RGB frame from the GPU pipeline.
        Called from the main thread via a queued signal.
        No overlay painting needed — blending was done on GPU.
        """
        qimg = QImage(raw_bytes, w, h, 3 * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg.copy())   # .copy() detaches from the bytes buffer
        self._pixmap_item.setPixmap(pixmap)

    def set_mask_overlay(self, mask: Optional[np.ndarray]) -> None:
        """
        Set a SAM2 registration mask (H×W uint8) to render as a semi-transparent
        overlay. Pass None to clear.
        """
        self._mask_overlay = mask
        if self._current_pixmap is not None:
            self._render_with_overlay(self._current_pixmap)

    def clear_prompts(self) -> None:
        """Remove all prompt points and the mask overlay."""
        self._prompt_points.clear()
        self._mask_overlay = None
        if self._current_pixmap is not None:
            self._render_with_overlay(self._current_pixmap)

    def get_prompt_points(self) -> list[tuple[float, float, int]]:
        """Return a copy of the current prompt points [(x, y, label), ...]."""
        return list(self._prompt_points)

    # ------------------------------------------------------------------
    # Internal rendering
    # ------------------------------------------------------------------

    def _render_with_overlay(self, base_pixmap: QPixmap) -> None:
        """Paint mask overlay and prompt dots onto a copy of base_pixmap."""
        if not self._mask_overlay is not None and not self._prompt_points:
            self._pixmap_item.setPixmap(base_pixmap)
            return

        result = base_pixmap.copy()
        painter = QPainter(result)
        painter.setRenderHint(QPainter.Antialiasing)

        # --- Mask overlay -------------------------------------------------
        if self._mask_overlay is not None:
            mask = self._mask_overlay
            mh, mw = mask.shape[:2]
            overlay_rgba = np.zeros((mh, mw, 4), dtype=np.uint8)
            overlay_rgba[mask > 0] = [0, 210, 80, 110]   # green, ~43% opacity
            overlay_img = QImage(
                overlay_rgba.tobytes(), mw, mh, 4 * mw, QImage.Format_RGBA8888
            )
            painter.drawImage(0, 0, overlay_img)

            # Contour outline
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            pen = QPen(QColor(0, 255, 100), 2.0)
            pen.setCosmetic(True)
            painter.setPen(pen)
            for contour in contours:
                pts = [QPointF(float(p[0][0]), float(p[0][1])) for p in contour]
                n = len(pts)
                if n > 1:
                    for i in range(n - 1):
                        painter.drawLine(pts[i], pts[i + 1])
                    painter.drawLine(pts[-1], pts[0])

        # --- Prompt dots --------------------------------------------------
        for x_vid, y_vid, label in self._prompt_points:
            fill = QColor(0, 210, 0) if label == 1 else QColor(220, 30, 30)
            painter.setBrush(QBrush(fill))
            painter.setPen(QPen(QColor(255, 255, 255), 1.5))
            painter.drawEllipse(QPointF(x_vid, y_vid), 6.0, 6.0)
            # Inner dot for clarity
            painter.setBrush(QBrush(QColor(255, 255, 255)))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPointF(x_vid, y_vid), 2.0, 2.0)

        painter.end()
        self._pixmap_item.setPixmap(result)

    # ------------------------------------------------------------------
    # Event overrides
    # ------------------------------------------------------------------

    def mousePressEvent(self, event) -> None:
        if self._mode != "registration":
            super().mousePressEvent(event)
            return
        if event.button() not in (Qt.LeftButton, Qt.RightButton):
            super().mousePressEvent(event)
            return

        scene_pt = self.mapToScene(event.pos())
        x_vid = max(0.0, min(float(scene_pt.x()), float(self._vid_w - 1)))
        y_vid = max(0.0, min(float(scene_pt.y()), float(self._vid_h - 1)))
        label = 1 if event.button() == Qt.LeftButton else 0

        self._prompt_points.append((x_vid, y_vid, label))
        if self._current_pixmap is not None:
            self._render_with_overlay(self._current_pixmap)

        self.point_added.emit(x_vid, y_vid, label)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.fitInView(
            QRectF(0.0, 0.0, float(self._vid_w), float(self._vid_h)),
            Qt.KeepAspectRatio,
        )


# ---------------------------------------------------------------------------
# TimelinePanel
# ---------------------------------------------------------------------------

class TimelinePanel(QWidget):
    """
    Bottom bar: frame scrubber, play/pause, FPS readout, export progress bar.
    """

    seek_requested  = Signal(int)   # emitted when user moves scrubber
    play_requested  = Signal()
    pause_requested = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._total_frames = 1
        self._playing = False
        self._scrubbing = False          # true while slider is being dragged
        self._seek_timer = QTimer()
        self._seek_timer.setSingleShot(True)
        self._seek_timer.timeout.connect(self._emit_seek)
        self._pending_seek_value = 0

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 6)
        layout.setSpacing(4)

        # --- Scrubber row -------------------------------------------------
        scrub_row = QHBoxLayout()

        self.frame_label = QLabel("0 / 0")
        self.frame_label.setFixedWidth(100)
        self.frame_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        scrub_row.addWidget(self.frame_label)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.setValue(0)
        self.slider.setEnabled(False)
        self.slider.sliderPressed.connect(self._on_slider_pressed)
        self.slider.sliderReleased.connect(self._on_slider_released)
        self.slider.valueChanged.connect(self._on_slider_value_changed)
        scrub_row.addWidget(self.slider, stretch=1)

        self.fps_label = QLabel("-- FPS")
        self.fps_label.setFixedWidth(70)
        self.fps_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        scrub_row.addWidget(self.fps_label)

        layout.addLayout(scrub_row)

        # --- Controls row -------------------------------------------------
        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(6)

        self.play_btn = QPushButton("▶  Play")
        self.play_btn.setFixedWidth(90)
        self.play_btn.setEnabled(False)
        self.play_btn.clicked.connect(self._on_play_pause)
        ctrl_row.addWidget(self.play_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Exporting… %p%")
        self.progress_bar.setVisible(False)
        ctrl_row.addWidget(self.progress_bar, stretch=1)

        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        ctrl_row.addWidget(self.status_label, stretch=1)

        layout.addLayout(ctrl_row)

        self.setFixedHeight(72)

    # ------------------------------------------------------------------
    # Public slots (called from MainWindow, safe from any thread via signals)
    # ------------------------------------------------------------------

    def set_video_info(self, total_frames: int, fps: float) -> None:
        self._total_frames = max(1, total_frames)
        self.slider.setMaximum(self._total_frames - 1)
        self.slider.setEnabled(True)
        self.play_btn.setEnabled(True)
        self.frame_label.setText(f"0 / {self._total_frames - 1}")

    def update_position(self, frame_idx: int) -> None:
        """Update slider and label without triggering seek_requested."""
        self._scrubbing = True
        self.slider.setValue(frame_idx)
        self._scrubbing = False
        self.frame_label.setText(f"{frame_idx} / {self._total_frames - 1}")

    def update_fps(self, fps: float) -> None:
        self.fps_label.setText(f"{fps:.1f} FPS")

    def show_export_progress(self, visible: bool) -> None:
        self.progress_bar.setVisible(visible)
        if visible:
            self.progress_bar.setValue(0)

    def set_export_progress(self, pct: int) -> None:
        self.progress_bar.setValue(pct)

    def set_playing(self, playing: bool) -> None:
        self._playing = playing
        self.play_btn.setText("⏸  Pause" if playing else "▶  Play")

    # ------------------------------------------------------------------
    # Internal slots
    # ------------------------------------------------------------------

    def _on_slider_pressed(self) -> None:
        self._scrubbing = True

    def _on_slider_released(self) -> None:
        self._scrubbing = False
        self._emit_seek()

    def _on_slider_value_changed(self, value: int) -> None:
        self.frame_label.setText(f"{value} / {self._total_frames - 1}")
        if not self._scrubbing:
            # Debounce rapid changes (e.g. keyboard arrow keys)
            self._pending_seek_value = value
            self._seek_timer.start(40)

    def _emit_seek(self) -> None:
        self.seek_requested.emit(self._pending_seek_value)

    def _on_play_pause(self) -> None:
        if self._playing:
            self.pause_requested.emit()
        else:
            self.play_requested.emit()


# ---------------------------------------------------------------------------
# ControlPanel
# ---------------------------------------------------------------------------

class ControlPanel(QWidget):
    """
    Left sidebar housing all user controls.

    Signals are emitted instead of acting directly so that MainWindow can
    maintain the application state machine.
    """

    load_video_requested  = Signal(str)    # absolute file path
    register_requested    = Signal()
    track_start_requested = Signal()
    track_stop_requested  = Signal()
    export_requested      = Signal()
    clear_points_requested = Signal()

    sam_weights_changed    = Signal(str)
    fastsam_weights_changed = Signal(str)
    reid_weights_changed   = Signal(str)

    threshold_changed = Signal(float)   # match threshold 0.50–0.99
    alpha_changed     = Signal(float)   # overlay opacity 0.10–0.90
    stride_changed    = Signal(int)     # inference stride (every N frames)
    live_preview_toggled = Signal(bool)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFixedWidth(290)
        self._build_ui()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def set_register_enabled(self, enabled: bool) -> None:
        self.register_btn.setEnabled(enabled)

    def set_track_controls_enabled(self, enabled: bool) -> None:
        self.track_btn.setEnabled(enabled)
        self.stop_btn.setEnabled(enabled)
        self.export_btn.setEnabled(enabled)

    def set_tracking_active(self, active: bool) -> None:
        self.track_btn.setEnabled(not active)
        self.stop_btn.setEnabled(active)

    def get_sam_weights(self) -> str:
        return self.sam_path_edit.text().strip()

    def get_fastsam_weights(self) -> str:
        return self.fastsam_path_edit.text().strip()

    def get_reid_weights(self) -> str:
        return self.reid_path_edit.text().strip()

    def get_prompt_mode(self) -> int:
        """Returns 1 for positive, 0 for negative."""
        return 1 if self.pos_radio.isChecked() else 0

    def get_threshold(self) -> float:
        return self.threshold_spin.value()

    def get_alpha(self) -> float:
        return self.alpha_spin.value()

    def get_stride(self) -> int:
        return self.stride_spin.value()

    def get_live_preview(self) -> bool:
        return self.live_preview_check.isChecked()

    # ------------------------------------------------------------------
    # Build UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # Title
        title = QLabel("OptimisedOneShot")
        title.setFont(QFont("Segoe UI", 11, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        root.addWidget(title)

        # ── Video ────────────────────────────────────────────────────
        root.addWidget(self._make_separator("Video"))
        self.load_btn = QPushButton("Load Video…")
        self.load_btn.clicked.connect(self._on_load_clicked)
        root.addWidget(self.load_btn)

        self.video_label = QLabel("No video loaded.")
        self.video_label.setWordWrap(True)
        self.video_label.setStyleSheet("color: #888; font-size: 10px;")
        root.addWidget(self.video_label)

        # ── Model Weights ─────────────────────────────────────────────
        root.addWidget(self._make_separator("Model Weights"))

        # SAM2
        root.addWidget(QLabel("SAM2 Checkpoint:"))
        sam_row = QHBoxLayout()
        self.sam_path_edit = QLineEdit()
        self.sam_path_edit.setPlaceholderText("sam2.1_hiera_large.pt")
        self.sam_path_edit.textChanged.connect(self.sam_weights_changed)
        sam_row.addWidget(self.sam_path_edit)
        sam_browse = QPushButton("…")
        sam_browse.setFixedWidth(28)
        sam_browse.clicked.connect(lambda: self._browse_weight(self.sam_path_edit))
        sam_row.addWidget(sam_browse)
        root.addLayout(sam_row)

        # FastSAM
        root.addWidget(QLabel("FastSAM Weights:"))
        fsam_row = QHBoxLayout()
        self.fastsam_path_edit = QLineEdit()
        self.fastsam_path_edit.setPlaceholderText("FastSAM-s.pt")
        self.fastsam_path_edit.textChanged.connect(self.fastsam_weights_changed)
        fsam_row.addWidget(self.fastsam_path_edit)
        fsam_browse = QPushButton("…")
        fsam_browse.setFixedWidth(28)
        fsam_browse.clicked.connect(lambda: self._browse_weight(self.fastsam_path_edit))
        fsam_row.addWidget(fsam_browse)
        root.addLayout(fsam_row)

        # Re-ID
        root.addWidget(QLabel("Re-ID Weights:"))
        reid_row = QHBoxLayout()
        self.reid_path_edit = QLineEdit()
        self.reid_path_edit.setPlaceholderText("osnet_x0_25_imagenet.pth (optional)")
        self.reid_path_edit.textChanged.connect(self.reid_weights_changed)
        reid_row.addWidget(self.reid_path_edit)
        reid_browse = QPushButton("…")
        reid_browse.setFixedWidth(28)
        reid_browse.clicked.connect(lambda: self._browse_weight(self.reid_path_edit))
        reid_row.addWidget(reid_browse)
        root.addLayout(reid_row)

        # ── Point Prompts ─────────────────────────────────────────────
        root.addWidget(self._make_separator("Point Prompts"))

        prompt_row = QHBoxLayout()
        self.pos_radio = QRadioButton("✚ Positive")
        self.neg_radio = QRadioButton("✖ Negative")
        self.pos_radio.setChecked(True)
        prompt_grp = QButtonGroup(self)
        prompt_grp.addButton(self.pos_radio, 1)
        prompt_grp.addButton(self.neg_radio, 0)
        prompt_row.addWidget(self.pos_radio)
        prompt_row.addWidget(self.neg_radio)
        root.addLayout(prompt_row)

        self.clear_pts_btn = QPushButton("Clear Points")
        self.clear_pts_btn.setEnabled(False)
        self.clear_pts_btn.clicked.connect(self.clear_points_requested)
        root.addWidget(self.clear_pts_btn)

        # ── Registration ──────────────────────────────────────────────
        root.addWidget(self._make_separator("Registration"))

        self.register_btn = QPushButton("Register Target (SAM2)")
        self.register_btn.setEnabled(False)
        self.register_btn.setStyleSheet(
            "QPushButton { background: #1a6b3c; color: white; font-weight: bold; }"
            "QPushButton:hover { background: #22894d; }"
            "QPushButton:disabled { background: #333; color: #666; }"
        )
        self.register_btn.clicked.connect(self.register_requested)
        root.addWidget(self.register_btn)

        # ── Tracking Parameters ───────────────────────────────────────
        root.addWidget(self._make_separator("Tracking Parameters"))

        # Similarity threshold
        thr_row = QHBoxLayout()
        thr_row.addWidget(QLabel("Match Threshold:"))
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.50, 0.99)
        self.threshold_spin.setSingleStep(0.01)
        self.threshold_spin.setValue(0.85)
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.valueChanged.connect(self.threshold_changed)
        thr_row.addWidget(self.threshold_spin)
        root.addLayout(thr_row)

        # Overlay alpha
        alp_row = QHBoxLayout()
        alp_row.addWidget(QLabel("Overlay Alpha:"))
        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.10, 0.90)
        self.alpha_spin.setSingleStep(0.05)
        self.alpha_spin.setValue(0.50)
        self.alpha_spin.setDecimals(2)
        self.alpha_spin.valueChanged.connect(self.alpha_changed)
        alp_row.addWidget(self.alpha_spin)
        root.addLayout(alp_row)

        # Inference stride
        str_row = QHBoxLayout()
        str_row.addWidget(QLabel("Infer Every N Frames:"))
        self.stride_spin = QSpinBox()
        self.stride_spin.setRange(1, 30)
        self.stride_spin.setValue(1)
        self.stride_spin.setToolTip(
            "Run full segmentation every N frames. "
            "Intermediate frames use tracker propagation."
        )
        self.stride_spin.valueChanged.connect(self.stride_changed)
        str_row.addWidget(self.stride_spin)
        root.addLayout(str_row)

        # Live preview toggle
        self.live_preview_check = QCheckBox("Live Preview Mode")
        self.live_preview_check.setChecked(True)
        self.live_preview_check.toggled.connect(self.live_preview_toggled)
        root.addWidget(self.live_preview_check)

        # ── Execution ─────────────────────────────────────────────────
        root.addWidget(self._make_separator("Execution"))

        self.track_btn = QPushButton("▶  Start Live Preview")
        self.track_btn.setEnabled(False)
        self.track_btn.setStyleSheet(
            "QPushButton { background: #1a4b8c; color: white; font-weight: bold; }"
            "QPushButton:hover { background: #1f5aa8; }"
            "QPushButton:disabled { background: #333; color: #666; }"
        )
        self.track_btn.clicked.connect(self.track_start_requested)
        root.addWidget(self.track_btn)

        self.stop_btn = QPushButton("■  Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.track_stop_requested)
        root.addWidget(self.stop_btn)

        self.export_btn = QPushButton("⬇  Export Rendered Video…")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self.export_requested)
        root.addWidget(self.export_btn)

        # ── Status ────────────────────────────────────────────────────
        root.addStretch(1)
        self.status_label = QLabel("Load a video to begin.")
        self.status_label.setWordWrap(True)
        self.status_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.status_label.setStyleSheet("color: #aaa; font-size: 10px;")
        root.addWidget(self.status_label)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_separator(text: str) -> QLabel:
        lbl = QLabel(f"── {text} ──")
        lbl.setStyleSheet(
            "color: #666; font-size: 9px; letter-spacing: 1px; margin-top: 4px;"
        )
        return lbl

    def _on_load_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Video File",
            "",
            "Video Files (*.mp4 *.mov *.avi *.mkv *.webm);;All Files (*)",
        )
        if path:
            self.video_label.setText(os.path.basename(path))
            self.load_video_requested.emit(path)

    def _browse_weight(self, line_edit: QLineEdit) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Weight File",
            "",
            "Model Weights (*.pt *.pth *.bin);;All Files (*)",
        )
        if path:
            line_edit.setText(path)


# ---------------------------------------------------------------------------
# RegistrationThread
# ---------------------------------------------------------------------------

class RegistrationThread(QThread):
    """
    Runs in a dedicated QThread so the Qt main loop stays responsive during
    heavy SAM2 inference (~3–8 seconds on first run including model load).

    Execution order
    ---------------
    1. Open video at _video_path, seek to _frame_idx, read one frame.
    2. Load HeavySAMRegistrar (SAM2 Large fp16, ~3.5 GB VRAM).
    3. Run register() with the user's point prompts → mask + bbox.
    4. Load ReIDEmbedder (OSNet / ResNet18, ~0.3 GB VRAM).
    5. Crop the target region, extract normalised reference embedding.
    6. Unload SAM2 + ReID embedder (del + empty_cache).
    7. Emit registration_done with all serialised CPU results.
    """

    registration_done = Signal(dict)   # {mask, bbox, reid_emb, score, frame_bgr, frame_idx}
    progress          = Signal(str)
    error             = Signal(str)

    def __init__(
        self,
        video_path: str,
        frame_idx: int,
        points: list[tuple[float, float, int]],
        sam_weights: str,
        fastsam_weights: str,
        reid_weights: str,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._video_path    = video_path
        self._frame_idx     = frame_idx
        self._points        = list(points)   # [(x, y, label), ...]
        self._sam_weights   = sam_weights
        self._fastsam_weights = fastsam_weights
        self._reid_weights  = reid_weights

    def run(self) -> None:
        import torch

        try:
            self._run_inner()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            self.error.emit(
                "GPU Out of Memory during registration.\n"
                "Close other GPU applications and try again, or switch to a "
                "smaller SAM2 checkpoint (e.g. sam2.1_hiera_base_plus.pt)."
            )
        except Exception:
            self.error.emit(
                f"Registration failed:\n{traceback.format_exc()}"
            )

    def _run_inner(self) -> None:
        import torch
        from models import HeavySAMRegistrar, ReIDEmbedder

        pos_pts = [(x, y) for x, y, l in self._points if l == 1]
        if not pos_pts:
            self.error.emit("At least one positive point is required.")
            return

        # ── Read target frame ────────────────────────────────────────
        self.progress.emit("Reading frame…")
        cap = cv2.VideoCapture(self._video_path, cv2.CAP_MSMF)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self._video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, self._frame_idx)
        ret, frame_bgr = cap.read()
        cap.release()

        if not ret or frame_bgr is None:
            self.error.emit(f"Could not read frame {self._frame_idx}.")
            return

        # ── SAM2 registration ────────────────────────────────────────
        self.progress.emit("Loading SAM2 Large (fp16)… this may take a moment.")
        registrar = HeavySAMRegistrar(
            checkpoint=self._sam_weights or None,
            device="cuda",
        )
        registrar.load()

        self.progress.emit("Running SAM2 segmentation…")
        result = registrar.register(frame_bgr, self._points)
        mask  = result["mask"]    # np.uint8  H×W
        bbox  = result["bbox"]    # (x, y, w, h) in pixels
        score = result["score"]   # float

        # ── Re-ID embedding ──────────────────────────────────────────
        self.progress.emit("Extracting reference embedding…")
        embedder = ReIDEmbedder(
            weights_path=self._reid_weights or None,
            device="cuda",
        )
        embedder.load()

        fh, fw = frame_bgr.shape[:2]
        x, y, w, h = bbox
        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(fw, int(x + w))
        y2 = min(fh, int(y + h))
        crop = frame_bgr[y1:y2, x1:x2]

        if crop.size == 0:
            self.error.emit("Registration bounding box collapsed to zero area.")
            embedder.unload()
            registrar.unload()
            return

        reid_emb = embedder.embed(crop)  # np.float32 (D,)

        # ── Unload heavy models ──────────────────────────────────────
        self.progress.emit("Purging SAM2 + Re-ID from VRAM…")
        registrar.unload()
        embedder.unload()

        free_gb = torch.cuda.mem_get_info()[0] / 1e9
        self.progress.emit(
            f"Target registered  ·  {free_gb:.1f} GB VRAM free  ·  "
            f"SAM2 score {score:.3f}"
        )

        self.registration_done.emit({
            "mask":      mask,
            "bbox":      bbox,
            "reid_emb":  reid_emb,
            "score":     score,
            "frame_bgr": frame_bgr.copy(),
            "frame_idx": self._frame_idx,
        })


# ---------------------------------------------------------------------------
# VideoReaderThread
# ---------------------------------------------------------------------------

class VideoReaderThread(QThread):
    """
    Manages cv2.VideoCapture and operates in two modes:

    PREVIEW mode (default)
        Processes SEEK commands from _cmd_q, reads a single frame per command,
        emits preview_frame(frame_idx, frame_bgr).  Used during scrubbing.

    TRACKING mode
        Reads frames continuously and pushes (frame_idx, frame_bgr) tuples into
        raw_frame_queue (mp.Queue) for the GPU Pipeline Process to consume.

    The mode transition is triggered by start_tracking() / stop_tracking()
    which enqueue commands on _cmd_q.
    """

    preview_frame = Signal(int, object)   # (frame_idx, np.ndarray BGR)
    reader_error  = Signal(str)
    end_of_video  = Signal()

    def __init__(
        self,
        video_path: str,
        raw_frame_queue,             # mp.Queue
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._path            = video_path
        self._raw_frame_queue = raw_frame_queue
        self._cmd_q           = queue.Queue()         # threading.Queue (in-process)
        self._stop_event      = threading.Event()

    # ------------------------------------------------------------------
    # Thread-safe control methods (called from main thread)
    # ------------------------------------------------------------------

    def seek(self, frame_idx: int) -> None:
        """Request a single-frame seek while in PREVIEW mode."""
        self._cmd_q.put(Cmd("SEEK", frame_idx))

    def start_tracking(self, start_frame: int) -> None:
        """Switch to TRACKING mode and begin feeding raw_frame_queue."""
        self._cmd_q.put(Cmd("START_TRACKING", start_frame))

    def stop_tracking(self) -> None:
        """Return to PREVIEW mode and send None sentinel to raw_frame_queue."""
        self._cmd_q.put(Cmd("STOP_TRACKING", None))

    def request_stop(self) -> None:
        """Terminate the thread entirely."""
        self._stop_event.set()
        self._cmd_q.put(Cmd("STOP", None))

    # ------------------------------------------------------------------
    # Thread entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        # Open capture with hardware-accelerated MSMF on Windows; fallback to FFMPEG.
        cap = cv2.VideoCapture(self._path, cv2.CAP_MSMF)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self._path)
        if not cap.isOpened():
            self.reader_error.emit(f"Cannot open video: {self._path}")
            return

        fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_delay  = 1.0 / fps
        tracking     = False
        current_idx  = 0

        while not self._stop_event.is_set():
            # ── Drain command queue ──────────────────────────────────
            try:
                cmd = self._cmd_q.get(timeout=0.0 if tracking else 0.05)
            except queue.Empty:
                cmd = None

            if cmd is not None:
                if cmd.type == "STOP":
                    break

                elif cmd.type == "SEEK":
                    idx = int(cmd.payload)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        current_idx = idx
                        self.preview_frame.emit(idx, frame.copy())

                elif cmd.type == "START_TRACKING":
                    idx = int(cmd.payload)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    current_idx = idx
                    tracking = True

                elif cmd.type == "STOP_TRACKING":
                    tracking = False
                    # Signal GPU process to stop consuming
                    try:
                        self._raw_frame_queue.put_nowait(None)
                    except Exception:
                        pass

            # ── TRACKING: push next frame ────────────────────────────
            if tracking:
                t0 = time.monotonic()
                ret, frame = cap.read()
                if not ret:
                    # End of video — send sentinel
                    try:
                        self._raw_frame_queue.put(None, timeout=1.0)
                    except Exception:
                        pass
                    tracking = False
                    self.end_of_video.emit()
                    continue

                try:
                    self._raw_frame_queue.put(
                        (current_idx, frame.copy()), timeout=0.15
                    )
                except queue.Full:
                    pass   # GPU pipeline is behind; drop this frame

                current_idx += 1

                # Pace output to video FPS
                elapsed = time.monotonic() - t0
                sleep_t = frame_delay - elapsed
                if sleep_t > 0.001:
                    time.sleep(sleep_t)

        cap.release()


# ---------------------------------------------------------------------------
# FrameDisplayWorker
# ---------------------------------------------------------------------------

class FrameDisplayWorker(QThread):
    """
    Drains the display_queue produced by the GPU Pipeline Process and emits
    signals to update the VideoCanvas and timeline in the main thread.

    display_queue payload: (raw_bytes: bytes, W: int, H: int,
                            frame_idx: int, meta: dict | None)
    meta keys: bbox, sim_score, accepted, mode, fps_gpu
    """

    frame_ready    = Signal(bytes, int, int, int)    # raw_rgb_bytes, W, H, frame_idx
    metadata_ready = Signal(dict)

    def __init__(self, display_queue, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._display_queue = display_queue
        self._stop_event    = threading.Event()
        self._fps_times: list[float] = []

    def request_stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                item = self._display_queue.get(timeout=0.1)
            except Exception:
                continue

            if item is None:
                break   # sentinel from GPU process

            raw_bytes, W, H, frame_idx, meta = item

            # FPS calculation
            now = time.monotonic()
            self._fps_times.append(now)
            self._fps_times = [t for t in self._fps_times if now - t <= 1.0]
            display_fps = float(len(self._fps_times))

            self.frame_ready.emit(raw_bytes, W, H, frame_idx)

            if meta is None:
                meta = {}
            meta["display_fps"] = display_fps
            self.metadata_ready.emit(meta)


# ---------------------------------------------------------------------------
# MainWindow
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    """
    Top-level application window and state machine.

    States
    ------
    idle          : No video loaded.
    video_loaded  : Video open; user can scrub and add point prompts.
    registered    : Target registered; GPU process ready to launch.
    tracking      : Live preview or batch render running.
    """

    def __init__(self, queues: dict) -> None:
        super().__init__()
        self._queues = queues

        # Application state
        self._video_path:   Optional[str]  = None
        self._total_frames: int            = 0
        self._video_fps:    float          = 30.0
        self._vid_w:        int            = 1280
        self._vid_h:        int            = 720
        self._current_frame_idx: int       = 0
        self._reg_result:   Optional[dict] = None

        # Worker references
        self._reg_thread:    Optional[RegistrationThread]  = None
        self._reader_thread: Optional[VideoReaderThread]   = None
        self._display_worker: Optional[FrameDisplayWorker] = None
        self._gpu_process    = None    # GPUPipelineProcess (imported lazily)

        # FPS tracking (display side)
        self._fps_history: list[float] = []
        self._fps_timer = QTimer()
        self._fps_timer.timeout.connect(self._refresh_fps_label)
        self._fps_timer.start(500)

        # GPU process watchdog
        self._watchdog = QTimer()
        self._watchdog.timeout.connect(self._check_gpu_process)

        self._build_ui()
        self._connect_signals()

        self.setWindowTitle("OptimisedOneShot — Video Object Segmentation")
        self.setMinimumSize(1200, 740)
        self.resize(1400, 860)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Left sidebar ──────────────────────────────────────────────
        self.control = ControlPanel()
        layout.addWidget(self.control)

        # ── Right: canvas + timeline ──────────────────────────────────
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        self.canvas = VideoCanvas()
        right_layout.addWidget(self.canvas, stretch=1)

        self.timeline = TimelinePanel()
        right_layout.addWidget(self.timeline)

        layout.addWidget(right, stretch=1)

        # ── Status bar ────────────────────────────────────────────────
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("Ready — load a video to begin.")

    def _connect_signals(self) -> None:
        cp = self.control
        tl = self.timeline
        cv = self.canvas

        # Control panel → MainWindow
        cp.load_video_requested.connect(self._on_load_video)
        cp.register_requested.connect(self._on_register)
        cp.track_start_requested.connect(self._on_start_tracking)
        cp.track_stop_requested.connect(self._on_stop_tracking)
        cp.export_requested.connect(self._on_export)
        cp.clear_points_requested.connect(self._on_clear_points)
        cp.threshold_changed.connect(self._on_threshold_changed)
        cp.alpha_changed.connect(self._on_alpha_changed)
        cp.stride_changed.connect(self._on_stride_changed)

        # Timeline → MainWindow
        tl.seek_requested.connect(self._on_seek)
        tl.play_requested.connect(self._on_play)
        tl.pause_requested.connect(self._on_pause)

        # Canvas → MainWindow
        cv.point_added.connect(self._on_point_added)

    # ------------------------------------------------------------------
    # Slot: Load Video
    # ------------------------------------------------------------------

    def _on_load_video(self, path: str) -> None:
        self._video_path = path

        cap = cv2.VideoCapture(path, cv2.CAP_MSMF)
        if not cap.isOpened():
            cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            self._show_error(f"Cannot open video:\n{path}")
            return

        self._vid_w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._vid_h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._video_fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Read first frame for immediate display
        ret, frame = cap.read()
        cap.release()

        if not ret:
            self._show_error("Video file appears to be empty or unreadable.")
            return

        self._current_frame_idx = 0
        self.canvas.set_video_size(self._vid_w, self._vid_h)
        self.canvas.clear_prompts()
        self.canvas.set_mode("registration")
        self.canvas.load_frame(frame)

        self.timeline.set_video_info(self._total_frames, self._video_fps)
        self.timeline.update_position(0)

        self.control.set_register_enabled(True)
        self.control.clear_pts_btn.setEnabled(True)
        self.control.set_track_controls_enabled(False)
        self.control.set_status(
            f"{self._vid_w}×{self._vid_h}  ·  {self._video_fps:.2f} fps  ·  "
            f"{self._total_frames} frames"
        )
        self._status_bar.showMessage("Video loaded. Left-click positive / right-click negative points, then Register.")

        # Start reader thread for scrubbing
        self._start_reader_thread()

    # ------------------------------------------------------------------
    # Slot: Seek / Play / Pause (preview mode)
    # ------------------------------------------------------------------

    def _on_seek(self, frame_idx: int) -> None:
        self._current_frame_idx = frame_idx
        if self._reader_thread and self._reader_thread.isRunning():
            self._reader_thread.seek(frame_idx)

    def _on_play(self) -> None:
        if self._reader_thread:
            self._reader_thread.start_tracking(self._current_frame_idx)
            self.timeline.set_playing(True)

    def _on_pause(self) -> None:
        if self._reader_thread:
            self._reader_thread.stop_tracking()
            self.timeline.set_playing(False)

    # ------------------------------------------------------------------
    # Slot: Point Added
    # ------------------------------------------------------------------

    def _on_point_added(self, x: float, y: float, label: int) -> None:
        # Points are stored in canvas; just update UI feedback
        n_pos = sum(1 for _, _, l in self.canvas.get_prompt_points() if l == 1)
        n_neg = sum(1 for _, _, l in self.canvas.get_prompt_points() if l == 0)
        self._status_bar.showMessage(
            f"Prompts: {n_pos} positive, {n_neg} negative  ·  "
            "Right-click for negative points  ·  Click 'Register Target' when ready."
        )

    def _on_clear_points(self) -> None:
        self.canvas.clear_prompts()
        self._status_bar.showMessage("Points cleared.")

    # ------------------------------------------------------------------
    # Slot: Register Target
    # ------------------------------------------------------------------

    def _on_register(self) -> None:
        if not self._video_path:
            return

        points = self.canvas.get_prompt_points()
        if not any(l == 1 for _, _, l in points):
            self._show_error("Add at least one positive point (left-click) before registering.")
            return

        self.control.register_btn.setEnabled(False)
        self.control.set_status("Loading SAM2…")
        self._status_bar.showMessage("Registering target — please wait…")

        self._reg_thread = RegistrationThread(
            video_path    = self._video_path,
            frame_idx     = self._current_frame_idx,
            points        = points,
            sam_weights   = self.control.get_sam_weights(),
            fastsam_weights = self.control.get_fastsam_weights(),
            reid_weights  = self.control.get_reid_weights(),
            parent        = self,
        )
        self._reg_thread.registration_done.connect(self._on_registration_done)
        self._reg_thread.progress.connect(self._on_registration_progress)
        self._reg_thread.error.connect(self._on_registration_error)
        self._reg_thread.start()

    def _on_registration_progress(self, msg: str) -> None:
        self.control.set_status(msg)
        self._status_bar.showMessage(msg)

    def _on_registration_error(self, msg: str) -> None:
        self.control.register_btn.setEnabled(True)
        self.control.set_status("Registration failed.")
        self._show_error(f"Registration failed:\n\n{msg}")

    def _on_registration_done(self, result: dict) -> None:
        self._reg_result = result

        # Show mask overlay on canvas
        self.canvas.set_mask_overlay(result["mask"])
        self.canvas.set_mode("tracking")   # disable further point-clicking

        self.control.register_btn.setEnabled(True)
        self.control.set_track_controls_enabled(True)
        self.control.set_status(
            f"Registered ✓  score={result['score']:.3f}  "
            f"bbox={result['bbox']}"
        )
        self._status_bar.showMessage(
            "Target registered. Click 'Start Live Preview' to begin tracking."
        )

    # ------------------------------------------------------------------
    # Slot: Start / Stop Tracking
    # ------------------------------------------------------------------

    def _on_start_tracking(self) -> None:
        if self._reg_result is None:
            self._show_error("Register a target before starting tracking.")
            return

        self._launch_gpu_process()
        self._launch_display_worker()

        # Tell the reader thread to start feeding raw_frame_queue
        if self._reader_thread and self._reader_thread.isRunning():
            self._reader_thread.start_tracking(self._current_frame_idx)

        self.control.set_tracking_active(True)
        self.timeline.set_playing(True)
        self._status_bar.showMessage("Tracking…")

        # Watchdog: check GPU process health every 5 s
        self._watchdog.start(5000)

    def _on_stop_tracking(self) -> None:
        self._watchdog.stop()

        # Stop the frame reader (sends sentinel to raw_frame_queue)
        if self._reader_thread and self._reader_thread.isRunning():
            self._reader_thread.stop_tracking()

        # Terminate GPU pipeline process
        if self._gpu_process is not None and self._gpu_process.is_alive():
            try:
                self._queues["pipeline_cmd_queue"].put_nowait(
                    _PipelineCmd("STOP", None)
                )
            except Exception:
                pass
            self._gpu_process.join(timeout=3.0)
            if self._gpu_process.is_alive():
                self._gpu_process.terminate()
            self._gpu_process = None

        # Stop display worker
        if self._display_worker and self._display_worker.isRunning():
            self._display_worker.request_stop()
            self._display_worker.wait(2000)

        self.control.set_tracking_active(False)
        self.timeline.set_playing(False)
        self._status_bar.showMessage("Tracking stopped.")

    # ------------------------------------------------------------------
    # Slot: Export
    # ------------------------------------------------------------------

    def _on_export(self) -> None:
        if self._reg_result is None:
            self._show_error("Register a target before exporting.")
            return

        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save Rendered Video", "", "MP4 Video (*.mp4);;All Files (*)"
        )
        if not out_path:
            return
        if not out_path.lower().endswith(".mp4"):
            out_path += ".mp4"

        self.timeline.show_export_progress(True)
        self._status_bar.showMessage("Batch rendering — do not close the application…")

        # Launch GPU process in BATCH_RENDER mode
        self._launch_gpu_process(batch_render=True, output_path=out_path)

        # The GPU process will emit progress updates via display_queue meta dicts.
        self._launch_display_worker()
        if self._reader_thread and self._reader_thread.isRunning():
            self._reader_thread.start_tracking(0)   # always render from frame 0

        self.control.set_tracking_active(True)

    # ------------------------------------------------------------------
    # Threshold / Alpha / Stride updates → GPU process
    # ------------------------------------------------------------------

    def _on_threshold_changed(self, value: float) -> None:
        self._send_pipeline_cmd("UPDATE_THRESHOLD", value)

    def _on_alpha_changed(self, value: float) -> None:
        self._send_pipeline_cmd("UPDATE_ALPHA", value)

    def _on_stride_changed(self, value: int) -> None:
        self._send_pipeline_cmd("UPDATE_STRIDE", value)

    def _send_pipeline_cmd(self, cmd_type: str, payload: Any = None) -> None:
        try:
            self._queues["pipeline_cmd_queue"].put_nowait(
                _PipelineCmd(cmd_type, payload)
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Frame display slots (from FrameDisplayWorker — QueuedConnection)
    # ------------------------------------------------------------------

    def _on_frame_ready(self, raw_bytes: bytes, W: int, H: int, frame_idx: int) -> None:
        self.canvas.load_composited_frame(raw_bytes, W, H)
        self.timeline.update_position(frame_idx)
        self._current_frame_idx = frame_idx

    def _on_metadata_ready(self, meta: dict) -> None:
        mode       = meta.get("mode", "")
        sim_score  = meta.get("sim_score", 0.0)
        bbox       = meta.get("bbox", ())
        fps_gpu    = meta.get("fps_gpu", 0.0)
        display_fps = meta.get("display_fps", 0.0)
        pct        = meta.get("export_pct", None)

        self.timeline.update_fps(display_fps)

        if pct is not None:
            self.timeline.set_export_progress(int(pct))
            if pct >= 100:
                self.timeline.show_export_progress(False)
                self._on_stop_tracking()
                self._status_bar.showMessage("Export complete.")
                return

        status_parts = []
        if mode:
            status_parts.append(mode)
        if sim_score > 0:
            status_parts.append(f"sim={sim_score:.3f}")
        if fps_gpu > 0:
            status_parts.append(f"{fps_gpu:.1f} GPU fps")
        self._status_bar.showMessage("  ·  ".join(status_parts) if status_parts else "Tracking…")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _start_reader_thread(self) -> None:
        """Start (or restart) the VideoReaderThread for the current video."""
        if self._reader_thread and self._reader_thread.isRunning():
            self._reader_thread.request_stop()
            self._reader_thread.wait(2000)

        self._reader_thread = VideoReaderThread(
            self._video_path,
            self._queues["raw_frame_queue"],
            parent=self,
        )
        self._reader_thread.preview_frame.connect(self._on_preview_frame)
        self._reader_thread.reader_error.connect(self._show_error)
        self._reader_thread.end_of_video.connect(self._on_end_of_video)
        self._reader_thread.start()

    def _on_preview_frame(self, frame_idx: int, frame_bgr: object) -> None:
        self.canvas.load_frame(frame_bgr)
        self.timeline.update_position(frame_idx)
        self._current_frame_idx = frame_idx

    def _on_end_of_video(self) -> None:
        self.timeline.set_playing(False)
        self._status_bar.showMessage("End of video.")

    def _launch_gpu_process(
        self,
        batch_render: bool = False,
        output_path: str = "",
    ) -> None:
        """Spawn the GPU Pipeline Process with current config."""
        # Import lazily — never at module level in main process
        from pipeline import GPUPipelineProcess

        if self._gpu_process is not None and self._gpu_process.is_alive():
            self._gpu_process.terminate()
            self._gpu_process.join(timeout=2)

        # Drain stale items from queues
        for q_name in ("raw_frame_queue", "display_queue"):
            q = self._queues[q_name]
            while True:
                try:
                    q.get_nowait()
                except Exception:
                    break

        config = {
            "match_threshold": self.control.get_threshold(),
            "ema_threshold":   0.92,
            "ema_alpha":       0.90,
            "overlay_alpha":   self.control.get_alpha(),
            "stride":          self.control.get_stride(),
            "fastsam_weights": self.control.get_fastsam_weights() or "FastSAM-s.pt",
            "reid_weights":    self.control.get_reid_weights() or None,
            "batch_render":    batch_render,
            "output_path":     output_path,
            "video_path":      self._video_path,
            "vid_w":           self._vid_w,
            "vid_h":           self._vid_h,
            "video_fps":       self._video_fps,
        }

        self._gpu_process = GPUPipelineProcess(
            queues     = self._queues,
            reg_result = self._reg_result,
            config     = config,
        )
        self._gpu_process.start()

    def _launch_display_worker(self) -> None:
        if self._display_worker and self._display_worker.isRunning():
            self._display_worker.request_stop()
            self._display_worker.wait(1000)

        self._display_worker = FrameDisplayWorker(
            self._queues["display_queue"], parent=self
        )
        self._display_worker.frame_ready.connect(self._on_frame_ready)
        self._display_worker.metadata_ready.connect(self._on_metadata_ready)
        self._display_worker.start()

    def _check_gpu_process(self) -> None:
        if self._gpu_process is not None and not self._gpu_process.is_alive():
            self._watchdog.stop()
            exit_code = self._gpu_process.exitcode
            self._gpu_process = None
            self.control.set_tracking_active(False)
            self.timeline.set_playing(False)
            if exit_code != 0:
                self._show_error(
                    f"GPU Pipeline Process exited unexpectedly (code {exit_code}).\n"
                    "Check the terminal for CUDA error details."
                )

    def _refresh_fps_label(self) -> None:
        pass  # FPS is updated via metadata_ready signal; timer kept for future use

    def _show_error(self, msg: str) -> None:
        QMessageBox.critical(self, "Error", str(msg))

    # ------------------------------------------------------------------
    # Close event
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        """Graceful shutdown: drain threads, terminate GPU process."""
        self._watchdog.stop()

        # Stop reader
        if self._reader_thread and self._reader_thread.isRunning():
            self._reader_thread.request_stop()
            self._reader_thread.wait(2000)

        # Stop display worker
        if self._display_worker and self._display_worker.isRunning():
            self._display_worker.request_stop()
            self._display_worker.wait(1000)

        # Terminate GPU process
        if self._gpu_process is not None:
            try:
                self._queues["pipeline_cmd_queue"].put_nowait(
                    _PipelineCmd("STOP", None)
                )
            except Exception:
                pass
            self._gpu_process.join(timeout=3.0)
            if self._gpu_process.is_alive():
                self._gpu_process.terminate()

        event.accept()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _PipelineCmd(NamedTuple):
    """Command routed through pipeline_cmd_queue to the GPU Pipeline Process."""
    type:    str
    payload: Any = None
