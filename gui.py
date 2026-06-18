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

    Registration workflow:
      1. Left-click drag → draw bounding box (rubber-band).
      2. After bbox drawn: left-click → positive point, right-click → negative point.
      3. Click "Register Target" to run FastSAM on the bbox + optional point hints.

    Signals
    -------
    point_added(x_vid, y_vid, label)
        Emitted on each point click when mode == 'registration' and bbox is drawn.
    bbox_drawn(x1, y1, x2, y2)
        Emitted when a bounding box drag is finalized.
    """

    point_added = Signal(float, float, int)
    bbox_drawn  = Signal(float, float, float, float)   # x1, y1, x2, y2 frame-space

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
        # Rubber-band bounding box state (registration phase 1)
        self._reg_bbox: Optional[tuple[float, float, float, float]] = None  # xyxy
        self._dragging: bool = False
        self._drag_origin: Optional[QPointF] = None
        # 4-corner polygon fitted to the live mask (set by MaskPreviewThread result)
        self._reg_quad: Optional[np.ndarray] = None   # (4, 2) int32

        # Embedding preview overlay — shown top-right after registration
        self._thumb_widget = QWidget(self)
        self._thumb_widget.setAttribute(Qt.WA_TransparentForMouseEvents)
        self._thumb_widget.setStyleSheet(
            "background: rgba(18,18,20,210); border: 1px solid #555; border-radius: 4px;"
        )
        _tl = QVBoxLayout(self._thumb_widget)
        _tl.setContentsMargins(5, 4, 5, 5)
        _tl.setSpacing(3)
        self._thumb_title = QLabel("Registered Target")
        self._thumb_title.setStyleSheet("color: #aaa; font-size: 9px; border: none;")
        self._thumb_title.setAlignment(Qt.AlignCenter)
        _tl.addWidget(self._thumb_title)
        self._thumb_img = QLabel()
        self._thumb_img.setStyleSheet("border: none;")
        self._thumb_img.setAlignment(Qt.AlignCenter)
        _tl.addWidget(self._thumb_img)
        self._thumb_widget.hide()

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
        _frame_buf = frame_rgb.tobytes()   # pin buffer; QImage takes a raw pointer
        qimg = QImage(_frame_buf, w, h, 3 * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)   # QPixmap copies pixels — safe after this
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
        """Set a mask (H×W uint8) as the semi-transparent overlay. Pass None to clear."""
        self._mask_overlay = mask
        self._reg_quad = None   # raw mask replaces any existing quad
        if self._current_pixmap is not None:
            self._render_with_overlay(self._current_pixmap)

    def set_quad_overlay(self, quad_pts: Optional[np.ndarray]) -> None:
        """Replace the raw mask fill with a fitted 4-corner polygon overlay.

        quad_pts: (4, 2) int32 array of corner coordinates in frame space, or None to clear.
        """
        self._reg_quad = quad_pts
        self._mask_overlay = None   # polygon replaces the blob fill
        if self._current_pixmap is not None:
            self._render_with_overlay(self._current_pixmap)

    def clear_prompts(self) -> None:
        """Remove all prompt points, bounding box, mask overlay, and quad polygon."""
        self._prompt_points.clear()
        self._mask_overlay = None
        self._reg_quad = None
        self._reg_bbox = None
        self._dragging = False
        self._drag_origin = None
        if self._current_pixmap is not None:
            self._render_with_overlay(self._current_pixmap)

    def get_reg_bbox(self) -> Optional[tuple[float, float, float, float]]:
        """Return the drawn bounding box as (x1, y1, x2, y2) or None."""
        return self._reg_bbox

    def set_embedding_preview(self, frame_bgr: np.ndarray, bbox: tuple) -> None:
        """Show a thumbnail of the registered crop in the top-right corner."""
        x, y, w, h = (int(v) for v in bbox)
        fh, fw = frame_bgr.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(fw, x + w), min(fh, y + h)
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return
        # Scale to fit within 160×120 preserving aspect ratio
        ch, cw = crop.shape[:2]
        scale = min(160 / cw, 120 / ch)
        nw, nh = max(1, int(cw * scale)), max(1, int(ch * scale))
        crop_small = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
        crop_rgb = cv2.cvtColor(crop_small, cv2.COLOR_BGR2RGB)
        _buf = crop_rgb.tobytes()
        qimg = QImage(_buf, nw, nh, 3 * nw, QImage.Format_RGB888)
        self._thumb_img.setPixmap(QPixmap.fromImage(qimg.copy()))
        self._thumb_img.setFixedSize(nw, nh)
        self._thumb_widget.adjustSize()
        self._reposition_thumb()
        self._thumb_widget.show()
        self._thumb_widget.raise_()

    def get_prompt_points(self) -> list[tuple[float, float, int]]:
        """Return a copy of the current prompt points [(x, y, label), ...]."""
        return list(self._prompt_points)

    # ------------------------------------------------------------------
    # Internal rendering
    # ------------------------------------------------------------------

    def _render_with_overlay(self, base_pixmap: QPixmap) -> None:
        """Paint mask overlay, quad polygon, bounding box, and prompt dots onto base_pixmap."""
        nothing_to_draw = (
            self._mask_overlay is None
            and self._reg_quad is None
            and not self._prompt_points
            and self._reg_bbox is None
        )
        if nothing_to_draw:
            self._pixmap_item.setPixmap(base_pixmap)
            return

        result = base_pixmap.copy()
        painter = QPainter(result)
        painter.setRenderHint(QPainter.Antialiasing)

        # --- Raw mask overlay (cyan blob, shown when no quad fitted yet) -----
        if self._mask_overlay is not None and self._reg_quad is None:
            mask = self._mask_overlay
            mh, mw = mask.shape[:2]
            overlay_rgba = np.zeros((mh, mw, 4), dtype=np.uint8)
            overlay_rgba[mask > 0] = [0, 210, 210, 110]   # cyan, ~43% opacity
            # Pin the buffer — QImage holds a raw C pointer.
            _overlay_buf = overlay_rgba.tobytes()
            overlay_img = QImage(_overlay_buf, mw, mh, 4 * mw, QImage.Format_RGBA8888)
            painter.drawImage(0, 0, overlay_img)

            # Contour outline
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            pen = QPen(QColor(0, 210, 210), 2.0)
            pen.setCosmetic(True)
            painter.setPen(pen)
            for contour in contours:
                pts = [QPointF(float(p[0][0]), float(p[0][1])) for p in contour]
                n = len(pts)
                if n > 1:
                    for i in range(n - 1):
                        painter.drawLine(pts[i], pts[i + 1])
                    painter.drawLine(pts[-1], pts[0])

        # --- 4-corner polygon (replaces raw mask once quad is computed) ------
        if self._reg_quad is not None:
            pts_np = self._reg_quad   # (4, 2) int32
            h_px = base_pixmap.height()
            w_px = base_pixmap.width()

            # Semi-transparent cyan fill via numpy → QImage overlay
            poly_rgba = np.zeros((h_px, w_px, 4), dtype=np.uint8)
            cv2.fillPoly(poly_rgba, [pts_np], (0, 210, 210, 80))
            _poly_buf = poly_rgba.tobytes()
            poly_img = QImage(_poly_buf, w_px, h_px, 4 * w_px, QImage.Format_RGBA8888)
            painter.drawImage(0, 0, poly_img)

            # Solid cyan outline
            outline_pen = QPen(QColor(0, 230, 230), 2.5)
            outline_pen.setCosmetic(True)
            painter.setPen(outline_pen)
            painter.setBrush(Qt.NoBrush)
            qpts = [QPointF(float(p[0]), float(p[1])) for p in pts_np]
            n = len(qpts)
            for i in range(n):
                painter.drawLine(qpts[i], qpts[(i + 1) % n])

            # Corner circles
            painter.setBrush(QBrush(QColor(0, 230, 230)))
            painter.setPen(QPen(QColor(255, 255, 255), 1.5))
            for p in qpts:
                painter.drawEllipse(p, 5.0, 5.0)

        # --- Bounding box (dashed cyan) -----------------------------------
        if self._reg_bbox is not None:
            x1, y1, x2, y2 = self._reg_bbox
            box_pen = QPen(QColor(0, 210, 210), 2.0)
            box_pen.setStyle(Qt.DashLine)
            box_pen.setCosmetic(True)
            painter.setPen(box_pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(QRectF(x1, y1, x2 - x1, y2 - y1))

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

        if event.button() == Qt.LeftButton:
            if self._reg_bbox is None:
                # Phase 1: start rubber-band bbox drag
                self._dragging = True
                self._drag_origin = QPointF(x_vid, y_vid)
            else:
                # Phase 2: add positive point refinement
                self._prompt_points.append((x_vid, y_vid, 1))
                if self._current_pixmap is not None:
                    self._render_with_overlay(self._current_pixmap)
                self.point_added.emit(x_vid, y_vid, 1)
        elif event.button() == Qt.RightButton and self._reg_bbox is not None:
            # Phase 2: add negative point refinement (only after bbox drawn)
            self._prompt_points.append((x_vid, y_vid, 0))
            if self._current_pixmap is not None:
                self._render_with_overlay(self._current_pixmap)
            self.point_added.emit(x_vid, y_vid, 0)

    def mouseMoveEvent(self, event) -> None:
        if self._dragging and self._drag_origin is not None:
            scene_pt = self.mapToScene(event.pos())
            x_vid = max(0.0, min(float(scene_pt.x()), float(self._vid_w - 1)))
            y_vid = max(0.0, min(float(scene_pt.y()), float(self._vid_h - 1)))
            ox, oy = self._drag_origin.x(), self._drag_origin.y()
            self._reg_bbox = (min(ox, x_vid), min(oy, y_vid),
                              max(ox, x_vid), max(oy, y_vid))
            if self._current_pixmap is not None:
                self._render_with_overlay(self._current_pixmap)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._mode == "registration" and event.button() == Qt.LeftButton and self._dragging:
            self._dragging = False
            self._drag_origin = None
            if self._reg_bbox is not None:
                x1, y1, x2, y2 = self._reg_bbox
                if (x2 - x1) < 5 or (y2 - y1) < 5:
                    # Too small — treat as a stray click, clear and ignore
                    self._reg_bbox = None
                    if self._current_pixmap is not None:
                        self._render_with_overlay(self._current_pixmap)
                else:
                    self.bbox_drawn.emit(x1, y1, x2, y2)
        else:
            super().mouseReleaseEvent(event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.fitInView(
            QRectF(0.0, 0.0, float(self._vid_w), float(self._vid_h)),
            Qt.KeepAspectRatio,
        )
        self._reposition_thumb()

    def _reposition_thumb(self) -> None:
        if not self._thumb_widget.isHidden():
            self._thumb_widget.move(
                self.width() - self._thumb_widget.width() - 10, 10
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
        self._pending_seek_value = self.slider.value()
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

    fastsam_weights_changed = Signal(str)
    reid_weights_changed   = Signal(str)

    threshold_changed = Signal(float)   # match threshold 0.50–0.99
    alpha_changed     = Signal(float)   # overlay opacity 0.10–0.90
    stride_changed    = Signal(int)     # inference stride (every N frames)
    live_preview_toggled = Signal(bool)
    sep_split_toggled  = Signal(bool)   # split merged masks at black separators
    sep_thresh_changed = Signal(int)    # separator darkness threshold 0–120
    detection_mode_toggled = Signal(bool)  # skip tracker, full-frame detect every frame

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

    def get_fastsam_weights(self) -> str:
        return self.fastsam_path_edit.text().strip()

    def get_reid_weights(self) -> str:
        return self.reid_path_edit.text().strip()

    def get_threshold(self) -> float:
        return self.threshold_spin.value()

    def get_alpha(self) -> float:
        return self.alpha_spin.value()

    def get_stride(self) -> int:
        return self.stride_spin.value()

    def get_live_preview(self) -> bool:
        return self.live_preview_check.isChecked()

    def get_separator_split(self) -> bool:
        return self.sep_split_check.isChecked()

    def get_detection_mode(self) -> bool:
        return self.detection_mode_check.isChecked()

    def get_separator_thresh(self) -> int:
        return self.sep_thresh_spin.value()

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

        # FastSAM
        root.addWidget(QLabel("FastSAM Weights:"))
        fsam_row = QHBoxLayout()
        self.fastsam_path_edit = QLineEdit()
        self.fastsam_path_edit.setPlaceholderText("FastSAM-x.pt")
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

        # ── Registration ──────────────────────────────────────────────
        root.addWidget(self._make_separator("Registration"))

        hint = QLabel("1. Drag bbox  2. +/- click refine  3. Register")
        hint.setStyleSheet("color: #666; font-size: 9px;")
        root.addWidget(hint)

        self.clear_pts_btn = QPushButton("Clear / Reset")
        self.clear_pts_btn.setEnabled(False)
        self.clear_pts_btn.clicked.connect(self.clear_points_requested)
        root.addWidget(self.clear_pts_btn)

        self.register_btn = QPushButton("Register Target (FastSAM)")
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

        # Separator splitting — cut merged masks at black lines between objects
        self.sep_split_check = QCheckBox("Split by separators")
        self.sep_split_check.setChecked(True)
        self.sep_split_check.setToolTip(
            "Split touching objects apart along the near-black lines that "
            "separate them. Disable for videos without separator lines."
        )
        self.sep_split_check.toggled.connect(self.sep_split_toggled)
        root.addWidget(self.sep_split_check)

        sep_row = QHBoxLayout()
        sep_row.addWidget(QLabel("Separator darkness ≤:"))
        self.sep_thresh_spin = QSpinBox()
        self.sep_thresh_spin.setRange(0, 120)
        self.sep_thresh_spin.setValue(40)
        self.sep_thresh_spin.setToolTip(
            "Pixels darker than this (0–255 grayscale) are treated as separator "
            "lines and cut out when splitting masks."
        )
        self.sep_thresh_spin.valueChanged.connect(self.sep_thresh_changed)
        sep_row.addWidget(self.sep_thresh_spin)
        root.addLayout(sep_row)

        # Live preview toggle
        self.live_preview_check = QCheckBox("Live Preview Mode")
        self.live_preview_check.setChecked(True)
        self.live_preview_check.toggled.connect(self.live_preview_toggled)
        root.addWidget(self.live_preview_check)

        # Detection mode — skip tracker, full-frame FastSAM every frame
        self.detection_mode_check = QCheckBox("Detection Mode (no tracker)")
        self.detection_mode_check.setChecked(False)
        self.detection_mode_check.setToolTip(
            "Skip the tracker entirely. Run full-frame FastSAM on every frame "
            "and match against the registered embedding — like a per-frame detector. "
            "More accurate for fast/erratic motion; slower than tracking mode."
        )
        self.detection_mode_check.toggled.connect(self.detection_mode_toggled)
        root.addWidget(self.detection_mode_check)

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
# MaskPreviewThread
# ---------------------------------------------------------------------------

class MaskPreviewThread(QThread):
    """
    Runs FastSAM on the current frame + bbox + points and emits the mask for
    immediate canvas overlay.  Re-ID is NOT run here — that only happens when
    the user clicks "Register Target".

    Triggered automatically (with debounce) whenever the user draws or adjusts
    the bounding box or adds / removes prompt points.
    """

    preview_ready = Signal(object, tuple, object)   # (mask_np H×W uint8*255, bbox_xywh, quad_pts|(4,2)int32|None)
    preview_error = Signal(str)
    progress      = Signal(str)

    def __init__(
        self,
        video_path: str,
        frame_idx: int,
        bbox: tuple,
        points: list,
        fastsam_weights: str,
        separator_thresh: int = 40,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._video_path      = video_path
        self._frame_idx       = frame_idx
        self._bbox            = bbox
        self._points          = list(points)
        self._fastsam_weights = fastsam_weights
        self._separator_thresh = separator_thresh

    def run(self) -> None:
        try:
            self._run_inner()
        except Exception:
            self.preview_error.emit(traceback.format_exc())

    def _run_inner(self) -> None:
        from models import FastSAMTracker, HeavySAMRegistrar, _pick_best_fastsam_mask

        cap = cv2.VideoCapture(self._video_path, cv2.CAP_MSMF)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self._video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, self._frame_idx)
        ret, frame_bgr = cap.read()
        cap.release()
        if not ret:
            return

        self.progress.emit("Loading FastSAM for preview…")
        fast_sam = FastSAMTracker(
            weights=self._fastsam_weights or "FastSAM-x.pt",
            device="cuda",
        )
        fast_sam.load(skip_compile=True)

        self.progress.emit("Running FastSAM…")
        masks_cuda, _ = fast_sam.predict_roi(frame_bgr, self._bbox)
        fast_sam.unload()

        if not masks_cuda:
            self.preview_error.emit("no_masks")
            return

        pos_pts = [(x, y) for x, y, l in self._points if l == 1]
        neg_pts = [(x, y) for x, y, l in self._points if l == 0]
        best_mask_cuda = _pick_best_fastsam_mask(masks_cuda, self._bbox, pos_pts, neg_pts)
        best_mask_np   = (best_mask_cuda > 0.5).cpu().numpy().astype(np.uint8)

        if pos_pts:
            best_mask_np = HeavySAMRegistrar._isolate_clicked_component(
                best_mask_np.astype(bool), frame_bgr, self._points, self._separator_thresh
            ).astype(np.uint8)

        ys, xs = np.where(best_mask_np > 0)
        if len(xs) == 0:
            x1, y1, x2, y2 = (int(v) for v in self._bbox)
            bbox_xywh = (x1, y1, x2 - x1, y2 - y1)
        else:
            bx1, by1 = int(xs.min()), int(ys.min())
            bx2, by2 = int(xs.max()) + 1, int(ys.max()) + 1
            bbox_xywh = (bx1, by1, bx2 - bx1, by2 - by1)

        from models import _mask_to_quad
        quad_pts = _mask_to_quad(best_mask_np)

        self.preview_ready.emit(best_mask_np * 255, bbox_xywh, quad_pts)


# ---------------------------------------------------------------------------
# RegistrationThread
# ---------------------------------------------------------------------------

class RegistrationThread(QThread):
    """
    Runs in a dedicated QThread so the Qt main loop stays responsive during
    FastSAM inference and Re-ID embedding.

    Execution order
    ---------------
    1. Open video at _video_path, seek to _frame_idx, read one frame.
    2. Load FastSAMTracker (~12 MB), run predict_roi() on the user-drawn bbox.
    3. Select the best candidate mask (by positive-point hit or IoU).
    4. Optionally isolate the clicked connected component along separator lines.
    5. Load ReIDEmbedder (OSNet / ResNet18, ~0.3 GB VRAM).
    6. Crop the target region, extract normalised reference embedding.
    7. Unload FastSAM + ReID embedder (del + empty_cache).
    8. Emit registration_done with all serialised CPU results.
    """

    registration_done = Signal(dict)   # {mask, bbox, reid_emb, score, frame_bgr, frame_idx}
    progress          = Signal(str)
    error             = Signal(str)

    def __init__(
        self,
        video_path: str,
        frame_idx: int,
        bbox: tuple,
        points: list[tuple[float, float, int]],
        fastsam_weights: str,
        reid_weights: str,
        separator_thresh: int = 40,
        precomputed_mask: Optional[np.ndarray] = None,
        precomputed_bbox: Optional[tuple] = None,
        precomputed_quad: Optional[np.ndarray] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._video_path       = video_path
        self._frame_idx        = frame_idx
        self._bbox             = bbox          # (x1, y1, x2, y2) drawn by user
        self._points           = list(points)  # [(x, y, label), ...]
        self._fastsam_weights  = fastsam_weights
        self._reid_weights     = reid_weights
        self._separator_thresh = separator_thresh
        self._precomputed_mask = precomputed_mask   # H×W uint8 *255, from MaskPreviewThread
        self._precomputed_bbox = precomputed_bbox   # (x, y, w, h)
        self._precomputed_quad = precomputed_quad   # (4, 2) int32, from MaskPreviewThread

    def run(self) -> None:
        import torch

        try:
            self._run_inner()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            self.error.emit(
                "GPU Out of Memory during registration.\n"
                "Close other GPU applications and try again."
            )
        except Exception:
            self.error.emit(
                f"Registration failed:\n{traceback.format_exc()}"
            )

    def _run_inner(self) -> None:
        import torch
        from models import FastSAMTracker, ReIDEmbedder, HeavySAMRegistrar, _pick_best_fastsam_mask

        if self._bbox is None:
            self.error.emit("Draw a bounding box on the frame before registering.")
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

        # ── FastSAM registration (skipped when live preview already ran) ─
        if self._precomputed_mask is not None and self._precomputed_bbox is not None:
            self.progress.emit("Using live-preview mask (skipping FastSAM)…")
            best_mask_np = (self._precomputed_mask > 0).astype(np.uint8)
            bbox_xywh    = self._precomputed_bbox
        else:
            self.progress.emit("Loading FastSAM…")
            fast_sam = FastSAMTracker(
                weights=self._fastsam_weights or "FastSAM-x.pt",
                device="cuda",
            )
            fast_sam.load(progress_cb=self.progress.emit, skip_compile=True)

            self.progress.emit("Running FastSAM segmentation…")
            masks_cuda, _ = fast_sam.predict_roi(frame_bgr, self._bbox)

            if not masks_cuda:
                self.error.emit(
                    "FastSAM found no masks inside the drawn bounding box.\n"
                    "Try drawing a larger box or adjusting FastSAM weights."
                )
                fast_sam.unload()
                return

            pos_pts = [(x, y) for x, y, l in self._points if l == 1]
            neg_pts = [(x, y) for x, y, l in self._points if l == 0]
            best_mask_cuda = _pick_best_fastsam_mask(masks_cuda, self._bbox, pos_pts, neg_pts)
            best_mask_np   = (best_mask_cuda > 0.5).cpu().numpy().astype(np.uint8)

            if pos_pts:
                best_mask_np = HeavySAMRegistrar._isolate_clicked_component(
                    best_mask_np.astype(bool), frame_bgr, self._points, self._separator_thresh
                ).astype(np.uint8)

            ys, xs = np.where(best_mask_np > 0)
            if len(xs) == 0:
                x1, y1, x2, y2 = (int(v) for v in self._bbox)
                bbox_xywh = (x1, y1, x2 - x1, y2 - y1)
            else:
                bx1, by1 = int(xs.min()), int(ys.min())
                bx2, by2 = int(xs.max()) + 1, int(ys.max()) + 1
                bbox_xywh = (bx1, by1, bx2 - bx1, by2 - by1)

            fast_sam.unload()

        # ── Re-ID embedding ──────────────────────────────────────────
        self.progress.emit("Extracting reference embedding…")
        embedder = ReIDEmbedder(
            weights_path=self._reid_weights or None,
            device="cuda",
        )
        embedder.load()

        fh, fw = frame_bgr.shape[:2]

        # Determine the quad to use: prefer precomputed; fall back to bbox rect.
        quad = self._precomputed_quad
        if quad is None and best_mask_np is not None:
            from models import _mask_to_quad
            quad = _mask_to_quad(best_mask_np)

        if quad is not None:
            # Crop to the polygon's axis-aligned bounding rect, then zero
            # pixels outside the polygon so Re-ID focuses on the object shape.
            qx, qy, qw, qh = cv2.boundingRect(quad)
            x1 = max(0, qx)
            y1 = max(0, qy)
            x2 = min(fw, qx + qw)
            y2 = min(fh, qy + qh)
            poly_mask = np.zeros((fh, fw), dtype=np.uint8)
            cv2.fillPoly(poly_mask, [quad], 255)
            crop_roi    = frame_bgr[y1:y2, x1:x2].copy()
            mask_roi    = poly_mask[y1:y2, x1:x2].astype(bool)
            # Fill non-polygon pixels with the mean object colour so the Re-ID
            # embedding is not biased toward black (which the background shares).
            if mask_roi.any():
                mean_col = crop_roi[mask_roi].mean(axis=0).astype(np.uint8)
                crop_roi[~mask_roi] = mean_col
            crop = crop_roi
        else:
            x, y, w, h = bbox_xywh
            x1 = max(0, int(x))
            y1 = max(0, int(y))
            x2 = min(fw, int(x + w))
            y2 = min(fh, int(y + h))
            crop = frame_bgr[y1:y2, x1:x2]

        if crop.size == 0:
            self.error.emit("Registration bounding box collapsed to zero area.")
            embedder.unload()
            return

        reid_emb = embedder.embed(crop)  # np.float32 (D,)

        # ── Unload models ────────────────────────────────────────────
        self.progress.emit("Purging Re-ID from VRAM…")
        embedder.unload()

        free_gb = torch.cuda.mem_get_info()[0] / 1e9
        self.progress.emit(f"Target registered  ·  {free_gb:.1f} GB VRAM free")

        self.registration_done.emit({
            "mask":      best_mask_np * 255,
            "bbox":      bbox_xywh,
            "reid_emb":  reid_emb,
            "score":     1.0,
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

    def play(self, start_frame: int) -> None:
        """Continuous preview playback at video FPS; emits preview_frame each frame."""
        self._cmd_q.put(Cmd("PLAY", start_frame))

    def pause(self) -> None:
        """Pause continuous preview playback."""
        self._cmd_q.put(Cmd("PAUSE", None))

    def start_tracking(self, start_frame: int, batch: bool = False) -> None:
        """
        Switch to TRACKING mode and begin feeding raw_frame_queue.

        batch=False (live preview): pace at video FPS and drop frames if the
        GPU pipeline falls behind — keeps the UI real-time.
        batch=True (export): never drop and never pace — every frame is pushed
        with a blocking put so the rendered output is complete.
        """
        self._cmd_q.put(Cmd("START_TRACKING", (start_frame, batch)))

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
        batch        = False   # export mode — no drop, no FPS pacing
        playing      = False   # preview playback — emits preview_frame
        current_idx  = 0

        while not self._stop_event.is_set():
            # ── Drain command queue ──────────────────────────────────
            active = tracking or playing
            try:
                cmd = self._cmd_q.get(timeout=0.0 if active else 0.05)
            except queue.Empty:
                cmd = None

            if cmd is not None:
                if cmd.type == "STOP":
                    break

                elif cmd.type == "SEEK":
                    playing = False   # explicit seek interrupts playback
                    idx = int(cmd.payload)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        current_idx = idx
                        self.preview_frame.emit(idx, frame.copy())

                elif cmd.type == "PLAY":
                    playing = True
                    tracking = False
                    idx = int(cmd.payload) if cmd.payload is not None else current_idx
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    current_idx = idx

                elif cmd.type == "PAUSE":
                    playing = False

                elif cmd.type == "START_TRACKING":
                    playing = False
                    if isinstance(cmd.payload, tuple):
                        start_frame, batch = cmd.payload
                    else:
                        start_frame, batch = cmd.payload, False
                    idx = int(start_frame)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    current_idx = idx
                    tracking = True

                elif cmd.type == "STOP_TRACKING":
                    tracking = False
                    batch = False
                    # Signal GPU process to stop consuming
                    try:
                        self._raw_frame_queue.put_nowait(None)
                    except Exception:
                        pass

            # ── PLAYING: emit preview frames at video FPS ────────────
            if playing:
                t0 = time.monotonic()
                ret, frame = cap.read()
                if not ret:
                    playing = False
                    self.end_of_video.emit()
                    continue
                self.preview_frame.emit(current_idx, frame.copy())
                current_idx += 1
                elapsed = time.monotonic() - t0
                sleep_t = frame_delay - elapsed
                if sleep_t > 0.001:
                    time.sleep(sleep_t)

            # ── TRACKING: push next frame to GPU queue ───────────────
            elif tracking:
                t0 = time.monotonic()
                ret, frame = cap.read()
                if not ret:
                    # End of video — send sentinel
                    try:
                        self._raw_frame_queue.put(None, timeout=1.0)
                    except Exception:
                        pass
                    tracking = False
                    batch = False
                    self.end_of_video.emit()
                    continue

                if batch:
                    # Export: never drop. Block until the GPU pipeline accepts
                    # the frame (with periodic stop checks) so the rendered
                    # output contains every frame. If the pipeline stops draining
                    # for too long (e.g. the GPU process died) bail out instead
                    # of spinning forever.
                    delivered = False
                    blocked_for = 0.0
                    while not self._stop_event.is_set() and blocked_for < 30.0:
                        try:
                            self._raw_frame_queue.put(
                                (current_idx, frame.copy()), timeout=0.5
                            )
                            delivered = True
                            break
                        except queue.Full:
                            blocked_for += 0.5
                    if not delivered:
                        self.reader_error.emit(
                            "Export stalled: the GPU pipeline stopped accepting "
                            "frames. The rendered file may be incomplete."
                        )
                        tracking = False
                        batch = False
                        continue
                else:
                    try:
                        self._raw_frame_queue.put(
                            (current_idx, frame.copy()), timeout=0.15
                        )
                    except queue.Full:
                        pass   # GPU pipeline is behind; drop this frame

                current_idx += 1

                # Pace output to video FPS — live preview only. Export runs
                # as fast as the GPU pipeline can consume frames.
                if not batch:
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
        self._reg_thread:     Optional[RegistrationThread]  = None
        self._preview_thread: Optional[MaskPreviewThread]   = None
        self._reader_thread:  Optional[VideoReaderThread]   = None
        self._display_worker: Optional[FrameDisplayWorker]  = None
        self._gpu_process     = None    # GPUPipelineProcess (imported lazily)

        # Live mask preview state (populated by MaskPreviewThread)
        self._live_mask:      Optional[np.ndarray] = None
        self._live_bbox_xywh: Optional[tuple]      = None
        self._live_quad:      Optional[np.ndarray] = None   # (4, 2) int32

        # Debounce timer — fires 400 ms after the last bbox/point change
        self._preview_debounce = QTimer()
        self._preview_debounce.setSingleShot(True)
        self._preview_debounce.timeout.connect(self._run_mask_preview)

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
        cp.sep_split_toggled.connect(self._on_sep_split_toggled)
        cp.sep_thresh_changed.connect(self._on_sep_thresh_changed)
        cp.detection_mode_toggled.connect(self._on_detection_mode_toggled)

        # Timeline → MainWindow
        tl.seek_requested.connect(self._on_seek)
        tl.play_requested.connect(self._on_play)
        tl.pause_requested.connect(self._on_pause)

        # Canvas → MainWindow
        cv.point_added.connect(self._on_point_added)
        cv.bbox_drawn.connect(self._on_bbox_drawn)

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
        self._status_bar.showMessage("Video loaded. Drag to draw a bounding box around the target, then click Register.")

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
        if self._reader_thread and self._reader_thread.isRunning():
            self._reader_thread.play(self._current_frame_idx)
            self.timeline.set_playing(True)

    def _on_pause(self) -> None:
        if self._reader_thread and self._reader_thread.isRunning():
            self._reader_thread.pause()
            self.timeline.set_playing(False)

    # ------------------------------------------------------------------
    # Slot: Point Added
    # ------------------------------------------------------------------

    def _on_point_added(self, x: float, y: float, label: int) -> None:
        n_pos = sum(1 for _, _, l in self.canvas.get_prompt_points() if l == 1)
        n_neg = sum(1 for _, _, l in self.canvas.get_prompt_points() if l == 0)
        self._status_bar.showMessage(
            f"Refinement points: {n_pos} positive, {n_neg} negative  ·  "
            "Computing mask preview…"
        )
        self._preview_debounce.start(400)

    def _on_clear_points(self) -> None:
        self._preview_debounce.stop()
        if self._preview_thread and self._preview_thread.isRunning():
            self._preview_thread.quit()
            self._preview_thread.wait(500)
        self._live_mask = None
        self._live_bbox_xywh = None
        self._live_quad = None
        self.canvas.clear_prompts()
        self._status_bar.showMessage("Cleared. Draw a new bounding box to start over.")

    def _on_bbox_drawn(self, x1: float, y1: float, x2: float, y2: float) -> None:
        w, h = int(x2 - x1), int(y2 - y1)
        self._status_bar.showMessage(
            f"Bbox drawn ({w}×{h} px)  ·  Computing mask preview…"
        )
        self._live_mask = None
        self._live_bbox_xywh = None
        self._live_quad = None
        self._preview_debounce.start(400)

    # ------------------------------------------------------------------
    # Live mask preview (MaskPreviewThread)
    # ------------------------------------------------------------------

    def _run_mask_preview(self) -> None:
        """Fired by debounce timer — cancel any running preview, start a new one."""
        if not self._video_path:
            return
        bbox = self.canvas.get_reg_bbox()
        if bbox is None:
            return

        if self._preview_thread and self._preview_thread.isRunning():
            self._preview_thread.quit()
            self._preview_thread.wait(500)

        self._preview_thread = MaskPreviewThread(
            video_path       = self._video_path,
            frame_idx        = self._current_frame_idx,
            bbox             = bbox,
            points           = self.canvas.get_prompt_points(),
            fastsam_weights  = self.control.get_fastsam_weights(),
            separator_thresh = self.control.get_separator_thresh(),
            parent           = self,
        )
        self._preview_thread.preview_ready.connect(self._on_mask_preview_ready)
        self._preview_thread.preview_error.connect(self._on_mask_preview_error)
        self._preview_thread.progress.connect(
            lambda msg: self._status_bar.showMessage(msg)
        )
        self._preview_thread.start()

    def _on_mask_preview_ready(self, mask: np.ndarray, bbox_xywh: tuple, quad_pts) -> None:
        self._live_mask      = mask
        self._live_bbox_xywh = bbox_xywh
        self._live_quad      = quad_pts   # None or (4,2) int32
        if quad_pts is not None:
            self.canvas.set_quad_overlay(quad_pts)
        else:
            self.canvas.set_mask_overlay(mask)
        pts = self.canvas.get_prompt_points()
        n_pos = sum(1 for _, _, l in pts if l == 1)
        n_neg = sum(1 for _, _, l in pts if l == 0)
        pt_str = f"  ·  {n_pos}+ {n_neg}− pts" if pts else ""
        quad_str = "  ·  polygon fitted" if quad_pts is not None else ""
        self._status_bar.showMessage(
            f"Mask preview ready{pt_str}{quad_str}  ·  Click 'Register Target' to embed."
        )

    def _on_mask_preview_error(self, msg: str) -> None:
        if msg == "no_masks":
            self._status_bar.showMessage(
                "FastSAM found no mask in that region — try a larger bounding box."
            )
        else:
            self._status_bar.showMessage("Mask preview failed — check FastSAM weights.")

    # ------------------------------------------------------------------
    # Slot: Register Target
    # ------------------------------------------------------------------

    def _on_register(self) -> None:
        if not self._video_path:
            return

        bbox = self.canvas.get_reg_bbox()
        if bbox is None:
            self._show_error(
                "Draw a bounding box first.\n"
                "Left-click and drag on the video to outline the target."
            )
            return

        points = self.canvas.get_prompt_points()

        self.control.register_btn.setEnabled(False)
        self.control.set_status("Loading FastSAM…")
        self._status_bar.showMessage("Registering target — please wait…")

        self._reg_thread = RegistrationThread(
            video_path        = self._video_path,
            frame_idx         = self._current_frame_idx,
            bbox              = bbox,
            points            = points,
            fastsam_weights   = self.control.get_fastsam_weights(),
            reid_weights      = self.control.get_reid_weights(),
            separator_thresh  = self.control.get_separator_thresh(),
            precomputed_mask  = self._live_mask,
            precomputed_bbox  = self._live_bbox_xywh,
            precomputed_quad  = self._live_quad,
            parent            = self,
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

        # Show polygon (if fitted) or raw mask overlay on canvas
        if self._live_quad is not None:
            self.canvas.set_quad_overlay(self._live_quad)
        else:
            self.canvas.set_mask_overlay(result["mask"])
        self.canvas.set_embedding_preview(result["frame_bgr"], result["bbox"])
        self.canvas.set_mode("tracking")   # disable further point-clicking

        self.control.register_btn.setEnabled(True)
        self.control.set_track_controls_enabled(True)
        self.control.set_status(
            f"Registered ✓  bbox={result['bbox']}"
        )
        self._status_bar.showMessage(
            "Target registered (FastSAM). Click 'Start Live Preview' to begin tracking."
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

        # Seek back to the beginning so the user can restart from frame 0
        self._current_frame_idx = 0
        self.timeline.update_position(0)
        if self._reader_thread and self._reader_thread.isRunning():
            self._reader_thread.seek(0)

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
            # batch=True → feed every frame, no drops, no FPS pacing.
            self._reader_thread.start_tracking(0, batch=True)

        self.control.set_tracking_active(True)

        # Watchdog: surface a dead GPU process instead of a frozen progress bar.
        self._watchdog.start(5000)

    # ------------------------------------------------------------------
    # Threshold / Alpha / Stride updates → GPU process
    # ------------------------------------------------------------------

    def _on_threshold_changed(self, value: float) -> None:
        self._send_pipeline_cmd("UPDATE_THRESHOLD", value)

    def _on_alpha_changed(self, value: float) -> None:
        self._send_pipeline_cmd("UPDATE_ALPHA", value)

    def _on_stride_changed(self, value: int) -> None:
        self._send_pipeline_cmd("UPDATE_STRIDE", value)

    def _on_sep_split_toggled(self, enabled: bool) -> None:
        self._send_pipeline_cmd("UPDATE_SEP_SPLIT", enabled)

    def _on_sep_thresh_changed(self, value: int) -> None:
        self._send_pipeline_cmd("UPDATE_SEP_THRESH", value)

    def _on_detection_mode_toggled(self, enabled: bool) -> None:
        self._send_pipeline_cmd("UPDATE_DETECTION_MODE", enabled)

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
            "fastsam_weights": self.control.get_fastsam_weights() or "FastSAM-x.pt",
            "reid_weights":    self.control.get_reid_weights() or None,
            "batch_render":    batch_render,
            "output_path":     output_path,
            "video_path":      self._video_path,
            "vid_w":           self._vid_w,
            "vid_h":           self._vid_h,
            "video_fps":       self._video_fps,
            "total_frames":    self._total_frames,
            "separator_split":  self.control.get_separator_split(),
            "separator_thresh": self.control.get_separator_thresh(),
            "detection_mode":   self.control.get_detection_mode(),
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

            # Stop the frame feed and clear any export progress UI so a dead
            # process never leaves the reader blocking or the bar frozen.
            if self._reader_thread and self._reader_thread.isRunning():
                self._reader_thread.stop_tracking()
            self.timeline.show_export_progress(False)
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
        self._preview_debounce.stop()

        # Stop live preview thread
        if self._preview_thread and self._preview_thread.isRunning():
            self._preview_thread.quit()
            self._preview_thread.wait(1000)

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
                self._gpu_process.join(timeout=2.0)  # reap to avoid zombie
            self._gpu_process = None

        # mp.Queue keeps an internal feeder thread alive until all buffered
        # items are flushed.  cancel_join_thread() tells it to abandon the
        # flush so the main process can exit immediately.
        for q in self._queues.values():
            try:
                q.cancel_join_thread()
            except AttributeError:
                pass  # threading.Queue — no feeder thread, nothing to do

        event.accept()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _PipelineCmd(NamedTuple):
    """Command routed through pipeline_cmd_queue to the GPU Pipeline Process."""
    type:    str
    payload: Any = None
