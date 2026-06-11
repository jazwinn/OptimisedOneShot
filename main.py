"""
OptimisedOneShot — One-Shot Video Object Segmentation
======================================================
Entry point. Responsibilities:
  1. Configure multiprocessing to use the 'spawn' start method (mandatory on
     Windows and required for CUDA safety across process boundaries).
  2. Create all cross-process mp.Queue instances before any child Process is
     spawned — queues must be picklable and created in the parent process.
  3. Bootstrap the Qt application and show MainWindow.

IMPORTANT: Do NOT import torch at module level here. The GPU Pipeline Process
initialises its own CUDA context inside its run() method; any premature CUDA
call in the parent process would pollute the CUDA context inherited by child
processes (on Linux with fork) or cause serialisation errors (on Windows/spawn).
"""

import sys
import multiprocessing as mp


def configure_multiprocessing() -> None:
    """
    Sets the multiprocessing start method to 'spawn' and calls freeze_support()
    for Windows PyInstaller compatibility.

    Must be invoked before any mp.Queue, mp.Event, or mp.Process creation.
    On Windows 'spawn' is the default, but setting it explicitly prevents
    accidental fallback if the environment changes.
    """
    mp.freeze_support()
    try:
        mp.set_start_method("spawn", force=False)
    except RuntimeError:
        # Already set — safe to continue.
        pass


def build_queues() -> dict:
    """
    Instantiate every cross-process bounded queue used by the application.

    All queues carry CPU-resident data only — CUDA tensors are NEVER placed
    into mp.Queue because CUDA IPC on Windows is unreliable and the copy
    overhead defeats the purpose. Tensors are converted to numpy bytes before
    enqueue and reconstructed on the receiving side.

    Returns
    -------
    dict
        raw_frame_queue    : (frame_idx: int, frame_bgr: np.ndarray uint8 H×W×3)
                             VideoReaderThread → GPU Pipeline Process (Decode Thread)
        reg_result_queue   : dict {mask, bbox, reid_emb, score, frame_bgr, frame_idx}
                             RegistrationThread → GPU Pipeline Process
        pipeline_cmd_queue : Cmd(type, payload) — START | STOP | UPDATE_THRESHOLD
                             | UPDATE_ALPHA | UPDATE_STRIDE | BATCH_RENDER
                             MainWindow → GPU Pipeline Process command loop
        display_queue      : (raw_bytes: bytes, W: int, H: int, frame_idx: int,
                              meta: dict | None)
                             GPU Pipeline Process (Blend Thread) → FrameDisplayWorker
    """
    return {
        "raw_frame_queue":    mp.Queue(maxsize=4),
        "reg_result_queue":   mp.Queue(maxsize=1),
        "pipeline_cmd_queue": mp.Queue(maxsize=8),
        "display_queue":      mp.Queue(maxsize=3),
    }


def main() -> None:
    """Application entry point."""
    configure_multiprocessing()
    queues = build_queues()

    # Qt must be imported after freeze_support / set_start_method so that the
    # Qt platform plugin is not accidentally initialised in a worker process.
    from qtpy.QtWidgets import QApplication
    from gui import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("OptimisedOneShot")
    app.setApplicationDisplayName("OptimisedOneShot")
    app.setStyle("Fusion")

    # Dark palette for a professional, low-distraction look.
    from qtpy.QtGui import QPalette, QColor
    from qtpy.QtCore import Qt
    palette = QPalette()
    palette.setColor(QPalette.Window,          QColor(30, 30, 32))
    palette.setColor(QPalette.WindowText,      QColor(220, 220, 220))
    palette.setColor(QPalette.Base,            QColor(22, 22, 24))
    palette.setColor(QPalette.AlternateBase,   QColor(38, 38, 40))
    palette.setColor(QPalette.ToolTipBase,     QColor(220, 220, 220))
    palette.setColor(QPalette.ToolTipText,     QColor(30, 30, 32))
    palette.setColor(QPalette.Text,            QColor(220, 220, 220))
    palette.setColor(QPalette.Button,          QColor(50, 50, 54))
    palette.setColor(QPalette.ButtonText,      QColor(220, 220, 220))
    palette.setColor(QPalette.BrightText,      QColor(255, 80, 80))
    palette.setColor(QPalette.Link,            QColor(80, 160, 255))
    palette.setColor(QPalette.Highlight,       QColor(0, 140, 255))
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    palette.setColor(QPalette.Disabled, QPalette.Text,       QColor(90, 90, 90))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(90, 90, 90))
    app.setPalette(palette)

    window = MainWindow(queues)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
