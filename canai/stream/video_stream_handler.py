import time
import queue
import threading
import logging
import numpy as np
from typing import Any, Optional

logger = logging.getLogger(__name__)

class VideoStreamHandler:
    """
    Continuously captures frames from the camera, runs detection,
    and buffers frames for event-based processing.

    Attributes
    ----------
    camera : Any
        The camera instance used to capture frames.
    detector : Any
        The AI detector instance used for object detection.
    frame_buffer : queue.Queue
        A queue that stores the last `max_pre_frames` frames.
    current_frame : Optional[np.ndarray]
        The most recently processed frame.
    lock : threading.Lock
        Lock to ensure thread-safe access to the current frame.
    """

    def __init__(
        self,
        camera: Any,
        detector: Any,
        max_pre_frames: int
    ) -> None:
        self.camera: Any = camera
        self.detector: Any = detector
        self.frame_buffer: queue.Queue = queue.Queue(maxsize=max_pre_frames)
        self.current_frame: Optional[np.ndarray] = None
        self.lock: threading.Lock = threading.Lock()

    def get_current_frame(self) -> Optional[np.ndarray]:
        """
        Retrieves the latest processed frame in a thread-safe manner.

        Returns
        -------
        Optional[np.ndarray]
            A copy of the most recently processed frame, or None if no frame is available.
        """
        with self.lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
            return None

    def run(self) -> None:
        """
        Continuously captures frames, runs detection, and updates the frame buffer.

        This function runs indefinitely while the camera is active.
        It processes each frame, applies object detection, and stores the frame
        in the buffer for pre-event recording.

        If the buffer is full, the oldest frame is removed before adding a new one.
        """
        time.sleep(2)  # Allow the camera to initialize
        logger.info("VideoStreamHandler started.")

        while self.camera.running:
            try:
                frame = self.camera.get_frame()
                if frame is None:
                    logger.warning("Received an empty frame from the camera.")
                    continue

                # Run detection and annotate the frame.
                detected, frame_with_boxes = self.detector.detect(frame)

                with self.lock:
                    self.current_frame = frame_with_boxes.copy()

                # Buffer the processed frame; if full, drop the oldest frame.
                try:
                    self.frame_buffer.put_nowait(frame_with_boxes.copy())
                except queue.Full:
                    self.frame_buffer.get_nowait()  # Remove oldest frame
                    self.frame_buffer.put_nowait(frame_with_boxes.copy())

            except RuntimeError as e:
                logger.error("Camera error: %s", e, exc_info=True)
                break
        logger.info("VideoStreamHandler stopped.")
