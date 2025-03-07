import time
import queue
import threading
import logging
import numpy as np
from typing import Any, Optional, List, Tuple

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
    frame_timestamps : List[float]
        A list of timestamps corresponding to the frames in the buffer.
    fps : float
        The frames per second of the camera.
    last_frame_time : float
        The timestamp of the last processed frame.
    target_frame_interval : float
        The target interval between frames to maintain the desired FPS.
    recording_fps : float
        The frames per second of the recording.
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
        self.frame_timestamps: List[float] = []
        self.fps: float = 30.0  # Default FPS, should be set from camera config
        self.last_frame_time: float = time.time()
        self.target_frame_interval: float = 1.0 / self.fps
        self.recording_fps: float = 30.0

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

    def _maintain_buffer(self) -> None:
        """
        Maintains the frame buffer to store exactly pre_event_seconds worth of frames.
        """
        while len(self.frame_timestamps) > 0 and \
              time.time() - self.frame_timestamps[0] > 2.0:  # 2 seconds pre-event
            self.frame_buffer.get()
            self.frame_timestamps.pop(0)

    def add_frame(self, frame: np.ndarray) -> None:
        """
        Adds a frame to the buffer while maintaining timing constraints.
        """
        with self.lock:
            self._maintain_buffer()
            if self.frame_buffer.full():
                self.frame_buffer.get()
            self.frame_buffer.put(frame)
            self.frame_timestamps.append(time.time())
            self.current_frame = frame

    def _get_timed_frames(self) -> List[Tuple[np.ndarray, float]]:
        """
        Returns frames with their corresponding timestamps.
        """
        with self.lock:
            return list(zip(list(self.frame_buffer.queue), self.frame_timestamps))

    def _sync_frame_rate(self) -> None:
        """
        Synchronizes frame processing to maintain target frame rate.
        """
        current_time = time.time()
        elapsed = float(current_time - self.last_frame_time)
        sleep_time = max(0.0, float(self.target_frame_interval - elapsed))
        time.sleep(sleep_time)
        self.last_frame_time = time.time()

    def run(self) -> None:
        """
        Continuously captures frames, runs detection, and updates the frame buffer.
        """
        logger.info("Video stream handler started.")

        try:
            while True:
                self._sync_frame_rate()
                frame = self.camera.get_frame()
                if frame is None:
                    logger.warning("Received an empty frame from the camera.")
                    continue

                # Run detection and annotate the frame.
                detected, frame_with_boxes = self.detector.detect(frame)

                self.add_frame(frame_with_boxes)

        except RuntimeError as e:
            logger.error("Camera error: %s", e, exc_info=True)
        finally:
            logger.info("Video stream handler stopped.")