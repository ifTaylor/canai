import threading
import cv2
import logging
import numpy as np
from typing import Any
import time
import psutil

logger = logging.getLogger(__name__)

class CanAI:
    """
    Main application orchestrator for object detection and event recording.

    Attributes
    ----------
    camera : Any
        The camera instance (e.g., RealSenseCamera or WebcamCamera).
    detector : Any
        The AI detector instance (e.g., AIDetector).
    recorder : Any
        The event clip recorder instance.
    stream_handler : Any
        The video stream handler instance.
    config : dict
        Configuration parameters for the application.
    recorded_detection : bool
        Flag indicating whether a detection has been recorded.
    frame_counter : int
        Counter for frames processed.
    start_time : float
        Start time of the application.
    last_detection_time : float
        Timestamp of the last detection event.
    recording_in_progress : bool
        Flag indicating whether a recording is in progress.
    """

    def __init__(
        self,
        camera: Any,
        detector: Any,
        recorder: Any,
        stream_handler: Any,
        config: dict
    ) -> None:
        self.camera: Any = camera
        self.detector: Any = detector
        self.recorder: Any = recorder
        self.stream_handler: Any = stream_handler
        self.config: dict = config
        self.recorded_detection: bool = False
        self.frame_counter: int = 0
        self.start_time: float = time.time()
        self.last_detection_time: float = 0.0
        self.recording_in_progress: bool = False
        self._validate_config()

    def run(self) -> None:
        """
        Runs the object detection application with optimized performance.
        """
        capture_thread = threading.Thread(target=self.stream_handler.run, daemon=True)
        capture_thread.start()
        logger.info("Detection system started. Press 'q' to exit.")

        try:
            while True:
                frame = self.stream_handler.get_current_frame()
                if frame is None:
                    continue

                # Scale down frame for processing
                scale_factor = 0.5
                processing_frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)

                self._handle_detection(processing_frame)
                self.frame_counter += 1

                if self.frame_counter % 60 == 0:  # Log performance every 60 frames
                    self._log_performance()
                    self._monitor_resources()

                cv2.imshow("Live Feed", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Exiting application...")
                    break
        except Exception as e:
            logger.error("Unexpected error in main loop: %s", e, exc_info=True)
        finally:
            logger.info("Stopping camera and closing all windows.")
            self.camera.stop()
            cv2.destroyAllWindows()

    def _handle_detection(self, frame: np.ndarray) -> None:
        """
        Handles object detection and recording logic.
        """
        current_time = time.time()
        detected, confidence = self.detector.detect(frame)
        high_threshold = self.config.get('detection_threshold', 0.7)
        low_threshold = self.config.get('low_detection_threshold', 0.3)

        if (confidence >= high_threshold).any() and not self.recording_in_progress:
            if current_time - self.last_detection_time > self.config['pre_event_seconds']:
                logger.info("Detection triggered, starting recording...")
                pre_event_frames = list(self.stream_handler.frame_buffer.queue)

                self.recording_in_progress = True
                threading.Thread(
                    target=self._record_clip,
                    args=(pre_event_frames,),
                    daemon=True
                ).start()
                self.last_detection_time = current_time
            elif (confidence >= low_threshold).any():
                self.last_detection_time = current_time
            elif (confidence < low_threshold).all():
                self.recorded_detection = False

    def _record_clip(self, pre_event_frames: list) -> None:
        """
        Records a video clip starting from pre-event frames.
        """
        self.recorder.record_clip(pre_event_frames, self.stream_handler.get_current_frame)
        self.recording_in_progress = False

    def _log_performance(self) -> None:
        """
        Logs performance metrics for the application.
        """
        elapsed_time = time.time() - self.start_time
        fps = self.frame_counter / elapsed_time if elapsed_time > 0 else 0
        logger.info("Current FPS: %.2f", fps)

    def _validate_config(self) -> None:
        """
        Validates the application configuration.
        """
        required_keys = ['fps', 'pre_event_seconds', 'post_event_seconds']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required configuration key: {key}")

        if not (1 <= self.config['fps'] <= 60):
            raise ValueError("FPS must be between 1 and 60")

        if self.config['pre_event_seconds'] < 0:
            raise ValueError("Pre-event seconds cannot be negative")

        if self.config['post_event_seconds'] < 0:
            raise ValueError("Post-event seconds cannot be negative")

    def _monitor_resources(self) -> None:
        """
        Monitors system resources and logs usage.
        """
        cpu_percent = psutil.cpu_percent()
        memory_info = psutil.virtual_memory()
        logger.info("CPU Usage: %.1f%%, Memory Usage: %.1f%%", cpu_percent, memory_info.percent)
