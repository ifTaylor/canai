import threading
import cv2
import logging
from typing import Any

logger = logging.getLogger(__name__)

class ObjectDetectionApp:
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

    def run(self) -> None:
        """
        Runs the object detection application, continuously capturing frames,
        detecting objects, and triggering event recording.

        This function starts the video stream handler in a separate thread
        and continuously processes video frames to detect objects. When an object
        is detected, a recording session is triggered.

        The application can be stopped by pressing the 'q' key.
        """
        capture_thread = threading.Thread(target=self.stream_handler.run, daemon=True)
        capture_thread.start()
        logger.info("Detection system started. Press 'q' to exit.")

        try:
            while True:
                frame = self.stream_handler.get_current_frame()
                if frame is None:
                    continue

                # Run detection on the current frame.
                detected, _ = self.detector.detect(frame)
                if detected and not self.recorded_detection:
                    logger.info("Detection triggered, starting recording...")
                    pre_event_frames = list(self.stream_handler.frame_buffer.queue)

                    threading.Thread(
                        target=self.recorder.record_clip,
                        args=(pre_event_frames, self.stream_handler.get_current_frame),
                        daemon=True
                    ).start()
                    self.recorded_detection = True
                elif not detected:
                    self.recorded_detection = False

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
