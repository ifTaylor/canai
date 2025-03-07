import cv2
import numpy as np
import logging
import time
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class WebcamCamera:
    """
    Handles video capture using OpenCV's VideoCapture for a built-in or external webcam.

    Attributes
    ----------
    cam_index : int
        The index of the webcam device.
    width : Optional[int]
        The width of the video frame.
    height : Optional[int]
        The height of the video frame.
    cap : cv2.VideoCapture
        The OpenCV VideoCapture object used to interface with the webcam.
    """

    def __init__(
            self,
            config: Dict[str, int] = {}
    ) -> None:
        self.cam_index = config.get("cam_index", 0)
        self.width = config.get("width")
        self.height = config.get("height")
        self.fps = config.get("fps", 30)

        self.cap: Optional[cv2.VideoCapture] = None
        self.last_frame_time: float = 0.0

        self.target_frame_interval: float = 1.0 / self.fps

        self.cap = cv2.VideoCapture(self.cam_index)

        if not self.cap.isOpened():
            logger.critical("Could not open webcam at index %s", self.cam_index)
            raise RuntimeError(f"Could not open webcam at index {self.cam_index}")

        if self.width and self.height:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        logger.info("Webcam initialized at index %d with resolution %dx%d at %d FPS",
                    self.cam_index, actual_width, actual_height, actual_fps)

    def get_frame(self) -> Optional[np.ndarray]:
        """
        Captures a single frame from the webcam.

        Returns
        -------
        Optional[np.ndarray]
            The captured frame as a NumPy array, or None if the frame could not be retrieved.
        """
        ret, frame = self.cap.read()
        if ret:
            return frame

        logger.warning("Failed to capture frame from webcam at index %s", self.cam_index)
        return None

    def stop(self) -> None:
        """
        Releases the webcam resource.
        """
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            logger.info("Webcam at index %s has been released", self.cam_index)
