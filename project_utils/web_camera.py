import cv2
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class WebcamCamera:
    """
    Handles video capture using OpenCV's VideoCapture for a built-in or external webcam.

    Attributes
    ----------
    cam_index : int
        The index of the webcam device.
    width : int
        The width of the video frame.
    height : int
        The height of the video frame.
    cap : cv2.VideoCapture
        The OpenCV VideoCapture object used to interface with the webcam.
    """

    def __init__(
        self, cam_index: int = 0,
        width: int = 640,
        height: int = 480
    ) -> None:
        self.cam_index: int = cam_index
        self.width: int = width
        self.height: int = height
        self.cap: cv2.VideoCapture = cv2.VideoCapture(self.cam_index)

        if not self.cap.isOpened():
            logger.critical("Could not open webcam at index %d", self.cam_index)
            raise RuntimeError(f"Could not open webcam at index {self.cam_index}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        logger.info("Webcam initialized at index %d with resolution %dx%d", self.cam_index, self.width, self.height)

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

        logger.warning("Failed to capture frame from webcam at index %d", self.cam_index)
        return None

    def stop(self) -> None:
        """
        Releases the webcam resource.
        """
        if self.cap.isOpened():
            self.cap.release()
            logger.info("Webcam at index %d has been released", self.cam_index)
