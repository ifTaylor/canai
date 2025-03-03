import cv2
import datetime
import time
import os
import logging
import numpy as np
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)

class EventClipRecorder:
    """
    Handles recording of video clips that include pre-event and post-event frames.

    Attributes
    ----------
    fps : int
        Frames per second for the recorded clip.
    post_event_seconds : int
        Duration (in seconds) to record after the detection event.
    output_dir : str
        Directory where the recorded clips are saved.
    """

    def __init__(
        self,
        fps: int,
        post_event_seconds: int,
        output_dir: Optional[str] = None
    ) -> None:
        self.fps: int = fps
        self.post_event_seconds: int = post_event_seconds
        self.output_dir: str = output_dir or ""

    def _sync_frame_rate(
        self,
        last_frame_time: float,
        frame_interval: float
    ) -> float:
        """
        Ensures recorded frames are spaced correctly to maintain playback speed.

        Parameters
        ----------
        last_frame_time : float
            Timestamp of the last recorded frame.
        frame_interval : float
            Expected time interval between frames.

        Returns
        -------
        float
            The updated timestamp of the last recorded frame.
        """
        next_frame_time = last_frame_time + frame_interval
        sleep_time = max(0, next_frame_time - time.time())
        time.sleep(sleep_time)
        return time.time()

    def _write_frames(
        self,
        frames: List[np.ndarray],
        out: cv2.VideoWriter,
        start_time: float
    ) -> float:
        """
        Writes pre-event frames to the video file.

        Parameters
        ----------
        frames : List[np.ndarray]
            List of frames to be written to the video.
        out : cv2.VideoWriter
            Video writer object.
        start_time : float
            Start time of the recording.

        Returns
        -------
        float
            Timestamp of the last recorded frame.
        """
        frame_interval = 1.0 / self.fps
        last_frame_time = start_time
        for frame in frames:
            out.write(frame)
            last_frame_time = self._sync_frame_rate(last_frame_time, frame_interval)
        return last_frame_time

    def _write_post_frames(
        self,
        current_frame_func: Callable[[], Optional[np.ndarray]],
        out: cv2.VideoWriter,
        last_frame_time: float
    ) -> float:
        """
        Captures and writes post-event frames after detection.

        Parameters
        ----------
        current_frame_func : Callable[[], Optional[np.ndarray]]
            Function that retrieves the current frame.
        out : cv2.VideoWriter
            Video writer object.
        last_frame_time : float
            Timestamp of the last recorded frame.

        Returns
        -------
        float
            Timestamp of the last recorded frame after post-event frames.
        """
        frame_interval = 1.0 / self.fps
        num_post_frames = int(self.fps * self.post_event_seconds)

        for _ in range(num_post_frames):
            frame = current_frame_func()
            if frame is not None:
                out.write(frame)
            last_frame_time = self._sync_frame_rate(last_frame_time, frame_interval)

        return last_frame_time

    def record_clip(
        self,
        pre_frames: List[np.ndarray],
        current_frame_func: Callable[[], Optional[np.ndarray]]
    ) -> None:
        """
        Records a video clip that includes pre-event and post-event frames.

        Parameters
        ----------
        pre_frames : List[np.ndarray]
            List of pre-event frames captured before the detection event.
        current_frame_func : Callable[[], Optional[np.ndarray]]
            Function to retrieve current frames for post-event recording.
        """
        if not pre_frames:
            logger.error("No pre-event frames to record.")
            return

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"event_{timestamp}.avi"
        if self.output_dir:
            filename = os.path.join(self.output_dir, filename)

        h, w, _ = pre_frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(filename, fourcc, self.fps, (w, h))

        if not out.isOpened():
            logger.error("Could not open video writer for file: %s", filename)
            return

        logger.info("Recording started: %s", filename)
        start_time = time.time()
        last_frame_time = self._write_frames(pre_frames, out, start_time)
        self._write_post_frames(current_frame_func, out, last_frame_time)
        out.release()

        logger.info("Recording saved: %s", filename)
