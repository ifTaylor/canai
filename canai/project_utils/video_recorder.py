import cv2
import datetime
import time
import os
import logging
import numpy as np
from typing import Callable, List, Optional, Tuple, Any

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
        self.output_dir: str = output_dir or "./clips/"

    def record_clip(self, pre_event_frames: List[np.ndarray], get_current_frame: Callable[[], np.ndarray]) -> None:
        """
        Record a video clip starting from pre-event frames.

        Parameters
        ----------
        pre_event_frames : List[np.ndarray]
            List of frames captured before the event
        get_current_frame : Callable[[], np.ndarray]
            Function to get the current frame from the stream
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(self.output_dir, exist_ok=True)

            # Generate timestamped filename
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = os.path.join(self.output_dir, f'{timestamp}.avi')

            # Get frame dimensions from first frame
            if not pre_event_frames:
                logger.warning("No pre-event frames available")
                return

            frame_height, frame_width = pre_event_frames[0].shape[:2]

            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                output_path,
                fourcc,
                self.fps,
                (frame_width, frame_height)
            )

            # Write pre-event frames
            for frame in pre_event_frames:
                video_writer.write(frame)

            # Record post-event frames
            start_time = time.time()
            while time.time() - start_time < self.post_event_seconds:
                frame = get_current_frame()
                if frame is not None:
                    video_writer.write(frame)
                time.sleep(1/self.fps)

            logger.info(f"Successfully recorded clip to {output_path}")

        except Exception as e:
            logger.error(f"Error recording video clip: {e}", exc_info=True)
        finally:
            if 'video_writer' in locals():
                video_writer.release()
