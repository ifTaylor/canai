import pyrealsense2 as rs
import numpy as np
import time
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class RealSenseCamera:
    """
    A class for interfacing with an Intel RealSense camera.

    Attributes
    ----------
    config : yaml
        Configuration parameters for the RealSense camera.
    pipeline : rs.pipeline
        The RealSense pipeline for handling camera streaming.
    color_sensor : Optional[rs.sensor]
        The RGB color sensor if available.
    running : bool
        Indicates whether the camera is running.
    """

    def __init__(
        self,
        config: Dict[str, int] = {}
    ) -> None:
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.last_frame_time: float = 0.0

        self.width = config.get("width")
        self.height = config.get("height")
        self.fps = config.get("fps")
        self.depth_enabled = config.get("depth_enabled", False)
        self.exposure: int = config.get("exposure", 100)
        self.gain: int = config.get("gain", 50)
        self.white_balance: int = config.get("white_balance", 4500)
        self.contrast: int = config.get("contrast", 50)
        self.saturation: int = config.get("saturation", 50)
        self.gamma: int = config.get("gamma", 50)
        self.sharpness: int = config.get("sharpness", 50)

        self.target_frame_interval: float = 1.0 / self.fps

        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        if self.depth_enabled:
            self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)

        self.profile = self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color) if self.depth_enabled else None

        self.running = True
        time.sleep(2)
        self._configure_sensor()
        logger.info("RealSense Camera initialized successfully.")

    def _is_fps_supported(
        self,
        width: int,
        height: int,
        fps: int
    ) -> bool:
        """
        Checks if the specified resolution and FPS are supported by the RealSense camera.

        Parameters
        ----------
        width : int
            The width of the resolution.
        height : int
            The height of the resolution.
        fps : int
            The desired frames per second.

        Returns
        -------
        bool
            True if the resolution and FPS are supported, False otherwise.
        """
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = self.config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()

        for sensor in device.sensors:
            for profile in sensor.get_stream_profiles():
                vprofile = profile.as_video_stream_profile()
                if vprofile.width() == width and vprofile.height() == height and vprofile.fps() == fps:
                    return True
        return False

    def _configure_sensor(self) -> None:
        """
        Configures the camera sensor with manual settings.
        """
        if not self.pipeline:
            logger.error("Cannot configure sensor. Pipeline is not initialized.")
            return

        device = self.profile.get_device()
        for sensor in device.sensors:
            if sensor.get_info(rs.camera_info.name) == 'RGB Camera':
                self.color_sensor = sensor
                break

        if self.color_sensor is None:
            logger.error("Could not find the RGB color sensor.")
            raise RuntimeError("Could not find the RGB color sensor.")

        self.color_sensor.set_option(rs.option.enable_auto_exposure, 0)
        self.color_sensor.set_option(rs.option.enable_auto_white_balance, 0)
        self._set_sensor_options()
        logger.info("Camera sensor configured with manual settings.")

    def _set_sensor_options(self) -> None:
        """
        Applies the user-specified sensor settings to the camera.
        """
        if not self.color_sensor:
            logger.error("Sensor is not initialized.")
            return

        self.color_sensor.set_option(rs.option.exposure, self.exposure)
        self.color_sensor.set_option(rs.option.gain, self.gain)
        self.color_sensor.set_option(rs.option.white_balance, self.white_balance)
        self.color_sensor.set_option(rs.option.contrast, self.contrast)
        self.color_sensor.set_option(rs.option.saturation, self.saturation)
        self.color_sensor.set_option(rs.option.gamma, self.gamma)
        self.color_sensor.set_option(rs.option.sharpness, self.sharpness)
        logger.info("Camera sensor settings applied.")

    def _sync_frame_rate(self) -> None:
        """
        Synchronizes frame capture to maintain target frame rate.
        """
        current_time = time.time()
        elapsed = float(current_time - self.last_frame_time)
        sleep_time = max(0.0, float(self.target_frame_interval - elapsed))
        time.sleep(sleep_time)
        self.last_frame_time = time.time()

    def get_frame(self) -> Optional[np.ndarray]:
        """
        Captures a single frame from the camera.

        Returns
        -------
        Optional[np.ndarray]
            The captured frame as a NumPy array, or None if no frame was received.
        """
        if not self.pipeline or not self.running:
            logger.error("Camera pipeline is not initialized or has stopped.")
            return None

        try:
            self._sync_frame_rate()
            frames = self.pipeline.wait_for_frames()
            if self.depth_enabled:
                aligned_frames = self.align.process(frames)
                color_frame = aligned_frames.get_color_frame()
            else:
                color_frame = frames.get_color_frame()

            if not color_frame:
                logger.warning("No color frame received from RealSense.")
                return None

            return np.asanyarray(color_frame.get_data())

        except RuntimeError as e:
            logger.error("RealSense Runtime Error: %s", e, exc_info=True)
            return None

    def stop(self) -> None:
        """
        Stops the RealSense camera pipeline.
        """
        if self.pipeline and self.running:
            self.pipeline.stop()
            self.running = False
            logger.info("RealSense Camera stopped.")
