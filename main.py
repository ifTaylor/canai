import yaml
import logging
from typing import Any, Dict

from canai.detectors.ai_detector import AIDetector
from canai.project_utils.video_recorder import EventClipRecorder
from canai.project_utils.web_camera import WebcamCamera
from canai.stream.video_stream_handler import VideoStreamHandler
from canai.canai import CanAI

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def load_yaml_config(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error("Failed to load configuration file %s: %s", path, e, exc_info=True)
        raise

def main() -> None:
    logger.info("Starting object detection application...")
    try:
        camera_config = load_yaml_config("configs/webcam_config.yaml")
        camera = WebcamCamera(config=camera_config)
        logger.info("Webcam initialized with config: %s", camera_config)

        # Load application configuration
        app_config = load_yaml_config("configs/app_config.yaml")

        # Create AI detector instance
        detector = AIDetector(
            model_path=app_config.get("model_path", app_config.get("model_path")),
            detection_threshold=app_config.get("detection_threshold", 0.5),
            target_class_id=app_config.get("target_class_id", None)
        )
        logger.info("AI Detector initialized with model: %s", app_config.get("model_path", "models/best14.pt"))

        # Extract application settings
        fps = app_config.get("fps", 60)
        pre_event_seconds = app_config.get("pre_event_seconds", 2)
        post_event_seconds = app_config.get("post_event_seconds", 20)
        max_pre_frames = int(pre_event_seconds * fps)

        # Create event recorder
        recorder = EventClipRecorder(fps=fps, post_event_seconds=post_event_seconds)
        logger.info("EventClipRecorder initialized with FPS: %d, post_event_seconds: %d", fps, post_event_seconds)

        # Create video stream handler
        stream_handler = VideoStreamHandler(camera, detector, max_pre_frames)
        logger.info("VideoStreamHandler initialized with max_pre_frames: %d", max_pre_frames)

        # Create and run object detection application
        app = CanAI(camera, detector, recorder, stream_handler, app_config)
        logger.info("CanAI initialized. Running application...")

        app.run()

    except Exception as e:
        logger.critical("Fatal error in main application loop: %s", e, exc_info=True)
        raise

if __name__ == "__main__":
    main()
