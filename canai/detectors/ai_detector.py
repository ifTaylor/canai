import torch
import cv2
import warnings
import logging
import numpy as np
from typing import Optional, Tuple

warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)


class AIDetector:
    """
    A class for object detection using a YOLOv5 model.

    Attributes
    ----------
    detection_threshold : float
        Confidence threshold for detections to be considered valid.
    target_class_id : Optional[int]
        The target class ID to filter detections, or None for all detections.
    model : torch.nn.Module
        The YOLOv5 model loaded from a local path or the Ultralytics repository.
    """

    def __init__(
            self,
            model_path: Optional[str] = None,
            detection_threshold: float = 0.5,
            target_class_id: Optional[int] = None
    ) -> None:
        self.detection_threshold: float = detection_threshold
        self.target_class_id: Optional[int] = target_class_id

        try:
            if torch.cuda.is_available():
                device = 'cuda:0'
            else:
                device = 'cpu'

            if model_path is None:
                logger.info("Loading pretrained YOLOv5n model from Ultralytics hub.")
                self.model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True, device=device)
            else:
                logger.info("Loading custom YOLOv5 model from: %s", model_path)
                self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True,
                                            device=device)

            self.model.to(device).eval()
            logger.info("YOLOv5 model loaded successfully on %s.", device)
        except Exception as e:
            logger.critical("Failed to load YOLOv5 model: %s", e, exc_info=True)
            raise

    def detect(
            self,
            frame: np.ndarray
    ) -> Tuple[bool, np.ndarray]:
        """
        Detects objects in a given frame using the YOLOv5 model.

        Parameters
        ----------
        frame : np.ndarray
            The input frame in BGR format.

        Returns
        -------
        Tuple[bool, np.ndarray]
            A tuple containing:
            - `True` if an object is detected, `False` otherwise.
            - The frame with bounding boxes and labels drawn.
        """
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, (640, 640))  # Standard YOLOv5 input size

            # Run inference
            results = self.model(frame_rgb, size=640)  # Specify size for faster processing
            detections = results.xyxy[0].cpu().numpy()

            output_frame = frame.copy()
            detected = False

            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                if conf < self.detection_threshold:
                    continue

                label = self.model.names[int(cls)]
                label_text = f"{label} {conf:.2f}"

                cv2.rectangle(output_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(output_frame,
                              (int(x1), int(y1) - text_height - baseline),
                              (int(x1) + text_width, int(y1)),
                              (0, 0, 255), thickness=-1)
                cv2.putText(output_frame, label_text,
                            (int(x1), int(y1) - baseline),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                detected = True

            if not detected:
                logger.debug("No objects detected in the frame.")

            return detected, output_frame

        except Exception as e:
            logger.error("Error during detection: %s", e, exc_info=True)
            return False, frame