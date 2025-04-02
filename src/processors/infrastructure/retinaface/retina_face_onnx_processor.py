from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.processors.domain.onnx_processor import OnnxProcessor
from src.processors.infrastructure.retinaface.box_utils import decode, decode_landm, py_cpu_nms
from src.processors.infrastructure.retinaface.configs import cfg_mnet
from src.processors.infrastructure.retinaface.prior_box import PriorBox
from src.types.base import BoundingBox, Point
from src.types.face import Face, FaceLandmarks
from src.types.selfie_data import SelfieData


class RetinaFaceOnnxProcessor(OnnxProcessor):
    """Processor for face detection using ONNX models."""

    def __init__(
        self,
        model_path: str | Path = ".models/retinaface.onnx",
        confidence_threshold: float = 0.7,
        nms_threshold: float = 0.5,
        top_k: int = 100,
        **kwargs,
    ) -> None:
        """Initialize the face detector processor.

        Args:
            model_path: Path to the ONNX model file
            confidence_threshold: Confidence threshold for face detection
            nms_threshold: Intersection over union threshold for NMS
            top_k: Keep top k results. If k <= 0, keep all results
            providers: List of execution providers to use
        """
        super().__init__(model_path, **kwargs)
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self.input_size = 640
        self.image_mean = np.array([104.0, 117.0, 123.0])

        self.config = cfg_mnet

    def preprocess(self, selfie_data: SelfieData) -> np.ndarray:
        """Preprocess the input image for face detection.

        Args:
            selfie_data: Input data containing the image

        Returns:
            Preprocessed input tensor
        """

        image = selfie_data.image.copy()

        # Resize
        ratio = self.input_size / max(image.shape[:2])
        image = cv2.resize(selfie_data.image, (0, 0), fx=ratio, fy=ratio)

        # Normalize using mean and scale
        image = image - self.image_mean

        # Convert to NCHW format with batch dimension
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)

        return image

    def postprocess(self, output: Any, selfie_data: SelfieData) -> SelfieData:
        """Postprocess the model output to get face detections.

        Args:
            output: Model output containing confidences and boxes
            selfie_data: Original input data

        Returns:
            Updated SelfieData with detected faces
        """

        # Get image dimensions
        height, width = selfie_data.image.shape[:2]

        loc, conf, landms = output

        ratio = self.input_size / max(height, width)
        new_size = (self.input_size, int(width * ratio)) if height >= width else (int(height * ratio), self.input_size)
        priors = PriorBox(self.config, image_size=(new_size), format="numpy").forward()

        # Process model output using custom prediction pipeline
        boxes = decode(loc[0], priors, self.config["variance"])
        landms = decode_landm(landms[0], priors, self.config["variance"])
        scores = conf[0, :, 1]

        # ignore low scores
        inds = np.where(scores > self.confidence_threshold)[0]
        if len(inds) == 0:
            return selfie_data
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][: self.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        keep = py_cpu_nms(boxes, scores, self.nms_threshold)
        boxes = boxes[keep]
        landms = landms[keep]
        scores = scores[keep]

        scale = np.array([width, height], dtype=np.float64)
        boxes *= np.tile(scale, 2)
        boxes = boxes.astype(int)
        landms *= np.tile(scale, 5)
        landms = landms.astype(int)

        if len(boxes) == 0:
            return selfie_data

        # Create Face objects for each detection
        faces = []
        for box, landm, confidence in zip(boxes, landms, scores, strict=True):
            # Create bounding box
            bb = BoundingBox(top_left=Point(x=box[0], y=box[1]), bottom_right=Point(x=box[2], y=box[3]))

            landmarks = FaceLandmarks(
                left_eye=Point(x=landm[0], y=landm[1]),
                right_eye=Point(x=landm[2], y=landm[3]),
                nose=Point(x=landm[4], y=landm[5]),
                left_mouth=Point(x=landm[6], y=landm[7]),
                right_mouth=Point(x=landm[8], y=landm[9]),
            )

            # Create face object
            face = Face(bb=bb, landmarks=landmarks, identity=None, metadata={"confidence": float(confidence)})

            faces.append(face)

        # Update selfie_data with detected faces
        selfie_data.faces = faces
        return selfie_data
