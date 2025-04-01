from pathlib import Path
from src.types.base import BoundingBox, Point
from src.types.face import Face, FaceLandmarks
from src.types.selfie_data import SelfieData
from typing import Any

import cv2
import numpy as np

from src.processors.domain.onnx_processor import OnnxProcessor


class FaceDetectorOnnxProcessor(OnnxProcessor):
    """Processor for face detection using ONNX models."""

    def __init__(
        self,
        model_path: str | Path,
        confidence_threshold: float = 0.7,
        nms_threshold: float = 0.5,
        top_k: int = -1,
        providers: list[str] | None = None,
    ) -> None:
        """Initialize the face detector processor.

        Args:
            model_path: Path to the ONNX model file
            confidence_threshold: Confidence threshold for face detection
            nms_threshold: Intersection over union threshold for NMS
            top_k: Keep top k results. If k <= 0, keep all results
            providers: List of execution providers to use
        """
        super().__init__(model_path, providers)
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self.input_size = (320, 240)  # Model expects 320x240 input
        self.image_mean = np.array([127, 127, 127])

    def preprocess(self, selfie_data: SelfieData) -> np.ndarray:
        """Preprocess the input image for face detection.

        Args:
            selfie_data: Input data containing the image

        Returns:
            Preprocessed input tensor
        """

        # Resize
        image = cv2.resize(selfie_data.image, self.input_size)

        # Normalize using mean and scale
        image = (image - self.image_mean) / 128

        # Convert to NCHW format with batch dimension
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)

        return image

    def _area_of(self, left_top: np.ndarray, right_bottom: np.ndarray) -> np.ndarray:
        """Compute the areas of rectangles given two corners.

        Args:
            left_top (N, 2): left top corner
            right_bottom (N, 2): right bottom corner

        Returns:
            area (N): return the area
        """
        hw = np.clip(right_bottom - left_top, 0.0, None)
        return hw[..., 0] * hw[..., 1]

    def _iou_of(self, boxes0: np.ndarray, boxes1: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Return intersection-over-union (Jaccard index) of boxes.

        Args:
            boxes0 (N, 4): ground truth boxes
            boxes1 (N or 1, 4): predicted boxes
            eps: a small number to avoid 0 as denominator

        Returns:
            iou (N): IoU values
        """
        overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
        overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

        overlap_area = self._area_of(overlap_left_top, overlap_right_bottom)
        area0 = self._area_of(boxes0[..., :2], boxes0[..., 2:])
        area1 = self._area_of(boxes1[..., :2], boxes1[..., 2:])
        return overlap_area / (area0 + area1 - overlap_area + eps)

    def _hard_nms(
        self, box_scores: np.ndarray, iou_threshold: float, top_k: int = -1, candidate_size: int = 200
    ) -> np.ndarray:
        """Perform hard non-maximum-supression to filter out boxes with iou greater than threshold.

        Args:
            box_scores (N, 5): boxes in corner-form and probabilities
            iou_threshold: intersection over union threshold
            top_k: keep top_k results. If k <= 0, keep all the results
            candidate_size: only consider the candidates with the highest scores

        Returns:
            picked: a list of indexes of the kept boxes
        """
        scores = box_scores[:, -1]
        boxes = box_scores[:, :-1]
        picked = []
        indexes = np.argsort(scores)
        indexes = indexes[-candidate_size:]
        while len(indexes) > 0:
            current = indexes[-1]
            picked.append(current)
            if 0 < top_k == len(picked) or len(indexes) == 1:
                break
            current_box = boxes[current, :]
            indexes = indexes[:-1]
            rest_boxes = boxes[indexes, :]
            iou = self._iou_of(
                rest_boxes,
                np.expand_dims(current_box, axis=0),
            )
            indexes = indexes[iou <= iou_threshold]

        return box_scores[picked, :]

    def _predict(
        self, width: int, height: int, confidences: np.ndarray, boxes: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Select boxes that contain human faces.

        Args:
            width: original image width
            height: original image height
            confidences (N, 2): confidence array
            boxes (N, 4): boxes array in corner-form

        Returns:
            boxes (k, 4): an array of boxes kept
            labels (k): an array of labels for each boxes kept
            probs (k): an array of probabilities for each boxes being in corresponding labels
        """
        boxes = boxes[0]
        confidences = confidences[0]

        picked_box_probs = []
        picked_labels = []
        for class_index in range(1, confidences.shape[1]):
            probs = confidences[:, class_index]
            mask = probs > self.confidence_threshold
            probs = probs[mask]

            if probs.shape[0] == 0:
                continue
            subset_boxes = boxes[mask, :]
            box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
            box_probs = self._hard_nms(
                box_probs,
                iou_threshold=self.nms_threshold,
                top_k=self.top_k,
            )
            picked_box_probs.append(box_probs)
            picked_labels.extend([class_index] * box_probs.shape[0])
        if not picked_box_probs:
            return np.array([]), np.array([]), np.array([])
        picked_box_probs = np.concatenate(picked_box_probs)
        picked_box_probs[:, 0] *= width
        picked_box_probs[:, 1] *= height
        picked_box_probs[:, 2] *= width
        picked_box_probs[:, 3] *= height
        return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]

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

        # Process model output using custom prediction pipeline
        boxes, labels, confidences = self._predict(width, height, output[0], output[1])

        if len(boxes) == 0:
            return selfie_data

        # Create Face objects for each detection
        faces = []
        for box, confidence in zip(boxes, confidences, strict=True):
            x1, y1, x2, y2 = box

            # Create bounding box
            bb = BoundingBox(top_left=Point(x=x1, y=y1), bottom_right=Point(x=x2, y=y2))

            # Create face landmarks (placeholder for now)
            landmarks = FaceLandmarks(
                left_eye=Point(x=x1 + (x2 - x1) // 4, y=y1 + (y2 - y1) // 3),
                right_eye=Point(x=x1 + 3 * (x2 - x1) // 4, y=y1 + (y2 - y1) // 3),
                nose=Point(x=x1 + (x2 - x1) // 2, y=y1 + (y2 - y1) // 2),
                left_mouth=Point(x=x1 + (x2 - x1) // 4, y=y1 + 2 * (y2 - y1) // 3),
                right_mouth=Point(x=x1 + 3 * (x2 - x1) // 4, y=y1 + 2 * (y2 - y1) // 3),
            )

            # Create face object
            face = Face(bb=bb, landmarks=landmarks, identity=None, metadata={"confidence": float(confidence)})

            faces.append(face)

        # Update selfie_data with detected faces
        selfie_data.faces = faces
        return selfie_data
