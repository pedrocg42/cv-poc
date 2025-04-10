from pathlib import Path

import numpy as np
from ultralytics import YOLOWorld
from ultralytics.engine.results import Results

from src.processors.domain.processor import Processor
from src.types.base import BoundingBox, Point
from src.types.face import FaceAttribute
from src.types.selfie_data import SelfieData


class FaceYOLOWorldProcessor(Processor):
    """Processor for object detection using YOLO World model."""

    def __init__(
        self,
        model_path: str | Path = "yolov8s-worldv2.pt",
        classes: list[str] | None = None,
        **kwargs,
    ) -> None:
        """Initialize the YOLO World processor.

        Args:
            model_path: Path to the ONNX model file
            classes: List of classes to detect. If None, uses default classes
            providers: List of execution providers to use
        """
        super().__init__(**kwargs)

        self.model_path = model_path
        self.default_classes = [
            "person",
            # "safety glasses",
            "glasses",
            # "safety helmet",
            "helmet",
            "ear plug",
        ]
        self.classes = classes if classes else self.default_classes

        self.model = YOLOWorld(str(model_path))
        self.model.set_classes(self.classes)

    def preprocess(self, selfie_data: SelfieData) -> list[np.ndarray]:
        """Preprocess the input image for face attribute detection.

        Args:
            selfie_data: Input data containing the image and face detections

        Returns:
            Preprocessed input tensor
        """
        if not selfie_data.faces:
            return None

        return selfie_data.image.copy()

    def postprocess(self, result: Results, selfie_data: SelfieData) -> SelfieData:
        face_attributes = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].numpy().astype(int).tolist()
            confidence = float(box.conf[0])
            if confidence > 0.3:
                face_attributes.append(
                    FaceAttribute(
                        bb=BoundingBox(top_left=Point(x=x1, y=y1), bottom_right=Point(x=x2, y=y2)),
                        confidence=confidence,
                        name=result.names[int(box.cls[0])],
                    )
                )
        selfie_data.face_attributes = face_attributes
        return selfie_data

    async def execute(self, selfie_data: SelfieData) -> SelfieData:
        """Run inference on the input data.

        Args:
            selfie_data: Input data to process

        Returns:
            Model output with detections
        """
        # Preprocess input
        processed_input = self.preprocess(selfie_data)

        if processed_input is None:
            return selfie_data

        # Run inference
        results = self.model(processed_input)

        # Postprocess output
        return self.postprocess(results[0], selfie_data)
