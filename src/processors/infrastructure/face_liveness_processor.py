from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.processors.domain.onnx_processor import OnnxProcessor
from src.types.selfie_data import SelfieData


# https://github.com/ffletcherr/face-recognition-liveness
class FaceLivenessProcessor(OnnxProcessor):
    """Processor for face liveness detection using ONNX models."""

    def __init__(self, model_path: str | Path = ".models/OULU_Protocol_2_model_0_0.onnx", **kwargs) -> None:
        """Initialize the face liveness processor.

        Args:
            model_path: Path to the ONNX model file
            providers: List of execution providers to use
        """
        super().__init__(model_path, **kwargs)
        self.input_size = (224, 224)
        self.image_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.image_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def preprocess(self, selfie_data: SelfieData) -> np.ndarray:
        """Preprocess the input image for face liveness detection.

        Args:
            selfie_data: Input data containing the image

        Returns:
            Preprocessed input tensor
        """
        # Get the image
        image = selfie_data.image.copy()

        if selfie_data.faces is None:
            return None

        face_images = [None] * len(selfie_data.faces)
        for i, face in enumerate(selfie_data.faces):
            face_image = face.bb.crop_image(image)
            # Resize to model input size
            face_image = cv2.resize(face_image, self.input_size)

            # Convert to float32 and normalize
            face_image = face_image.astype(np.float32) / 255.0
            face_image = (face_image - self.image_mean) / self.image_std

            # Convert to NCHW format with batch dimension
            face_image = np.transpose(face_image, [2, 0, 1])
            face_image = np.expand_dims(face_image, axis=0)
            face_images[i] = face_image

        return face_images

    def postprocess(self, outputs: Any, selfie_data: SelfieData) -> SelfieData:
        """Postprocess the model output to get liveness scores.

        Args:
            output: Model output containing liveness scores
            selfie_data: Original input data

        Returns:
            Updated SelfieData with liveness scores
        """
        if not outputs:
            return selfie_data

        # Assuming the model outputs a single score between 0 and 1
        # where 1 means real face and 0 means fake/spoof
        for face, output in zip(selfie_data.faces, outputs, strict=True):
            output_pixel, output_binary = output
            face.liveness_score = (np.mean(output_pixel.flatten()) + np.mean(output_binary.flatten())) / 2.0

        return selfie_data

    async def execute(self, selfie_data: SelfieData) -> SelfieData:
        """Run inference on the input data.

        Args:
            selfie_data: Input data to process

        Returns:
            Model output
        """
        # Preprocess input
        processed_input = self.preprocess(selfie_data)

        if processed_input is None:
            return selfie_data

        outputs = []
        for cropped_face in processed_input:
            # Run inference
            output = self.session.run(self.output_names, {self.input_name: cropped_face})
            outputs.append(output)

        # Postprocess output
        return self.postprocess(outputs, selfie_data)
