from pathlib import Path
from typing import Any

import numpy as np

from src.processors.domain.onnx_processor import OnnxProcessor
from src.types.face import Identity
from src.types.selfie_data import SelfieData


class FaceRecognitionProcessor(OnnxProcessor):
    """Processor for face recognition using ArcFace model."""

    def __init__(self, model_path: str | Path = ".models/arcfaceresnet100-11-int8.onnx", **kwargs) -> None:
        """Initialize the face recognition processor.

        Args:
            model_path: Path to the ONNX model file
            providers: List of execution providers to use
        """
        super().__init__(model_path, **kwargs)
        self.input_size = 112  # ArcFace typically expects 112x112 input
        self.image_mean = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.image_std = np.array([1.0, 1.0, 1.0], np.float32)

    def preprocess(self, selfie_data: SelfieData) -> np.ndarray:
        """Preprocess the input image for face recognition.

        Args:
            selfie_data: Input data containing the image and face detections

        Returns:
            Preprocessed input tensor
        """
        if not selfie_data.faces:
            return None

        # Get the first face for now (can be extended to handle multiple faces)
        aligned_faces = []
        for face in selfie_data.faces:
            if face.aligned_face is not None:
                aligned_face = face.aligned_face
            else:
                image = selfie_data.image.copy()

                # Extract face region
                aligned_face = face.align_face(image, self.input_size)
                face.aligned_face = aligned_face.copy()

            # Convert to NCHW format with batch dimension
            aligned_face = aligned_face.astype(np.float32)
            aligned_face -= self.image_mean
            aligned_face /= self.image_std
            aligned_face = np.transpose(aligned_face, [2, 0, 1])
            aligned_face = np.expand_dims(aligned_face, axis=0)
            aligned_faces.append(aligned_face)

        return aligned_faces

    def postprocess(self, outputs: Any, selfie_data: SelfieData) -> SelfieData:
        """Postprocess the model output to get face embeddings.

        Args:
            output: Model output containing face embeddings
            selfie_data: Original input data

        Returns:
            Updated SelfieData with face embeddings
        """
        if not selfie_data.faces or not outputs:
            return selfie_data

        for face, output in zip(selfie_data.faces, outputs, strict=True):
            face.identity = Identity(embedding=output[0])

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
        for aligned_face in processed_input:
            # Run inference
            output = self.session.run(self.output_names, {self.input_name: aligned_face})
            outputs.append(output)

        # Postprocess output
        return self.postprocess(outputs, selfie_data)
