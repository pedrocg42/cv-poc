from abc import abstractmethod
from pathlib import Path
from src.types.selfie_data import SelfieData
from typing import Any

import onnxruntime as ort

from src.processors.domain.processor import Processor


class OnnxProcessor(Processor):
    """Processor for ONNX models."""

    def __init__(
        self,
        model_path: str | Path | None,
        providers: list[str] | None = None,
        session_options: ort.SessionOptions | None = None,
    ) -> None:
        """Initialize the ONNX processor.

        Args:
            model_path: Path to the ONNX model file
            providers: List of execution providers to use (e.g., ['CUDAExecutionProvider', 'CPUExecutionProvider'])
            session_options: Optional ONNX Runtime session options
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Configure session options
        if session_options is None:
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Configure providers
        if providers is None:
            providers = [
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
                "CoreMLExecutionProvider",
                "CPUExecutionProvider",
            ]

        # Create ONNX Runtime session
        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=providers,
            sess_options=session_options,
        )

        # Get model metadata
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.output_shapes = [output.shape for output in self.session.get_outputs()]

    @abstractmethod
    def preprocess(self, selfie_data: SelfieData) -> SelfieData:
        raise NotImplementedError("Child classes should implement this method")

    @abstractmethod
    def postprocess(self, output: Any, selfie_data: SelfieData) -> SelfieData:
        raise NotImplementedError("Child classes should implement this method")

    async def execute(self, selfie_data: SelfieData) -> SelfieData:
        """Run inference on the input data.

        Args:
            selfie_data: Input data to process

        Returns:
            Model output
        """
        # Preprocess input
        processed_input = self.preprocess(selfie_data)

        # Run inference
        output = self.session.run(self.output_names, {self.input_name: processed_input})

        # Postprocess output
        return self.postprocess(output, selfie_data)

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the loaded model.

        Returns:
            Dictionary containing model information
        """
        return {
            "input_name": self.input_name,
            "output_name": self.output_name,
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
            "providers": self.session.get_providers(),
        }
