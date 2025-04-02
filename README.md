# Face Analysis POC

This project demonstrates a proof of concept for face detection, recognition, and attribute extraction using multiple models and inference backends. The implementation uses asynchronous methods for efficient processing and includes visualization utilities.

## Features

- Face detection using InsightFace
- Face recognition using InsightFace
- Face attribute extraction using MediaPipe
- Support for multiple inference backends (CPU, CUDA, MPS)
- Asynchronous processing pipeline
- Visualization utilities for results
- Support for both image and video processing

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd face-analysis-poc
```

2. Create a virtual environment (recommended):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv venv
source .venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
uv pip install -r requirements.txt
```

## Dev

```bash
uv pip install -r dev-requirements.txt
pre-commit install
```

## Usage

### Face Attributes

```bash
python face_attributes_cam.py
```

### Using the Model Manager Directly

```python
from src.model_manager import ModelManager, InferenceBackend
from src.visualization import Visualizer

# Initialize the model manager
model_manager = ModelManager()
await model_manager.initialize(backend=InferenceBackend.CPU)  # or CUDA/MPS

# Process an image
image = cv2.imread('path/to/image.jpg')
detections = await model_manager.detect_faces(image)
recognitions = await model_manager.recognize_faces(image)
attributes = await model_manager.extract_attributes(image)

# Visualize results
visualizer = Visualizer()
result_image = visualizer.visualize_results(image, detections, recognitions, attributes)
```

## Project Structure

```
face-analysis-poc/
├── src/
│   ├── model_manager.py    # Model management and inference
│   ├── visualization.py    # Visualization utilities
│   └── main.py            # Example usage
├── requirements.txt       # Project dependencies
└── README.md             # This file
```

## Supported Inference Backends

- CPU (default)
- CUDA (for NVIDIA GPUs)
- MPS (for Apple Silicon)

## Requirements

- Python 3.8+
- OpenCV
- PyTorch
- InsightFace
- MediaPipe
- NumPy
- Other dependencies listed in requirements.txt

## Notes

- The project uses InsightFace for face detection and recognition
- MediaPipe is used for face attribute extraction
- All processing is done asynchronously for better performance
- The visualization module provides utilities for drawing bounding boxes, landmarks, and text overlays

## License

This project is licensed under the MIT License - see the LICENSE file for details.
