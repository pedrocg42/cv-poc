import asyncio
from glob import glob
from pathlib import Path

import cv2
import numpy as np

from src.processors.infrastructure.face_recognition_processor import FaceRecognitionProcessor
from src.processors.infrastructure.retinaface.retina_face_onnx_processor import RetinaFaceOnnxProcessor
from src.types.face import Identity
from src.types.selfie_data import SelfieData


async def extract_identity(
    image_path: str, face_detector: RetinaFaceOnnxProcessor, face_recognizer: FaceRecognitionProcessor
) -> Identity | None:
    """Extract face identity from an image.

    Args:
        image_path: Path to the image file
        face_detector: Face detection processor
        face_recognizer: Face recognition processor

    Returns:
        Identity object if face is detected, None otherwise
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return None

    # Convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create selfie data
    selfie_data = SelfieData(image=image)

    # Run face detection
    selfie_data = await face_detector(selfie_data)

    if not selfie_data.faces:
        print(f"No faces detected in: {image_path}")
        return None

    # Run face recognition
    selfie_data = await face_recognizer(selfie_data)

    # Get the first face's identity
    if selfie_data.faces and selfie_data.faces[0].identity:
        return selfie_data.faces[0].identity

    return None


async def main():
    # Initialize processors
    face_detector = RetinaFaceOnnxProcessor()
    face_recognizer = FaceRecognitionProcessor()

    # Create identities directory if it doesn't exist
    identities_dir = Path(".identities")
    identities_dir.mkdir(exist_ok=True)

    for identity_id in identities_dir.glob("*"):
        if identity_id.is_dir():
            images_paths = glob(str(identity_id / "*.jpeg"))

            embeddings = []
            for image_path in images_paths:
                print(f"Processing: {image_path}")
                identity = await extract_identity(str(image_path), face_detector, face_recognizer)

                if identity:
                    # Save embeddings as numpy array
                    embeddings.append(identity.embedding)

            embeddings = np.asarray(embeddings, dtype=np.float32)
            np.save(identity_id / "face_recognition_embeddings.npy", embeddings)


if __name__ == "__main__":
    asyncio.run(main())
