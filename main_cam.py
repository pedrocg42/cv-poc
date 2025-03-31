import asyncio
import cv2
import numpy as np
from model_manager import ModelManager, InferenceBackend
from visualization import Visualizer
import argparse
from pathlib import Path


async def main():
    # Initialize visualizer
    visualizer = Visualizer()

    # Open video
    cap = cv2.VideoCapture()
    if not cap.isOpened():
        raise ValueError("Could not open camera")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run inference
            detections = await model_manager.detect_faces(frame)
            recognitions = await model_manager.recognize_faces(frame)
            attributes = await model_manager.extract_attributes(frame)

            # Visualize results
            result_frame = visualizer.visualize_results(frame, detections, recognitions, attributes)

            # Write frame
            out.write(result_frame)

            # Optional: Display frame
            cv2.imshow("Face Analysis", result_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(main())
