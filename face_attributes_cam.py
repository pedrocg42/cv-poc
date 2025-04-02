import asyncio
import time

import cv2

from src.identity_manager import IdentityManager
from src.processors.infrastructure.face_recognition_processor import FaceRecognitionProcessor
from src.processors.infrastructure.retinaface.retina_face_onnx_processor import RetinaFaceOnnxProcessor
from src.types.selfie_data import SelfieData


async def main():
    face_detector_processor = RetinaFaceOnnxProcessor()
    face_recognition_processor = FaceRecognitionProcessor()

    identity_manager = IdentityManager()

    # Open video
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        raise ValueError("Could not open camera")

    # Initialize FPS variables
    prev_frame_time = time.time()
    new_frame_time = time.time()
    fps = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Calculate FPS
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            fps = int(fps)

            selfie_data = SelfieData(image=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Run inference
            selfie_data = await face_detector_processor(selfie_data)
            selfie_data = await face_recognition_processor(selfie_data)
            selfie_data = identity_manager.identify(selfie_data)

            # Show
            draw_frame = selfie_data.draw()

            # Display FPS on frame
            cv2.putText(draw_frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Optional: Display frame
            cv2.imshow("Face Analysis", cv2.cvtColor(draw_frame, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(main())
