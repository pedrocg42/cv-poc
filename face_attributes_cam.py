import asyncio
from src.types.selfie_data import SelfieData
import cv2
import time
from src.processors.infrastructure.face_detector_onnx_processor import FaceDetectorOnnxProcessor


async def main():
    face_detector_processor = FaceDetectorOnnxProcessor(".models/version-RFB-320-int8.onnx")

    # Open video
    cap = cv2.VideoCapture(0)
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
