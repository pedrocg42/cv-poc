import argparse
import asyncio

import cv2
from model_manager import InferenceBackend, ModelManager

from visualization import Visualizer


async def process_image(image_path: str, output_path: str, model_manager: ModelManager):
    # Initialize visualizer
    visualizer = Visualizer()

    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Run inference
    detections = await model_manager.detect_faces(image)
    recognitions = await model_manager.recognize_faces(image)
    attributes = await model_manager.extract_attributes(image)

    # Visualize results
    result_image = visualizer.visualize_results(image, detections, recognitions, attributes)

    # Save result
    cv2.imwrite(output_path, result_image)
    print(f"Results saved to: {output_path}")


async def process_video(video_path: str, output_path: str, model_manager: ModelManager):
    # Initialize visualizer
    visualizer = Visualizer()

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

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


async def add_identities(model_manager: ModelManager, identities_dir: str):
    """Add identities from a directory of images."""
    print(f"Adding identities from directory: {identities_dir}")
    results = model_manager.add_identities_from_directory(identities_dir)

    for name, success in results.items():
        if success:
            print(f"Successfully added identity: {name}")
        else:
            print(f"Failed to add identity: {name}")


async def main():
    parser = argparse.ArgumentParser(description="Face Analysis POC")
    parser.add_argument(
        "--mode",
        choices=["image", "video", "add_identities"],
        required=True,
        help="Operation mode: process image, video, or add identities",
    )
    parser.add_argument("--input", required=True, help="Input image/video file or directory of identity images")
    parser.add_argument("--output", required=False, help="Output file path (required for image/video mode)")
    parser.add_argument("--backend", choices=["cpu", "cuda", "mps"], default="cpu", help="Inference backend to use")

    args = parser.parse_args()

    # Initialize model manager
    model_manager = ModelManager()
    await model_manager.initialize(backend=InferenceBackend(args.backend))

    try:
        if args.mode == "add_identities":
            await add_identities(model_manager, args.input)
        elif args.mode == "image":
            if not args.output:
                parser.error("--output is required for image mode")
            await process_image(args.input, args.output, model_manager)
        elif args.mode == "video":
            if not args.output:
                parser.error("--output is required for video mode")
            await process_video(args.input, args.output, model_manager)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
