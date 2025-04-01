import cv2
import numpy as np

from .model_manager import AttributeResult, DetectionResult, RecognitionResult


class Visualizer:
    def __init__(self):
        self.colors = {
            "detection": (0, 255, 0),  # Green for detection
            "landmarks": (255, 0, 0),  # Blue for landmarks
            "text": (255, 255, 255),  # White for text
        }
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5
        self.thickness = 2

    def draw_detection(self, image: np.ndarray, detection: DetectionResult) -> np.ndarray:
        """Draw bounding box and landmarks for a face detection."""
        x1, y1, x2, y2 = map(int, detection.bbox)

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), self.colors["detection"], self.thickness)

        # Draw confidence score
        conf_text = f"{detection.confidence:.2f}"
        cv2.putText(image, conf_text, (x1, y1 - 10), self.font, self.font_scale, self.colors["text"], self.thickness)

        # Draw landmarks if available
        if detection.landmarks is not None:
            for i in range(0, len(detection.landmarks), 2):
                x, y = int(detection.landmarks[i]), int(detection.landmarks[i + 1])
                cv2.circle(image, (x, y), 2, self.colors["landmarks"], -1)

        return image

    def draw_attributes(self, image: np.ndarray, attributes: AttributeResult, position: tuple[int, int]) -> np.ndarray:
        """Draw face attributes at the specified position."""
        y_offset = position[1]

        if attributes.age is not None:
            age_text = f"Age: {attributes.age:.1f}"
            cv2.putText(
                image,
                age_text,
                (position[0], y_offset),
                self.font,
                self.font_scale,
                self.colors["text"],
                self.thickness,
            )
            y_offset += 20

        if attributes.gender is not None:
            gender_text = f"Gender: {attributes.gender}"
            cv2.putText(
                image,
                gender_text,
                (position[0], y_offset),
                self.font,
                self.font_scale,
                self.colors["text"],
                self.thickness,
            )
            y_offset += 20

        if attributes.emotion is not None:
            emotion_text = f"Emotion: {attributes.emotion}"
            cv2.putText(
                image,
                emotion_text,
                (position[0], y_offset),
                self.font,
                self.font_scale,
                self.colors["text"],
                self.thickness,
            )
            y_offset += 20

        if attributes.race is not None:
            race_text = f"Race: {attributes.race}"
            cv2.putText(
                image,
                race_text,
                (position[0], y_offset),
                self.font,
                self.font_scale,
                self.colors["text"],
                self.thickness,
            )

        return image

    def draw_recognition(
        self, image: np.ndarray, recognition: RecognitionResult, position: tuple[int, int]
    ) -> np.ndarray:
        """Draw face recognition results at the specified position."""
        if recognition.identity is not None:
            identity_text = f"Identity: {recognition.identity}"
            cv2.putText(
                image,
                identity_text,
                (position[0], position[1]),
                self.font,
                self.font_scale,
                self.colors["text"],
                self.thickness,
            )

        if recognition.confidence is not None:
            conf_text = f"Confidence: {recognition.confidence:.2f}"
            cv2.putText(
                image,
                conf_text,
                (position[0], position[1] + 20),
                self.font,
                self.font_scale,
                self.colors["text"],
                self.thickness,
            )

        return image

    def visualize_results(
        self,
        image: np.ndarray,
        detections: list[DetectionResult],
        recognitions: list[RecognitionResult],
        attributes: list[AttributeResult],
    ) -> np.ndarray:
        """Visualize all results on the image."""
        result_image = image.copy()

        # Draw detections and their corresponding attributes/recognition
        for _i, (detection, recognition, attribute) in enumerate(
            zip(detections, recognitions, attributes, strict=False)
        ):
            # Draw detection box and landmarks
            result_image = self.draw_detection(result_image, detection)

            # Calculate position for attributes and recognition
            x1, y1, x2, y2 = map(int, detection.bbox)
            position = (x1, y2 + 10)  # Position below the detection box

            # Draw attributes and recognition
            result_image = self.draw_attributes(result_image, attribute, position)
            result_image = self.draw_recognition(result_image, recognition, (position[0], position[1] + 80))

        return result_image
