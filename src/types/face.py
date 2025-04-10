import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from src.types.base import Annotation, BoundingBox, Point


class Identity(BaseModel):
    embedding: np.ndarray | None = None
    id: int | None = None
    name: str | None = None
    last_name: str | None = None
    metadata: dict | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class FaceLandmarks(BaseModel):
    left_eye: Point
    right_eye: Point
    nose: Point
    left_mouth: Point
    right_mouth: Point

    def numpy(self) -> np.ndarray:
        return np.asarray(
            [
                [self.left_eye.x, self.left_eye.y],
                [self.right_eye.x, self.right_eye.y],
                [self.nose.x, self.nose.y],
                [self.left_mouth.x, self.left_mouth.y],
                [self.right_mouth.x, self.right_mouth.y],
            ],
            dtype=np.float32,
        )

    def scale(self, factor: float) -> "FaceLandmarks":
        return FaceLandmarks(
            left_eye=self.left_eye.scale(factor),
            right_eye=self.right_eye.scale(factor),
            nose=self.nose.scale(factor),
            left_mouth=self.left_mouth.scale(factor),
            right_mouth=self.right_mouth.scale(factor),
        )

    def draw(self, image: np.ndarray) -> np.ndarray:
        image = self.left_eye.draw(image)
        image = self.right_eye.draw(image)
        image = self.nose.draw(image)
        image = self.left_mouth.draw(image)
        image = self.right_mouth.draw(image)
        return image


class FaceAttribute(BaseModel):
    bb: BoundingBox
    confidence: float
    name: str

    def draw(self, image: np.ndarray) -> np.ndarray:
        if self.bb is not None:
            # Draw bounding box
            image = self.bb.draw(image)

        if self and self.name:
            # Draw name on top of bounding box
            text_position = (int(self.bb.top_left.x), int(self.bb.top_left.y - 10))  # Position above the box
            cv2.putText(
                image,
                f"{self.name}",
                text_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        return image


class Face(Annotation):
    bb: BoundingBox | None = None
    landmarks: FaceLandmarks | None = None
    identity: Identity | None = None
    liveness_score: float | None = None
    attributes: list[FaceAttribute] = Field(default_factory=list)
    reference_points: np.ndarray = np.array(
        [
            [0.2704875, 0.4615741],
            [0.58510536, 0.45983392],
            [0.42879644, 0.6405054],
            [0.29954734, 0.82469195],
            [0.5600884, 0.8232509],
        ],
        dtype=np.float64,
    )
    aligned_face: np.ndarray | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def scale(self, factor: float) -> "Face":
        return Face(
            bb=self.bb.scale(factor),
            landmarks=self.landmarks.scale(factor),
            identity=self.identity,
        )

    def draw(self, image: np.ndarray) -> np.ndarray:
        if self.bb is not None:
            # Draw bounding box
            image = self.bb.draw(image)

        if self.landmarks is not None:
            # Draw landmarks
            image = self.landmarks.draw(image)

        if self.identity and self.identity.name:
            # Draw name on top of bounding box
            text_position = (int(self.bb.top_left.x), int(self.bb.top_left.y - 10))  # Position above the box
            cv2.putText(
                image,
                f"{self.identity.name} {self.identity.last_name}",
                text_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        if self.liveness_score < 0.1:
            print(self.liveness_score)
            # Add red tint to face region only
            # face_crop = self.bb.crop_image(image)
            # red_overlay = np.zeros_like(face_crop)
            # red_overlay[:, :, 0] = 20  # Red channel

            # Put tinted face back into image
            # face_crop += red_overlay

            # Add "Attack!" text below bounding box
            text_position = (self.bb.top_left.x, self.bb.bottom_right.y + 25)  # Position below the box
            cv2.putText(image, "Attack!", text_position, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

        return image

    def crop(self, image: np.ndarray) -> np.ndarray:
        return self.bb.crop_image(image)

    def align_face(self, image: np.ndarray, image_size: int = 112) -> np.ndarray:
        reference_points = self.reference_points * image_size
        tform_cv2, _ = cv2.estimateAffinePartial2D(self.landmarks.numpy(), reference_points)
        aligned = cv2.warpAffine(image, tform_cv2, (112, 112), flags=cv2.INTER_CUBIC)
        return aligned
