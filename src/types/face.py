import numpy as np
from pydantic import BaseModel, ConfigDict

from src.types.base import Annotation, BoundingBox, Point


class Identity(BaseModel):
    name: str
    embeddings: list[np.ndarray] | None = None
    metadata: dict | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class FaceLandmarks(BaseModel):
    left_eye: Point
    right_eye: Point
    nose: Point
    left_mouth: Point
    right_mouth: Point

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


class Face(Annotation):
    bb: BoundingBox | None = None
    landmarks: FaceLandmarks | None = None
    identity: Identity | None = None

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

        return image

    def crop(self, image: np.ndarray) -> np.ndarray:
        return self.bb.crop_image(image)
