from abc import ABC, abstractmethod

import cv2
import numpy as np
from pydantic import BaseModel, Field


class Annotation(BaseModel, ABC):
    metadata: dict = Field(default_factory=dict)

    @abstractmethod
    def scale(self, factor: float) -> "Annotation":
        """Scale the annotation by a factor"""
        pass

    @abstractmethod
    def draw(self, image: np.ndarray) -> np.ndarray:
        """Draw the annotation on an image"""
        pass

    @abstractmethod
    def crop(self, image: np.ndarray) -> np.ndarray:
        """Crop the image according to this annotation"""
        pass


class Point(Annotation):
    x: int
    y: int

    def scale(self, factor: float) -> "Point":
        return Point(x=int(self.x * factor), y=int(self.y * factor))

    def draw(self, image: np.ndarray) -> np.ndarray:
        cv2.circle(image, (self.x, self.y), 2, (0, 255, 0), -1)
        return image

    def crop(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Cannot crop image with just a point")


class BoundingBox(Annotation):
    top_left: Point
    bottom_right: Point

    @property
    def width(self) -> int:
        return self.bottom_right.x - self.top_left.x

    @property
    def height(self) -> int:
        return self.bottom_right.y - self.top_left.y

    def scale(self, factor: float) -> "BoundingBox":
        return BoundingBox(top_left=self.top_left.scale(factor), bottom_right=self.bottom_right.scale(factor))

    def crop(self, bb: "BoundingBox") -> "BoundingBox":
        self.top_left.x -= bb.top_left.x
        self.top_left.y -= bb.top_left.y
        self.bottom_right.x -= bb.top_left.x
        self.bottom_right.y -= bb.top_left.y
        return self

    def draw(self, image: np.ndarray) -> np.ndarray:
        cv2.rectangle(
            image, (self.top_left.x, self.top_left.y), (self.bottom_right.x, self.bottom_right.y), (0, 255, 0), 2
        )
        return image

    def crop_image(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        y1 = min(max(self.top_left.y, 0), height)
        y2 = min(max(self.bottom_right.y, 0), height)
        x1 = min(max(self.top_left.x, 0), width)
        x2 = min(max(self.bottom_right.x, 0), width)
        return image[y1:y2, x1:x2]

    def area(self) -> float:
        return (self.bottom_right.x - self.top_left.x) * (self.bottom_right.y - self.top_left.y)

    def iou(self, bb: "BoundingBox") -> float:
        x_left = max(self.top_left.x, bb.top_left.x)
        y_top = max(self.top_left.y, bb.top_left.y)
        x_right = min(self.bottom_right.x, bb.bottom_right.x)
        y_bottom = min(self.bottom_right.y, bb.bottom_right.y)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        area1 = self.area()
        area2 = bb.area()
        union_area = area1 + area2 - intersection_area
        return intersection_area / union_area


class Polygon(Annotation):
    points: list[Point]

    def scale(self, factor: float) -> "Polygon":
        return Polygon(points=[p.scale(factor) for p in self.points])

    def draw(self, image: np.ndarray) -> np.ndarray:
        points = np.array([(p.x, p.y) for p in self.points])
        cv2.polylines(image, [points], True, (0, 255, 0), 2)
        return image

    def crop(self, image: np.ndarray) -> np.ndarray:
        # Create mask and crop using min/max bounds
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        points = np.array([(p.x, p.y) for p in self.points])
        cv2.fillPoly(mask, [points], 255)
        x1, y1 = np.min(points, axis=0)
        x2, y2 = np.max(points, axis=0)
        masked = cv2.bitwise_and(image, image, mask=mask)
        return masked[y1:y2, x1:x2]
