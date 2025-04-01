from src.types.face import Face

import numpy as np
from pydantic import BaseModel, ConfigDict


class SelfieData(BaseModel):
    image: np.ndarray
    faces: list[Face] | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def draw(self) -> np.ndarray:
        draw_image = self.image.copy()

        if self.faces:
            for face in self.faces:
                face.draw(draw_image)

        return draw_image
