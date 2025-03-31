from pydantic import BaseModel

from types.face import Face


class SelfieData(BaseModel):
    faces: list[Face]
