from abc import ABC, abstractmethod
from src.types.selfie_data import SelfieData


class Processor(ABC):
    async def __call__(self, selfie_data: SelfieData) -> SelfieData:
        return await self.execute(selfie_data)

    @abstractmethod
    async def execute(selfie_data: SelfieData) -> SelfieData:
        raise NotImplementedError("Child classes should implement execute method")
