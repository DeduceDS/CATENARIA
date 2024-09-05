# app/application/interfaces.py
from abc import ABC, abstractmethod
from typing import List, Dict
from app.presentation.rest.schemas.models import Vano, Linea
from app.presentation.rest.schemas.response_models import VanoPrediction


class VanoRepository(ABC):
    @abstractmethod
    async def save(self, vano: Vano) -> None:
        pass

    @abstractmethod
    async def get_all(self) -> List[Vano]:
        pass


class LineaDataService(ABC):
    @abstractmethod
    async def save_vano(self, vano: Vano) -> None:
        pass

    async def save_linea(self, linea: Linea) -> None:
        pass


class LineaPredictService(ABC):
    @abstractmethod
    async def predict_vano(self, vano: Vano) -> VanoPrediction:
        pass
