# app/application/interfaces.py
from abc import ABC, abstractmethod
from typing import List, Dict
from app.domain.models import Vano, ElectraData


class VanoRepository(ABC):
    @abstractmethod
    async def save(self, vano: Vano) -> None:
        pass

    @abstractmethod
    async def get_all(self) -> List[Vano]:
        pass


class ElectraDataService(ABC):
    @abstractmethod
    async def process_electra_data(self, data: ElectraData) -> None:
        pass


class ElectraPredictService(ABC):
    @abstractmethod
    async def predict_data_from_json(self, data: Dict) -> Dict:
        pass
