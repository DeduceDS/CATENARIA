# app/application/services.py

from typing import Dict
import numpy as np

from fastapi import UploadFile

from app.presentation.rest.schemas.models import Linea, Vano
from app.presentation.rest.schemas.response_models import VanoPrediction
from app.domain.exceptions.file_exceptions import InvalidFileFormatException
from app.application.interfaces import LineaDataService, LineaPredictService
from app.application.interfaces import VanoRepository

from electra_package.release_2 import process_vano


# Data Service
class LineaDataServiceImpl(LineaDataService):
    def __init__(self, vano_repository: VanoRepository):
        self.vano_repository = vano_repository

    async def save_vano(self, vano: Vano) -> None:
        await self.vano_repository.save(vano)

    async def save_linea(self, linea: Linea) -> None:
        for vano in linea.vanos:
            await self.vano_repository.save(vano)


# Predict Service
class LineaPredictServiceImpl(LineaPredictService):
    def __init__(self):
        pass

    async def predict_vano(self, vano: Vano) -> VanoPrediction:

        # json serializer
        def json_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, (list, tuple)):
                return [json_serializable(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: json_serializable(value) for key, value in obj.items()}
            else:
                return obj

        # getting required out_data destructuring the returned tuple (prediction_dict, rmses, maxes, correlations)
        prediction_dict, *others = process_vano(vano.model_dump())

        result = json_serializable(prediction_dict)

        return VanoPrediction(**result)


class FileCheckerServiceImpl:
    @staticmethod
    async def validate_file_type(file: UploadFile, filetype: str) -> None:
        raise (
            InvalidFileFormatException()
            if not file.filename.endswith(filetype)
            else None
        )
