# app/presentation/rest/routes/vano_routes.py
from fastapi import APIRouter, File, UploadFile, Depends
from fastapi.responses import JSONResponse, FileResponse
from app.application.services import LineaPredictServiceImpl, FileCheckerServiceImpl
from app.application.interfaces import LineaPredictService
from app.presentation.rest.schemas.models import Linea, Vano
from app.presentation.rest.schemas.response_models import VanoPrediction
from app.presentation.rest.exceptions.Files import InvaildFileFormatException

from app.config import settings


# TODO implement dependency injection like this, all over the project.
def get_predict_service() -> LineaPredictService:
    return LineaPredictServiceImpl()


# Repository deps
if settings.DATABASE_FEATURE:
    from app.infrastructure.database import get_db
    from app.infrastructure.repositories import VanoRepositoryImpl
    from app.application.services import LineaDataServiceImpl
    from sqlalchemy.ext.asyncio import AsyncSession

import json
import tempfile

# Routers
vano_router = APIRouter(prefix="/vano", tags=["Vano"])
vano_database_router = APIRouter(prefix="/vano", tags=["Database"])


# Predict
@vano_router.post("/predict")
async def predict_vano_handler(
    vano: Vano, predict_service: LineaPredictService = Depends(get_predict_service)
) -> VanoPrediction:
    prediction = await predict_service.predict_vano(vano)
    return prediction


# Database vano operations
if settings.DATABASE_FEATURE:
    # Upload vanos to DB by Linea json file
    @vano_database_router.post("/upload_linea_file")
    async def upload_file_handler(
        file: UploadFile = File(...), db: AsyncSession = Depends(get_db)
    ):
        # FIXME this service isn't even triggering when database.db file gives 500 error due to UTF8 encoding error on UploadFile
        file_checker_service = FileCheckerServiceImpl()
        vano_repository = VanoRepositoryImpl(db)
        electra_service = LineaDataServiceImpl(vano_repository)

        file_checker_service.validate_file_type(file, "json")

        content = await file.read()  # file to bytes
        data = json.loads(content)  # bytes to dict

        # Convert the list of dictionaries to a list of Vano objects
        # TODO is this line necessary?
        vanos = [Vano(**vano_data) for vano_data in data]
        linea = Linea(vanos=vanos)

        await electra_service.save_linea(linea)

        return {"message": "File processed successfully"}

    @vano_database_router.get("/vanos_file")
    async def get_vanos_file_handler(db: AsyncSession = Depends(get_db)):
        vano_repository = VanoRepositoryImpl(db)
        vanos = await vano_repository.get_all()

        # Convert vanos to a list of dictionaries
        vanos_data = [vano.model_dump() for vano in vanos.vanos]

        # Create a temporary file to store the JSON data
        with tempfile.NamedTemporaryFile(
            mode="w+", delete=False, suffix=".json"
        ) as tmp_file:
            json.dump(vanos_data, tmp_file, indent=2)

        # Return the file as a downloadable response
        return FileResponse(
            tmp_file.name, media_type="application/json", filename="vanos.json"
        )

    @vano_database_router.get("/vanos/{id}")
    async def get_vano_by_id_handler(
        id: str, db: AsyncSession = Depends(get_db)
    ) -> Vano:
        vano_repository = VanoRepositoryImpl(db)
        vano = await vano_repository.get_vano(id)
        return vano

    @vano_database_router.get("/vanos/predict/{id}")
    async def predict_vano_by_id_handler(
        id: str, db: AsyncSession = Depends(get_db)
    ) -> VanoPrediction:
        vano_repository = VanoRepositoryImpl(db)
        vano = await vano_repository.get_vano(id)

        prediction_service = LineaPredictServiceImpl()
        prediction = await prediction_service.predict_vano(vano)
        return prediction
