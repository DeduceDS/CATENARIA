# app/presentation/rest/routes/vano_routes.py
from fastapi import (
    APIRouter,
    # File,
    # UploadFile,
    # HTTPException,
    # Depends,
    # BackgroundTasks
)
from fastapi.responses import JSONResponse
from app.application.services import LineaPredictServiceImpl
from app.domain.models import Linea, Vano
from app.domain.response_models import VanoPrediction

## Repository deps
# from app.infrastructure.database import get_db
# from app.infrastructure.repositories import VanoRepositoryImpl
# from app.application.services import ElectraDataServiceImpl
# from sqlalchemy.ext.asyncio import AsyncSession

import json

vano_router = APIRouter(prefix="/vano", tags=["Vano"])


@vano_router.post("/predict")
async def predict_vano(vano: Vano) -> VanoPrediction:

    predict_service = LineaPredictServiceImpl()
    prediction = await predict_service.predict_vano(vano)

    return prediction


# DATABASE
# @vano_router.post("/upload")
# async def upload_file(file: UploadFile = File(...), db: AsyncSession = Depends(get_db)):
#     content = await file.read()
#     data = json.loads(content)

#     # Convert the list of dictionaries to a list of Vano objects
#     vanos = [Vano(**vano_data) for vano_data in data]
#     electra_data = ElectraData(vanos=vanos)

#     vano_repository = VanoRepositoryImpl(db)
#     electra_service = ElectraDataServiceImpl(vano_repository)
#     await electra_service.process_electra_data(electra_data)

#     return {"message": "File processed successfully"}


# @router.get("/vanos")
# async def get_vanos(db: AsyncSession = Depends(get_db)):
#     vano_repository = VanoRepositoryImpl(db)
#     vanos = await vano_repository.get_all()
#     return vanos
