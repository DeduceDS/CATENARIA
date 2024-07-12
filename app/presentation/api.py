# app/presentation/api.py
from fastapi import APIRouter, File, UploadFile, Depends, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from app.application.services import ElectraDataServiceImpl, ElectraPredictServiceImpl
from app.infrastructure.repositories import VanoRepositoryImpl
from app.infrastructure.database import get_db
from app.domain.models import ElectraData, Vano
import json
import tempfile
import os

router = APIRouter()


@router.post("/upload")
async def upload_file(file: UploadFile = File(...), db: AsyncSession = Depends(get_db)):
    content = await file.read()
    data = json.loads(content)

    # Convert the list of dictionaries to a list of Vano objects
    vanos = [Vano(**vano_data) for vano_data in data]
    electra_data = ElectraData(vanos=vanos)

    vano_repository = VanoRepositoryImpl(db)
    electra_service = ElectraDataServiceImpl(vano_repository)
    await electra_service.process_electra_data(electra_data)

    return {"message": "File processed successfully"}


@router.post("/predict_json_download")
async def predict_json(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    content = await file.read()
    data = json.loads(content)

    predict_service = ElectraPredictServiceImpl()
    prediction = await predict_service.predict_data_from_json(data)

    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp:
        # Write the JSON data to the temporary file
        json.dump(prediction, tmp, indent=2)
        tmp_path = tmp.name
        print(tmp_path)

    # def cleanup(path: str):
    #     os.unlink(path)

    # # Add the cleanup task to background tasks
    # background_tasks.add_task(cleanup, tmp_path)

    # Return the file as a downloadable response
    return FileResponse(
        tmp_path, media_type="application/octet-stream", filename="vano_result.json"
    )


@router.post("/predict_json")
async def predict_json(file: UploadFile = File(...)):
    content = await file.read()
    data = json.loads(content)

    predict_service = ElectraPredictServiceImpl()
    prediction = await predict_service.predict_data_from_json(data)

    return JSONResponse(prediction)


@router.get("/vanos")
async def get_vanos(db: AsyncSession = Depends(get_db)):
    vano_repository = VanoRepositoryImpl(db)
    vanos = await vano_repository.get_all()
    return vanos
