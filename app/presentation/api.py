# app/presentation/api.py
from fastapi import APIRouter, File, UploadFile, Depends, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from app.application.services import ElectraDataServiceImpl, ElectraPredictServiceImpl
from app.infrastructure.repositories import VanoRepositoryImpl
from app.infrastructure.database import get_db
from app.domain.models import ElectraData, Vano
from app.tasks.celery_app import process_electra_data

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


# Queue endpoints


@router.post("/queue_predict_json")
async def queue_predict_json(file: UploadFile = File(...)):
    content = await file.read()
    data = json.loads(content)

    # Queue the task
    task = process_electra_data.delay(data)

    return {"message": "Task queued successfully", "task_id": task.id}


@router.get("/task_status/{task_id}")
async def get_task_status(task_id: str):
    task = process_electra_data.AsyncResult(task_id)

    if task.state == "PENDING":
        response = {"state": task.state, "status": "Task is waiting for execution"}
    elif task.state == "STARTED":
        response = {"state": task.state, "status": "Task has been started"}
    elif task.state == "PROGRESS":
        response = {
            "state": task.state,
            "status": task.info.get("status", ""),
            "progress": task.info.get("progress", 0),
        }
    elif task.state == "SUCCESS":
        response = {
            "state": task.state,
            "status": "Task has been completed",
            "result": task.result,
        }
    else:
        response = {
            "state": task.state,
            "status": str(task.info),  # this is the exception raised
        }
    return response


@router.get("/task_result/{task_id}")
async def get_task_result(task_id: str):
    task = process_electra_data.AsyncResult(task_id)

    if task.state == "SUCCESS":
        return {"result": task.result}
    elif task.state == "FAILURE":
        raise HTTPException(status_code=400, detail="Task failed")
    else:
        raise HTTPException(status_code=404, detail="Task not completed yet")
