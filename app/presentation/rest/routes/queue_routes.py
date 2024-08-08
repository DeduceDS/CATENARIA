# app/presentation/api.py
from fastapi import APIRouter, File, UploadFile, Depends, HTTPException, BackgroundTasks
from app.tasks.celery_app import predict_linea


import json

# /quque/<-->
queue_router = APIRouter(prefix="/queue", tags=["Queue"])


@queue_router.post("/predict_linea_json_file")
async def predict_linea_json_file(file: UploadFile = File(...)):
    content = await file.read()
    data = json.loads(content)

    # Queue the task
    task = predict_linea.delay(data)

    return {"message": "Task queued successfully", "task_id": task.id}


@queue_router.get("/task_status/{task_id}")
async def get_task_status(task_id: str):
    task = predict_linea.AsyncResult(task_id)

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


@queue_router.get("/task_result/{task_id}")
async def get_task_result(task_id: str):
    task = predict_linea.AsyncResult(task_id)

    if task.state == "SUCCESS":
        return {"result": task.result}
    elif task.state == "FAILURE":
        raise HTTPException(status_code=400, detail="Task failed")
    else:
        raise HTTPException(status_code=404, detail="Task not completed yet")
