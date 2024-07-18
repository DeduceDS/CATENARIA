# app/main.py
from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.presentation.api import router
from app.infrastructure.database import init_db
from app.tasks.celery_app import celery_app


@asynccontextmanager
async def lifespan(app: FastAPI):
    # await init_db()
    yield


app = FastAPI(lifespan=lifespan)


app.include_router(router)


app.celery_app = celery_app

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
