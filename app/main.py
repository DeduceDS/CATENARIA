# app/main.py
from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.presentation.rest.routes.vano_routes import vano_router
from app.presentation.rest.routes.queue_routes import queue_router
from app.tasks.celery_app import celery_app
from app.config import settings

# from app.infrastructure.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    # await init_db()
    yield


app = FastAPI(lifespan=lifespan)


app.include_router(vano_router)
app.include_router(queue_router)


app.celery_app = celery_app

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
