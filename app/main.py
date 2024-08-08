# app/main.py
from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.presentation.rest.routes.vano_routes import vano_router, vano_database_router
from app.presentation.rest.routes.queue_routes import queue_router
from app.config import settings

# TODO improve feature flags -___-

# Queue feature imports
if settings.QUEUE_FEATURE:
    from app.tasks.celery_app import celery_app
# Database feature imports
if settings.DATABASE_FEATURE:
    from app.infrastructure.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    # init db if feature enabled
    await init_db() if settings.DATABASE_FEATURE else None
    yield


# Fastapi instace
app = FastAPI(lifespan=lifespan)

# Include routers (feature dependant)
app.include_router(vano_router)
if settings.QUEUE_FEATURE:
    app.include_router(queue_router)
    app.celery_app = celery_app
if settings.DATABASE_FEATURE:
    app.include_router(vano_database_router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
