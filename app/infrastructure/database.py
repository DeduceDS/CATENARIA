# app/infrastructure/database.py
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from app.config import settings

engine = create_async_engine(settings.DB_URL, echo=True)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session


# Create tables
async def init_db():
    async with engine.begin() as conn:
        if settings.SCHEMA_NAME == "":
            raise Exception("SCHEMA_NAME is not set")
        await conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {settings.SCHEMA_NAME}"))
        await conn.run_sync(Base.metadata.create_all)
