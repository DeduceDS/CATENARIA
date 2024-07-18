# app/config.py
import os
from dotenv import load_dotenv

load_dotenv()

# Existing database configuration
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_URL = f"postgresql+asyncpg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
SCHEMA_NAME = os.getenv("DB_SCHEMA", "")

# KeyDB configuration
KEYDB_HOST = os.getenv("KEYDB_HOST", "redis")
KEYDB_PORT = os.getenv("KEYDB_PORT", "6379")
KEYDB_URL = f"redis://{KEYDB_HOST}:{KEYDB_PORT}/0"

# Celery configuration
CELERY_BROKER_URL = KEYDB_URL
CELERY_RESULT_BACKEND = KEYDB_URL
