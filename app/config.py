from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    # Database configuration
    DB_USER: str = "postgres"
    DB_PASSWORD: str = "postgres"
    DB_HOST: str = "localhost"
    DB_PORT: str = "5432"
    DB_NAME: str = "postgres"
    DB_SCHEMA: Optional[str] = None

    # KeyDB configuration
    KEYDB_HOST: str = "redis"
    KEYDB_PORT: str = "6379"

    # Feature flags
    QUEUE_FEATURE: bool = Field(default=False, description="Enable queue feature")
    ANOTHER_BOOL_FEATURE: bool = Field(
        default=True, description="Another boolean feature"
    )

    # Computed properties
    @property
    def DB_URL(self) -> str:
        return f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    @property
    def KEYDB_URL(self) -> str:
        return f"redis://{self.KEYDB_HOST}:{self.KEYDB_PORT}/0"

    @property
    def CELERY_BROKER_URL(self) -> str:
        return self.KEYDB_URL

    @property
    def CELERY_RESULT_BACKEND(self) -> str:
        return self.KEYDB_URL

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=True
    )

    # Validators
    @field_validator("QUEUE_FEATURE", mode="before")
    @classmethod
    def boolean_validator(cls, value):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("true", "1", "t", "y", "yes", "on", "enabled")
        return bool(value)


# Create an instance of the Settings class
settings = Settings()
