"""Configuration settings loaded from environment variables."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env")

    database_url: str = "postgresql://samui:samui@localhost:5432/samui"
    azure_storage_connection_string: str = ""
    azure_container_name: str = "samui-images"
    sam3_model_name: str = "sam3_hiera_large"


settings = Settings()
