from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    APP_NAME: str = "brain-mri-segmentation"
    APP_VERSION: str = "1.0.0"
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

def get_settings():
    return Settings()
