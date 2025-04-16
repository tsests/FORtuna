from pydantic_settings import BaseSettings  # Новый импорт

class Settings(BaseSettings):
    db_name: str
    db_user: str
    db_password: str
    db_host: str
    db_port: str

    minio_endpoint: str
    minio_access_key: str
    minio_secret_key: str
    minio_secure: bool = False

    class Config:
        env_file = ".env"

settings = Settings()

