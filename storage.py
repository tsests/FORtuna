# storage.py
from minio import Minio
from config import settings

minio_client = Minio(
    settings.minio_endpoint,
    access_key=settings.minio_access_key,
    secret_key=settings.minio_secret_key,
    secure=settings.minio_secure
)

def ensure_bucket(name: str):
    if not minio_client.bucket_exists(name):
        minio_client.make_bucket(name)

