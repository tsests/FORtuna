import hashlib
import os
import tempfile
import subprocess
import io
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from minio import Minio
from psycopg2 import connect
from config import settings
from db import get_db_conn, save_to_db
from storage import minio_client, ensure_bucket
from init_db import init_tables
from contextlib import asynccontextmanager
from typing import List

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_tables()
    for bucket in ["videos", "frames"]:
        ensure_bucket(bucket)
    yield

app = FastAPI(lifespan=lifespan)

def calculate_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

@app.post("/upload/")
async def upload_video(file: UploadFile, fps: float = Form(0.2)):
    try:
        content = await file.read()
        video_hash = calculate_hash(content)

        # Проверяем, существует ли уже такая комбинация видео+FPS
        with get_db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM videos WHERE hash = %s AND fps = %s",
                    (video_hash, fps)
                )
                if cur.fetchone():
                    return JSONResponse({
                        "status": "exists", 
                        "video_hash": video_hash,
                        "fps": fps
                    })

        # Если видео с таким FPS ещё не обрабатывалось
        video_filename = f"{video_hash}.mp4"
        
        # Проверяем, есть ли уже оригинал в MinIO
        try:
            minio_client.stat_object("videos", video_filename)
        except:
            # Если видео нет в хранилище - сохраняем
            file_obj = io.BytesIO(content)
            minio_client.put_object(
                "videos", 
                video_filename, 
                data=file_obj, 
                length=len(content), 
                content_type="video/mp4"
            )

        if fps <= 0 or fps > 60:
            return JSONResponse(
                {"status": "error", "message": "FPS must be between 0 and 60"},
                status_code=400
            )

        # Сохраняем метаданные (включая FPS)
        save_to_db(
            """INSERT INTO videos (hash, filename, fps, processed) 
               VALUES (%s, %s, %s, FALSE)""",
            (video_hash, file.filename, fps)
        )

        # Обработка кадров (остаётся без изменений, но сохраняем FPS в frames)
        with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
            tmp.write(content)
            tmp.flush()
            
            out_dir = tempfile.mkdtemp()
            try:
                frame_pattern = os.path.join(out_dir, "frame_%04d.jpg")
                subprocess.run([
                    "ffmpeg", "-i", tmp.name, "-vf", f"fps={fps}",
                    frame_pattern
                ], check=True)

                # Загружаем кадры в MinIO
                frame_files = sorted(f for f in os.listdir(out_dir) if f.endswith('.jpg'))
                for idx, frame_name in enumerate(frame_files, start=1):
                    frame_path = os.path.join(out_dir, frame_name)
                    with open(frame_path, "rb") as f:
                        minio_client.put_object(
                            "frames", 
                            f"{video_hash}/{fps}/{frame_name}",  # Добавляем FPS в путь
                            data=f, 
                            length=os.path.getsize(frame_path), 
                            content_type="image/jpeg"
                        )
                    save_to_db(
                        """INSERT INTO frames 
                           (video_hash, fps, frame_number, frame_key) 
                           VALUES (%s, %s, %s, %s)""",
                        (video_hash, fps, idx, f"{video_hash}/{fps}/{frame_name}")
                    )

                # Помечаем как обработанное
                save_to_db(
                    "UPDATE videos SET processed = TRUE WHERE hash = %s AND fps = %s",
                    (video_hash, fps)
                )

                return {
                    "status": "processed", 
                    "video_hash": video_hash,
                    "fps": fps,
                    "frames": len(frame_files)
                }

            finally:
                # Очистка временных файлов
                for f in frame_files:
                    try:
                        os.remove(os.path.join(out_dir, f))
                    except:
                        pass
                try:
                    os.rmdir(out_dir)
                except:
                    pass

    except subprocess.CalledProcessError as e:
        return JSONResponse(
            {"status": "error", "message": f"FFmpeg processing failed: {str(e)}"},
            status_code=500
        )
    except Exception as e:
        return JSONResponse(
            {"status": "error", "message": str(e)},
            status_code=500
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
