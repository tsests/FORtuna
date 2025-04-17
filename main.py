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
from fastapi.middleware.cors import CORSMiddleware
import cv2
from typing import Optional



@asynccontextmanager
async def lifespan(app: FastAPI):
    init_tables()
    for bucket in ["videos", "frames"]:
        ensure_bucket(bucket)
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def calculate_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()

@app.get("/")
def read_root():
    return FileResponse("static/index.html")


@app.post("/upload/")
async def upload_video(
    file: UploadFile,
    fps: Optional[float] = Form(None)  # Теперь параметр необязательный
):
    # Устанавливаем значение по умолчанию, если fps не передан
    if fps is None or fps == 0:
        fps = 0.2 
    try:
        content = await file.read()
        video_hash = calculate_hash(content)

        # Проверка существующей записи (остаётся без изменений)
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

        video_filename = f"{video_hash}.mp4"
        
        # Проверка и сохранение в MinIO (остаётся без изменений)
        try:
            minio_client.stat_object("videos", video_filename)
        except:
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

        # Сохраняем метаданные (без изменений)
        save_to_db(
            """INSERT INTO videos (hash, filename, fps, processed) 
               VALUES (%s, %s, %s, FALSE)""",
            (video_hash, file.filename, fps)
        )

        # Используем OpenCV вместо FFmpeg
        with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
            tmp.write(content)
            tmp.flush()
            
            out_dir = tempfile.mkdtemp()
            frame_files = []
            
            try:
                cap = cv2.VideoCapture(tmp.name)
                if not cap.isOpened():
                    raise Exception("Could not open video file")
                
                frame_rate = cap.get(cv2.CAP_PROP_FPS)
                frame_interval = int(round(frame_rate / fps))  # Исправлено: используем frame_interval вместо interval
                frame_count = 0
                saved_count = 0

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Сохраняем кадр только если пришло время согласно FPS
                    if frame_count % frame_interval == 0:
                        saved_count += 1
                        frame_name = f"frame_{saved_count:04d}.jpg"
                        frame_path = os.path.join(out_dir, frame_name)
                        success = cv2.imwrite(frame_path, frame)
                        if not success:
                            raise Exception(f"Failed to save frame {frame_name}")
                        frame_files.append(frame_name)
                    
                    frame_count += 1
                
                cap.release()

                # Загрузка кадров в MinIO (остаётся без изменений)
                for idx, frame_name in enumerate(frame_files, start=1):
                    frame_path = os.path.join(out_dir, frame_name)
                    with open(frame_path, "rb") as f:
                        minio_client.put_object(
                            "frames", 
                            f"{video_hash}/{fps}/{frame_name}",
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
                # Очистка временных файлов (остаётся без изменений)
                for f in frame_files:
                    try:
                        os.remove(os.path.join(out_dir, f))
                    except:
                        pass
                try:
                    os.rmdir(out_dir)
                except:
                    pass

    except Exception as e:
        return JSONResponse(
            {"status": "error", "message": str(e)},
            status_code=500
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
