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
from fastapi import HTTPException
from fastapi import Form, File, UploadFile



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
    file: UploadFile = File(...),
    fps: Optional[float] = Form(None)
):
    # Устанавливаем и валидируем значение FPS
    if fps is None:
        fps = 0.2  # Значение по умолчанию
    elif fps <= 0 or fps > 60:
        return JSONResponse(
            {"status": "error", "message": "FPS must be between 0.01 and 60"},
            status_code=400
        )

    try:
        content = await file.read()
        video_hash = calculate_hash(content)

        # Проверка существующей записи
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
        
        # Проверка и сохранение в MinIO
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

        # Сохраняем метаданные
        save_to_db(
            """INSERT INTO videos (hash, filename, fps, processed) 
               VALUES (%s, %s, %s, FALSE)""",
            (video_hash, file.filename, fps)
        )

        # Используем OpenCV для обработки видео
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
                if frame_rate <= 0:
                    raise Exception("Invalid frame rate in video file")
                
                # Защита от деления на ноль и корректный расчет интервала
                frame_interval = max(1, int(round(frame_rate / fps)))
                frame_count = 0
                saved_count = 0

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
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

                # Загрузка кадров в MinIO
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

    except Exception as e:
        return JSONResponse(
            {"status": "error", "message": str(e)},
            status_code=500
        )


@app.get("/frames/{video_hash}/{fps}/")
async def get_frames_list(video_hash: str, fps: float):
    """
    Получение списка всех кадров для видео с указанным hash и fps
    """
    try:
        # Проверяем существование видео
        with get_db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM videos WHERE hash = %s AND fps = %s AND processed = TRUE",
                    (video_hash, fps)
                )
                if not cur.fetchone():
                    raise HTTPException(status_code=404, detail="Video not found or not processed")

                # Получаем список кадров
                cur.execute(
                    "SELECT frame_number, frame_key FROM frames WHERE video_hash = %s AND fps = %s ORDER BY frame_number",
                    (video_hash, fps)
                )
                frames = cur.fetchall()
                
                if not frames:
                    raise HTTPException(status_code=404, detail="No frames found")

        # Формируем список URL для доступа к кадрам
        frames_list = [
            {
                "frame_number": frame[0],
                "url": f"/frames/{video_hash}/{fps}/{frame[0]}"
            }
            for frame in frames
        ]

        return {
            "video_hash": video_hash,
            "fps": fps,
            "frames_count": len(frames_list),
            "frames": frames_list
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/frames/{video_hash}/{fps}/{frame_number}")
async def get_single_frame(video_hash: str, fps: float, frame_number: int):
    """
    Получение конкретного кадра по номеру
    """
    try:
        # Получаем ключ кадра из базы данных
        with get_db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT frame_key FROM frames 
                       WHERE video_hash = %s AND fps = %s AND frame_number = %s""",
                    (video_hash, fps, frame_number)
                )
                frame_key = cur.fetchone()
                
                if not frame_key:
                    raise HTTPException(status_code=404, detail="Frame not found")

        # Получаем объект из MinIO
        try:
            frame_object = minio_client.get_object("frames", frame_key[0])
            return FileResponse(
                frame_object,
                media_type="image/jpeg",
                headers={"Content-Disposition": f"inline; filename=frame_{frame_number}.jpg"}
            )
        finally:
            frame_object.close()
            frame_object.release_conn()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/preview/{video_hash}/{fps}/")
async def get_video_preview(video_hash: str, fps: float):
    return await get_single_frame(video_hash, fps, 1)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
