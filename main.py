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
import torch
from torchvision import transforms
from torchvision.models import mobilenet_v2
from PIL import Image

# —————————————————————————
#        ФИКСИРОВАННЫЕ НАСТРОЙКИ
# —————————————————————————
MODEL_PATH    = "best_model.pth"
IMG_SIZE      = 224
GRID_SIZE     = (2, 2)
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
BUCKET_VIDEOS = "videos"
BUCKET_FRAMES = "frames"
# —————————————————————————

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
    if fps is None or fps <= 0:
        fps = 0.2  # Значение по умолчанию
    elif fps > 60:
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
                "url": f"/frames/{video_hash}/{fps}/frame_{frame[0]:04d}.jpg"
                #"filename": f"frame_{frame[0]:04d}.jpg"  # Добавлено форматирование имени файла
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
                headers={"Content-Disposition": f"inline; filename=frame_{frame_number:04d}.jpg"}  # Исправлено форматирование
            )
        finally:
            frame_object.close()
            frame_object.release_conn()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/preview/{video_hash}/{fps}/")
async def get_video_preview(video_hash: str, fps: float):
    return await get_single_frame(video_hash, fps, 1)

# Функции для классификации
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

def load_model(path: str):
    model = mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(model.last_channel, 1)
    sd = torch.load(path, map_location=DEVICE)
    model.load_state_dict(sd)
    model.eval().to(DEVICE)
    return model

async def fetch_frame_image(key: str) -> Image.Image:
    """Загружает и возвращает PIL.Image из MinIO"""
    resp = minio_client.get_object(BUCKET_FRAMES, key)
    data = resp.read()
    resp.close()
    return Image.open(io.BytesIO(data)).convert("RGB")

async def get_keys(video_hash: str, fps: float) -> list:
    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT frame_key FROM frames WHERE video_hash=%s AND fps=%s ORDER BY frame_number",
            (video_hash, fps)
        )
        rows = cur.fetchall()
    return [r[0] for r in rows]

@app.post("/classify/")
async def classify_video(
    video_hash: str = Form(...),
    fps: float = Form(...),
    loyalty: float = Form(...)
):
    if not 0 <= loyalty <= 100:
        raise HTTPException(status_code=400, detail="Loyalty must be between 0 and 100")
    thr = 100.0 - loyalty

    # Проверяем, есть ли уже результаты для этих параметров
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT fc.frame_number, fc.is_sharp, fc.sharpness_percentage
                FROM frame_classifications fc
                JOIN frames f ON fc.video_hash = f.video_hash 
                    AND fc.fps = f.fps 
                    AND fc.frame_number = f.frame_number
                WHERE fc.video_hash = %s 
                    AND fc.fps = %s 
                    AND fc.loyalty_threshold = %s
                ORDER BY fc.frame_number
            """, (video_hash, fps, thr))
            
            existing_results = cur.fetchall()
            
            if existing_results:
                return JSONResponse({
                    "status": "from_cache",
                    "results": [{
                        "frame_number": r[0],
                        "is_sharp": r[1],
                        "sharpness_percentage": r[2]
                    } for r in existing_results]
                })

    # Если результатов нет - выполняем классификацию
    model = load_model(MODEL_PATH)
    keys = await get_keys(video_hash, fps)
    
    if not keys:
        raise HTTPException(status_code=404, detail="No frames found")

    results = []
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            for key in keys:
                frame_number = int(key.split('/')[-1].split('_')[1].split('.')[0])
                
                img = await fetch_frame_image(key)
                patches = split_image_to_patches(img)
                batch = torch.stack([val_transform(p) for p in patches]).to(DEVICE)
                
                with torch.no_grad():
                    logits = model(batch)
                    probs = torch.sigmoid(logits).squeeze(1).cpu().numpy() * 100.0
                
                sharp_pct = (probs >= thr).sum() / len(probs) * 100.0
                is_sharp = bool(sharp_pct >= thr)  # Явное преобразование в Python bool
                
                # Сохраняем результат в БД
                cur.execute("""
                    INSERT INTO frame_classifications
                    (video_hash, fps, frame_number, is_sharp, sharpness_percentage, loyalty_threshold)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    video_hash,
                    fps,
                    frame_number,
                    is_sharp,  # Теперь обычный Python bool
                    float(sharp_pct),  # Явное преобразование в float
                    float(thr)  # Явное преобразование в float
                ))
                
                results.append({
                    "frame_number": frame_number,
                    "is_sharp": is_sharp,
                    "sharpness_percentage": round(sharp_pct, 2)
                })
            
            conn.commit()

    return JSONResponse({
        "status": "processed",
        "results": results
    })
    
def split_image_to_patches(img: Image.Image) -> list:
    """Разбивает изображение на патчи согласно GRID_SIZE"""
    w, h = img.size
    pw, ph = w // GRID_SIZE[0], h // GRID_SIZE[1]
    return [img.crop((ix*pw, iy*ph, (ix+1)*pw, (iy+1)*ph))
            for iy in range(GRID_SIZE[1]) for ix in range(GRID_SIZE[0])]
            
@app.get("/classification/{video_hash}/{fps}/{loyalty}/")
async def get_classification_results(
    video_hash: str,
    fps: float,
    loyalty: float
):
    if not 0 <= loyalty <= 100:
        raise HTTPException(status_code=400, detail="Loyalty must be between 0 and 100")
    thr = 100.0 - loyalty

    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT frame_number, is_sharp, sharpness_percentage
                FROM frame_classifications
                WHERE video_hash = %s AND fps = %s AND loyalty_threshold = %s
                ORDER BY frame_number
            """, (video_hash, fps, thr))
            
            results = cur.fetchall()
            
            if not results:
                raise HTTPException(
                    status_code=404,
                    detail="No classification results found for these parameters"
                )

    return {
        "video_hash": video_hash,
        "fps": fps,
        "loyalty_threshold": thr,
        "results": [{
            "frame_number": r[0],
            "is_sharp": r[1],
            "sharpness_percentage": r[2]
        } for r in results]
    }

@app.post("/correct-classification/")
async def correct_classification(
    video_hash: str = Form(...),
    fps: float = Form(...),
    frame_number: int = Form(...),
    is_sharp: bool = Form(...),
    sharpness_percentage: float = Form(...),
    user_comment: Optional[str] = Form(None)
):
    """
    Ручная коррекция результатов классификации кадра
    """
    # Проверяем существование кадра
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            # Получаем текущие значения
            cur.execute("""
                SELECT is_sharp, sharpness_percentage 
                FROM frame_classifications
                WHERE video_hash = %s AND fps = %s AND frame_number = %s
            """, (video_hash, fps, frame_number))
            
            current_data = cur.fetchone()
            if not current_data:
                raise HTTPException(
                    status_code=404,
                    detail="Frame classification not found"
                )

            original_is_sharp, original_sharpness = current_data

            # Обновляем классификацию
            cur.execute("""
                UPDATE frame_classifications
                SET 
                    is_sharp = %s,
                    sharpness_percentage = %s
                WHERE 
                    video_hash = %s AND 
                    fps = %s AND 
                    frame_number = %s
                RETURNING *
            """, (
                is_sharp, 
                sharpness_percentage,
                video_hash,
                fps,
                frame_number
            ))

            # Сохраняем историю изменений
            cur.execute("""
                INSERT INTO manual_classification_corrections
                (
                    video_hash, fps, frame_number,
                    original_is_sharp, corrected_is_sharp,
                    original_sharpness, corrected_sharpness,
                    corrected_by
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                video_hash,
                fps,
                frame_number,
                original_is_sharp,
                is_sharp,
                original_sharpness,
                sharpness_percentage,
                "user"  # Можно заменить на реальное имя пользователя из auth
            ))

            conn.commit()

    return {
        "status": "success",
        "message": "Classification updated",
        "video_hash": video_hash,
        "fps": fps,
        "frame_number": frame_number,
        "new_is_sharp": is_sharp,
        "new_sharpness": sharpness_percentage,
        "previous_values": {
            "is_sharp": original_is_sharp,
            "sharpness": original_sharpness
        }
    }

@app.get("/classification-history/{video_hash}/{fps}/{frame_number}/")
async def get_classification_history(
    video_hash: str,
    fps: float,
    frame_number: int
):
    """
    Получение истории изменений классификации для конкретного кадра
    """
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            # Текущая классификация
            cur.execute("""
                SELECT is_sharp, sharpness_percentage
                FROM frame_classifications
                WHERE video_hash = %s AND fps = %s AND frame_number = %s
            """, (video_hash, fps, frame_number))
            
            current = cur.fetchone()
            if not current:
                raise HTTPException(
                    status_code=404,
                    detail="Frame not found"
                )

            # История изменений
            cur.execute("""
                SELECT 
                    corrected_at,
                    original_is_sharp,
                    corrected_is_sharp,
                    original_sharpness,
                    corrected_sharpness,
                    corrected_by
                FROM manual_classification_corrections
                WHERE 
                    video_hash = %s AND 
                    fps = %s AND 
                    frame_number = %s
                ORDER BY corrected_at DESC
            """, (video_hash, fps, frame_number))
            
            history = cur.fetchall()

    return {
        "current": {
            "is_sharp": current[0],
            "sharpness_percentage": current[1]
        },
        "history": [{
            "timestamp": record[0].isoformat(),
            "from": {
                "is_sharp": record[1],
                "sharpness": record[3]
            },
            "to": {
                "is_sharp": record[2],
                "sharpness": record[4]
            },
            "corrected_by": record[5]
        } for record in history]
    }



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
