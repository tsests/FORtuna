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
from fastapi import Form, File, UploadFile, Body
import torch
from torchvision import transforms
from torchvision.models import mobilenet_v2
from PIL import Image
import csv
import datetime
import io
from fastapi import Response, HTTPException
from fastapi.responses import JSONResponse


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

tags_metadata = [
    {
        "name": "videos",
        "description": "Operations with video files - upload, list, delete.",
    },
    {
        "name": "frames",
        "description": "Operations with extracted video frames.",
    },
    {
        "name": "classifications",
        "description": "Frame quality classification operations.",
    },
    {
        "name": "corrections",
        "description": "Manual classification corrections.",
    },
]

app = FastAPI(
    openapi_tags=tags_metadata,
    title="Video Processing API",
    description="API for video frame extraction and classification",
    version="1.0.0",
    lifespan=lifespan
)


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

@app.post("/videos/")
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
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in ('.mp4', '.mov', '.avi'):
            return JSONResponse(
                {"status": "error", "message": "Unsupported video format. Supported formats: .mp4, .mov, .avi"},
                status_code=400
            )
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

        video_filename = f"{video_hash}{file_ext}"
        
        # Проверка и сохранение в MinIO
        try:
            minio_client.stat_object("videos", video_filename)
        except:
            file_obj = io.BytesIO(content)
            # Устанавливаем правильный content_type в зависимости от формата
            content_type = {
                '.mp4': 'video/mp4',
                '.mov': 'video/quicktime',
                '.avi': 'video/x-msvideo'
            }.get(file_ext, 'video/mp4')
            
            minio_client.put_object(
                "videos", 
                video_filename, 
                data=file_obj, 
                length=len(content), 
                content_type=content_type
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

                return JSONResponse(
                    {
                        "status": "processed", 
                        "video_hash": video_hash,
                        "fps": fps,
                        "frames": len(frame_files),
                        "links": {
                            "self": f"/videos/{video_hash}/",
                            "frames": f"/frames/{video_hash}/{fps}/",
                            "classify": f"/classify/{video_hash}/{fps}/"
                        }
                    },
                    status_code=201
                )

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

@app.get("/videos/")
async def list_videos():
    """Получение списка всех загруженных видео"""
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT hash, filename, fps, processed FROM videos")
            videos = cur.fetchall()
    
    return [{
        "hash": v[0],
        "filename": v[1],
        "fps": v[2],
        "processed": v[3],
        "links": {
            "self": f"/videos/{v[0]}/",
            "frames": f"/frames/{v[0]}/{v[2]}/"
        }
    } for v in videos]

@app.get("/videos/{video_hash}/")
async def get_video_details(video_hash: str):
    """Получение деталей конкретного видео"""
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT hash, filename, fps, processed FROM videos WHERE hash = %s",
                (video_hash,)
            )
            video = cur.fetchone()
            
            if not video:
                raise HTTPException(status_code=404, detail="Video not found")
                
            cur.execute(
                "SELECT DISTINCT fps FROM frames WHERE video_hash = %s",
                (video_hash,)
            )
            available_fps = [f[0] for f in cur.fetchall()]
    
    return {
        "hash": video[0],
        "filename": video[1],
        "fps": video[2],
        "processed": video[3],
        "available_fps": available_fps,
        "links": {
            "frames": f"/frames/{video[0]}/",
            "classifications": f"/classifications/{video[0]}/"
        }
    }

@app.delete("/videos/{video_hash}/")
async def delete_video(video_hash: str):
    """Удаление видео и всех связанных данных"""
    try:
        with get_db_conn() as conn:
            with conn.cursor() as cur:
                # Получаем все связанные кадры
                cur.execute(
                    "SELECT frame_key FROM frames WHERE video_hash = %s",
                    (video_hash,)
                )
                frame_keys = [f[0] for f in cur.fetchall()]
                
                # Удаляем из MinIO
                for key in frame_keys:
                    try:
                        minio_client.remove_object(BUCKET_FRAMES, key)
                    except:
                        pass
                
                # Удаляем видео из MinIO
                try:
                    minio_client.remove_object(BUCKET_VIDEOS, f"{video_hash}.mp4")
                except:
                    pass
                
                # Удаляем из базы данных
                cur.execute("DELETE FROM frame_classifications WHERE video_hash = %s", (video_hash,))
                cur.execute("DELETE FROM frames WHERE video_hash = %s", (video_hash,))
                cur.execute("DELETE FROM videos WHERE hash = %s", (video_hash,))
                conn.commit()
        
        return {"status": "deleted", "video_hash": video_hash}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
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

@app.delete("/frames/{video_hash}/{fps}/{frame_number}")
async def delete_frame(video_hash: str, fps: float, frame_number: int):
    """Удаление конкретного кадра"""
    try:
        with get_db_conn() as conn:
            with conn.cursor() as cur:
                # Получаем ключ кадра
                cur.execute(
                    "SELECT frame_key FROM frames WHERE video_hash = %s AND fps = %s AND frame_number = %s",
                    (video_hash, fps, frame_number)
                )
                frame_key = cur.fetchone()
                
                if not frame_key:
                    raise HTTPException(status_code=404, detail="Frame not found")
                
                # Удаляем из MinIO
                minio_client.remove_object(BUCKET_FRAMES, frame_key[0])
                
                # Удаляем из базы
                cur.execute(
                    "DELETE FROM frames WHERE video_hash = %s AND fps = %s AND frame_number = %s",
                    (video_hash, fps, frame_number)
                )
                conn.commit()
        
        return {"status": "deleted", "frame_number": frame_number}
    
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

@app.post("/classifications/")
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

    return JSONResponse(
        {
            "status": "processed",
            "video_hash": video_hash,
            "fps": fps,
            "loyalty_threshold": thr,
            "results": results,
            "links": {
                "self": f"/classifications/{video_hash}/{fps}/{loyalty}/",
                "video": f"/videos/{video_hash}/"
            }
        },
        status_code=201
    )
    
def split_image_to_patches(img: Image.Image) -> list:
    """Разбивает изображение на патчи согласно GRID_SIZE"""
    w, h = img.size
    pw, ph = w // GRID_SIZE[0], h // GRID_SIZE[1]
    return [img.crop((ix*pw, iy*ph, (ix+1)*pw, (iy+1)*ph))
            for iy in range(GRID_SIZE[1]) for ix in range(GRID_SIZE[0])]
            
@app.get("/classifications/{video_hash}/{fps}/{loyalty}/")
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
            "sharpness_percentage": r[2],
            "links": {
                "frame": f"/frames/{video_hash}/{fps}/{r[0]}",
                "corrections": f"/classification-history/{video_hash}/{fps}/{r[0]}/"
            }
        } for r in results],
        "links": {
            "video": f"/videos/{video_hash}/",
            "frames": f"/frames/{video_hash}/{fps}/"
        }
    }
@app.post("/correct-classification/")
async def correct_classification(
    video_hash: str = Form(...),
    fps: float = Form(...),
    frame_number: int = Form(...),
    loyalty: float = Form(...),
    is_sharp: bool = Form(...),
    user_comment: Optional[str] = Form(None)
):
    """
    Ручная коррекция is_sharp для кадра с учетом loyalty_threshold
    Изменения применяются только к указанному порогу (loyalty_threshold)
    """
    if not 0 <= loyalty <= 100:
        raise HTTPException(status_code=400, detail="Loyalty must be between 0 and 100")
    thr = 100.0 - loyalty

    with get_db_conn() as conn:
        with conn.cursor() as cur:
            # 1. Получаем текущую классификацию ТОЛЬКО для указанного порога
            cur.execute("""
                SELECT is_sharp, sharpness_percentage 
                FROM frame_classifications
                WHERE 
                    video_hash = %s AND 
                    fps = %s AND 
                    frame_number = %s AND
                    loyalty_threshold = %s
                FOR UPDATE
            """, (video_hash, fps, frame_number, thr))
            
            current = cur.fetchone()
            if not current:
                raise HTTPException(
                    status_code=404,
                    detail="Frame classification not found for these parameters"
                )

            original_is_sharp, original_sharpness = current

            # 2. Обновляем ТОЛЬКО запись с указанным порогом
            cur.execute("""
                UPDATE frame_classifications
                SET is_sharp = %s
                WHERE 
                    video_hash = %s AND 
                    fps = %s AND 
                    frame_number = %s AND
                    loyalty_threshold = %s
            """, (
                is_sharp,
                video_hash,
                fps,
                frame_number,
                thr
            ))

            # 3. Логируем изменение с указанием порога
            cur.execute("""
                INSERT INTO manual_classification_corrections
                (video_hash, fps, frame_number, loyalty_threshold,
                 original_is_sharp, corrected_is_sharp,
                 original_sharpness, corrected_sharpness,
                 comment)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                video_hash,
                fps,
                frame_number,
                thr,
                original_is_sharp,
                is_sharp,
                original_sharpness,
                original_sharpness,  # Мы не меняем sharpness_percentage
                user_comment
            ))

            conn.commit()

    return {
        "status": "success",
        "message": f"is_sharp updated for loyalty_threshold={thr}",
        "video_hash": video_hash,
        "fps": fps,
        "frame_number": frame_number,
        "loyalty_threshold": thr,
        "new_is_sharp": is_sharp,
        "previous_is_sharp": original_is_sharp,
        "sharpness_percentage": original_sharpness
    }


@app.get("/classifications-history/{video_hash}/{fps}/{frame_number}/")
async def get_classification_history(
    video_hash: str,
    fps: float,
    frame_number: int,
    loyalty: Optional[float] = None
):
    """
    Получение истории изменений классификации для конкретного кадра
    Если loyalty указан - возвращает историю только для этого порога
    """
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            # Базовый запрос для истории
            query = """
                SELECT 
                    corrected_at,
                    loyalty_threshold,
                    original_is_sharp,
                    corrected_is_sharp,
                    original_sharpness,
                    corrected_sharpness,
                    corrected_by,
                    comment
                FROM manual_classification_corrections
                WHERE 
                    video_hash = %s AND 
                    fps = %s AND 
                    frame_number = %s
            """
            params = [video_hash, fps, frame_number]

            # Если указан loyalty - фильтруем по нему
            if loyalty is not None:
                if not 0 <= loyalty <= 100:
                    raise HTTPException(status_code=400, detail="Loyalty must be between 0 and 100")
                thr = 100.0 - loyalty
                query += " AND loyalty_threshold = %s"
                params.append(thr)

            query += " ORDER BY corrected_at DESC"
            cur.execute(query, params)
            history = cur.fetchall()

            # Получаем текущие классификации
            current_query = """
                SELECT 
                    is_sharp, 
                    sharpness_percentage,
                    loyalty_threshold
                FROM frame_classifications
                WHERE 
                    video_hash = %s AND 
                    fps = %s AND 
                    frame_number = %s
            """
            current_params = [video_hash, fps, frame_number]

            if loyalty is not None:
                current_query += " AND loyalty_threshold = %s"
                current_params.append(thr)

            cur.execute(current_query, current_params)
            current_classifications = cur.fetchall()

    return {
        "current_classifications": [{
            "loyalty_threshold": c[2],
            "is_sharp": c[0],
            "sharpness_percentage": c[1]
        } for c in current_classifications],
        "history": [{
            "timestamp": record[0].isoformat(),
            "loyalty_threshold": record[1],
            "from": {
                "is_sharp": record[2],
                "sharpness": record[4]
            },
            "to": {
                "is_sharp": record[3],
                "sharpness": record[5]
            },
            "corrected_by": record[6],
            "comment": record[7]
        } for record in history]
    }

@app.patch("/classifications/{video_hash}/{fps}/{frame_number}/")
async def partially_update_classification(
    video_hash: str,
    fps: float,
    frame_number: int,
    loyalty: float = Body(...),
    is_sharp: Optional[bool] = Body(None),
    sharpness_percentage: Optional[float] = Body(None),
    user_comment: Optional[str] = Body(None)
):
    """Частичное обновление классификации с учетом loyalty_threshold"""
    if not 0 <= loyalty <= 100:
        raise HTTPException(status_code=400, detail="Loyalty must be between 0 and 100")
    thr = 100.0 - loyalty

    updates = {}
    if is_sharp is not None:
        updates["is_sharp"] = is_sharp
    if sharpness_percentage is not None:
        updates["sharpness_percentage"] = sharpness_percentage
    
    if not updates:
        raise HTTPException(
            status_code=400,
            detail="No fields provided for update"
        )

    with get_db_conn() as conn:
        with conn.cursor() as cur:
            # Получаем текущие значения
            cur.execute("""
                SELECT is_sharp, sharpness_percentage 
                FROM frame_classifications
                WHERE 
                    video_hash = %s AND 
                    fps = %s AND 
                    frame_number = %s AND
                    loyalty_threshold = %s
            """, (video_hash, fps, frame_number, thr))
            
            current = cur.fetchone()
            if not current:
                raise HTTPException(status_code=404, detail="Classification not found for these parameters")
            
            # Применяем обновления
            set_clause = ", ".join([f"{k} = %s" for k in updates.keys()])
            values = list(updates.values()) + [video_hash, fps, frame_number, thr]
            
            cur.execute(f"""
                UPDATE frame_classifications
                SET {set_clause}
                WHERE 
                    video_hash = %s AND 
                    fps = %s AND 
                    frame_number = %s AND
                    loyalty_threshold = %s
                RETURNING *
            """, values)
            
            updated = cur.fetchone()
            
            # Сохраняем в историю
            cur.execute("""
                INSERT INTO manual_classification_corrections
                (video_hash, fps, frame_number, loyalty_threshold,
                 original_is_sharp, corrected_is_sharp,
                 original_sharpness, corrected_sharpness,
                 comment)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                video_hash, fps, frame_number, thr,
                current[0], updated[0],
                current[1], updated[1],
                user_comment
            ))
            
            conn.commit()
    
    return {
        "status": "updated",
        "frame_number": frame_number,
        "loyalty_threshold": thr,
        "changes": updates,
        "links": {
            "self": f"/classifications/{video_hash}/{fps}/{frame_number}/",
            "history": f"/classification-history/{video_hash}/{fps}/{frame_number}/"
        }
    }


@app.get("/reports/{video_hash}/full/", tags=["reports"])
async def generate_full_report(video_hash: str, format: str = "json"):
    """
    Генерация полного отчета о видео
    
    Параметры:
    - format: формат отчета (json или csv)
    
    Возвращает:
    - Полную информацию о видео, всех кадрах и их классификациях
    - История ручных изменений классификации
    - Статистика по качеству кадров
    """
    try:
        # Получаем базовую информацию о видео
        with get_db_conn() as conn:
            with conn.cursor() as cur:
                # Основная информация о видео
                cur.execute("""
                    SELECT hash, filename, fps, processed, created_at 
                    FROM videos 
                    WHERE hash = %s
                """, (video_hash,))
                video_info = cur.fetchone()
                
                if not video_info:
                    raise HTTPException(status_code=404, detail="Video not found")
                
                # Все доступные FPS для этого видео
                cur.execute("""
                    SELECT DISTINCT fps 
                    FROM frames 
                    WHERE video_hash = %s 
                    ORDER BY fps
                """, (video_hash,))
                available_fps = [row[0] for row in cur.fetchall()]
                
                # Собираем данные для каждого FPS
                fps_reports = []
                for fps in available_fps:
                    # Информация о кадрах для этого FPS
                    cur.execute("""
                        SELECT 
                            f.frame_number,
                            f.frame_key,
                            fc.is_sharp,
                            fc.sharpness_percentage,
                            fc.loyalty_threshold,
                            CASE WHEN mcc.corrected_at IS NOT NULL THEN TRUE ELSE FALSE END as was_corrected
                        FROM frames f
                        LEFT JOIN frame_classifications fc 
                            ON f.video_hash = fc.video_hash 
                            AND f.fps = fc.fps 
                            AND f.frame_number = fc.frame_number
                        LEFT JOIN manual_classification_corrections mcc
                            ON f.video_hash = mcc.video_hash 
                            AND f.fps = mcc.fps 
                            AND f.frame_number = mcc.frame_number
                        WHERE f.video_hash = %s AND f.fps = %s
                        ORDER BY f.frame_number
                    """, (video_hash, fps))
                    
                    frames_data = []
                    for row in cur.fetchall():
                        frames_data.append({
                            "frame_number": row[0],
                            "frame_key": row[1],
                            "is_sharp": row[2],
                            "sharpness_percentage": row[3],
                            "loyalty_threshold": row[4],
                            "was_corrected": row[5]
                        })
                    
                    # Статистика для этого FPS
                    cur.execute("""
                        SELECT 
                            COUNT(*) as total_frames,
                            SUM(CASE WHEN fc.is_sharp THEN 1 ELSE 0 END) as sharp_frames,
                            SUM(CASE WHEN NOT fc.is_sharp THEN 1 ELSE 0 END) as blurry_frames,
                            SUM(CASE WHEN mcc.corrected_at IS NOT NULL THEN 1 ELSE 0 END) as corrected_frames,
                            AVG(fc.sharpness_percentage) as avg_sharpness
                        FROM frames f
                        LEFT JOIN frame_classifications fc 
                            ON f.video_hash = fc.video_hash 
                            AND f.fps = fc.fps 
                            AND f.frame_number = fc.frame_number
                        LEFT JOIN manual_classification_corrections mcc
                            ON f.video_hash = mcc.video_hash 
                            AND f.fps = mcc.fps 
                            AND f.frame_number = mcc.frame_number
                        WHERE f.video_hash = %s AND f.fps = %s
                    """, (video_hash, fps))
                    
                    stats = cur.fetchone()
                    fps_stats = {
                        "total_frames": stats[0],
                        "sharp_frames": stats[1],
                        "blurry_frames": stats[2],
                        "corrected_frames": stats[3],
                        "avg_sharpness": float(stats[4]) if stats[4] else 0,
                        "sharp_percentage": (stats[1] / stats[0] * 100) if stats[0] > 0 else 0,
                        "correction_percentage": (stats[3] / stats[0] * 100) if stats[0] > 0 else 0
                    }
                    
                    fps_reports.append({
                        "fps": fps,
                        "frames_count": len(frames_data),
                        "frames": frames_data,
                        "statistics": fps_stats
                    })
                
                # История всех ручных изменений
                cur.execute("""
                    SELECT 
                        fps,
                        frame_number,
                        original_is_sharp,
                        corrected_is_sharp,
                        corrected_by,
                        corrected_at,
                        comment
                    FROM manual_classification_corrections
                    WHERE video_hash = %s
                    ORDER BY corrected_at DESC
                """, (video_hash,))
                
                correction_history = []
                for row in cur.fetchall():
                    correction_history.append({
                        "fps": row[0],
                        "frame_number": row[1],
                        "original_classification": row[2],
                        "new_classification": row[3],
                        "corrected_by": row[4],
                        "timestamp": row[5].isoformat(),
                        "comment": row[6]
                    })
        
        # Формируем полный отчет
        full_report = {
            "video": {
                "hash": video_info[0],
                "filename": video_info[1],
                "upload_date": video_info[4].isoformat(),
                "processed": video_info[3],
                "available_fps": available_fps
            },
            "fps_reports": fps_reports,
            "correction_history": correction_history,
            "summary_statistics": {
                "total_fps_options": len(available_fps),
                "total_frames": sum([r['statistics']['total_frames'] for r in fps_reports]),
                "total_sharp_frames": sum([r['statistics']['sharp_frames'] for r in fps_reports]),
                "total_blurry_frames": sum([r['statistics']['blurry_frames'] for r in fps_reports]),
                "total_corrected_frames": sum([r['statistics']['corrected_frames'] for r in fps_reports]),
                "avg_sharpness_across_fps": sum([r['statistics']['avg_sharpness'] for r in fps_reports]) / len(fps_reports) if fps_reports else 0
            },
            "generated_at": datetime.datetime.now().isoformat()
        }
        
        if format.lower() == "csv":
            # Конвертируем в CSV (упрощенная версия)
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Заголовки
            writer.writerow([
                "Video Hash", "Filename", "FPS", "Frame Number", 
                "Is Sharp", "Sharpness Percentage", "Loyalty Threshold",
                "Was Corrected", "Correction History"
            ])
            
            # Данные
            for fps_report in fps_reports:
                for frame in fps_report["frames"]:
                    # Находим историю изменений для этого кадра
                    frame_history = [
                        h for h in correction_history 
                        if h["fps"] == fps_report["fps"] and h["frame_number"] == frame["frame_number"]
                    ]
                    
                    writer.writerow([
                        video_info[0],
                        video_info[1],
                        fps_report["fps"],
                        frame["frame_number"],
                        frame["is_sharp"],
                        frame["sharpness_percentage"],
                        frame["loyalty_threshold"],
                        frame["was_corrected"],
                        "; ".join([
                            f"{h['timestamp']}: {h['original_classification']}->{h['new_classification']}"
                            for h in frame_history
                        ])
                    ])
            
            return Response(
                content=output.getvalue(),
                media_type="text/csv",
                headers={
                    "Content-Disposition": f"attachment; filename=report_{video_hash}.csv"
                }
            )
        else:
            return JSONResponse(content=full_report)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
