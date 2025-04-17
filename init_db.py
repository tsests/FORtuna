from db import get_db_conn

def init_tables():
    TABLES = {
        "videos": """
            CREATE TABLE IF NOT EXISTS videos (
                id SERIAL PRIMARY KEY,
                hash TEXT NOT NULL,
                filename TEXT,
                fps FLOAT NOT NULL,
                processed BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (hash, fps)
            );
        """,
        "frames": """
            CREATE TABLE IF NOT EXISTS frames (
                id SERIAL PRIMARY KEY,
                video_hash TEXT NOT NULL,
                fps FLOAT NOT NULL,
                frame_number INTEGER NOT NULL,
                frame_key TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (video_hash, fps) REFERENCES videos(hash, fps) ON DELETE CASCADE,
                UNIQUE (video_hash, fps, frame_number)
            );
        """,
        "frame_classifications": """
            CREATE TABLE IF NOT EXISTS frame_classifications (
                id SERIAL PRIMARY KEY,
                video_hash TEXT NOT NULL,
                fps FLOAT NOT NULL,
                frame_number INTEGER NOT NULL,
                is_sharp BOOLEAN NOT NULL,
                sharpness_percentage FLOAT NOT NULL,
                loyalty_threshold FLOAT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (video_hash, fps, frame_number) 
                    REFERENCES frames(video_hash, fps, frame_number) ON DELETE CASCADE,
                UNIQUE (video_hash, fps, frame_number, loyalty_threshold)
            );
        """,
        "manual_classification_corrections": """
            CREATE TABLE IF NOT EXISTS manual_classification_corrections (
                id SERIAL PRIMARY KEY,
                video_hash TEXT NOT NULL,
                fps FLOAT NOT NULL,
                frame_number INTEGER NOT NULL,
                original_is_sharp BOOLEAN NOT NULL,
                corrected_is_sharp BOOLEAN NOT NULL,
                original_sharpness FLOAT NOT NULL,
                corrected_sharpness FLOAT NOT NULL,
                corrected_by TEXT DEFAULT 'user',
                corrected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (video_hash, fps, frame_number) 
                    REFERENCES frames(video_hash, fps, frame_number) ON DELETE CASCADE
            );
        """
    }

    with get_db_conn() as conn:
        with conn.cursor() as cur:
            for ddl in TABLES.values():
                cur.execute(ddl)
            conn.commit()
