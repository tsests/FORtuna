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
                frame_number INTEGER,
                frame_key TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (video_hash, fps) REFERENCES videos(hash, fps) ON DELETE CASCADE
            );
        """
    }

    with get_db_conn() as conn:
        with conn.cursor() as cur:
            for ddl in TABLES.values():
                cur.execute(ddl)
            conn.commit()
