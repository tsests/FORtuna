import psycopg2
from config import settings

def get_db_conn():
    return psycopg2.connect(
        dbname=settings.db_name,
        user=settings.db_user,
        password=settings.db_password,
        host=settings.db_host,
        port=settings.db_port
    )

def save_to_db(query: str, values: tuple):
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute(query, values)
    conn.commit()
    cur.close()
    conn.close()

