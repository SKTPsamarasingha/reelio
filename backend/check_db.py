import os
import psycopg
from dotenv import load_dotenv

load_dotenv()
url = os.environ["DATABASE_URL"]
try:
    with psycopg.connect(url, connect_timeout=10) as conn:
        print(conn.execute("select version()").fetchone()[0])
except Exception as e:
    print(type(e).__name__, str(e)[:300])