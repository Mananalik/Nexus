# app/scripts/list_tables.py
import asyncio
import os
from urllib.parse import urlparse
import asyncpg
from dotenv import load_dotenv
load_dotenv()

def normalize_dsn_for_asyncpg(dsn: str) -> str:
    """
    Convert SQLAlchemy-style DSN with +asyncpg into a plain asyncpg DSN.
    e.g. postgresql+asyncpg://user:pass@host:port/db -> postgresql://user:pass@host:port/db
    """
    if not dsn:
        raise ValueError("DATABASE_URL is empty")
    # Remove the +asyncpg part if present
    return dsn.replace("postgresql+asyncpg://", "postgresql://").replace("postgres+asyncpg://", "postgres://")

async def main():
    url = os.getenv("DATABASE_URL")
    print("DATABASE_URL:", url)
    if not url:
        print("ERROR: DATABASE_URL env var not set")
        return

    dsn = normalize_dsn_for_asyncpg(url)
    print("Using DSN for asyncpg:", dsn)

    try:
        conn = await asyncpg.connect(dsn)
    except Exception as e:
        print("Failed to connect:", e)
        return

    try:
        rows = await conn.fetch(
            "SELECT table_schema, table_name FROM information_schema.tables WHERE table_schema='public' ORDER BY table_name;"
        )
        if not rows:
            print("No tables found in public schema.")
        else:
            print("Tables in public schema:")
            for r in rows:
                print(f" - {r['table_schema']}.{r['table_name']}")
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(main())
