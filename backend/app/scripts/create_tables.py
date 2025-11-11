# app/scripts/create_tables.py
import asyncio
import os
import sys
import importlib

from app.db import engine, Base
from dotenv import load_dotenv
load_dotenv()
print("DATABASE_URL (from env):", os.getenv("DATABASE_URL"))

# Import model modules so that SQLAlchemy sees the Table mappings and registers them
# Add any other modules that define models here if you create more later.
try:
    importlib.import_module("app.models")
    print("Imported app.models")
except Exception as e:
    print("Warning: could not import app.models:", e)

async def create():
    # show which tables SQLAlchemy currently knows about
    print("Before create_all, registered tables:", list(Base.metadata.tables.keys()))

    async with engine.begin() as conn:
        print("Connected, creating tables...")
        await conn.run_sync(Base.metadata.create_all)
        print("create_all finished")

    # re-print registered tables (should be same names) and verify
    print("After create_all, registered tables:", list(Base.metadata.tables.keys()))

if __name__ == "__main__":
    asyncio.run(create())
