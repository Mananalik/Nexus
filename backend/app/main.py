from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os


# Use relative imports to get dependencies from other package files
from .api import router as api_router  # Import the router from api.py
from .config import logger             # Import the configured logger

# --- FastAPI Configuration ---

app = FastAPI(title="Transaction Processing API", version="1.0.0")

default_origins = [
    "http://localhost:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:3001",
]

frontend_origins_env = os.getenv("FRONTEND_ORIGINS", "")


def _normalize_origin(origin: str) -> str:
    # CORS origin values must be scheme + host (+ optional port), without trailing slash.
    return origin.strip().rstrip("/")


frontend_origins = [_normalize_origin(o) for o in frontend_origins_env.split(",") if o.strip()]
allow_origins = frontend_origins or default_origins
allow_origin_regex = os.getenv("FRONTEND_ORIGIN_REGEX", r"https://.*\.vercel\.app")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_origin_regex=allow_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info(f"CORS allow_origins configured: {allow_origins}")
logger.info(f"CORS allow_origin_regex configured: {allow_origin_regex}")

# Include all the routes from api.py
# All routes in api_router will be prefixed with /api
app.include_router(api_router)


# --- Root Endpoint ---

@app.get("/")
async def root():
    """Root endpoint with API information"""
    logger.info("Root endpoint was hit")
    return {
        "message": "Transaction Processing API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/api/health",
            "process_transactions": "/api/process-transactions (POST)",
            "financial_advisor": "/api/financial-advisor (POST)"
        }
    }