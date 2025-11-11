from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


# Use relative imports to get dependencies from other package files
from .api import router as api_router  # Import the router from api.py
from .config import logger             # Import the configured logger

# --- FastAPI Configuration ---

app = FastAPI(title="Transaction Processing API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allows your frontend to connect
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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