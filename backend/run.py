import uvicorn
from app.main import app      # Import the 'app' instance from app/main.py
from app.config import logger # Import the configured logger

if __name__ == "__main__":
    logger.info("Starting Transaction Processing API...")
    
    # This runs the Uvicorn server, pointing it to your main 'app' object
    uvicorn.run(
        "app.main:app",  # The import string: 'directory.filename:app_variable'
        host="0.0.0.0",  # Listen on all network interfaces
        port=8000,       # Run on port 8000
        log_level="info",
        reload=True      # Automatically restart server on code changes (great for dev)
    )