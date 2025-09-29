from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.post("/api/process-transactions")
async def process_transactions(file: UploadFile = File(...)):
    if file.content_type != 'text/html':
         raise HTTPException(status_code=400, detail="Invalid file type. Please upload an HTML file.")
    print(f"Received file: {file.filename}")

    return {
         "filename": file.filename,
        "content_type": file.content_type,
        "message": "Backend has successfully received the file!"
    }