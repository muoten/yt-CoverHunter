# export an api to detect if 2 youtube videos are covers of each other
import os
from app.parse_config import config
import numpy as np
from app.youtube_cover_detector import YoutubeCoverDetector, prepare_cover_detection, cover_detection
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import asyncio
from typing import Dict
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import pathlib
import tempfile
from pathlib import Path
import requests
import time

# Use Render's persistent storage if available
if os.getenv('RENDER'):
    WAV_DIR = Path("/opt/render/project/wav_files")
else:
    WAV_DIR = Path("/tmp/youtube_cover_detector_api_wav")

# Create WAV directory with proper permissions
WAV_DIR.mkdir(exist_ok=True, parents=True)
os.chmod(str(WAV_DIR), 0o777)  # Give full permissions

# Update config
config['WAV_FOLDER'] = str(WAV_DIR)

# Update the THRESHOLD setting
THRESHOLD = float(os.getenv('THRESHOLD', config.get('THRESHOLD', 0.3)))

# Add more logging
def download_checkpoint():
    CHECKPOINT_URL = os.getenv('CHECKPOINT_URL')
    if CHECKPOINT_URL:
        print("Starting model download from:", CHECKPOINT_URL)
        try:
            os.makedirs('pretrain_model', exist_ok=True)
            response = requests.get(CHECKPOINT_URL)
            if response.status_code == 200:
                with open('pretrain_model/checkpoint.pt', 'wb') as f:
                    f.write(response.content)
                print("Model checkpoint downloaded successfully!")
            else:
                print(f"Failed to download model: {response.status_code}")
        except Exception as e:
            print(f"Error downloading model: {str(e)}")
            raise

print("Starting application...")
download_checkpoint()
print("Checkpoint download complete, initializing FastAPI app...")

# Create the FastAPI app
app = FastAPI()

# Store results in memory
detection_results: Dict[str, Dict] = {}

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://muoten-yt-cover-detector.hf.space",  # Hugging Face Space URL
        "http://localhost:7860",  # Local development
        "*"  # Allow all origins for testing
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define directories
BASE_DIR = pathlib.Path(__file__).parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

# Create directories if they don't exist
TEMPLATES_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

# Add VideoRequest model
class VideoRequest(BaseModel):
    video_url1: str
    video_url2: str

# Create cookies file from environment variable
def setup_cookies():
    cookies_content = os.environ.get('YOUTUBE_COOKIES', '')
    print("Cookies environment variable present:", bool(cookies_content))  # Debug line
    if cookies_content:
        cookies_path = '/tmp/cookies.txt'
        try:
            with open(cookies_path, 'w') as f:
                f.write(cookies_content)
            print(f"Cookies file created at: {cookies_path}")  # Debug line
            # Verify content
            with open(cookies_path, 'r') as f:
                print("First few lines of cookies file:", f.read()[:100])  # Debug line
            return cookies_path
        except Exception as e:
            print(f"Error setting up cookies: {str(e)}")  # Debug line
            return None
    return None

# Call this when your app starts
cookies_file = setup_cookies()
print(f"Cookies file path: {cookies_file}")  # Debug line

# Add API endpoints
@app.post("/api/detect-cover")
async def detect_cover(request: VideoRequest, background_tasks: BackgroundTasks):
    # Generate a unique task ID
    task_id = str(hash(request.video_url1 + request.video_url2 + str(asyncio.get_event_loop().time())))
    
    # Initialize the task status
    detection_results[task_id] = {
        "status": "processing",
        "result": None,
        "error": None
    }
    
    # Add the task to background processing
    background_tasks.add_task(process_video, request.video_url1, request.video_url2, task_id)
    
    return {"task_id": task_id, "status": "processing"}

@app.get("/api/detection-status/{task_id}")
async def get_detection_status(task_id: str):
    if task_id not in detection_results:
        raise HTTPException(status_code=404, detail="Task not found")
    return detection_results[task_id]

@app.post("/api/get-thumbnails")
async def get_thumbnails(request: VideoRequest):
    try:
        detector = YoutubeCoverDetector()
        video_id1 = detector._get_video_id(request.video_url1)
        video_id2 = detector._get_video_id(request.video_url2)
        
        return {
            "video_urls": {
                "url1": request.video_url1,
                "url2": request.video_url2
            },
            "thumbnails": {
                "video1": detector._get_thumbnail_url(video_id1),
                "video2": detector._get_thumbnail_url(video_id2)
            }
        }
    except Exception as e:
        print(f"Error in get_thumbnails: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

async def process_video(video_url1: str, video_url2: str, task_id: str):
    try:
        # Clean up old files first
        cleanup_old_files(WAV_DIR)
        
        detector = YoutubeCoverDetector()
        detector.ydl_opts = {  # Set ydl_opts as an instance variable
            'format': 'bestaudio/best',
            'cookiefile': cookies_file,
        }
        result = await detector.detect_cover(video_url1, video_url2)
        
        detection_results[task_id] = {
            "status": "completed",
            "result": result,
            "error": None
        }
    except Exception as e:
        detection_results[task_id] = {
            "status": "failed",
            "error": str(e),
            "result": None
        }
    finally:
        # Clean up files after processing
        cleanup_old_files(WAV_DIR, max_age_hours=0)

def cleanup_old_files(directory: Path, max_age_hours: int = 1):
    """Remove files older than max_age_hours"""
    current_time = time.time()
    for file_path in directory.glob("*.wav"):
        if current_time - file_path.stat().st_mtime > (max_age_hours * 3600):
            try:
                file_path.unlink()
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")

# Serve static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Serve index.html at root
@app.get("/")
async def read_root():
    return FileResponse(str(TEMPLATES_DIR / "index.html"))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=7860) 