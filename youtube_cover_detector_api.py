import os
from app.utils.logging_config import setup_logger
from fastapi import FastAPI, HTTPException, BackgroundTasks, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import get_swagger_ui_html
from pydantic import BaseModel
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
import time
import asyncio
from typing import Dict
import pathlib
import tempfile
from pathlib import Path
import requests
import yt_dlp
import joblib
import librosa
import numpy as np

YT_DLP_USE_COOKIES = False

# Setup logger
logger = setup_logger('api')

# Configure environment before importing libraries that might use it
os.environ["JOBLIB_TEMP_FOLDER"] = "/tmp"
os.environ["XDG_CACHE_HOME"] = "/tmp/.cache"  # Ensure this is set to a writable directory
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)
os.chmod(os.environ["XDG_CACHE_HOME"], 0o777)
logger.info("Environment configured")

def get_fresh_cookies():
    """Get fresh cookies from YouTube using Selenium"""
    print("Getting fresh YouTube cookies...")
    print(f"Chrome binary: {os.environ.get('CHROME_BIN')}")
    print(f"ChromeDriver path: {os.environ.get('CHROMEDRIVER_PATH')}")
    
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.binary_location = os.environ.get('CHROME_BIN', '/usr/bin/chromium')
    
    try:
        print("Initializing Chrome driver...")
        driver = webdriver.Chrome(options=chrome_options)
        
        # Visit YouTube with a popular video
        print("Visiting YouTube...")
        driver.get("https://www.youtube.com/watch?v=dQw4w9WgXcQ")  # Never gonna give you up :)
        time.sleep(5)  # Wait for cookies
        
        # Format cookies in Netscape format
        cookie_string = "# Netscape HTTP Cookie File\n"
        cookie_string += "# https://curl.haxx.se/rfc/cookie_spec.html\n"
        cookie_string += "# This is a generated file!  Do not edit.\n\n"
        
        for cookie in driver.get_cookies():
            domain = cookie.get('domain', '')
            if not domain.startswith('.'):
                domain = '.' + domain
            
            expiry = cookie.get('expiry', '0')
            if isinstance(expiry, float):
                expiry = int(expiry)
            
            cookie_string += (
                f"{domain}\t"
                "TRUE\t"
                f"{cookie.get('path', '/')}\t"
                f"{'TRUE' if cookie.get('secure', False) else 'FALSE'}\t"
                f"{expiry}\t"
                f"{cookie.get('name', '')}\t"
                f"{cookie.get('value', '')}\n"
            )
        
        return cookie_string
    
    except Exception as e:
        print(f"Error getting fresh cookies: {e}")
        return None
    
    finally:
        if 'driver' in locals():
            driver.quit()

# Get cookies from environment variable or get fresh ones
COOKIE_FILE = '/tmp/youtube_cookies.txt'

def setup_cookies():
    try:
        # Try environment variable first
        cookie_content = os.getenv('YOUTUBE_COOKIES')
        if not cookie_content:
            # Get fresh cookies if no environment variable
            cookie_content = get_fresh_cookies()
        
        if cookie_content:
            with open(COOKIE_FILE, 'w') as f:
                f.write(cookie_content)
            print("Cookies saved successfully")
            return COOKIE_FILE
        
    except Exception as e:
        print(f"Error setting up cookies: {e}")
    
    return None

# Set up cookies on startup
if YT_DLP_USE_COOKIES:
    COOKIE_FILE = setup_cookies()

# Import local modules after environment setup
from app.parse_config import config
from app.youtube_cover_detector import YoutubeCoverDetector, prepare_cover_detection, cover_detection

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

print("Starting application...")

# Create the FastAPI app
app = FastAPI()

# Store results in memory
detection_results: Dict[str, Dict] = {}

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://muoten-youtube-cover-detector4.hf.space",  # Hugging Face Space URL
        "http://localhost:8080",  # Local development
        "https://yt-coverhunter.fly.dev",  # Fly.dev URL
        "https://yt-coverhunter.fly.dev/api",  # Fly.dev API URL
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

# Add API router
api_router = APIRouter()  # Remove prefix to match existing frontend paths

# Add API endpoints
@api_router.post("/api/detect-cover")
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

@api_router.get("/api/detection-status/{task_id}")
async def get_detection_status(task_id: str):
    if task_id not in detection_results:
        raise HTTPException(status_code=404, detail="Task not found")
    return detection_results[task_id]

@api_router.post("/api/get-thumbnails")
async def get_thumbnails(request: VideoRequest):
    try:
        detector = YoutubeCoverDetector()
        video_id1 = detector._get_video_id(request.video_url1)
        video_id2 = detector._get_video_id(request.video_url2)
        
        print(f"Getting thumbnails for videos: {video_id1}, {video_id2}")
        
        response_data = {
            "video_urls": {
                "url1": request.video_url1,
                "url2": request.video_url2
            },
            "thumbnails": {
                "video1": detector._get_thumbnail_url(video_id1),
                "video2": detector._get_thumbnail_url(video_id2)
            }
        }
        
        print(f"Thumbnail response: {response_data}")
        return JSONResponse(
            content=response_data,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type",
            }
        )
    except Exception as e:
        print(f"Error in get_thumbnails: {str(e)}")
        return JSONResponse(
            status_code=400,
            content={"error": str(e)},
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type",
            }
        )

# Add OPTIONS handler for CORS preflight requests
@app.options("/api/get-thumbnails")
async def get_thumbnails_options():
    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        }
    )

async def process_video(video_url1: str, video_url2: str, task_id: str):
    try:
        logger.info(f"Processing videos: {video_url1}, {video_url2}")
        # Clean up old files first
        cleanup_old_files(WAV_DIR)
        
        detector = YoutubeCoverDetector()
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

# Include API router
app.include_router(api_router)

@app.get("/api/docs")
async def get_docs():
    return get_swagger_ui_html(openapi_url="/openapi.json", title="API Documentation")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "api_version": "1.0",
        "endpoints": [
            "/api/detect-cover",
            "/api/detection-status/{task_id}",
            "/api/get-thumbnails"
        ]
    }

@app.get("/")
async def root():
    return {"status": "ok", "message": "YouTube Cover Detector API is running"}

@app.get("/healthz")
async def healthcheck():
    return {"status": "healthy"}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=7860, log_level="debug") 
