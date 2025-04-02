import os

# Configure environment before importing libraries that might use it
os.environ["JOBLIB_TEMP_FOLDER"] = "/tmp"
os.environ["XDG_CACHE_HOME"] = "/tmp/.cache"  # Ensure this is set to a writable directory
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)
os.chmod(os.environ["XDG_CACHE_HOME"], 0o777)

from app.parse_config import config
import numpy as np
from app.youtube_cover_detector import YoutubeCoverDetector, prepare_cover_detection, cover_detection
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import asyncio
from typing import Dict
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import pathlib
import tempfile
from pathlib import Path
import requests
import time
import yt_dlp
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
# Import these after setting environment variable
import joblib
import librosa

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

# Define cookie file path first
cookie_file_path = '/tmp/youtube_cookies.txt'

def get_youtube_cookies():
    # Create a unique temporary directory for user data
    with tempfile.TemporaryDirectory() as chrome_data_dir:
        chrome_cache_dir = f'{chrome_data_dir}/cache'
        os.makedirs(chrome_cache_dir, exist_ok=True)
        
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument(f"--user-data-dir={chrome_data_dir}")
        chrome_options.add_argument(f"--disk-cache-dir={chrome_cache_dir}")
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--remote-debugging-port=9222")
        chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        
        print(f"Chrome options configured with data dir: {chrome_data_dir} and cache dir: {chrome_cache_dir}")
        
        driver = None
        try:
            # Initialize driver
            driver = webdriver.Chrome(service=Service("/usr/bin/chromedriver"), options=chrome_options)
            print("Chrome driver created successfully")
            
            # Visit YouTube and wait
            print("Visiting YouTube...")
            driver.get('https://www.youtube.com/watch?v=HzC2-GJu1Q8')
            time.sleep(5)
            
            # Get all cookies
            cookies = driver.get_cookies()
            print(f"Found {len(cookies)} cookies")
            
            # Save cookies in Netscape format
            cookies_file_path = "/tmp/youtube_cookies.txt"
            with open(cookies_file_path, "w") as f:
                f.write("# Netscape HTTP Cookie File\n")
                for cookie in cookies:
                    # Ensure all necessary attributes are present and convert to string
                    domain = cookie.get("domain", "")
                    name = cookie.get("name", "")
                    value = cookie.get("value", "")
                    path = cookie.get("path", "/")
                    secure = "TRUE" if cookie.get("secure") else "FALSE"
                    expiry = str(cookie.get("expiry", 9999999999))  # fallback
                    
                    # Log cookie details
                    print(f"Writing cookie: domain={domain}, name={name}, value={value}, path={path}, secure={secure}, expiry={expiry}")
                    
                    # Write to file if all necessary attributes are present
                    if domain and name and value:
                        f.write(f"{domain}\tTRUE\t{path}\t{secure}\t{expiry}\t{name}\t{value}\n")
                    else:
                        print(f"Skipping cookie with missing attributes: {cookie}")
            
            print("Cookies saved to /tmp/youtube_cookies.txt")
            
            # Log the contents of the cookies file
            with open(cookies_file_path, "r") as f:
                print("Contents of /tmp/youtube_cookies.txt:")
                print(f.read())
        
        except Exception as e:
            print(f"Error in get_youtube_cookies: {e}")
            raise
        
        finally:
            if driver is not None:
                try:
                    driver.quit()
                    print("Chrome driver closed successfully")
                except Exception as e:
                    print(f"Error closing driver: {e}")

print("Starting YouTube cookie collection with Selenium...")
# Get cookies and write them to file
try:
    print("Initializing Chrome driver...")
    cookie_string = get_youtube_cookies()
    print(f"Got cookies successfully, writing to {cookie_file_path}")
    with open(cookie_file_path, 'w') as f:
        f.write(cookie_string)
    print("Cookies written to file successfully")
except Exception as e:
    print(f"Error getting cookies with Selenium: {e}")
    print("Falling back to environment variable if available...")
    # Fallback to environment variable if available
    if 'YOUTUBE_NETSCAPE_COOKIE' in os.environ:
        with open(cookie_file_path, 'w') as cookie_file:
            cookie_file.write(os.environ['YOUTUBE_NETSCAPE_COOKIE'])
        print("Used cookie from environment variable")
    else:
        print("No cookie fallback available!")

print("Cookie setup complete, starting application...")

# let's try directly download the audio with cookies
command = "yt-dlp --cookies /tmp/youtube_cookies.txt -o '/tmp/yt-dlp/%(id)s.%(ext)s' --extract-audio --audio-format mp3 https://www.youtube.com/watch?v=HzC2-GJu1Q8"
print("command: ", command)

try:
    os.system(command)
except Exception as e:
    print(f"Test download failed: {e}")
    print("Continuing with application startup...")

# Set up yt-dlp options with proper paths
ydl_opts = {
    'cookiefile': cookie_file_path,
    'verbose': True,
    'quiet': False,
    'no_warnings': False,
    'paths': {
        'home': '/tmp/yt-dlp',
        'temp': '/tmp/yt-dlp-temp',
    },
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
    }],
    'cachedir': '/tmp/yt-dlp-cache',
    'outtmpl': '/tmp/yt-dlp/%(id)s.%(ext)s',
}

# Create necessary directories with proper permissions
for directory in ['/tmp/yt-dlp', '/tmp/yt-dlp-temp', '/tmp/yt-dlp-cache']:
    os.makedirs(directory, exist_ok=True)
    os.chmod(directory, 0o777)

# Use yt-dlp to download or process YouTube data
try:
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download(['https://www.youtube.com/watch?v=HzC2-GJu1Q8'])
except Exception as e:
    print(f"Test download with yt-dlp failed: {e}")
    print("Continuing with application startup...")

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

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8080) 