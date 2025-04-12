import os
import logging
import sys
from fastapi import FastAPI, HTTPException, BackgroundTasks, APIRouter, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import get_swagger_ui_html
from pydantic import BaseModel, validator
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
import csv
from app.youtube_cover_detector import (
    extract_video_id, 
    _cosine_distance, 
    _generate_embeddings_from_filepaths,
    _generate_dataset_txt_from_files,
    get_average_processing_time
)
from app.parse_config import config
from contextlib import suppress
from app.youtube_cover_detector import CoverDetector, cleanup_temp_files, logger
import math

#First version that works! Though it takes more than 3 minutes to run in fly.dev free tier
YT_DLP_USE_COOKIES = os.getenv('YT_DLP_USE_COOKIES', False)

def setup_logger():
    logger = logging.getLogger('api')
    
    # Clear any existing handlers to prevent duplicates
    if logger.hasHandlers():
        logger.handlers.clear()
    
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger

# Initialize logger once at module level
logger = setup_logger()

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

# Initialize CSV files if they don't exist
def init_csv_files():
    logger.info("Initializing CSV files...")
    csv_files = {
        config['SCORES_CSV_FILE']: ['url1', 'url2', 'result', 'score', 'feedback'],
        config['VECTORS_CSV_FILE']: ['url', 'vector']
    }
    
    for filepath, headers in csv_files.items():
        try:
            if not os.path.exists(filepath):
                logger.info(f"Creating CSV file: {filepath}")
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                with open(filepath, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(headers)
        except Exception as e:
            logger.error(f"Error creating CSV file {filepath}: {e}")
            raise

init_csv_files()

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

    @validator('video_url1', 'video_url2')
    def validate_urls(cls, v):
        if not v.startswith('http'):
            raise ValueError('URL must start with http')
        if 'youtube.com' not in v and 'youtu.be' not in v:
            raise ValueError('Must be a YouTube URL')
        return v

# Add API router
api_router = APIRouter()  # Remove prefix to match existing frontend paths

# Store active tasks
active_tasks: Dict[str, asyncio.Task] = {}

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
    
    # Cancel any existing task for these videos
    video_key = f"{request.video_url1}_{request.video_url2}"
    if video_key in active_tasks:
        active_tasks[video_key].cancel()
        with suppress(asyncio.CancelledError):
            await active_tasks[video_key]
        del active_tasks[video_key]
    
    # Store the task so it can be cancelled if needed
    task = asyncio.create_task(process_video(request.video_url1, request.video_url2))
    active_tasks[video_key] = task
    
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

async def process_video(video_url1: str, video_url2: str):
    try:
        logger.info(f"Processing videos: {video_url1}, {video_url2}")
        # Clean up old files first
        cleanup_old_files(WAV_DIR)
        
        detector = YoutubeCoverDetector()
        result = await detector.detect_cover(video_url1, video_url2)
        
        detection_results[video_url1 + "_" + video_url2] = {
            "status": "completed",
            "result": result,
            "error": None
        }
    except Exception as e:
        detection_results[video_url1 + "_" + video_url2] = {
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

# Mount the static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

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

# Update the CSV_FILE path
CSV_FILE = os.getenv('CSV_FILE', '/data/compared_videos.csv')

def read_compared_videos():
    compared_videos = []
    try:
        with open(CSV_FILE, mode='r', newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                compared_videos.append(row)
    except FileNotFoundError:
        logger.debug("CSV file not found. Creating a new file with headers.")
        with open(CSV_FILE, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['url1', 'url2', 'result', 'score', 'feedback'])
            writer.writeheader()
    return compared_videos

def write_compared_video(url1, url2, result, score):
    logger.debug(f"Starting write_compared_video with params: {url1}, {url2}, {result}, {score}")
    try:
        with open(CSV_FILE, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['url1', 'url2', 'result', 'score', 'feedback'])
            if file.tell() == 0:
                writer.writeheader()
                logger.debug("Writing header to CSV file")
            row_data = {
                'url1': url1, 
                'url2': url2, 
                'result': result, 
                'score': score,
                'feedback': ''
            }
            writer.writerow(row_data)
            logger.debug(f"Successfully wrote row to CSV: {row_data}")
    except Exception as e:
        logger.error(f"Error writing to CSV: {e}")
        raise

def log_csv_contents():
    logger.debug("##### Current contents of the CSV file #####")
    compared_videos = read_compared_videos()
    for video in compared_videos:
        logger.debug(video)
    logger.debug("############################################")

@app.get('/api/compared-videos')
async def get_compared_videos(request: Request):
    compared_videos = read_compared_videos()
    return JSONResponse(content=compared_videos)

""" @app.post('/api/detect-cover-dummy')
async def detect_cover_route(request: Request):
    data = await request.json()
    url1 = data.get('video_url1')
    url2 = data.get('video_url2')
    
    # Simulate a cover detection result and score
    result = 'Cover' if url1 and url2 else 'Not a Cover'
    score = 0.85 if result == 'Cover' else 0.15  # Example score
    
    write_compared_video(url1, url2, result, score)
    return JSONResponse(content={'result': result, 'score': score}) """

@app.post("/api/check-if-cover")
async def check_if_cover(request: VideoRequest, background_tasks: BackgroundTasks):
    video_key = f"{request.video_url1}_{request.video_url2}"
    
    # Use dynamic timeout based on recent performance
    timeout = math.ceil(get_average_processing_time() * 1.1)
    
    logger.info(f"Starting cover detection for videos: {request.video_url1} and {request.video_url2}")
    
    # Cancel any existing task for these videos
    if video_key in active_tasks:
        active_tasks[video_key].cancel()
        with suppress(asyncio.CancelledError):
            await active_tasks[video_key]
        del active_tasks[video_key]
    
    # Store the task so it can be cancelled if needed
    task = asyncio.create_task(process_videos(request.video_url1, request.video_url2))
    active_tasks[video_key] = task
    
    try:
        result = await asyncio.wait_for(task, timeout=timeout)
        del active_tasks[video_key]
        return result
    except asyncio.TimeoutError:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        del active_tasks[video_key]
        if config['CLEANUP_ON_TIMEOUT']:
            cleanup_temp_files(request.video_url1, request.video_url2)
        raise HTTPException(status_code=408, detail="Request timeout")
    except Exception as e:
        logger.error(f"Error in check_if_cover: {str(e)}")
        logger.exception(e)  # This will log the full traceback
        if video_key in active_tasks:
            active_tasks[video_key].cancel()
            del active_tasks[video_key]
        raise HTTPException(status_code=500, detail=f"Failed to start detection process: {str(e)}")

async def process_videos(url1: str, url2: str):
    detector = CoverDetector()
    start_time = time.time()
    try:
        result = await detector.compare_videos(url1, url2)
        elapsed_time = time.time() - start_time
        result['elapsed_time'] = round(elapsed_time, 2)  # Add elapsed time to result
        return result
    except Exception as e:
        logger.error(f"Error in process_videos: {str(e)}")
        logger.exception(e)
        raise HTTPException(status_code=500, detail=f"Error processing videos: {str(e)}")

def reset_compared_videos():
    """Reset the CSV file by creating a new empty file with just the header"""
    logger.debug("Resetting compared videos CSV file")
    try:
        with open(CSV_FILE, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['url1', 'url2', 'result', 'score'])
            writer.writeheader()
        logger.debug("CSV file has been reset successfully")
    except Exception as e:
        logger.error(f"Error resetting CSV file: {e}")
        raise

@app.post('/api/reset-compared-videos')
async def reset_compared_videos_endpoint():
    """Endpoint to reset the compared videos CSV file"""
    try:
        reset_compared_videos()
        return JSONResponse(content={"status": "success", "message": "Compared videos have been reset"})
    except Exception as e:
        logger.error(f"Error in reset_compared_videos_endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/feedback")
async def submit_feedback(request: Request):
    logger.info("Received feedback request")
    try:
        data = await request.json()
        url1 = data['url1']
        url2 = data['url2']
        feedback = data['feedback']
        
        logger.info(f"Processing feedback: {url1}, {url2}, {feedback}")
        
        # Read current CSV
        videos = read_compared_videos()
        logger.debug(f"Read {len(videos)} entries from CSV")
        
        # Update the feedback for matching entry
        found = False
        for video in videos:
            if ((video['url1'] == url1 and video['url2'] == url2) or 
                (video['url1'] == url2 and video['url2'] == url1)):
                video['feedback'] = feedback
                found = True
                logger.info(f"Updated feedback for entry")
                break
        
        if not found:
            logger.error(f"No matching entry found for URLs: {url1}, {url2}")
            raise HTTPException(status_code=404, detail="No matching entry found")
        
        # Write back to CSV
        csv_file = config['SCORES_CSV_FILE']
        os.makedirs(os.path.dirname(csv_file), exist_ok=True)
        
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['url1', 'url2', 'result', 'score', 'feedback', 'elapsed_time'])
            writer.writeheader()
            writer.writerows(videos)
            logger.info(f"Successfully wrote {len(videos)} entries back to CSV")
        
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error in submit_feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def _generate_audio_from_youtube_id(youtube_id):
    try:
        logger.debug("Starting audio generation process")
        wav_folder = Path(WAV_DIR)
        wav_folder.mkdir(exist_ok=True, parents=True)
        os.chmod(str(wav_folder), 0o777)  # Give full permissions
        
        logger.info(f"Downloading video {youtube_id}...")
        
        # Download with yt-dlp
        ydl_opts = {
            'format': 'bestaudio/best',
            'quiet': False,
            'no_warnings': True,
            'logger': logger,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
            }],
            'outtmpl': f'{WAV_DIR}/{youtube_id}.%(ext)s',
            'external_downloader_args': ['--max-tries=10'],  # Added retry attempts
            'socket_timeout': 30,
            'retries': 10,
            'fragment_retries': 10,
            'ignoreerrors': False  # Changed to False to catch errors
        }
        
        mp3_path = f'{WAV_DIR}/{youtube_id}.mp3'
        wav_path = f'{WAV_DIR}/{youtube_id}.wav'
        
        logger.info(f"Attempting download to {mp3_path}")
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                # Check if directory is writable
                if not os.access(WAV_DIR, os.W_OK):
                    logger.error(f"Directory {WAV_DIR} is not writable")
                    os.chmod(WAV_DIR, 0o777)
                    logger.info(f"Changed permissions on {WAV_DIR}")

                logger.info(f"Starting download with best quality...")
                ydl.download([f'https://www.youtube.com/watch?v={youtube_id}'])
                
                if not os.path.exists(mp3_path):
                    logger.error(f"Download completed but MP3 not found at {mp3_path}")
                    logger.error(f"Directory contents: {os.listdir(WAV_DIR)}")
                    raise ValueError("MP3 file not created after download")
                    
            except Exception as e:
                logger.error(f"First download attempt failed: {str(e)}")
                # Try alternate format if first attempt fails
                logger.info("Retrying with worst quality...")
                ydl_opts['format'] = 'worstaudio/worst'
                ydl.download([f'https://www.youtube.com/watch?v={youtube_id}'])
                
                if not os.path.exists(mp3_path):
                    logger.error(f"Both download attempts failed. MP3 not found at {mp3_path}")
                    logger.error(f"Directory contents: {os.listdir(WAV_DIR)}")
                    raise ValueError("MP3 file not created after both download attempts")
        
        # Convert to WAV
        logger.info(f"Converting {mp3_path} to {wav_path}...")
        
        # Use ffmpeg with more detailed output
        ffmpeg_cmd = f'ffmpeg -y -i "{mp3_path}" -ac 1 -ar 16000 -t 10 "{wav_path}" 2>&1'
        logger.info(f"Running ffmpeg command: {ffmpeg_cmd}")
        conversion_output = os.popen(ffmpeg_cmd).read()
        logger.info(f"FFmpeg output: {conversion_output}")
        
        if not os.path.exists(wav_path):
            logger.error(f"WAV file not created at {wav_path}")
            logger.error(f"Directory contents after conversion: {os.listdir(WAV_DIR)}")
            raise ValueError("Audio conversion failed")
        
        logger.info(f"Successfully created WAV file: {wav_path}")
        return f"{youtube_id}.wav"
        
    except Exception as e:
        logger.error(f"Error in _generate_audio_from_youtube_id: {str(e)}")
        logger.error(f"Full error details: {type(e).__name__}: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

@app.post("/api/cleanup")
async def cleanup_request(request: VideoRequest):
    video_key = f"{request.video_url1}_{request.video_url2}"
    if video_key in active_tasks:
        active_tasks[video_key].cancel()
        with suppress(asyncio.CancelledError):
            await active_tasks[video_key]
        del active_tasks[video_key]
    return {"status": "cleaned"}

@app.get("/api/avg-processing-time")
async def get_avg_time():
    avg_time = get_average_processing_time()
    return {"avg_time": avg_time}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=7860, log_level="debug") 
