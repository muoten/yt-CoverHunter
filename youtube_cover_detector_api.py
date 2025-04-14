import os
import logging
import sys
from fastapi import FastAPI, HTTPException, BackgroundTasks, APIRouter, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, Response
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
    get_average_processing_time,
    comparison_queue,
    get_result_from_csv,
    process_queue,
    read_compared_videos,
    logger
)
from app.parse_config import config
from contextlib import suppress
from app.youtube_cover_detector import (
    CoverDetector, 
    cleanup_temp_files, 
    logger
)
import math
from multiprocessing import Process, Queue, Manager
from app.background_worker import start_background_worker
from functools import lru_cache
from datetime import datetime, timedelta
import statistics

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

# Mount static files first - these should always be accessible
app.mount("/static", StaticFiles(directory="static"), name="static")

# Basic routes that should always work
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main page - this should always work"""
    with open('templates/index.html') as f:
        return f.read()

@app.get("/api/compared-videos")
async def get_compared_videos():
    """Get history - this should always work"""
    videos = read_compared_videos()
    return videos

@app.get("/api/queue-status")
async def get_queue_status():
    """Get current queue status"""
    # Count both queued tasks and tasks that are currently processing
    processing_tasks = len([t for t in shared_active_tasks.values() 
                          if t.get('status') in ['pending', 'downloading', 'downloading_first', 
                                               'downloading_second', 'processing']])
    queued_tasks = comparison_queue.qsize()
    
    # Read the CSV to count completed comparisons
    with open(config['SCORES_CSV_FILE'], 'r') as f:
        csv_reader = csv.reader(f)
        next(csv_reader)  # Skip header
        completed_count = sum(1 for row in csv_reader)  # Count actual rows
    
    return {
        "pending_tasks": queued_tasks + processing_tasks,
        "completed_comparisons": completed_count
    }

# Create shared queues and dictionaries
manager = Manager()
shared_active_tasks = manager.dict()
comparison_queue = Queue()

# Initialize the background worker
@app.on_event("startup")
async def startup_event():
    """Start the queue processor on app startup"""
    # Check if model exists
    model_path = os.path.join(config['MODEL_FOLDER'], 'checkpoints/g_00000043')
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}")
        # Exit the application if model is missing
        sys.exit(1)
        
    logger.info("Starting queue processor...")
    start_background_worker(comparison_queue, shared_active_tasks)
    logger.info("Queue processor started")

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
    if video_key in shared_active_tasks:
        shared_active_tasks[video_key].cancel()
        with suppress(asyncio.CancelledError):
            await shared_active_tasks[video_key]
        del shared_active_tasks[video_key]
    
    # Store the task so it can be cancelled if needed
    task = asyncio.create_task(process_video(request.video_url1, request.video_url2))
    shared_active_tasks[video_key] = task
    
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
        result = await detector.detect_cover(video_url1, video_2)
        
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

@app.post("/api/check-if-cover")
async def check_if_cover(request: VideoRequest, background_tasks: BackgroundTasks):
    try:
        # Check if videos were already compared
        existing_result = get_result_from_csv(request.video_url1, request.video_url2)
        if existing_result:
            return {
                "status": "completed",
                "result": {
                    "result": "Cover" if existing_result["is_cover"] else "Not Cover",
                    "score": existing_result["distance"]
                }
            }
        
        # Check if system is busy
        processing_tasks = len([t for t in shared_active_tasks.values() 
                              if t.get('status') in ['pending', 'downloading', 'downloading_first', 
                                                    'downloading_second', 'processing']])
        queued_tasks = comparison_queue.qsize()
        total_pending = queued_tasks + processing_tasks
        
        if total_pending > 0:  # Block if anything is being processed or queued
            return {
                "status": "busy",
                "message": "System is busy. Please try again in a few minutes."
            }
        
        # Add new request to queue
        request_id = f"{hash((request.video_url1, request.video_url2))}"
        task = {
            'id': request_id,
            'url1': request.video_url1,
            'url2': request.video_url2,
            'status': 'pending',
            'start_time': time.time()
        }
        
        logger.info(f"Adding request {request_id} to queue")
        comparison_queue.put(task)
        shared_active_tasks[request_id] = task
        logger.info(f"Request {request_id} added to queue. New queue size: {comparison_queue.qsize()}")
        
        return {
            "request_id": request_id,
            "status": "queued",
            "message": f"Request added to queue. Position: {comparison_queue.qsize()}"
        }
        
    except Exception as e:
        logger.error(f"Error in check_if_cover: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Add these variables
last_median_calculation = None
cached_median = None
MEDIAN_CACHE_DURATION = timedelta(minutes=5)

def get_recent_processing_times():
    """Get list of recent processing times from completed comparisons"""
    # Read from the CSV file
    try:
        with open('processing_times.csv', 'r') as f:
            reader = csv.reader(f)
            # Skip header
            next(reader)
            # Get last 3 entries
            times = []
            for row in reader:
                if len(row) >= 2:  # Ensure row has enough columns
                    try:
                        process_time = float(row[1])
                        times.append(process_time)
                    except (ValueError, IndexError):
                        continue
            # Return last 3 times
            return times[-3:]
    except (FileNotFoundError, IOError):
        return []

@lru_cache(maxsize=1)
def calculate_median_processing_time(recent_entries_tuple):
    """Calculate median processing time from recent entries"""
    times = list(recent_entries_tuple)
    if not times:
        return None
    return round(statistics.median(times))

def get_median_processing_time():
    """Get median processing time with caching"""
    global last_median_calculation, cached_median
    
    now = datetime.now()
    if (last_median_calculation is None or 
        now - last_median_calculation > MEDIAN_CACHE_DURATION):
        
        # Read recent processing times
        recent_times = get_recent_processing_times()
        
        # Convert to tuple for caching
        times_tuple = tuple(recent_times)
        
        # Calculate new median
        cached_median = calculate_median_processing_time(times_tuple)
        last_median_calculation = now
        
        logger.info(f"Updated median processing time: {cached_median} seconds")
    
    return cached_median

@app.get("/api/comparison-status/{request_id}")
async def get_comparison_status(request_id: str):
    """Get the status of a comparison request"""
    task = shared_active_tasks.get(request_id)
    if task:
        if task['status'] == 'completed':
            # Remove completed task from active tasks
            del shared_active_tasks[request_id]
        else:
            # Add estimated time remaining based on median of last 3 processes
            avg_time = get_median_processing_time()  # This gets median of last 3 entries
            start_time = task.get('start_time', time.time())
            elapsed = time.time() - start_time
            
            # Check for timeout
            if elapsed > config['REQUEST_TIMEOUT']:
                task['status'] = 'failed'
                task['error'] = 'Request timed out'
                return task
            
            # Calculate remaining time
            remaining = max(0, avg_time - elapsed)
            
            # If less than 1 second remaining but not done, extend to timeout
            if remaining < 1 and task['status'] != 'completed':
                remaining = max(0, config['REQUEST_TIMEOUT'] - elapsed)
                task['extended'] = True
            
            task['estimated_remaining'] = remaining
            
            # Update status based on elapsed time
            if elapsed < 10:
                task['status'] = 'downloading_videos'
                task['progress'] = (elapsed / 10) * 30
            elif elapsed < 25:
                task['status'] = 'processing_audio'
                task['progress'] = 30 + ((elapsed - 10) / 15) * 40
            else:
                task['status'] = 'calculating_similarity'
                task['progress'] = 70 + ((elapsed - 25) / (avg_time - 25)) * 25
            
            task['progress'] = min(95, task['progress'])
        
        return task
    return {"status": "not_found"}

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
    if video_key in shared_active_tasks:
        shared_active_tasks[video_key].cancel()
        with suppress(asyncio.CancelledError):
            await shared_active_tasks[video_key]
        del shared_active_tasks[video_key]
    return {"status": "cleaned"}

@app.get("/api/avg-processing-time")
async def get_avg_time():
    avg_time = get_average_processing_time()
    return {"avg_time": avg_time}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8080, workers=4, log_level="debug") 
