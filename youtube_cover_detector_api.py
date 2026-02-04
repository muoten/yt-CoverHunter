import os
import logging
import sys
from fastapi import FastAPI, HTTPException, BackgroundTasks, APIRouter, Request, Query
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
    cleanup_chrome_user_data_dirs,
    logger
)
import math
from multiprocessing import Process, Queue, Manager
from app.background_worker import start_background_worker
import psutil
from app.utils.memory_logger import log_detailed_memory
def cleanup_completed_tasks():
    """Remove completed tasks older than 5 minutes from shared_active_tasks"""
    current_time = time.time()
    tasks_to_remove = []
    
    for task_id, task in shared_active_tasks.items():
        if task.get('status') == 'completed':
            # Remove completed tasks older than 5 minutes
            completed_time = task.get('completed_time', task.get('start_time', current_time))
            if current_time - completed_time > 300:  # 5 minutes
                tasks_to_remove.append(task_id)
    
    # Remove old completed tasks
    for task_id in tasks_to_remove:
        del shared_active_tasks[task_id]
        logger.info(f'Cleaned up old completed task: {task_id}')
    
    return len(tasks_to_remove)



# Global cache for all videos
_all_videos_cache = None
_cache_timestamp = 0
CACHE_DURATION = 30  # Cache for 30 seconds

def cleanup_old_wav_files(max_age_hours: int = 2):
    """Remove WAV files older than specified hours to prevent disk space issues"""
    wav_folder = config['WAV_FOLDER']
    if not os.path.exists(wav_folder):
        return
    
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600  # Convert hours to seconds
    removed_count = 0
    total_size_freed = 0
    
    try:
        for filename in os.listdir(wav_folder):
            file_path = os.path.join(wav_folder, filename)
            
            # Only process WAV files
            if not filename.endswith('.wav'):
                continue
            
            try:
                # Get file modification time
                file_mtime = os.path.getmtime(file_path)
                file_age = current_time - file_mtime
                
                # Check if file is older than max_age_hours
                if file_age > max_age_seconds:
                    file_size = os.path.getsize(file_path)
                    os.remove(file_path)
                    removed_count += 1
                    total_size_freed += file_size
                    logger.info(f"Removed old WAV file: {filename} (age: {file_age/3600:.1f}h, size: {file_size/1024/1024:.1f}MB)")
                    
            except Exception as e:
                logger.error(f"Error processing file {filename}: {e}")
        
        if removed_count > 0:
            logger.info(f"WAV cleanup completed: removed {removed_count} files, freed {total_size_freed/1024/1024:.1f}MB")
        else:
            logger.debug("WAV cleanup completed: no old files found")
            
    except Exception as e:
        logger.error(f"Error during WAV cleanup: {e}")

# Chrome cleanup is imported from app.youtube_cover_detector
# Using alias for backward compatibility with existing code
cleanup_chrome_user_data = cleanup_chrome_user_data_dirs

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
async def get_compared_videos(
    page: int = Query(1, ge=1, description="Page number (1-based)"),
    per_page: int = Query(100, ge=1, le=500, description="Items per page")
):
    """Get history with true pagination - only reads needed lines from CSV"""
    try:
        # First, get total count efficiently without loading all data
        total_count = 0
        try:
            with open(config['SCORES_CSV_FILE'], 'r') as file:
                total_count = sum(1 for line in file) - 1  # Subtract header
        except FileNotFoundError:
            return {"videos": [], "pagination": {"current_page": 1, "per_page": per_page, "total_count": 0, "total_pages": 0, "has_prev": False, "has_next": False}}
        
        # Calculate pagination
        total_pages = (total_count + per_page - 1) // per_page
        
        # For reverse pagination (newest first), we need to calculate the actual line numbers
        # Since we want newest first, we read from the end of the file
        start_line_from_end = (page - 1) * per_page
        end_line_from_end = start_line_from_end + per_page
        
        # Read only the specific lines we need (optimized approach)
        paginated_videos = []
        try:
            # For reverse pagination, we need to read from the end of the file
            # Calculate the actual line numbers we need (from the end)
            start_line = max(0, total_count - end_line_from_end)
            end_line = total_count - start_line_from_end
            
            with open(config['SCORES_CSV_FILE'], 'r') as file:
                reader = csv.DictReader(file, fieldnames=['url1', 'url2', 'result', 'score', 'feedback', 'elapsed_time', 'ground_truth', 'timestamp'])
                next(reader)  # Skip header
                
                # Skip to the start line
                for i, row in enumerate(reader):
                    if i >= start_line:
                        if i < end_line:
                            paginated_videos.append(row)
                        else:
                            break
                
                # Reverse to show newest first
                paginated_videos = list(reversed(paginated_videos))
                
        except FileNotFoundError:
            paginated_videos = []
        
        return {
            "videos": paginated_videos,
            "pagination": {
                "current_page": page,
                "per_page": per_page,
                "total_count": total_count,
                "total_pages": total_pages,
                "has_prev": page > 1,
                "has_next": page < total_pages
            }
        }
    except Exception as e:
        logger.error(f"Error in paginated endpoint: {e}")
        return {"error": str(e)}


@app.get("/api/compared-videos-all")
async def get_all_compared_videos():
    """Get all videos for metrics calculation - optimized with caching"""
    global _all_videos_cache, _cache_timestamp
    
    current_time = time.time()
    
    # Return cached data if still fresh
    if (_all_videos_cache is not None and 
        current_time - _cache_timestamp < CACHE_DURATION):
        return _all_videos_cache
    
    # Load fresh data (this still loads all data, but it's cached)
    _all_videos_cache = read_compared_videos()
    _cache_timestamp = current_time
    
    return _all_videos_cache

@app.get("/api/queue-status")
async def get_queue_status():
    """Get current queue status"""
    try:
        # Get counts with caching
        current_time = time.time()
        if not hasattr(get_queue_status, 'last_update') or \
           current_time - get_queue_status.last_update > 5:  # Cache for 5 seconds
            
            get_queue_status.pending_tasks = len([t for t in shared_active_tasks.values() 
                                                if t.get('status') != 'completed'])
            # Optimized: count lines instead of loading all data
            try:
                with open(config['SCORES_CSV_FILE'], 'r') as file:
                    get_queue_status.completed_comparisons = sum(1 for line in file) - 1  # Subtract header
            except FileNotFoundError:
                get_queue_status.completed_comparisons = 0
            get_queue_status.last_update = current_time
            
        return {
            "pending_tasks": get_queue_status.pending_tasks,
            "completed_comparisons": get_queue_status.completed_comparisons
        }
    except Exception as e:
        logger.error(f"Error getting queue status: {str(e)}")
        return {"error": str(e)}

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
    
    # Start periodic cleanup task
    asyncio.create_task(periodic_cleanup())

async def periodic_cleanup():
    """Periodic cleanup task that runs every 15 minutes"""
    while True:
        try:
            await asyncio.sleep(900)  # Wait 15 minutes
            
            # Clean up completed tasks
            cleaned_tasks = cleanup_completed_tasks()
            if cleaned_tasks > 0:
                logger.info(f"Periodic cleanup: removed {cleaned_tasks} completed tasks")
            
            # Clean up old WAV files (older than 2 hours)
            cleanup_old_wav_files(max_age_hours=2)
            
            # Clean up Chrome user data directory
            cleanup_chrome_user_data()
            
        except Exception as e:
            logger.error(f"Error in periodic cleanup: {e}")
            await asyncio.sleep(60)  # Wait 1 minute before retrying


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
        "http://46.225.92.232:8080",  # Hetzner/Coolify (direct)
        "http://46.225.92.232",  # Hetzner/Coolify (via Traefik)
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


@app.post("/api/cleanup-completed")
async def cleanup_completed_tasks_endpoint():
    """Manually clean up all completed tasks"""
    try:
        cleaned_count = cleanup_completed_tasks()
        return {
            "status": "success",
            "cleaned_tasks": cleaned_count,
            "message": f"Cleaned up {cleaned_count} completed tasks"
        }
    except Exception as e:
        logger.error(f"Error cleaning up tasks: {str(e)}")
        return {"error": str(e)}

@app.post("/api/cleanup-wav-files")
async def cleanup_wav_files_endpoint(max_age_hours: int = 12):
    """Manually clean up old WAV files"""
    try:
        cleanup_old_wav_files(max_age_hours)
        return {
            "status": "success",
            "message": f"WAV cleanup completed for files older than {max_age_hours} hours"
        }
    except Exception as e:
        logger.error(f"Error cleaning up WAV files: {str(e)}")
        return {"error": str(e)}

@app.post("/api/cleanup-chrome-data")
async def cleanup_chrome_data_endpoint():
    """Manually clean up Chrome user data directory"""
    try:
        cleanup_chrome_user_data()
        return {
            "status": "success",
            "message": "Chrome user data cleanup completed"
        }
    except Exception as e:
        logger.error(f"Error cleaning up Chrome user data: {str(e)}")
        return {"error": str(e)}

@app.get("/health")
async def health_check():
    """Health check endpoint with WAV file information"""
    try:
        # Get disk usage for WAV folder
        wav_folder = config['WAV_FOLDER']
        disk_usage = psutil.disk_usage(wav_folder)
        
        # Count WAV files
        wav_count = 0
        wav_total_size = 0
        if os.path.exists(wav_folder):
            for filename in os.listdir(wav_folder):
                if filename.endswith('.wav'):
                    wav_count += 1
                    file_path = os.path.join(wav_folder, filename)
                    try:
                        wav_total_size += os.path.getsize(file_path)
                    except:
                        pass
        
        return {
            "status": "healthy",
            "api_version": "1.0",
            "wav_files": {
                "count": wav_count,
                "total_size": f"{wav_total_size / 1024 / 1024:.1f}MB",
                "folder": wav_folder
            },
            "disk": {
                "total": f"{disk_usage.total / 1024 / 1024 / 1024:.1f}GB",
                "free": f"{disk_usage.free / 1024 / 1024 / 1024:.1f}GB",
                "used_percent": f"{(disk_usage.used / disk_usage.total) * 100:.1f}%"
            },
            "endpoints": [
                "/api/detect-cover",
                "/api/detection-status/{task_id}",
                "/api/get-thumbnails"
            ]
        }
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return {
            "status": "healthy",
            "api_version": "1.0",
            "error": str(e)
        }

@app.post("/api/check-if-cover")
async def check_if_cover(request: VideoRequest):
    """Start a new comparison task"""
    try:
        video_key = f"{request.video_url1}_{request.video_url2}"
        
        # Calculate median time once at start
        avg_time = get_average_processing_time()
        
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
        
        if total_pending > 4:  # Block if anything is being processed or queued
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
            'start_time': time.time(),
            'estimated_time': avg_time,  # Add estimated time here
            'progress': 0
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

@app.get("/api/comparison-status/{request_id}")
async def get_comparison_status(request_id: str):
    """Get the status of a comparison request"""
    task = shared_active_tasks.get(request_id)
    if task:
        if task['status'] == 'completed':
            # Set completed_time if not already set
            if 'completed_time' not in task:
                task['completed_time'] = time.time()
            # Remove completed task from active tasks
            del shared_active_tasks[request_id]
        elif task['status'] == 'failed':
            # Task already failed - return as-is without overwriting status
            pass
        else:
            start_time = task.get('start_time', time.time())
            elapsed = time.time() - start_time
            avg_time = task.get('estimated_time')  # Use stored estimate

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
            writer = csv.DictWriter(file, fieldnames=['url1', 'url2', 'result', 'score', 'feedback', 'elapsed_time', 'ground_truth', 'timestamp'])
            writer.writeheader()
            writer.writerows(videos)
            logger.info(f"Successfully wrote {len(videos)} entries back to CSV")
        
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error in submit_feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))



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

# Add this right before your CoverDetector initialization
# Where you currently log "Memory Usage Before CoverDetector init"
log_detailed_memory()
