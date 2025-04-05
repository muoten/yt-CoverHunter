import os
import logging
import sys
from fastapi import FastAPI, HTTPException, BackgroundTasks, APIRouter, Request
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
import csv
from app.youtube_cover_detector import (
    extract_video_id, 
    _cosine_distance, 
    _generate_embeddings_from_filepaths,
    _generate_dataset_txt_from_files,
    cover_detection as youtube_cover_detection
)

#First version that works! Though it takes more than 3 minutes to run in fly.dev free tier
YT_DLP_USE_COOKIES = True

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
async def cover_detection_endpoint(request: Request):
    data = await request.json()
    youtube_url1 = data.get('video_url1')
    youtube_url2 = data.get('video_url2')
    
    logger.debug(f"Received request for URLs: {youtube_url1} and {youtube_url2}")
    
    if not youtube_url1 or not youtube_url2:
        raise HTTPException(status_code=400, detail="Both youtube_url1 and youtube_url2 parameters are required")

    try:
        # Get the WAV file paths
        video_id1 = extract_video_id(youtube_url1)
        video_id2 = extract_video_id(youtube_url2)
        
        wav_path1 = f"/tmp/youtube_cover_detector_api_wav/{video_id1}.wav"
        wav_path2 = f"/tmp/youtube_cover_detector_api_wav/{video_id2}.wav"
        print(f"WAV file paths: {wav_path1}, {wav_path2}")
        
        # Generate audio if needed
        if not os.path.exists(wav_path1):
            logger.debug(f"Generating audio for {video_id1}")
            from app.youtube_cover_detector import _generate_audio_from_youtube_id
            _generate_audio_from_youtube_id(video_id1)
        
        if not os.path.exists(wav_path2):
            logger.debug(f"Generating audio for {video_id2}")
            from app.youtube_cover_detector import _generate_audio_from_youtube_id
            _generate_audio_from_youtube_id(video_id2)
        
        # Call cover_detection and get the actual values
        result = cover_detection(youtube_url1, youtube_url2)
        distance = float(result["distance"])  # Convert to float
        is_cover = result["is_cover"]
        result_text = "Cover" if is_cover else "Not Cover"
        
        # Write to CSV with the actual values
        logger.debug(f"Writing result to CSV: Distance={distance}, Result={result_text}")
        write_compared_video(youtube_url1, youtube_url2, result_text, distance)
        
        return {
            "result": result_text,
            "score": distance
        }
    except Exception as e:
        logger.error(f"Error in cover detection: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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
    data = await request.json()
    url1 = data['url1']
    url2 = data['url2']
    feedback = data['feedback']
    
    try:
        # Read current CSV
        videos = read_compared_videos()
        
        # Update the feedback for matching entry
        for video in videos:
            if ((video['url1'] == url1 and video['url2'] == url2) or 
                (video['url1'] == url2 and video['url2'] == url1)):
                video['feedback'] = feedback
        
        # Write back to CSV
        with open(CSV_FILE, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['url1', 'url2', 'result', 'score', 'feedback'])
            writer.writeheader()
            writer.writerows(videos)
        
        return {"status": "success"}
    except Exception as e:
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

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=7860, log_level="debug") 
