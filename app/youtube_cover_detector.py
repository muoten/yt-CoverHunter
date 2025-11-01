# export an api to detect if 2 youtube videos are covers of each other

#import youtube_dl
import os
from app.parse_config import config
import logging
import sys
import numpy as np
# first method is GET /cover-detection?youtube_url1=...&youtube_url2=...
# it will return a json with the result
import json
import pickle
import time
import asyncio
import yt_dlp
import soundfile as sf
import librosa
import torch
import torchaudio
from pathlib import Path
import csv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import io
from typing import Dict, Any
import shutil
import glob
import psutil
from datetime import datetime
from app.utils.memory_logger import log_detailed_memory
import random

# Initialize queue and tasks at module level
comparison_queue = asyncio.Queue()
processing_tasks = {}
active_tasks = {}

# Anti-bot evasion: track recent downloads to implement cooldowns
_last_download_time = 0
_MIN_DOWNLOAD_DELAY = 30  # Minimum 30 seconds between downloads to avoid rate limits

THRESHOLD = config['THRESHOLD']


WAV_FOLDER = config['WAV_FOLDER']

CQT_FEAT_DIR = Path(WAV_FOLDER) / "cqt_feat"
CQT_FEAT_DIR.mkdir(exist_ok=True, parents=True)

#CSV_FILE = '/tmp/compared_videos.csv'
CSV_FILE = config['SCORES_CSV_FILE']

async def process_queue():
    while True:
        try:
            logger.info(f"Queue processor waiting for tasks. Current queue size: {comparison_queue.qsize()}")
            request = await comparison_queue.get()
            logger.info(f"Processing request {request['id']}")
            
            try:
                detector = CoverDetector()
                # Update status to downloading
                request['status'] = 'downloading'
                request['progress'] = 20
                active_tasks[request['id']] = request
                logger.info(f"Starting video comparison for request {request['id']}")
                
                result = await detector.compare_videos(request['url1'], request['url2'], request)
                
                # Update status to completed
                request['status'] = 'completed'
                request['result'] = result
                request['progress'] = 100
                active_tasks[request['id']] = request
                logger.info(f"Request {request['id']} completed")
            except Exception as e:
                request['status'] = 'failed'
                request['error'] = str(e)
                request['progress'] = 0
                logger.error(f"Error processing request {request['id']}: {e}")
                active_tasks[request['id']] = request
            finally:
                comparison_queue.task_done()
        except Exception as e:
            logger.error(f"Error in queue processor: {e}")
            await asyncio.sleep(1)  # Prevent tight loop on errors

def setup_logger():
    logger = logging.getLogger('api')
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger

logger = setup_logger()

def _generate_audio_from_youtube_id(youtube_id, request=None):
    """Generate audio from YouTube ID with user agent rotation and anti-bot evasion"""
    
    global _last_download_time
    
    # Anti-bot evasion: Add delay between downloads
    current_time = time.time()
    time_since_last = current_time - _last_download_time
    if time_since_last < _MIN_DOWNLOAD_DELAY:
        delay = _MIN_DOWNLOAD_DELAY - time_since_last
        logger.info(f"Anti-bot cooldown: waiting {delay:.1f}s before next download...")
        time.sleep(delay)
    _last_download_time = time.time()
    
    # List of realistic user agents to rotate through
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:120.0) Gecko/20100101 Firefox/120.0',
        'Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/120.0.0.0 Safari/537.36'
    ]
    
    # Randomly select a user agent
    selected_user_agent = random.choice(user_agents)
    logger.info(f"Using user agent: {selected_user_agent[:50]}...")
    
    try:
        logger.debug("Starting audio generation process")
        wav_folder = Path(WAV_FOLDER)
        wav_folder.mkdir(exist_ok=True, parents=True)
        os.chmod(str(wav_folder), 0o777)  # Give full permissions
        
        logger.info(f"Downloading video {youtube_id}...")
        
        def progress_hook(d):
            if request and d['status'] == 'downloading':
                try:
                    # Calculate download progress
                    total_bytes = d.get('total_bytes') or d.get('total_bytes_estimate', 0)
                    if total_bytes > 0:
                        downloaded = d.get('downloaded_bytes', 0)
                        progress = (downloaded / total_bytes) * 100
                        speed = d.get('speed', 0)  # bytes per second
                        eta = (total_bytes - downloaded) / speed if speed > 0 else 0
                        
                        request['download_progress'] = {
                            'progress': progress,
                            'speed': speed,
                            'eta': eta,
                            'total_bytes': total_bytes,
                            'downloaded_bytes': downloaded
                        }
                        active_tasks[request['id']] = request
                        logger.debug(f"Download progress: {progress:.1f}% at {speed/1024:.1f} KB/s, ETA: {eta:.1f}s")
                except Exception as e:
                    logger.error(f"Error in progress hook: {e}")

        # Check for cookie file - try multiple possible locations
        cookie_file = None
        possible_cookie_paths = [
            '/tmp/youtube_cookies.txt',
            '/data/youtube_cookies.txt',  # Persistent volume mount
            os.path.expanduser('~/.config/youtube_cookies.txt'),
            os.getenv('YOUTUBE_COOKIES_FILE', ''),
        ]
        
        for cookie_path in possible_cookie_paths:
            if cookie_path and os.path.exists(cookie_path):
                file_size = os.path.getsize(cookie_path)
                cookie_file = cookie_path
                logger.info(f"✓ Found cookie file: {cookie_file} ({file_size} bytes)")
                # Verify it's a valid cookie file (should have some content)
                if file_size < 100:
                    logger.warning(f"Cookie file seems too small ({file_size} bytes), might be invalid")
                break
        
        if not cookie_file:
            # Check if YOUTUBE_COOKIES env var exists - create cookie file on-demand if needed
            youtube_cookies_env = os.getenv('YOUTUBE_COOKIES')
            if youtube_cookies_env:
                logger.warning("YOUTUBE_COOKIES env var exists but cookie file not found at /tmp/youtube_cookies.txt")
                logger.info("Creating cookie file from YOUTUBE_COOKIES environment variable...")
                try:
                    with open('/tmp/youtube_cookies.txt', 'w') as f:
                        f.write(youtube_cookies_env)
                    # Verify it was created
                    if os.path.exists('/tmp/youtube_cookies.txt'):
                        file_size = os.path.getsize('/tmp/youtube_cookies.txt')
                        cookie_file = '/tmp/youtube_cookies.txt'
                        logger.info(f"✓ Created cookie file from env var: {cookie_file} ({file_size} bytes)")
                    else:
                        logger.error("Failed to create cookie file even though write succeeded")
                except Exception as e:
                    logger.error(f"Failed to create cookie file from YOUTUBE_COOKIES: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
            else:
                logger.debug("No YOUTUBE_COOKIES env var and no cookie file found")
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'external_downloader': 'aria2c',
            'http_headers': {
                'User-Agent': selected_user_agent,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            },
            'external_downloader_args': {
                'aria2c': [
                    # Increase concurrent connections
                    '-x', '4',              
                    '-s', '4',              
                    '--max-connection-per-server=4',
                    
                    # Fixed min-split-size to valid range
                    '--min-split-size=1M',  # Minimum allowed value
                    '--timeout=10',
                    '--connect-timeout=5',
                    '--retry-wait=1',
                    '--lowest-speed-limit=100K',
                    
                    # Keep the good settings
                    '--enable-http-keep-alive=true',
                    '--enable-http-pipelining=true',
                    '--http-accept-gzip=true',
                    
                    # Add these for better performance
                    '--optimize-concurrent-downloads=true',
                    '--conditional-get=true',
                    '--auto-file-renaming=false',
                    '--allow-overwrite=true'
                ]
            },
            'socket_timeout': 10,
            'retries': 5,
            'quiet': True,
            'no_warnings': True,
            'progress_hooks': [progress_hook],  # Add progress hook
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
            }],
            'outtmpl': f'{WAV_FOLDER}/{youtube_id}.%(ext)s',
        }
        
        # Add cookies if available
        if cookie_file:
            ydl_opts['cookiefile'] = cookie_file
            logger.info(f"Using cookies file: {cookie_file}")
        else:
            # Try cookies-from-browser as fallback - always try this if no cookie file
            # This can work even without a cookie file if browser cookies are accessible
            # Since we have Chrome in Docker, try using it
            logger.info("No cookie file found, attempting cookies-from-browser as fallback...")
            try:
                # Try Chrome - we have it installed in Docker
                ydl_opts['cookiesfrombrowser'] = ('chrome',)
                logger.info("Attempting to use cookies-from-browser (Chrome)")
            except Exception as e:
                logger.debug(f"Chrome cookies-from-browser failed: {e}")
                try:
                    # Fall back to Firefox if available
                    ydl_opts['cookiesfrombrowser'] = ('firefox',)
                    logger.info("Attempting to use cookies-from-browser (Firefox)")
                except Exception as e2:
                    logger.warning(f"Browser cookies not available: {e2}")
                    logger.warning("No cookies available - age-gated videos will likely fail")
                    logger.warning("Set YOUTUBE_COOKIES env var or USE_BROWSER_COOKIES=true")
        
        # Dynamic client rotation for anti-bot evasion
        # If cookies are available, prioritize iOS/TV clients which work better with cookies
        if cookie_file:
            client_options = [
                ['ios'],  # Best for age-gate with cookies
                ['tv_embedded'],  # Second best
                ['ios', 'tv_embedded'],  # iOS with TV fallback
                ['ios', 'android'],  # iOS with Android fallback
                ['android'],  # Mobile client
                ['web'],  # Web client
            ]
            logger.info("Using iOS/TV clients (cookies available)")
        else:
            client_options = [
                ['android'],  # Mobile client
                ['web'],  # Web client
                ['web_embedded'],  # Embedded player
                ['android', 'web_embedded'],  # Fallback chain
                ['web', 'android'],  # Alternative chain
            ]
        
        geo_countries = ['US', 'UK', 'CA', 'AU', 'DE']  # Rotate geo bypass
        
        last_error = None
        max_retries = 5  # Increased retries when cookies are available
        
        # Make sure cookie_file persists through retries
        for attempt in range(max_retries):
            # Re-check cookie file at start of each attempt (in case it was created)
            if not cookie_file:
                for cookie_path in possible_cookie_paths:
                    if cookie_path and os.path.exists(cookie_path):
                        cookie_file = cookie_path
                        ydl_opts['cookiefile'] = cookie_file
                        logger.info(f"Found cookie file on retry: {cookie_file}")
                        break
            
            # Select random client and geo for this attempt
            selected_client = random.choice(client_options)
            selected_geo = random.choice(geo_countries)
            
            ydl_opts['extractor_args'] = {
                'youtube': {
                    'player_client': selected_client
                }
            }
            ydl_opts['geo_bypass'] = True
            ydl_opts['geo_bypass_country'] = selected_geo
            
            # Ensure cookies are still set
            if cookie_file and 'cookiefile' not in ydl_opts:
                ydl_opts['cookiefile'] = cookie_file
            
            logger.info(f"Attempt {attempt + 1}/{max_retries}: Using client {selected_client}, geo {selected_geo}")
            
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    try:
                        logger.info(f"Attempting download with best quality...")
                        ydl.download([f'https://www.youtube.com/watch?v={youtube_id}'])
                        # Success! Break out of retry loop
                        break
                    except Exception as e:
                        error_str = str(e)
                        last_error = e
                        logger.warning(f"Download failed (attempt {attempt + 1}): {error_str[:200]}")
                        
                        # Detect age-gate errors
                        is_age_gate = ("sign in to confirm your age" in error_str.lower() or
                                      "inappropriate for some users" in error_str.lower() or
                                      "use --cookies" in error_str.lower())
                        
                        if is_age_gate:
                            logger.warning("Age-gate error detected!")
                            # If we have cookies but still got age-gate, try different client
                            if cookie_file:
                                logger.info("Cookies are available but age-gate still failed. Retrying with different client...")
                                if attempt < max_retries - 1:
                                    # Switch to iOS client which works better with cookies
                                    client_options = [['ios'], ['tv_embedded'], ['ios', 'tv_embedded']]
                                    continue
                            else:
                                # No cookies - check if cookie file exists but wasn't loaded
                                for cookie_path in possible_cookie_paths:
                                    if cookie_path and os.path.exists(cookie_path):
                                        cookie_file = cookie_path
                                        ydl_opts['cookiefile'] = cookie_file
                                        logger.info(f"Age-gate detected: Found cookie file, will use on retry: {cookie_file}")
                                        break
                        
                        # If this looks like an anti-bot block, try next client immediately
                        if "unavailable" in error_str.lower() or "private" in error_str.lower():
                            if attempt < max_retries - 1:
                                logger.info(f"Anti-bot block detected. Retrying immediately with different client...")
                                continue
                            else:
                                # Last attempt failed, try worst quality as final fallback
                                logger.info(f"Final attempt: trying worst quality...")
                                ydl_opts['format'] = 'worstaudio/worst'
                                try:
                                    ydl.download([f'https://www.youtube.com/watch?v={youtube_id}'])
                                    break  # Success with worst quality
                                except Exception as e2:
                                    raise e  # Re-raise original error
                        else:
                            # Non-anti-bot error, try worst quality
                            logger.info(f"Trying worst quality format...")
                            ydl_opts['format'] = 'worstaudio/worst'
                            try:
                                ydl.download([f'https://www.youtube.com/watch?v={youtube_id}'])
                                break  # Success
                            except Exception as e2:
                                raise e  # Re-raise original error
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    logger.info(f"Retrying immediately with different client...")
                    continue
                else:
                    # All retries exhausted
                    raise
        
        # If we get here without breaking, all retries failed
        if last_error:
            raise last_error
        
        # Convert to WAV
        mp3_path = f'{WAV_FOLDER}/{youtube_id}.mp3'
        wav_path = f'{WAV_FOLDER}/{youtube_id}.wav'
        
        logger.info(f"Converting {mp3_path} to {wav_path}...")
        
        # Check if mp3 exists
        if not os.path.exists(mp3_path):
            logger.error(f"Error: MP3 file not found at {mp3_path}")
            logger.error(f"Directory contents: {os.listdir(WAV_FOLDER)}")
            raise ValueError("MP3 file not found")
        
        # Use ffmpeg with more detailed output
        # extract first 10 seconds
        if config['PROCESS_ONLY_FIRST_N_SECONDS'] > 0:
            ffmpeg_cmd = f'ffmpeg -y -i "{mp3_path}" -ac 1 -ar 16000 -t {config["EXTRACT_ONLY_FIRST_N_SECONDS"]} "{wav_path}" 2>&1'
        else:
            ffmpeg_cmd = f'ffmpeg -y -i "{mp3_path}" -ac 1 -ar 16000 "{wav_path}" 2>&1'
        logger.info(f"Running ffmpeg command: {ffmpeg_cmd}")
        conversion_output = os.popen(ffmpeg_cmd).read()
        logger.info(f"FFmpeg output: {conversion_output}")
        
        if not os.path.exists(wav_path):
            logger.error(f"Error: WAV file not created at {wav_path}")
            logger.error(f"Directory contents after conversion: {os.listdir(WAV_FOLDER)}")
            raise ValueError("Audio conversion failed")
        
        logger.info(f"Successfully created WAV file: {wav_path}")
        logger.info(f"Removing mp3 file: {mp3_path}")
        os.remove(mp3_path)
        return f"{youtube_id}.wav"
        
    except Exception as e:
        logger.error(f"Error in _generate_audio_from_youtube_id: {str(e)}")
        logger.error(f"Full error details: {type(e).__name__}: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def _generate_dataset_txt_from_files(filename1, filename2):
	           
    dataset = []
    for i,filename in enumerate([filename1, filename2]):
        entry = {
            "perf": filename,
            "wav": filename,  # Path can be MP3 or WAV
            "dur_s": 0,
            "work": filename,
            "version": filename,
        	"work_id": i,
    	}
        dataset.append(entry)

    with open(f'{WAV_FOLDER}/dataset.txt', 'w') as f:
        for entry in dataset:
            f.write(json.dumps(entry) + '\n')

    print("dataset.txt has been successfully created!")


def _generate_embeddings_from_filepaths(audio_path1, audio_path2, embeddings1=None, embeddings2=None):
    COVERHUNTER_FOLDER = config['COVERHUNTER_FOLDER']
    MODEL_FOLDER = config['MODEL_FOLDER']
    os.system(f"mkdir -p {WAV_FOLDER}")
    os.system(f"rm -f {WAV_FOLDER}/*.txt")

    if embeddings1 is not None:
        audio_path1 = audio_path2
        os.system(f"rm {audio_path1}")
    if embeddings2 is not None:
        audio_path2 = audio_path1
        os.system(f"rm {audio_path2}")

    if embeddings1 is not None and embeddings2 is not None:
        return embeddings1, embeddings2

    _generate_dataset_txt_from_files(audio_path1, audio_path2)
    os.system(f"cp {WAV_FOLDER}/dataset.txt {WAV_FOLDER}/sp_aug.txt")
    os.system(f"rm -rf {WAV_FOLDER}/cqt_feat/")
    os.system(f"rm -rf {WAV_FOLDER}/sp_wav/")
    os.system(f"cp {COVERHUNTER_FOLDER}data/covers80_testset/hparams.yaml {WAV_FOLDER}/hparams.yaml")

    # first we get features with coverhunter
    command = f"PYTHONPATH={COVERHUNTER_FOLDER} python {COVERHUNTER_FOLDER}/tools/extract_csi_features.py {WAV_FOLDER}"
    if os.system(command) != 0:
        raise ValueError("Feature extraction failed") 

    # then we get the embeddings with coverhunter
    command = f"PYTHONPATH={COVERHUNTER_FOLDER} python {COVERHUNTER_FOLDER}/tools/make_embeds.py {WAV_FOLDER} {MODEL_FOLDER}"
    if os.system(command) != 0:
        raise ValueError("Embedding extraction failed")

    # if embeddings were generated, we can remove the wav and cqt.npy files
    os.system(f"rm {audio_path1}")
    os.system(f"rm {audio_path2}")
    os.system(f"rm {audio_path1}.cqt.npy")
    os.system(f"rm {audio_path2}.cqt.npy")

    # Path to the pickle file
    pickle_file_path = os.path.join(WAV_FOLDER, 'reference_embeddings.pkl')
    # get the embeddings from the file
    embeddings = pickle.load(open(pickle_file_path, 'rb'))
    return embeddings

def _cosine_distance(vec1, vec2):
    logger.debug(f"Calculating cosine distance between vectors: {vec1} and {vec2}")
    from numpy import dot
    from numpy.linalg import norm
    return 1 - dot(vec1, vec2) / (norm(vec1) * norm(vec2))

def prepare_cover_detection(youtube_url1, youtube_url2):
    # get the youtube id from the url
    youtube_id1 = youtube_url1.split('=')[1]
    youtube_id2 = youtube_url2.split('=')[1]

    # get the audio from the youtube video
    audio1 = _generate_audio_from_youtube_id(youtube_id1)
    audio2 = _generate_audio_from_youtube_id(youtube_id2)

    return audio1, audio2
    

async def download_audio(youtube_id):
    # Dynamic client rotation for anti-bot evasion (same as sync version)
    client_options = [
        ['android'],
        ['web'],
        ['web_embedded'],
        ['android', 'web_embedded'],
        ['web', 'android'],
    ]
    geo_countries = ['US', 'UK', 'CA', 'AU', 'DE']
    
    last_error = None
    max_retries = 3
    
    for attempt in range(max_retries):
        selected_client = random.choice(client_options)
        selected_geo = random.choice(geo_countries)
        
        # Check for cookie file (same as sync version)
        cookie_file = None
        possible_cookie_paths = [
            '/tmp/youtube_cookies.txt',
            os.path.expanduser('~/.config/youtube_cookies.txt'),
            os.getenv('YOUTUBE_COOKIES_FILE', ''),
        ]
        
        for cookie_path in possible_cookie_paths:
            if cookie_path and os.path.exists(cookie_path):
                cookie_file = cookie_path
                logger.info(f"Using cookies from: {cookie_file}")
                break
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'quiet': True,
            'external_downloader': 'aria2c',
            'external_downloader_args': ['-x', '16', '-k', '1M'],
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
            }],
            'outtmpl': f'{WAV_FOLDER}/{youtube_id}.%(ext)s',
            'extractor_args': {
                'youtube': {
                    'player_client': selected_client
                }
            },
            'geo_bypass': True,
            'geo_bypass_country': selected_geo,
        }
        
        # Add cookies if available
        if cookie_file:
            ydl_opts['cookiefile'] = cookie_file
            logger.info(f"Using cookies file: {cookie_file}")
        else:
            # Try cookies-from-browser if enabled
            use_browser_cookies = os.getenv('USE_BROWSER_COOKIES', 'false').lower() == 'true'
            if use_browser_cookies:
                try:
                    ydl_opts['cookiesfrombrowser'] = ('chrome',)
                    logger.info("Attempting to use cookies-from-browser (Chrome)")
                except Exception:
                    try:
                        ydl_opts['cookiesfrombrowser'] = ('firefox',)
                        logger.info("Attempting to use cookies-from-browser (Firefox)")
                    except Exception:
                        logger.debug("Browser cookies not available")
        
        try:
            logger.info(f"Async download attempt {attempt + 1}/{max_retries}: client {selected_client}, geo {selected_geo}")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                await asyncio.to_thread(ydl.download, [f'https://www.youtube.com/watch?v={youtube_id}'])
            break  # Success
        except Exception as e:
            last_error = e
            error_str = str(e)
            logger.warning(f"Async download failed (attempt {attempt + 1}): {error_str[:200]}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying immediately with different client...")
                continue
    
    if last_error and not os.path.exists(f'{WAV_FOLDER}/{youtube_id}.mp3'):
        raise last_error

async def process_videos(video_ids):
    tasks = [download_audio(video_id) for video_id in video_ids]
    await asyncio.gather(*tasks)

# Usage
# asyncio.run(process_videos(['video_id1', 'video_id2']))

# apart from the current csv file, we will also save another csv file named "vectors.csv" with key the youtube id and value the embeddings

def get_vectors_csv():
    VECTORS_CSV_FILE = config['VECTORS_CSV_FILE']
    if not os.path.exists(VECTORS_CSV_FILE):
        return {}
    
    vectors_csv = {}
    with open(VECTORS_CSV_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert string back to numpy array
            try:
                import numpy as np
                embeddings_str = row['embeddings']
                
                # Parse the format: "[ 0.05285105 -0.00633944  0.17383038 ... ]"
                # Remove brackets and normalize whitespace
                clean_str = embeddings_str.strip('[]')
                # Replace any newlines with spaces and normalize whitespace
                clean_str = ' '.join(clean_str.split())
                
                # Split by spaces and convert to float
                values = []
                for x in clean_str.split():
                    if x.strip():
                        try:
                            values.append(float(x))
                        except ValueError:
                            continue
                
                embeddings_array = np.array(values, dtype=np.float32)
                vectors_csv[row['youtube_id']] = embeddings_array
                
            except Exception as e:
                logger.error(f"Error parsing embeddings for {row['youtube_id']}: {e}")
                logger.error(f"Problematic embeddings string: {embeddings_str[:100]}...")
                continue
    
    return vectors_csv

def update_vectors_csv(youtube_id, embeddings):
    # check if the youtube_id is already in the csv file
    # create csv file if not exists
    logger.info(f"=== UPDATE_VECTORS_CSV TRACE ===")
    # get filepath from youtube_id
    video_filepath = os.path.join(WAV_FOLDER, f"{youtube_id}.wav")
    
    logger.info(f"Input video_filepath: '{video_filepath}' (type: {type(video_filepath)})")
    #youtube_id = video_filepath.split('/')[-1].split('.')[0]
    logger.info(f"Extracted youtube_id: '{youtube_id}'")
    
    logger.info(f"Input embeddings type: {type(embeddings)}")
    logger.info(f"Input embeddings length: {len(embeddings) if hasattr(embeddings, '__len__') else 'N/A'}")
    
    VECTORS_CSV_FILE = config['VECTORS_CSV_FILE']
    logger.info(f"VECTORS_CSV_FILE path: {VECTORS_CSV_FILE}")
    
    if not os.path.exists(VECTORS_CSV_FILE):
        logger.info("VECTORS_CSV_FILE does not exist, creating it...")
        with open(VECTORS_CSV_FILE, 'w') as f:
            f.write("youtube_id,embeddings\n")
    else:
        logger.info("VECTORS_CSV_FILE exists")
    
    # read the csv file
    vectors_csv = {}
    with open(VECTORS_CSV_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert string back to numpy array
            try:
                import numpy as np
                embeddings_str = row['embeddings']
                
                # Parse the format: "[ 0.05285105 -0.00633944  0.17383038 ... ]"
                clean_str = embeddings_str.strip('[]')
                # Replace any newlines with spaces and normalize whitespace
                clean_str = ' '.join(clean_str.split())
                
                # Split by spaces and convert to float
                values = []
                for x in clean_str.split():
                    if x.strip():
                        try:
                            values.append(float(x))
                        except ValueError:
                            continue
                
                embeddings_array = np.array(values, dtype=np.float32)
                vectors_csv[row['youtube_id']] = embeddings_array
                
            except Exception as e:
                logger.error(f"Error parsing embeddings for {row['youtube_id']}: {e}")
                continue
    
    logger.info(f"Current vectors_csv contents: {list(vectors_csv.keys())}")
    logger.info(f"Current vectors_csv keys: {list(vectors_csv.keys())}")
    
    if youtube_id in vectors_csv:
        logger.info(f"Youtube ID '{youtube_id}' already in vectors.csv, updating...")
        vectors_csv[youtube_id] = embeddings
    else:
        logger.info(f"Youtube ID '{youtube_id}' not in vectors.csv, adding...")
        vectors_csv[youtube_id] = embeddings

    logger.info(f"Updated vectors_csv: {list(vectors_csv.keys())}")
    logger.info(f"About to write {len(vectors_csv)} entries to CSV")

    # Write the updated data back to the file using the exact same format
    with open(VECTORS_CSV_FILE, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['youtube_id', 'embeddings'])
        writer.writeheader()
        for yt_id, emb in vectors_csv.items():
            logger.info(f"Writing youtube_id: '{yt_id}' to vectors.csv")
            # Convert numpy array to the exact same format with line breaks
            # Format: "[ 0.05285105 -0.00633944  0.17383038 ... ]"
            embeddings_list = [f'{x:.8f}' for x in emb]
            
            # Add line breaks every 6 values to match the original format
            formatted_values = []
            for i in range(0, len(embeddings_list), 6):
                chunk = embeddings_list[i:i+6]
                formatted_values.append('  '.join(chunk))
            
            embeddings_str = f'"[ {"  ".join(formatted_values)} ]"'
            writer.writerow({'youtube_id': yt_id, 'embeddings': embeddings_str})
    
    # Clean up video files after saving embeddings
    try:
        # Remove WAV file
        wav_path = f"{WAV_FOLDER}/{youtube_id}.wav"
        if os.path.exists(wav_path):
            os.remove(wav_path)
            logger.info(f"Cleaned up WAV file: {wav_path}")
        
        # Remove MP3 file if it exists
        mp3_path = f"{WAV_FOLDER}/{youtube_id}.mp3"
        if os.path.exists(mp3_path):
            os.remove(mp3_path)
            logger.info(f"Cleaned up MP3 file: {mp3_path}")
            
    except Exception as e:
        logger.error(f"Error cleaning up video files for {youtube_id}: {e}")
    
    logger.info(f"=== END UPDATE_VECTORS_CSV TRACE ===")

def read_compared_videos():
    compared_videos = []
    try:
        with open(CSV_FILE, mode='r', newline='') as file:
            reader = csv.DictReader(file, fieldnames=['url1', 'url2', 'result', 'score', 'feedback', 'elapsed_time', 'ground_truth', 'timestamp'])
            next(reader)  # Skip header row
            for row in reader:
                compared_videos.append(row)
    except FileNotFoundError:
        logger.debug("CSV file not found. Creating a new file with headers.")
        with open(CSV_FILE, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['url1', 'url2', 'result', 'score', 'feedback', 'elapsed_time', 'ground_truth', 'timestamp'])
            writer.writeheader()
    return compared_videos

def write_compared_video(url1: str, url2: str, result: str, score: float, elapsed_time: float = None) -> None:
    """Write the comparison result to a CSV file"""
    try:
        logger.debug(f"Starting write_compared_video with params: {url1}, {url2}, {result}, {score}, elapsed_time={elapsed_time}")
        
        # Create backup directories if they don't exist
        backup_dir = os.path.dirname(config['SCORES_CSV_FILE_BACKUP'])
        os.makedirs(backup_dir, exist_ok=True)
        
        # Always check and log last backup time
        # Find most recent backup by checking all backup files
        backup_pattern = f"{config['SCORES_CSV_FILE_BACKUP']}_*"
        backup_files = glob.glob(backup_pattern)
        
        if backup_files:
            # Get the most recent backup file's timestamp
            last_backup_time = max(os.path.getmtime(f) for f in backup_files)
        else:
            last_backup_time = 0
        
        logger.info(f"Last backup time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(last_backup_time))}")
        
        # Check if backup is needed (before writing new data)
        if os.path.exists(config['SCORES_CSV_FILE']):
            if time.time() - last_backup_time > config['BACKUP_INTERVAL']:  # Use config['BACKUP_INTERVAL']
                logger.info("Creating backup of CSV files...")
                backup_suffix = time.strftime('%Y%m%d_%H%M%S')
                shutil.copy(config['SCORES_CSV_FILE'], f"{config['SCORES_CSV_FILE_BACKUP']}_{backup_suffix}")
                shutil.copy(config['VECTORS_CSV_FILE'], f"{config['VECTORS_CSV_FILE_BACKUP']}_{backup_suffix}")
                logger.info("Backup completed successfully")
        
        # Write the new comparison result
        with open(config['SCORES_CSV_FILE'], 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['url1', 'url2', 'result', 'score', 'feedback', 'elapsed_time', 'ground_truth', 'timestamp'])
            writer.writerow({
                'url1': url1,
                'url2': url2, 
                'result': result,
                'score': str(score),
                'feedback': '',
                'elapsed_time': str(elapsed_time) if elapsed_time is not None else '',
                'ground_truth': '',  # Empty for new entries
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            })
    except Exception as e:
        logger.error(f"Error writing to CSV: {e}")
        raise
    
    # Log CSV contents after writing
    log_csv_contents()
    logger.debug("Finished write_compared_video function")

def log_csv_contents():
    try:
        compared_videos = read_compared_videos()
        if not compared_videos:
            logger.info("CSV file is empty")
    except Exception as e:
        logger.error(f"Error reading CSV contents: {e}")

class VideoURLs(BaseModel):
    youtube_url1: str
    youtube_url2: str

def get_result_from_csv(url1: str, url2: str):
    """Get the result from CSV if it exists"""
    compared_videos = read_compared_videos()
    for video in compared_videos:
        if ((video['url1'] == url1 and video['url2'] == url2) or 
            (video['url1'] == url2 and video['url2'] == url1)):
            return {
                "distance": float(video['score']),
                "is_cover": video['result'] == "Cover"
            }
    return None


def extract_video_id(url):
    """Extract video ID from YouTube URL"""
    try:
        if "youtu.be" in url:
            return url.split("/")[-1].split("?")[0]
        elif "youtube.com" in url:
            return url.split("v=")[1].split("&")[0]
        return url
    except Exception as e:
        logger.error(f"Error extracting video ID from {url}: {e}")
        return url

def cleanup_temp_files(url1: str, url2: str):
    """Remove temporary WAV files for the given video URLs."""
    wav_folder = config['WAV_FOLDER']
    for url in [url1, url2]:
        video_id = url.split("v=")[-1]
        wav_path = os.path.join(wav_folder, f"{video_id}.wav")
        if os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except Exception as e:
                logger.error(f"Error cleaning up {wav_path}: {e}")

def log_memory(tag: str = ""):
    """Log detailed memory usage with an optional tag"""
    process = psutil.Process(os.getpid())
    mem = process.memory_info()
    
    logger.info(f"""
=== Memory Usage {tag} ===
RSS (Actual memory used): {mem.rss / 1024 / 1024:.2f}MB
VMS (Virtual memory): {mem.vms / 1024 / 1024:.2f}MB
Shared: {mem.shared / 1024 / 1024:.2f}MB
Time: {datetime.now().strftime('%H:%M:%S')}
========================
""")
    log_detailed_memory()

class CoverDetector:
    def __init__(self):
        log_memory("Before CoverDetector init")
        
        # Model initialization
        self.model_path = os.path.join(config['MODEL_FOLDER'], 'checkpoints/g_00000043')
        log_memory("Before model load")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        logger.info(f"Using model at {self.model_path}")
        
        log_memory("After model load")
        
        self.threshold = config['THRESHOLD']
        self.wav_folder = config['WAV_FOLDER']
        self.model_folder = config['MODEL_FOLDER']
        self.coverhunter_folder = config['COVERHUNTER_FOLDER']
        self.scores_csv_file = config['SCORES_CSV_FILE']
        self.vectors_csv_file = config['VECTORS_CSV_FILE']
        self.process_only_first_n_seconds = config['PROCESS_ONLY_FIRST_N_SECONDS']
        
    async def compare_videos(self, url1: str, url2: str, request: Dict = None) -> Dict[str, Any]:
        logger.info("Start comparison")
        try:
            start_time = time.time()
            
            # Update progress for first video download
            if request:
                request['status'] = 'downloading_first'
                request['progress'] = 10
                active_tasks[request['id']] = request
            
            video_id1 = extract_video_id(url1)
            vectors_csv = get_vectors_csv()
            if video_id1 in vectors_csv:
                logger.info("Using existing embeddings for first video")
                embeddings1 = vectors_csv[video_id1]
            else:
                embeddings1 = None

                logger.info("Starting download of first video")
                wav1 = _generate_audio_from_youtube_id(video_id1, request=request)
                wav_path1 = os.path.join(self.wav_folder, wav1)
                if request:
                    request['progress'] = 25
                    active_tasks[request['id']] = request
            
                # Add random delay to avoid rate limiting
                delay = random.uniform(15, 60)
                time.sleep(delay)  # random delay between downloads
            
            # Update progress for second video download
            if request:
                request['status'] = 'downloading_second'
                request['progress'] = 30
                active_tasks[request['id']] = request
            

            video_id2 = extract_video_id(url2)
            if video_id2 in vectors_csv:
                logger.info("Using existing embeddings for second video")
                embeddings2 = vectors_csv[video_id2]
            else:
                embeddings2 = None

                logger.info("Starting download of second video")
                wav2 = _generate_audio_from_youtube_id(video_id2, request=request)
                wav_path2 = os.path.join(self.wav_folder, wav2)
                if request:
                    request['progress'] = 45
                    active_tasks[request['id']] = request
            
                # Update progress for processing
                if request:
                    request['status'] = 'processing'
                    request['progress'] = 50
                    active_tasks[request['id']] = request
            
            if request:
                request['status'] = 'generating_embeddings'
                request['progress'] = 60
                active_tasks[request['id']] = request
            
            # Generate embedding
            logger.info("Before embeddings")
            if embeddings1 is None or embeddings2 is None:
                if embeddings1 is not None:
                    wav_path1 = wav_path2
                if embeddings2 is not None:
                    wav_path2 = wav_path1
                embeddings = _generate_embeddings_from_filepaths(wav_path1, wav_path2, embeddings1=embeddings1, embeddings2=embeddings2)
                keys = list(embeddings.keys())
                
                # Ensure we have exactly 2 embeddings in the correct order
                if len(keys) == 1:
                    # We only have one embedding, need to add the existing one
                    if embeddings1 is not None:
                        # We have embeddings1, so the generated embedding is for video2
                        # Add embeddings1 at the beginning
                        new_embeddings = {}
                        new_embeddings[keys[0]] = embeddings1  # First position for video1
                        new_embeddings[keys[0] + "_2"] = embeddings[keys[0]]  # Second position for video2
                        embeddings = new_embeddings
                        keys = list(embeddings.keys())
                    elif embeddings2 is not None:
                        # We have embeddings2, so the generated embedding is for video1
                        # Add embeddings2 at the end
                        new_embeddings = {}
                        new_embeddings[keys[0]] = embeddings[keys[0]]  # First position for video1
                        new_embeddings[keys[0] + "_2"] = embeddings2  # Second position for video2
                        embeddings = new_embeddings
                        keys = list(embeddings.keys())
                
                # Now we should have exactly 2 embeddings
                if len(keys) >= 1:
                    embedding1 = embeddings[keys[0]]
                if len(keys) >= 2:
                    embedding2 = embeddings[keys[1]]
                
                # Save new embeddings to vectors.csv
                for youtube_id, embedding in embeddings.items():
                    update_vectors_csv(youtube_id, embedding)
                    
            else:
                embedding1 = embeddings1
                embedding2 = embeddings2

            logger.info("After embeddings")
            
            # Save embeddings to vectors.csv
       
            
            if request:
                request['status'] = 'comparing'
                request['progress'] = 80
                active_tasks[request['id']] = request
            
            # Calculate distance and determine if it's a cover
            logger.info("Before distance calculation")
            distance = _cosine_distance(embedding1, embedding2)
            logger.info("End comparison")
            
            is_cover = distance < self.threshold
            
            result = "Cover" if is_cover else "Not Cover"
            
            elapsed_time = time.time() - start_time
            # Write result to CSV
            write_compared_video(url1, url2, result, float(distance), elapsed_time=round(elapsed_time, 2))
            
            if request:
                request['status'] = 'saving_results'
                request['progress'] = 90
                active_tasks[request['id']] = request
            
            return {"result": result, "score": float(distance)}
            
        except Exception as e:
            logger.error(f"Error comparing videos: {e}")
            raise
        finally:
            # Cleanup
            log_memory("After cleanup")

    def cleanup_temp_files(self, url1: str, url2: str):
        for url in [url1, url2]:
            video_id = url.split("v=")[-1]
            wav_path = os.path.join(self.wav_folder, f"{video_id}.wav")
            if os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except Exception as e:
                    logger.error(f"Error cleaning up {wav_path}: {e}")

def get_average_processing_time() -> float:
    """Calculate median processing time from last 3 entries"""
    try:
        videos = read_compared_videos()
        logger.debug(f"Read {len(videos)} entries from CSV")
        if not videos:
            logger.info("No previous entries found, using default timeout of 50 seconds")
            return 50  # Default if no entries
        
        recent_times = []
        for v in videos[-3:]:
            try:
                elapsed = v.get('elapsed_time')
                if elapsed and elapsed != 'None':
                    # Try to convert to float, handling different formats
                    if isinstance(elapsed, (int, float)):
                        recent_times.append(float(elapsed))
                    elif isinstance(elapsed, str) and elapsed.strip():
                        recent_times.append(float(elapsed.split(',')[0]))
                    logger.debug(f"Successfully added time: {recent_times[-1]}")
            except (ValueError, TypeError, IndexError) as e:
                logger.debug(f"Skipping invalid elapsed_time value: {elapsed} - {str(e)}")
                continue
        
        if not recent_times:
            logger.info("No valid elapsed times found, using default timeout of 50 seconds")
            return 50  # Default if no valid times
        
        # Calculate median
        sorted_times = sorted(recent_times)
        if len(sorted_times) % 2 == 0:
            median_time = (sorted_times[len(sorted_times)//2 - 1] + sorted_times[len(sorted_times)//2]) / 2
        else:
            median_time = sorted_times[len(sorted_times)//2]
        
        logger.info(f"Calculated median processing time: {round(median_time)} seconds from {len(recent_times)} recent entries. Times used: {recent_times}")
        return round(median_time)
    except Exception as e:
        logger.error(f"Error calculating average time: {e}")
        logger.info("Using default timeout of 50 seconds due to error")
        return 50  # Default on error

# Make sure the function is exported
__all__ = [
    'extract_video_id',
    '_cosine_distance',
    '_generate_embeddings_from_filepaths',
    '_generate_dataset_txt_from_files',
    'get_average_processing_time',
    'comparison_queue',
    'get_result_from_csv',
    'CoverDetector',
    'cleanup_temp_files',
    'read_compared_videos',
    'process_queue',
    'logger'
]
