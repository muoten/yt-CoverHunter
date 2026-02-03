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
import subprocess
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
import uuid
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
_MIN_DOWNLOAD_DELAY = 6  # Minimum 6 seconds between downloads to avoid rate limits
_MAX_DOWNLOAD_DELAY = 30

# Cache of video IDs that permanently failed (geo-restricted, private, age-restricted)
# Avoids re-attempting the same broken videos. Maps video_id -> error reason.
_failed_video_cache = {}

# Cookie file for YouTube authentication (bypasses datacenter IP blocks)
_COOKIE_FILE = os.getenv('YT_COOKIE_FILE', config.get('YT_COOKIE_FILE', '/data/cookies.txt'))
_USE_COOKIES = os.path.exists(_COOKIE_FILE) if _COOKIE_FILE else False

# Residential proxy for bypassing hard datacenter IP blocks
# Uses sticky sessions so the same IP is used for metadata + download
_PROXY_URL_BASE = os.getenv('YT_PROXY_URL', config.get('YT_PROXY_URL', ''))
_USE_PROXY = bool(_PROXY_URL_BASE)

def _make_sticky_proxy_url(duration_minutes=10):
    """Build a Decodo sticky session proxy URL from the base URL.

    Sticky sessions ensure the same residential IP is used for both
    metadata extraction and stream download, preventing YouTube 403 errors
    from IP-bound signed URLs.

    Base URL format: http://USERNAME:PASSWORD@gate.decodo.com:7000
    Sticky format:   http://user-USERNAME-session-SESSID-sessionduration-MIN:PASSWORD@gate.decodo.com:7000
    """
    if not _PROXY_URL_BASE:
        return ''
    from urllib.parse import urlparse, urlunparse
    parsed = urlparse(_PROXY_URL_BASE)
    username = parsed.username or ''
    password = parsed.password or ''
    session_id = uuid.uuid4().hex[:12]
    sticky_user = f"user-{username}-session-{session_id}-sessionduration-{duration_minutes}"
    netloc = f"{sticky_user}:{password}@{parsed.hostname}"
    if parsed.port:
        netloc += f":{parsed.port}"
    return urlunparse((parsed.scheme, netloc, parsed.path, '', '', ''))

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
                # Cooldown between comparisons to avoid YouTube IP blocking
                delay = random.uniform(_MIN_DOWNLOAD_DELAY, _MAX_DOWNLOAD_DELAY)
                logger.info(f"Cooldown between comparisons: waiting {delay:.1f}s...")
                await asyncio.sleep(delay)
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

if _USE_COOKIES:
    logger.info(f"YouTube cookies loaded from {_COOKIE_FILE}")
else:
    logger.warning(f"No YouTube cookie file found at {_COOKIE_FILE} - downloads may fail from datacenter IPs")

if _USE_PROXY:
    logger.info(f"Residential proxy configured (will be used as fallback for blocked videos)")
else:
    logger.info("No proxy configured - set YT_PROXY_URL env var to enable proxy fallback")

def _generate_audio_from_youtube_id(youtube_id, request=None):
    """Generate audio from YouTube ID with user agent rotation and anti-bot evasion

    Args:
        youtube_id: YouTube video ID
        request: Optional request dict for progress tracking
    """
    
    global _last_download_time

    # Check if this video previously failed with a permanent error
    if youtube_id in _failed_video_cache:
        reason = _failed_video_cache[youtube_id]
        logger.warning(f"Skipping {youtube_id}: previously failed ({reason})")
        raise Exception(f"{reason}: {youtube_id}")

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

        # Use cookies if available (bypasses datacenter IP blocks)
        if _USE_COOKIES:
            ydl_opts['cookiefile'] = _COOKIE_FILE
            logger.info(f"Using cookie file: {_COOKIE_FILE}")

        last_error = None
        max_retries = 3

        geo_restriction_detected = False
        geo_countries = ['US', 'UK', 'DE', 'FR', 'NL', 'SE']

        for attempt in range(max_retries):
            # Use default web client - android/ios require PO tokens and fail
            if 'extractor_args' in ydl_opts:
                del ydl_opts['extractor_args']

            if geo_restriction_detected:
                selected_geo = geo_countries[attempt % len(geo_countries)]
                ydl_opts['geo_bypass'] = True
                ydl_opts['geo_bypass_country'] = selected_geo
                logger.info(f"Attempt {attempt + 1}/{max_retries}: default web client, geo {selected_geo}")
            else:
                logger.info(f"Attempt {attempt + 1}/{max_retries}: default web client")
            
            # Log equivalent command line for debugging
            def ydl_opts_to_cmd_list(youtube_id, opts):
                """Convert yt-dlp options to command line list (for subprocess)."""
                cmd_parts = []
                
                # Format
                if 'format' in opts:
                    cmd_parts.extend(['-f', opts['format']])
                
                # External downloader - must specify downloader name in args
                if 'external_downloader' in opts:
                    downloader = opts['external_downloader']
                    cmd_parts.extend(['--external-downloader', downloader])
                    if 'external_downloader_args' in opts and downloader in opts['external_downloader_args']:
                        args = opts['external_downloader_args'][downloader]
                        # Format: --external-downloader-args "aria2c:arg1 arg2 arg3"
                        # All aria2c arguments must be in a single string
                        args_list = [str(a) for a in args]
                        args_str = ' '.join(args_list)
                        # Quote the entire argument to prevent shell splitting
                        # Format: aria2c:-x 4 -s 4 --max-connection-per-server=4 ...
                        full_args = f'{downloader}:{args_str}'
                        cmd_parts.extend(['--external-downloader-args', full_args])
                
                # Cookie file
                if 'cookiefile' in opts:
                    cmd_parts.extend(['--cookies', opts['cookiefile']])
                
                # Cookies from browser
                if 'cookiesfrombrowser' in opts:
                    browser = opts['cookiesfrombrowser'][0] if isinstance(opts['cookiesfrombrowser'], tuple) else opts['cookiesfrombrowser']
                    cmd_parts.extend(['--cookies-from-browser', browser])
                
                # Extractor args (player client)
                if 'extractor_args' in opts and 'youtube' in opts['extractor_args']:
                    player_client = opts['extractor_args']['youtube'].get('player_client', [])
                    if player_client:
                        client_str = ','.join(player_client) if isinstance(player_client, list) else str(player_client)
                        cmd_parts.extend(['--extractor-args', f'youtube:player_client={client_str}'])
                
                # Geo bypass
                if opts.get('geo_bypass'):
                    if 'geo_bypass_country' in opts:
                        cmd_parts.extend(['--geo-bypass-country', opts['geo_bypass_country']])
                    else:
                        cmd_parts.append('--geo-bypass')
                
                # Output template
                if 'outtmpl' in opts:
                    cmd_parts.extend(['-o', opts['outtmpl']])
                
                # Postprocessor (audio extraction)
                if 'postprocessors' in opts:
                    for pp in opts['postprocessors']:
                        if pp.get('key') == 'FFmpegExtractAudio':
                            cmd_parts.extend(['--extract-audio'])
                            if 'preferredcodec' in pp:
                                cmd_parts.extend(['--audio-format', pp['preferredcodec']])
                
                # JS runtime for YouTube signature solving
                cmd_parts.extend(['--js-runtimes', 'node'])
                cmd_parts.extend(['--remote-components', 'ejs:github'])

                # Proxy
                if 'proxy' in opts and opts['proxy']:
                    cmd_parts.extend(['--proxy', opts['proxy']])

                # Add the URL
                cmd_parts.append(f'https://www.youtube.com/watch?v={youtube_id}')

                return cmd_parts
            
            cmd_list = ydl_opts_to_cmd_list(youtube_id, ydl_opts)
            equivalent_cmd_str = 'python3 -m yt_dlp ' + ' '.join(f'"{arg}"' if ' ' in arg or '=' in arg else arg for arg in cmd_list)
            logger.debug(f"Equivalent yt-dlp command: {equivalent_cmd_str}")
            
            # Check if command-line mode is enabled
            use_cli = os.getenv('YT_DLP_USE_CLI', 'false').lower() in ('true', '1', 'yes')
            use_cli_config = config.get('YT_DLP_USE_CLI', False)
            use_command_line = use_cli or use_cli_config
            
            try:
                if use_command_line:
                    # Use command-line yt-dlp (more reliable for some videos)
                    logger.info(f"Using command-line yt-dlp (CLI mode enabled)")
                    # Build command: python3 -m yt_dlp [args] [url]
                    full_cmd = ['python3', '-m', 'yt_dlp'] + cmd_list
                    
                    logger.debug(f"Executing: {' '.join(full_cmd)}")
                    result = subprocess.run(
                        full_cmd,
                        capture_output=True,
                        text=True,
                        check=False  # Don't raise on non-zero exit
                    )
                    
                    if result.returncode == 0:
                        logger.info("Download successful via CLI")
                        # Check for warnings but don't fail on them
                        if result.stderr:
                            warnings = result.stderr
                            # Log warnings but don't fail - PO token warnings are common but downloads can still succeed
                            if "WARNING" in warnings or "GVS PO Token" in warnings:
                                logger.debug(f"CLI warnings (non-fatal): {warnings[:300]}")
                        
                        # Verify the mp3 file was created (yt-dlp with --extract-audio should create it)
                        mp3_path = f'{WAV_FOLDER}/{youtube_id}.mp3'
                        # Also check for other possible formats (mp4, m4a, etc.) that might need conversion
                        possible_formats = ['.mp3', '.m4a', '.mp4', '.webm', '.opus']
                        file_found = False
                        for ext in possible_formats:
                            test_path = f'{WAV_FOLDER}/{youtube_id}{ext}'
                            if os.path.exists(test_path):
                                file_found = True
                                logger.debug(f"Downloaded file found: {test_path}")
                                # If it's not mp3, rename it or note that conversion might be needed
                                if ext != '.mp3' and not os.path.exists(mp3_path):
                                    logger.warning(f"Downloaded {ext} file but expected mp3. File may need conversion.")
                                break
                        
                        if not file_found:
                            logger.warning(f"CLI returned success but no output file found in {WAV_FOLDER}/")
                            logger.warning(f"Checking stdout for file location: {result.stdout[:500] if result.stdout else 'No stdout'}")
                            # Don't fail here - let the mp3 check later catch it
                        
                        break  # Success! Break out of retry loop
                    else:
                        # Parse error from stderr or stdout
                        error_str = (result.stderr or result.stdout or "").strip()
                        # Filter out warnings - only treat actual errors as failures
                        error_lines = error_str.split('\n')
                        actual_errors = [line for line in error_lines if line.startswith('ERROR:')]
                        if actual_errors:
                            error_str = '\n'.join(actual_errors)
                        logger.warning(f"CLI download failed (attempt {attempt + 1}): {error_str[:200]}")
                        
                        # Apply same error handling as Python API
                        # Detect geo-restriction errors - permanent failure from datacenter IPs
                        is_geo_restricted = ("not made this video available in your country" in error_str.lower() or
                                            "not available in your country" in error_str.lower() or
                                            "geo restricted" in error_str.lower())

                        if is_geo_restricted:
                            logger.warning(f"Video is geo-restricted, not retrying: {youtube_id}")
                            _failed_video_cache[youtube_id] = "Geo-restricted"
                            raise Exception(f"Geo-restricted: {youtube_id}")
                        
                        # Detect format availability errors
                        is_format_error = ("requested format is not available" in error_str.lower() or
                                         "signature extraction failed" in error_str.lower() or
                                         "sabr streaming" in error_str.lower() or
                                         "only images are available" in error_str.lower())
                        
                        if is_format_error:
                            logger.warning("Format extraction error detected. Retrying...")
                            if attempt < max_retries - 1:
                                delay = random.uniform(0.5, 1.5)
                                logger.info(f"Waiting {delay:.1f}s before retry...")
                                time.sleep(delay)
                                continue

                        # Private videos - don't retry
                        if "private video" in error_str.lower():
                            logger.warning(f"Video is private, not retrying: {youtube_id}")
                            _failed_video_cache[youtube_id] = "Private video"
                            raise Exception(f"Private video: {youtube_id}")

                        # "Video unavailable" - may be datacenter IP blocking, retry with delay
                        if "video unavailable" in error_str.lower():
                            if attempt < max_retries - 1:
                                delay = random.uniform(3, 6)
                                logger.warning(f"Video unavailable (may be datacenter IP block). Waiting {delay:.1f}s before retry...")
                                time.sleep(delay)
                                continue
                            else:
                                _failed_video_cache[youtube_id] = "Video unavailable"
                                raise Exception(f"Video unavailable: {youtube_id}")

                        # Age-restricted videos - can't bypass without cookies, don't retry
                        if "sign in to confirm your age" in error_str.lower() or "age" in error_str.lower() and "sign in" in error_str.lower():
                            logger.warning(f"Video is age-restricted, not retrying: {youtube_id}")
                            _failed_video_cache[youtube_id] = "Age-restricted"
                            raise Exception(f"Video is age-restricted: {youtube_id}")

                        # Actual anti-bot detection (403, bot check, etc.)
                        if "sign in" in error_str.lower() or "403" in error_str.lower():
                            if attempt < max_retries - 1:
                                delay = random.uniform(2, 5)
                                logger.info(f"Anti-bot block detected. Waiting {delay:.1f}s before retry...")
                                time.sleep(delay)
                                continue
                            else:
                                logger.info(f"Final attempt: trying worst quality...")
                                ydl_opts['format'] = 'worstaudio/worst'
                                cmd_list = ydl_opts_to_cmd_list(youtube_id, ydl_opts)
                                full_cmd = ['python3', '-m', 'yt_dlp'] + cmd_list
                                result = subprocess.run(full_cmd, capture_output=True, text=True, check=False)
                                if result.returncode == 0:
                                    logger.info("Download successful via CLI with worst quality")
                                    break
                                else:
                                    last_error = Exception(f"yt-dlp CLI failed: {error_str[:200]}")
                            # Non-anti-bot error, try worst quality
                            logger.info(f"Trying worst quality format...")
                            ydl_opts['format'] = 'worstaudio/worst'
                            cmd_list = ydl_opts_to_cmd_list(youtube_id, ydl_opts)
                            full_cmd = ['python3', '-m', 'yt_dlp'] + cmd_list
                            result = subprocess.run(full_cmd, capture_output=True, text=True, check=False)
                            if result.returncode == 0:
                                logger.info("Download successful via CLI with worst quality")
                                break
                            else:
                                last_error = Exception(f"yt-dlp CLI failed: {error_str[:200]}")
                        
                        # Store error for outer exception handler
                        last_error = Exception(f"yt-dlp CLI failed: {error_str[:200]}")
                        # Continue to next retry attempt (outer loop will handle max retries)
                        if attempt < max_retries - 1:
                            continue
                        else:
                            # Final attempt failed
                            raise last_error
                else:
                    # Use Python API (original method)
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
                        
                        # Detect geo-restriction errors - permanent failure from datacenter IPs
                        is_geo_restricted = ("not made this video available in your country" in error_str.lower() or
                                            "not available in your country" in error_str.lower() or
                                            "geo restricted" in error_str.lower())

                        if is_geo_restricted:
                            logger.warning(f"Video is geo-restricted, not retrying: {youtube_id}")
                            _failed_video_cache[youtube_id] = "Geo-restricted"
                            raise Exception(f"Geo-restricted: {youtube_id}")
                        
                        # Detect format availability errors (signature extraction, SABR streaming, etc.)
                        is_format_error = ("requested format is not available" in error_str.lower() or
                                         "signature extraction failed" in error_str.lower() or
                                         "sabr streaming" in error_str.lower() or
                                         "only images are available" in error_str.lower())
                        
                        if is_format_error:
                            logger.warning("Format extraction error detected. Retrying...")
                            if attempt < max_retries - 1:
                                delay = random.uniform(0.5, 1.5)
                                logger.info(f"Waiting {delay:.1f}s before retry...")
                                time.sleep(delay)
                                continue
                        
                        # Private videos - don't retry
                        if "private video" in error_str.lower():
                            logger.warning(f"Video is private, not retrying: {youtube_id}")
                            _failed_video_cache[youtube_id] = "Private video"
                            raise Exception(f"Private video: {youtube_id}")

                        # "Video unavailable" - may be datacenter IP blocking, retry with delay
                        if "video unavailable" in error_str.lower():
                            if attempt < max_retries - 1:
                                delay = random.uniform(3, 6)
                                logger.warning(f"Video unavailable (may be datacenter IP block). Waiting {delay:.1f}s before retry...")
                                time.sleep(delay)
                                continue
                            else:
                                _failed_video_cache[youtube_id] = "Video unavailable"
                                raise Exception(f"Video unavailable: {youtube_id}")

                        # Age-restricted videos - can't bypass without cookies, don't retry
                        if "sign in to confirm your age" in error_str.lower() or ("age" in error_str.lower() and "sign in" in error_str.lower()):
                            logger.warning(f"Video is age-restricted, not retrying: {youtube_id}")
                            _failed_video_cache[youtube_id] = "Age-restricted"
                            raise Exception(f"Video is age-restricted: {youtube_id}")

                        # Actual anti-bot detection (sign-in required, 403, etc.)
                        if "sign in" in error_str.lower() or "403" in error_str.lower():
                            if attempt < max_retries - 1:
                                delay = random.uniform(2, 5)
                                logger.info(f"Anti-bot block detected. Waiting {delay:.1f}s before retry...")
                                time.sleep(delay)
                                continue
                            else:
                                logger.info(f"Final attempt: trying worst quality...")
                                ydl_opts['format'] = 'worstaudio/worst'
                                try:
                                    ydl.download([f'https://www.youtube.com/watch?v={youtube_id}'])
                                    break
                                except Exception as e2:
                                    raise e
                        else:
                            # Non-anti-bot error, try worst quality
                            logger.info(f"Trying worst quality format...")
                            ydl_opts['format'] = 'worstaudio/worst'
                            try:
                                # ydl is still in scope within the with block
                                ydl.download([f'https://www.youtube.com/watch?v={youtube_id}'])
                                break  # Success
                            except Exception as e2:
                                raise e  # Re-raise original error
            except Exception as e:
                last_error = e
                # Don't retry errors that are known to be permanent
                error_msg = str(e).lower()
                is_permanent = ("age-restricted" in error_msg or
                               "private video" in error_msg or
                               "geo-restricted" in error_msg)
                if is_permanent:
                    raise
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
        error_msg = str(e).lower()
        # Proxy fallback: if video unavailable and proxy is configured, retry once via proxy
        if _USE_PROXY and "video unavailable" in error_msg:
            logger.info(f"Attempting proxy fallback for {youtube_id} via residential proxy (sticky session)...")
            try:
                sticky_proxy = _make_sticky_proxy_url(duration_minutes=10)
                logger.info(f"Using sticky proxy session for {youtube_id}")
                proxy_ydl_opts = {
                    'format': 'bestaudio/best',
                    'socket_timeout': 15,
                    'retries': 3,
                    'quiet': True,
                    'no_warnings': True,
                    'postprocessors': [{
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': 'mp3',
                    }],
                    'outtmpl': f'{WAV_FOLDER}/{youtube_id}.%(ext)s',
                    'proxy': sticky_proxy,
                }
                # No cookies with proxy - cookie session is tied to original
                # user's geo/IP and conflicts with proxy's different IP

                proxy_cmd_list = ydl_opts_to_cmd_list(youtube_id, proxy_ydl_opts)
                full_cmd = ['python3', '-m', 'yt_dlp'] + proxy_cmd_list
                logger.info(f"Proxy bestaudio (sticky) for {youtube_id}")
                try:
                    result = subprocess.run(full_cmd, capture_output=True, text=True, check=False, timeout=60)
                except subprocess.TimeoutExpired:
                    logger.warning(f"Proxy sticky session timed out after 60s for {youtube_id}")
                    for ext in ['mp3', 'mp4', 'mp4.part', 'mp4.ytdl', 'webm', 'webm.part']:
                        partial = f'{WAV_FOLDER}/{youtube_id}.{ext}'
                        if os.path.exists(partial):
                            os.remove(partial)
                    _failed_video_cache[youtube_id] = "Video unavailable (proxy timeout)"
                    raise ValueError("Proxy download timed out")

                if result.returncode == 0:
                    logger.info(f"Proxy download successful (sticky session) for {youtube_id}")
                    _failed_video_cache.pop(youtube_id, None)
                    mp3_path = f'{WAV_FOLDER}/{youtube_id}.mp3'
                    wav_path = f'{WAV_FOLDER}/{youtube_id}.wav'
                    if not os.path.exists(mp3_path):
                        raise ValueError(f"Proxy download succeeded but MP3 not found at {mp3_path}")
                    if config['PROCESS_ONLY_FIRST_N_SECONDS'] > 0:
                        ffmpeg_cmd = f'ffmpeg -y -i "{mp3_path}" -ac 1 -ar 16000 -t {config["EXTRACT_ONLY_FIRST_N_SECONDS"]} "{wav_path}" 2>&1'
                    else:
                        ffmpeg_cmd = f'ffmpeg -y -i "{mp3_path}" -ac 1 -ar 16000 "{wav_path}" 2>&1'
                    os.popen(ffmpeg_cmd).read()
                    if not os.path.exists(wav_path):
                        raise ValueError("Proxy download: audio conversion failed")
                    logger.info(f"Proxy fallback: successfully created {wav_path}")
                    os.remove(mp3_path)
                    return f"{youtube_id}.wav"
                else:
                    error_out = (result.stderr or result.stdout or "").strip()
                    logger.warning(f"Proxy sticky session failed for {youtube_id}: {error_out[:200]}")
                    for ext in ['mp3', 'mp4', 'mp4.part', 'mp4.ytdl', 'webm', 'webm.part']:
                        partial = f'{WAV_FOLDER}/{youtube_id}.{ext}'
                        if os.path.exists(partial):
                            os.remove(partial)
                    _failed_video_cache[youtube_id] = "Video unavailable (even via proxy)"
            except Exception as proxy_err:
                logger.error(f"Proxy fallback failed for {youtube_id}: {proxy_err}")
                _failed_video_cache[youtube_id] = "Video unavailable (proxy error)"

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

    if embeddings1 is not None and embeddings2 is not None:
        os.system(f"rm {audio_path1}")
        os.system(f"rm {audio_path2}")
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
    last_error = None
    max_retries = 3

    for attempt in range(max_retries):
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
        }

        try:
            logger.info(f"Async download attempt {attempt + 1}/{max_retries}: default web client")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                await asyncio.to_thread(ydl.download, [f'https://www.youtube.com/watch?v={youtube_id}'])
            break
        except Exception as e:
            last_error = e
            error_str = str(e)
            logger.warning(f"Async download failed (attempt {attempt + 1}): {error_str[:200]}")
            if "video unavailable" in error_str.lower():
                raise  # Don't retry unavailable videos
            if attempt < max_retries - 1:
                await asyncio.sleep(random.uniform(2, 5))
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

def cleanup_chrome_user_data_dirs():
    """Remove all Chrome user data directories from /tmp to free up disk space."""
    total_freed = 0
    removed_count = 0
    
    # Find all Chrome user data directories
    chrome_patterns = [
        "/tmp/chrome_user_data",
        "/tmp/chrome_user_data_*",
        "/tmp/chrome-test-profile-*"
    ]
    
    for pattern in chrome_patterns:
        for chrome_data_path in glob.glob(pattern):
            if not os.path.isdir(chrome_data_path):
                continue
            
            try:
                # Get directory size before removal
                dir_size = 0
                for dirpath, dirnames, filenames in os.walk(chrome_data_path):
                    for filename in filenames:
                        file_path = os.path.join(dirpath, filename)
                        try:
                            dir_size += os.path.getsize(file_path)
                        except:
                            pass
                
                # Remove the entire directory
                shutil.rmtree(chrome_data_path, ignore_errors=True)
                total_freed += dir_size
                removed_count += 1
                logger.debug(f"Removed Chrome user data: {chrome_data_path} ({dir_size/1024/1024:.1f}MB)")
                
            except Exception as e:
                logger.warning(f"Error removing Chrome user data {chrome_data_path}: {e}")
    
    if removed_count > 0:
        size_mb = total_freed / 1024 / 1024
        logger.info(f"Cleaned up {removed_count} Chrome user data directories, freed {size_mb:.1f}MB")

def cleanup_old_tmp_files(max_age_minutes: int = 30):
    """Clean up old files in WAV_FOLDER to prevent disk full.
    
    Removes WAV files, CQT files, and other temp files older than max_age_minutes.
    Also cleans up Chrome user data directories.
    """
    try:
        # First, clean up Chrome user data directories (they can be very large)
        cleanup_chrome_user_data_dirs()
        
        if not os.path.exists(WAV_FOLDER):
            return
        
        current_time = time.time()
        max_age_seconds = max_age_minutes * 60
        cleaned_count = 0
        cleaned_size = 0
        
        # Clean up old files in WAV_FOLDER (WAV, MP3, CQT, TXT, PKL, etc.)
        for filename in os.listdir(WAV_FOLDER):
            file_path = os.path.join(WAV_FOLDER, filename)
            if not os.path.isfile(file_path):
                continue
            
            # Skip active processing files - only clean files that are definitely old
            # Files ending in .wav, .mp3, .cqt.npy, .txt, .pkl are safe to clean if old
            if not any(filename.endswith(ext) for ext in ['.wav', '.mp3', '.cqt.npy', '.txt', '.pkl', '.npy']):
                continue
            
            try:
                file_mtime = os.path.getmtime(file_path)
                file_age = current_time - file_mtime
                
                # Remove files older than max_age_minutes
                if file_age > max_age_seconds:
                    file_size = os.path.getsize(file_path)
                    os.remove(file_path)
                    cleaned_count += 1
                    cleaned_size += file_size
                    logger.debug(f"Cleaned up old file: {filename} (age: {file_age/60:.1f}m, size: {file_size/1024:.1f}KB)")
            except Exception as e:
                logger.warning(f"Error cleaning file {filename}: {e}")
        
        # Also clean up CQT feature directory
        cqt_dir = os.path.join(WAV_FOLDER, "cqt_feat")
        if os.path.exists(cqt_dir):
            for filename in os.listdir(cqt_dir):
                file_path = os.path.join(cqt_dir, filename)
                if os.path.isfile(file_path):
                    try:
                        file_mtime = os.path.getmtime(file_path)
                        file_age = current_time - file_mtime
                        if file_age > max_age_seconds:
                            file_size = os.path.getsize(file_path)
                            os.remove(file_path)
                            cleaned_count += 1
                            cleaned_size += file_size
                    except Exception as e:
                        logger.warning(f"Error cleaning CQT file {filename}: {e}")
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old files, freed {cleaned_size/1024/1024:.1f}MB")
    except Exception as e:
        logger.error(f"Error in cleanup_old_tmp_files: {e}")

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
        start_time = time.time()  # Initialize at function start to ensure it's always defined
        
        # Cleanup old files at the start to prevent /tmp from filling up
        try:
            cleanup_old_tmp_files(max_age_minutes=30)
        except Exception as e:
            logger.warning(f"Error cleaning old tmp files at start: {e}")
        
        try:
            
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
            
                # Add longer random delay to avoid rate limiting after first download
                # YouTube is more likely to block the second download, so use longer delay
                delay = random.uniform(_MIN_DOWNLOAD_DELAY, _MAX_DOWNLOAD_DELAY)
                logger.info(f"Waiting {delay:.1f}s before second video download to avoid rate limiting...")
                await asyncio.sleep(delay)  # Use asyncio.sleep in async context
            
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
            # Always cleanup WAV files and temporary files, even on error
            try:
                self.cleanup_temp_files(url1, url2)
                # Also cleanup CQT feature files that might be left behind
                video_id1 = extract_video_id(url1)
                video_id2 = extract_video_id(url2)
                for video_id in [video_id1, video_id2]:
                    cqt_path = os.path.join(self.wav_folder, f"{video_id}.wav.cqt.npy")
                    if os.path.exists(cqt_path):
                        try:
                            os.remove(cqt_path)
                            logger.debug(f"Cleaned up CQT file: {cqt_path}")
                        except Exception as e:
                            logger.warning(f"Could not remove CQT file {cqt_path}: {e}")
            except Exception as cleanup_error:
                logger.error(f"Error during cleanup: {cleanup_error}")
            
            # Cleanup old files in /tmp to prevent disk full
            try:
                cleanup_old_tmp_files()
            except Exception as cleanup_error:
                logger.warning(f"Error cleaning old tmp files: {cleanup_error}")
            
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
