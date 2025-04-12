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

THRESHOLD = config['THRESHOLD']


WAV_FOLDER = config['WAV_FOLDER']

CQT_FEAT_DIR = Path(WAV_FOLDER) / "cqt_feat"
CQT_FEAT_DIR.mkdir(exist_ok=True, parents=True)

#CSV_FILE = '/tmp/compared_videos.csv'
CSV_FILE = config['SCORES_CSV_FILE']
# Initialize FastAPI app
app = FastAPI()

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

def _generate_audio_from_youtube_id(youtube_id):
    try:
        logger.debug("Starting audio generation process")
        wav_folder = Path(WAV_FOLDER)
        wav_folder.mkdir(exist_ok=True, parents=True)
        os.chmod(str(wav_folder), 0o777)  # Give full permissions
        
        logger.info(f"Downloading video {youtube_id}...")
        
        # Download with yt-dlp
        ydl_opts = {
            'format': 'bestaudio/best',
            'quiet': False,
            'no_warnings': True,
            'logger': logger,  # Add logger to yt-dlp
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
            }],
            'outtmpl': f'{WAV_FOLDER}/{youtube_id}.%(ext)s',
            'external_downloader': 'aria2c',
            'socket_timeout': 30,
            'retries': 10,
            'fragment_retries': 10,
            'ignoreerrors': True
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                logger.info(f"Attempting download with best quality...")
                ydl.download([f'https://www.youtube.com/watch?v={youtube_id}'])
            except Exception as e:
                logger.error(f"Download failed with error: {str(e)}")
                # Try alternate format if first attempt fails
                logger.info(f"Retrying with worst quality...")
                ydl_opts['format'] = 'worstaudio/worst'
                ydl.download([f'https://www.youtube.com/watch?v={youtube_id}'])
        
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


def _generate_embeddings_from_filepaths(audio_path1, audio_path2):
	COVERHUNTER_FOLDER = config['COVERHUNTER_FOLDER']
	MODEL_FOLDER = config['MODEL_FOLDER']
	os.system(f"mkdir -p {WAV_FOLDER}")
	os.system(f"rm -f {WAV_FOLDER}/*.txt")
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
    

class YoutubeCoverDetector:
    def __init__(self):
        """Initialize the model"""
        try:
            # Load model
            model_path = os.getenv('MODEL_PATH', '/code/pretrain_model/checkpoints/checkpoint.pt')
            logger.info(f"Attempting to load model from: {model_path}")
            
            if not os.path.exists(model_path):
                logger.error(f"Model file not found at {model_path}")
                raise FileNotFoundError(f"Model file not found at {model_path}")

            # Load the model with explicit error handling
            try:
                # Read the file into a BytesIO buffer first
                with open(model_path, 'rb') as f:
                    buffer = io.BytesIO(f.read())
                
                # Load model from buffer
                buffer.seek(0)
                self.model = torch.load(buffer, map_location='cpu')
                logger.info("Model loaded successfully")
                
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
                raise
            
        except Exception as e:
            logger.error(f"Error in YoutubeCoverDetector initialization: {str(e)}")
            raise

    def extract_features(self, audio_path):
        """Extract CQT features from audio file"""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=16000, mono=True)
            
            # Compute CQT
            C = np.abs(librosa.cqt(y, sr=sr, hop_length=512,
                                 bins_per_octave=12, n_bins=84))
            
            # Convert to log scale
            C = librosa.amplitude_to_db(C, ref=np.max)
            
            return torch.FloatTensor(C)
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            raise

    async def detect_cover(self, url1, url2):
        """Detect if two YouTube videos are covers"""
        try:
            # Get video IDs
            video_id1 = self._get_video_id(url1)
            video_id2 = self._get_video_id(url2)
            
            # Download and convert audio
            wav1 = _generate_audio_from_youtube_id(video_id1)
            wav2 = _generate_audio_from_youtube_id(video_id2)
            
            # Extract features
            feat1 = self.extract_features(f"{WAV_FOLDER}/{wav1}")
            feat2 = self.extract_features(f"{WAV_FOLDER}/{wav2}")
            
            # Compare features (simple cosine similarity for now)
            similarity = torch.nn.functional.cosine_similarity(
                feat1.mean(dim=1).unsqueeze(0),
                feat2.mean(dim=1).unsqueeze(0)
            ).item()
            
            distance = 1 - similarity
            is_cover = distance < THRESHOLD
            
            return {
                "distance": distance,
                "is_cover": is_cover,
                "threshold": THRESHOLD
            }
            
        except Exception as e:
            print(f"Error in detect_cover: {e}")
            raise

    def _get_video_id(self, url):
        """Extract video ID from YouTube URL"""
        try:
            if "youtu.be" in url:
                return url.split("/")[-1].split("?")[0]
            elif "youtube.com" in url:
                return url.split("v=")[1].split("&")[0]
            return url
        except Exception as e:
            print(f"Error extracting video ID from {url}: {e}")
            return url

    def _get_thumbnail_url(self, video_id):
        """Get thumbnail URL for a video ID"""
        return f"https://img.youtube.com/vi/{video_id}/0.jpg"

async def download_audio(youtube_id):
    ydl_opts = {
        'format': 'bestaudio/best',
        'quiet': True,
        'external_downloader': 'aria2c',  # Use aria2c for faster downloads
        'external_downloader_args': ['-x', '16', '-k', '1M'],  # Use 16 connections
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
        }],
        'outtmpl': f'{WAV_FOLDER}/{youtube_id}.%(ext)s',
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        await asyncio.to_thread(ydl.download, [f'https://www.youtube.com/watch?v={youtube_id}'])

async def process_videos(video_ids):
    tasks = [download_audio(video_id) for video_id in video_ids]
    await asyncio.gather(*tasks)

# Usage
# asyncio.run(process_videos(['video_id1', 'video_id2']))

# apart from the current csv file, we will also save another csv file named "vectors.csv" with key the youtube id and value the embeddings

def update_vectors_csv(youtube_id, embeddings):
    # check if the youtube_id is already in the csv file
    # create csv file if not exists
    VECTORS_CSV_FILE = config['VECTORS_CSV_FILE']
    if not os.path.exists(VECTORS_CSV_FILE):
        with open(VECTORS_CSV_FILE, 'w') as f:
            f.write("youtube_id,embeddings\n")
    # read the csv file
    vectors_csv = {}
    with open(VECTORS_CSV_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            vectors_csv[row['youtube_id']] = row['embeddings']
    if youtube_id in vectors_csv:
        logger.debug(f"Youtube ID {youtube_id} already in vectors.csv, updating...")
        vectors_csv[youtube_id] = embeddings
    else:
        logger.debug(f"Youtube ID {youtube_id} not in vectors.csv, adding...")

def read_compared_videos():
    compared_videos = []
    try:
        with open(CSV_FILE, mode='r', newline='') as file:
            reader = csv.DictReader(file, fieldnames=['url1', 'url2', 'result', 'score', 'feedback', 'elapsed_time'])
            next(reader)  # Skip header row
            for row in reader:
                compared_videos.append(row)
    except FileNotFoundError:
        logger.debug("CSV file not found. Creating a new file with headers.")
        with open(CSV_FILE, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['url1', 'url2', 'result', 'score', 'feedback', 'elapsed_time'])
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
            writer = csv.DictWriter(f, fieldnames=['url1', 'url2', 'result', 'score', 'feedback', 'elapsed_time'])
            writer.writerow({
                'url1': url1,
                'url2': url2, 
                'result': result,
                'score': str(score),
                'feedback': '',
                'elapsed_time': str(elapsed_time) if elapsed_time is not None else ''
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

class CoverDetector:
    def __init__(self):
        self.threshold = config['THRESHOLD']
        self.wav_folder = config['WAV_FOLDER']
        self.model_folder = config['MODEL_FOLDER']
        self.coverhunter_folder = config['COVERHUNTER_FOLDER']
        self.scores_csv_file = config['SCORES_CSV_FILE']
        self.vectors_csv_file = config['VECTORS_CSV_FILE']
        self.process_only_first_n_seconds = config['PROCESS_ONLY_FIRST_N_SECONDS']
        
    async def compare_videos(self, url1: str, url2: str) -> Dict[str, Any]:
        try:
            start_time = time.time()
            # Download and process videos
            if asyncio.current_task().cancelled():
                raise asyncio.CancelledError()
            
            video_id1 = extract_video_id(url1)
            wav1 = _generate_audio_from_youtube_id(video_id1)
            if asyncio.current_task().cancelled():
                raise asyncio.CancelledError()
            
            video_id2 = extract_video_id(url2)
            wav2 = _generate_audio_from_youtube_id(video_id2)
            if asyncio.current_task().cancelled():
                raise asyncio.CancelledError()
            
            # Get WAV file paths
            wav_path1 = os.path.join(self.wav_folder, wav1)
            wav_path2 = os.path.join(self.wav_folder, wav2)
            
            # Generate embeddings
            embeddings = _generate_embeddings_from_filepaths(wav_path1, wav_path2)
            keys = list(embeddings.keys())
            embedding1 = embeddings[keys[0]]
            embedding2 = embeddings[keys[1]]
            
            # Calculate distance and determine if it's a cover
            distance = _cosine_distance(embedding1, embedding2)
            is_cover = distance < self.threshold
            
            result = "Cover" if is_cover else "Not Cover"
            
            elapsed_time = time.time() - start_time
            # Write result to CSV
            write_compared_video(url1, url2, result, float(distance), elapsed_time=round(elapsed_time, 2))
            
            return {"result": result, "score": float(distance)}
            
        except Exception as e:
            logger.error(f"Error comparing videos: {e}")
            raise

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

# Ensure the FastAPI app runs
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
