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

THRESHOLD = config['THRESHOLD']


WAV_FOLDER = config['WAV_FOLDER']

CQT_FEAT_DIR = Path(WAV_FOLDER) / "cqt_feat"
CQT_FEAT_DIR.mkdir(exist_ok=True, parents=True)

CSV_FILE = '/tmp/compared_videos.csv'

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
        ffmpeg_cmd = f'ffmpeg -y -i "{mp3_path}" -ac 1 -ar 16000 -t 10 "{wav_path}" 2>&1'
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
	#MODEL_FOLDER = WAV_FOLDER
	os.system(f"mkdir -p {WAV_FOLDER}")
	#os.system(f"cp {WAV_FOLDER}/sp_aug.txt {WAV_FOLDER}/sp_aug.txt.bak")
	os.system(f"rm -f {WAV_FOLDER}/*.txt")
	_generate_dataset_txt_from_files(audio_path1, audio_path2)
	#os.system(f"touch {WAV_FOLDER}/sp_aug.txt")
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
    
""" def cover_detection(audio1, audio2):
    # generate the embeddings
    embeddings = _generate_embeddings_from_filepaths(audio1, audio2)
    # embeddings is a dictionary
    # get the keys
    keys = list(embeddings.keys())
    # get the embeddings
    embeddings1 = embeddings[keys[0]]
    embeddings2 = embeddings[keys[1]]
    # get the distance between the embeddings
    distance = _cosine_distance(embeddings1, embeddings2)

    # Determine if the videos are covers
    is_cover = distance < THRESHOLD
    result = "Cover" if is_cover else "Not Cover"

    distance = round(distance, 2)
    # Log and write the result to the CSV
    logger.debug(f"Distance: {distance}, Result: {result}")
    
    write_compared_video(audio1, audio2, result, distance)

    return distance """


def _debug_cover_detection():
  
    #YOUTUBE_ID1 = 'GbpnAGajyMc'
    YOUTUBE_ID1 = 'pFKiJDxBp4c'
    #YOUTUBE_ID1 = 'dQw4w9WgXcQ'
    #YOUTUBE_ID2 = 'Qr0-7Ds79zo'
    YOUTUBE_ID2 = 'kqXSBe-qMGo'
    #YOUTUBE_ID3 = '9egB_8-bvUY'
    ALREADY_DOWNLOADED = False
    if not ALREADY_DOWNLOADED:
        youtube_url1 = f'https://www.youtube.com/watch?v={YOUTUBE_ID1}'
        youtube_url2 = f'https://www.youtube.com/watch?v={YOUTUBE_ID2}'
        audio1, audio2 = prepare_cover_detection(youtube_url1, youtube_url2)
    else:
         audio1 = f'{YOUTUBE_ID1}.wav'
         audio2 = f'{YOUTUBE_ID2}.wav'	

    distance = cover_detection(audio1, audio2)
    print(distance) 
    if distance < THRESHOLD:
        print("COVER")
    else:
        print("NOT COVER")

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
                self.model = torch.load(model_path, map_location='cpu')
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
            writer = csv.DictWriter(file, fieldnames=['url1', 'url2', 'result', 'score'])
            writer.writeheader()
    return compared_videos

def write_compared_video(url1, url2, result, score):
    logger.debug(f"Starting write_compared_video with params: {url1}, {url2}, {result}, {score}")
    try:
        with open(CSV_FILE, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['url1', 'url2', 'result', 'score'])
            if file.tell() == 0:
                writer.writeheader()
                logger.debug("Writing header to CSV file")
            row_data = {'url1': url1, 'url2': url2, 'result': result, 'score': score}
            writer.writerow(row_data)
            logger.debug(f"Successfully wrote row to CSV: {row_data}")
    except Exception as e:
        logger.error(f"Error writing to CSV: {e}")
        raise
    
    # Log CSV contents after writing
    log_csv_contents()
    logger.debug("Finished write_compared_video function")

def log_csv_contents():
    logger.debug("##### Current contents of the CSV file #####")
    try:
        compared_videos = read_compared_videos()
        if not compared_videos:
            logger.debug("CSV file is empty")
        else:
            for video in compared_videos:
                logger.debug(f"CSV entry: {video}")
    except Exception as e:
        logger.error(f"Error reading CSV contents: {e}")
    logger.debug("############################################")

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

def cover_detection(youtube_url1: str, youtube_url2: str):
    logger.debug(f"Received request for URLs: {youtube_url1} and {youtube_url2}")
    
    if not youtube_url1 or not youtube_url2:
        raise HTTPException(status_code=400, detail="Both youtube_url1 and youtube_url2 parameters are required")

    try:
        # First check if we already have the result
        existing_result = get_result_from_csv(youtube_url1, youtube_url2)
        if existing_result is not None:
            logger.debug("Found existing result in CSV, returning cached result")
            return existing_result

        # If not found in CSV, perform the full analysis
        logger.debug("No existing result found, performing full analysis")
        wav_path1 = f"/tmp/youtube_cover_detector_api_wav/{extract_video_id(youtube_url1)}.wav"
        wav_path2 = f"/tmp/youtube_cover_detector_api_wav/{extract_video_id(youtube_url2)}.wav"
        
        # Generate embeddings and calculate distance
        logger.debug("Generating embeddings and calculating distance...")
        embeddings = _generate_embeddings_from_filepaths(wav_path1, wav_path2)
        keys = list(embeddings.keys())
        embeddings1 = embeddings[keys[0]]
        embeddings2 = embeddings[keys[1]]
        distance = _cosine_distance(embeddings1, embeddings2)
        logger.debug(f"Distance: {distance}")
        distance = float(distance)
        distance = round(distance, 2)
        logger.debug(f"Rounded distance: {distance}")
        
        # Determine if it's a cover
        is_cover = distance < THRESHOLD
        result = "Cover" if is_cover else "Not Cover"
        
        # Write to CSV
        logger.debug(f"Writing result to CSV: Distance={distance}, Result={result}")
        write_compared_video(youtube_url1, youtube_url2, result, distance)
        
        return {
            "distance": distance,
            "is_cover": is_cover
        }
    except Exception as e:
        logger.error(f"Error in cover detection: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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

# Ensure the FastAPI app runs
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
