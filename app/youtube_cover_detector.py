# export an api to detect if 2 youtube videos are covers of each other

#import youtube_dl
import os
from app.parse_config import config
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
THRESHOLD = config['THRESHOLD']


WAV_FOLDER = config['WAV_FOLDER']

CQT_FEAT_DIR = Path(WAV_FOLDER) / "cqt_feat"
CQT_FEAT_DIR.mkdir(exist_ok=True, parents=True)

def _generate_audio_from_youtube_id(youtube_id):
    try:
        wav_folder = Path(WAV_FOLDER)
        wav_folder.mkdir(exist_ok=True, parents=True)
        os.chmod(str(wav_folder), 0o777)  # Give full permissions
        
        # Download with yt-dlp
        ydl_opts = {
            'format': 'bestaudio/best',
            'quiet': True,
            'no_warnings': True,
            'geo_bypass': True,  # Try to bypass geo-restrictions
            'geo_bypass_country': 'US',
            'no_check_certificate': True,  # Skip HTTPS certificate validation
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
            }],
            'outtmpl': f'{WAV_FOLDER}/{youtube_id}.%(ext)s',
            # Add fallback options
            'external_downloader': 'aria2c',
            'socket_timeout': 30,
            'retries': 10,
            'fragment_retries': 10,
            'ignoreerrors': True
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                ydl.download([f'https://www.youtube.com/watch?v={youtube_id}'])
            except Exception as e:
                print(f"Download failed with error: {str(e)}")
                # Try alternate format if first attempt fails
                ydl_opts['format'] = 'worstaudio/worst'
                ydl.download([f'https://www.youtube.com/watch?v={youtube_id}'])
        
        # Convert to WAV
        mp3_path = f'{WAV_FOLDER}/{youtube_id}.mp3'
        wav_path = f'{WAV_FOLDER}/{youtube_id}.wav'
        
        os.system(f'ffmpeg -y -i "{mp3_path}" -ac 1 -ar 16000 "{wav_path}"')
        
        if not os.path.exists(wav_path):
            raise ValueError("Audio conversion failed")
            
        return f"{youtube_id}.wav"
        
    except Exception as e:
        print(f"Error in _generate_audio_from_youtube_id: {str(e)}")
        raise

def _generate_dataset_txt_from_files(filename1, filename2):
	           
    dataset = []
    for i,filename in enumerate([filename1, filename2]):
        entry = {
            "perf": filename,
            "wav": WAV_FOLDER + "/" + filename,  # Path can be MP3 or WAV
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
	os.system(f"rm {WAV_FOLDER}/*.txt")
	_generate_dataset_txt_from_files(audio_path1, audio_path2)
	#os.system(f"touch {WAV_FOLDER}/sp_aug.txt")
	os.system(f"cp {WAV_FOLDER}/dataset.txt {WAV_FOLDER}/sp_aug.txt")
	os.system(f"rm -r {WAV_FOLDER}/cqt_feat/")
	os.system(f"rm -r {WAV_FOLDER}/sp_wav/")

	
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
    
def cover_detection(audio1, audio2):

    # generate the embeddings
    embeddings = _generate_embeddings_from_filepaths(audio1, audio2)
    # embeddings is a dictionary
    # get the keys
    keys = list(embeddings.keys())
    # get the embeddings
    embeddings1 = embeddings[keys[0]]
    print(embeddings1)
    print(len(embeddings1))

    embeddings2 = embeddings[keys[1]]
    print(embeddings2)
    # get the distance between the embeddings
    distance = _cosine_distance(embeddings1, embeddings2)
    print(distance)

    return distance


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
        print("Initializing YoutubeCoverDetector...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model
        model_path = 'pretrain_model/checkpoint.pt'
        if os.path.exists(model_path):
            print(f"Loading model from {model_path}...")
            try:
                self.model = torch.load(model_path, map_location=self.device)
                print("Model loaded successfully!")
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                raise
        else:
            print(f"Model file not found at {model_path}")

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
        if "youtu.be" in url:
            return url.split("/")[-1]
        elif "youtube.com" in url:
            return url.split("v=")[1].split("&")[0]
        return url

    def _get_thumbnail_url(self, video_id):
        """Get thumbnail URL for a video ID"""
        return f"https://img.youtube.com/vi/{video_id}/0.jpg"

if __name__ == '__main__':
    DEBUG = True
    if DEBUG:
        _debug_cover_detection()
