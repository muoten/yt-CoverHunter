FROM python:3.11-slim

WORKDIR /code

# Install system dependencies and HF's supported Chromium setup
RUN apt-get update && apt-get install -y \
chromium-driver \
chromium \
ffmpeg \
fonts-liberation \
libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 \
libxcomposite1 libxdamage1 libxrandr2 libgbm1 \
libxshmfence1 libxss1 libasound2 libnspr4 xdg-utils \
libavcodec-extra \
aria2 \
git-lfs \
wget \
&& rm -rf /var/lib/apt/lists/*

# Verify ffmpeg installation
RUN ffmpeg -version

# Set up safe defaults for headless Chromium
ENV CHROME_BIN=/usr/bin/chromium
ENV CHROMEDRIVER_PATH=/usr/bin/chromedriver
ENV XDG_CACHE_HOME=/tmp/.cache

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories with proper permissions
RUN mkdir -p /code/pretrain_model/checkpoints && \
    mkdir -p /code/static && \
    mkdir -p /code/templates && \
    mkdir -p /code/data/covers80_testset && \
    mkdir -p /tmp/youtube_cover_detector_api_wav && \
    chmod -R 777 /tmp/youtube_cover_detector_api_wav

# Copy the application files
COPY app/ /code/app/
COPY tools/ /code/tools/
COPY src/ /code/src/
COPY youtube_cover_detector_api.py /code/
COPY templates/ /code/templates/
COPY data/ /code/data/
COPY pretrain_model/ /code/pretrain_model/

# Make sure the model file exists in the correct location
RUN test -f /code/pretrain_model/checkpoints/checkpoint.pt || echo "Warning: Model file not found"

# Set Python path to include app directory
ENV PYTHONPATH=/code
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
ENV HOST=0.0.0.0
ENV PYTHONIOENCODING=utf-8
ENV PYTHONBREAKPOINT=0
ENV PYTHONASYNCIODEBUG=1

RUN mkdir -p /tmp/.cache && chmod -R 777 /tmp/.cache

# Create a directory for persistent data
RUN mkdir -p /data
VOLUME /data

# Update the environment variable for the CSV file location
ENV CSV_FILE=/data/compared_videos.csv

# Run the application with logging
CMD ["sh", "-c", "echo 'Starting container...' && PYTHONUNBUFFERED=1 uvicorn youtube_cover_detector_api:app --host $HOST --port $PORT --workers 1 --log-level debug --reload --access-log --use-colors"]
#CMD ["python", "youtube_cover_detector_api.py"]