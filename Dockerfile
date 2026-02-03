FROM python:3.11-slim

WORKDIR /code

# Install system dependencies first - this layer can be cached
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
    gcc python3-dev \
    nodejs npm \
    && rm -rf /var/lib/apt/lists/*

# Verify ffmpeg installation
RUN ffmpeg -version

# Set up environment variables
ENV CHROME_BIN=/usr/bin/chromium \
    CHROMEDRIVER_PATH=/usr/bin/chromedriver \
    XDG_CACHE_HOME=/tmp/.cache \
    PYTHONPATH=/code \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    HOST=0.0.0.0 \
    PYTHONIOENCODING=utf-8 \
    PYTHONBREAKPOINT=0 \
    PYTHONASYNCIODEBUG=1 \
    CSV_FILE=/data/compared_videos.csv

# Copy requirements and install dependencies - this layer can be cached if requirements don't change
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Add build dependencies before pip install
RUN apt-get update && \
    apt-get install -y gcc python3-dev && \
    pip install psutil==5.9.8 && \
    # Clean up to reduce image size
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create necessary directories
RUN mkdir -p /code/pretrain_model/checkpoints \
    /code/pretrain_model/config \
    /code/static \
    /code/templates \
    /code/data/covers80_testset \
    /tmp/youtube_cover_detector_api_wav \
    /tmp/.cache \
    /data \
    && chmod -R 777 /tmp/youtube_cover_detector_api_wav \
    && chmod -R 777 /tmp/.cache

# Create model directories
RUN mkdir -p /code/pretrain_model/checkpoints /code/pretrain_model/config

COPY pretrain_model/config/hparams.yaml /code/pretrain_model/config/hparams.yaml

# Download model files from Hugging Face
RUN wget -O /code/pretrain_model/checkpoints/g_00000043 https://huggingface.co/muoten/yt-coverhunter/resolve/main/checkpoint.pt

# Install nano at the end to avoid cache invalidation of previous layers
RUN apt-get update && apt-get install -y nano && rm -rf /var/lib/apt/lists/*

# Copy application files (excluding config.yaml to preserve cache on config changes)
# Copy app directory but exclude config.yaml
COPY app/ /code/app/
# Copy config.yaml separately at the end so config changes don't invalidate the entire cache
COPY app/config.yaml /code/app/config.yaml
COPY tools/ /code/tools/
COPY src/ /code/src/
COPY youtube_cover_detector_api.py /code/
COPY templates/ /code/templates/
COPY data/ /code/data/

# Create volume for persistent data
VOLUME /data

# Run API, review script, then test script (in sequence)
CMD ["sh", "-c", "uvicorn youtube_cover_detector_api:app --host 0.0.0.0 --port 8080 --workers 1 --log-level debug --access-log --use-colors & sleep 10 && python /code/src/review_video_pairs.py > /code/review_video_pairs.log 2>&1 && nohup python /code/src/test_video_pairs.py > /code/test_video_pairs.log 2>&1 & wait"]
EXPOSE 8080