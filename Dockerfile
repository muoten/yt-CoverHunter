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

# Create necessary directories
RUN mkdir -p /code/pretrain_model/checkpoints \
    /code/static \
    /code/templates \
    /code/data/covers80_testset \
    /tmp/youtube_cover_detector_api_wav \
    /tmp/.cache \
    /data \
    && chmod -R 777 /tmp/youtube_cover_detector_api_wav \
    && chmod -R 777 /tmp/.cache

# Install nano at the end to avoid cache invalidation of previous layers
RUN apt-get update && apt-get install -y nano && rm -rf /var/lib/apt/lists/*

# Copy all application files at once - do this last to maximize cache usage
COPY app/ /code/app/
COPY tools/ /code/tools/
COPY src/ /code/src/
COPY youtube_cover_detector_api.py /code/
COPY templates/ /code/templates/
COPY data/ /code/data/
COPY pretrain_model/ /code/pretrain_model/

# Download model file if needed
RUN curl -L https://your-model-url/checkpoint.pt -o /code/pretrain_model/checkpoints/checkpoint.pt || echo "Warning: Model file not downloaded"

# Create volume for persistent data
VOLUME /data

# Run the application
CMD ["sh", "-c", "echo 'Starting container...' && PYTHONUNBUFFERED=1 uvicorn youtube_cover_detector_api:app --host $HOST --port $PORT --workers 4 --log-level debug --reload --access-log --use-colors"]
EXPOSE 8080