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

# Create necessary directories
RUN mkdir -p /code/static /code/templates

# Copy the application
COPY app/ /code/app/
RUN mkdir -p /code/app/utils && touch /code/app/utils/__init__.py
COPY youtube_cover_detector_api.py /code/
COPY templates/ /code/templates/

# Set Python path to include app directory
ENV PYTHONPATH=/code
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
ENV HOST=0.0.0.0
ENV PYTHONIOENCODING=utf-8
ENV PYTHONBREAKPOINT=0
ENV PYTHONASYNCIODEBUG=1

RUN mkdir -p /tmp/.cache && chmod -R 777 /tmp/.cache

# Create necessary directories with proper permissions
RUN mkdir -p /tmp/youtube_cover_detector_api_wav && \
    chmod -R 777 /tmp/youtube_cover_detector_api_wav

# Run the application with logging
CMD ["sh", "-c", "echo 'Starting container...' && PYTHONUNBUFFERED=1 uvicorn youtube_cover_detector_api:app --host $HOST --port $PORT --workers 1 --log-level debug --reload --access-log --use-colors"]
#CMD ["python", "youtube_cover_detector_api.py"]