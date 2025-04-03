FROM python:3.11-slim

WORKDIR /code

# Install system dependencies# Use HF's supported Chromium setup
RUN apt-get update && apt-get install -y \
    chromium-driver \
    chromium \
    ffmpeg \
    aria2 \
    libavcodec-extra \
    libav-tools \
    fonts-liberation \
    libnss3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libgbm1 \
    libxshmfence1 \
    libxss1 \
    libasound2 \
    libnspr4 \
    xdg-utils \
    && rm -rf /var/lib/apt/lists/*


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
COPY youtube_cover_detector_api.py /code/
COPY templates/ /code/templates/

# Set Python path to include app directory
ENV PYTHONPATH=/code
ENV PYTHONUNBUFFERED=1

RUN mkdir -p /tmp/.cache && chmod -R 777 /tmp/.cache

# Run the application with logging
CMD ["sh", "-c", "echo 'Starting container...' && uvicorn youtube_cover_detector_api:app --host 0.0.0.0 --port 8080"]
#CMD ["python", "youtube_cover_detector_api.py"]