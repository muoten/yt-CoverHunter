FROM python:3.11-slim

WORKDIR /code

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

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

# Add environment variable for more verbose logging
ENV PYTHONUNBUFFERED=1

# Run the application with logging
CMD ["sh", "-c", "echo 'Starting container...' && python -u youtube_cover_detector_api.py"] 