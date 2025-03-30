# export an api to detect if 2 youtube videos are covers of each other
from flask import Flask, request, jsonify
import os
from app.parse_config import config
import numpy as np
from app.youtube_cover_detector import YoutubeCoverDetector, prepare_cover_detection, cover_detection
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import asyncio
from typing import Dict
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import pathlib
import tempfile
from pathlib import Path
import requests

# Rest of your API code... 