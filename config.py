"""Configuration settings for Flickd AI Engine"""
import os
from pathlib import Path
from typing import List

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
FRAMES_DIR = BASE_DIR / "frames"
MODELS_DIR = BASE_DIR / "models"
MODELS_CACHE_DIR = BASE_DIR / "models" / "cache"
OUTPUTS_DIR = BASE_DIR / "outputs"
LOGS_DIR = BASE_DIR / "logs"

# API Configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
API_VERSION = "1.0.0"
API_WORKERS = int(os.getenv("API_WORKERS", 1))

# Model Configuration
YOLO_MODEL_SIZE = "m"  # Medium model for good balance of speed/accuracy
YOLO_CONFIDENCE_THRESHOLD = 0.25  # Lower threshold for better detection
YOLO_IOU_THRESHOLD = 0.4  # Lower for better overlapping detection
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
CLIP_DEVICE = os.getenv("CLIP_DEVICE", "cuda" if os.path.exists("/usr/local/cuda") else "cpu")

# Fashion item classes for YOLO
FASHION_CLASSES = [
    "person",      # Essential for fashion detection
    "top",         # Shirts, blouses, t-shirts
    "bottom",      # Pants, skirts, shorts  
    "dress",       # Dresses, gowns
    "outerwear",   # Jackets, coats
    "footwear",    # Shoes, boots
    "bag",         # Handbags, backpacks
    "accessories"  # Jewelry, hats, etc.
]

# Product Matching (Flickd Requirements)
FAISS_INDEX_PATH = MODELS_CACHE_DIR / "product_index.faiss"
MATCH_THRESHOLD_EXACT = 0.9     # Exact Match: > 0.9 (Flickd Requirement)
MATCH_THRESHOLD_SIMILAR = 0.75  # Similar Match: 0.75-0.9 (Flickd Requirement)

# Video Processing
MAX_FRAMES_TO_PROCESS = 30  # Process more frames for better detection
MAX_FRAMES_PER_VIDEO = 30  # Alias for compatibility
FRAME_EXTRACTION_INTERVAL = 1.0  # Extract every 1 second
FRAMES_DIR = BASE_DIR / "frames"  # Temporary frames directory
VIDEO_UPLOAD_MAX_SIZE_MB = int(os.getenv("VIDEO_UPLOAD_MAX_SIZE_MB", 100))
SUPPORTED_VIDEO_FORMATS = [".mp4", ".mov", ".avi", ".mkv", ".webm"]

# Vibe Classification
SUPPORTED_VIBES = [
    "Coquette",
    "Clean Girl", 
    "Cottagecore",
    "Streetcore",
    "Y2K",
    "Boho",
    "Party Glam"
]

# Vibe keywords for rule-based classification
VIBE_KEYWORDS = {
    "Coquette": ["coquette", "feminine", "bow", "ribbon", "pink", "lace", "romantic", "soft", "delicate", "pastel"],
    "Clean Girl": ["clean", "minimal", "simple", "fresh", "natural", "effortless", "neutral", "basic", "sleek"],
    "Cottagecore": ["cottage", "rural", "vintage", "floral", "garden", "countryside", "rustic", "homemade", "cozy"],
    "Streetcore": ["street", "urban", "edgy", "grunge", "punk", "alternative", "underground", "raw", "authentic"],
    "Y2K": ["y2k", "2000s", "retro", "cyber", "futuristic", "metallic", "low rise", "butterfly", "tech"],
    "Boho": ["boho", "bohemian", "hippie", "free spirit", "ethnic", "tribal", "flowing", "earthy", "natural"],
    "Party Glam": ["party", "glam", "glamour", "sparkle", "sequin", "night out", "club", "fancy", "dressy", "elegant"]
}

# Paths
PRODUCT_CATALOG_PATH = DATA_DIR / "catalog.csv"
MODELS_CACHE_DIR = MODELS_CACHE_DIR

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Create necessary directories
for directory in [DATA_DIR, FRAMES_DIR, MODELS_CACHE_DIR, OUTPUTS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True) 