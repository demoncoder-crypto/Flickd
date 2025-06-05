"""Configuration settings for Flickd AI Engine"""
import os
from pathlib import Path
from typing import List

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
FRAMES_DIR = BASE_DIR / "frames"
MODELS_DIR = BASE_DIR / "models"

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
API_WORKERS = int(os.getenv("API_WORKERS", 1))

# Model Configuration
YOLO_MODEL_SIZE = os.getenv("YOLO_MODEL_SIZE", "m")  # n, s, m, l, x
YOLO_CONFIDENCE_THRESHOLD = float(os.getenv("YOLO_CONFIDENCE_THRESHOLD", 0.5))
YOLO_IOU_THRESHOLD = float(os.getenv("YOLO_IOU_THRESHOLD", 0.45))
CLIP_MODEL_NAME = os.getenv("CLIP_MODEL_NAME", "ViT-B/32")
CLIP_DEVICE = os.getenv("CLIP_DEVICE", "cuda" if os.path.exists("/usr/local/cuda") else "cpu")

# Fashion item classes for YOLO
FASHION_CLASSES = [
    "top", "shirt", "blouse", "t-shirt",
    "bottom", "pants", "jeans", "skirt", "shorts",
    "dress", "jacket", "coat", "sweater",
    "bag", "handbag", "purse", "backpack",
    "shoes", "boots", "sneakers", "heels",
    "accessories", "jewelry", "hat", "sunglasses", "watch"
]

# Product Matching
FAISS_INDEX_PATH = Path(os.getenv("FAISS_INDEX_PATH", DATA_DIR / "product_embeddings.index"))
MATCH_THRESHOLD_EXACT = float(os.getenv("MATCH_THRESHOLD_EXACT", 0.9))
MATCH_THRESHOLD_SIMILAR = float(os.getenv("MATCH_THRESHOLD_SIMILAR", 0.75))

# Video Processing
MAX_FRAMES_PER_VIDEO = int(os.getenv("MAX_FRAMES_PER_VIDEO", 30))
FRAME_EXTRACTION_INTERVAL = float(os.getenv("FRAME_EXTRACTION_INTERVAL", 1.0))  # seconds
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
PRODUCT_CATALOG_PATH = Path(os.getenv("PRODUCT_CATALOG_PATH", DATA_DIR / "products.csv"))
MODELS_CACHE_DIR = Path(os.getenv("MODELS_CACHE_DIR", MODELS_DIR / "cache"))

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = Path(os.getenv("LOG_FILE", BASE_DIR / "logs" / "flickd_ai.log"))

# Create necessary directories
for directory in [DATA_DIR, FRAMES_DIR, MODELS_DIR, MODELS_CACHE_DIR, LOG_FILE.parent]:
    directory.mkdir(parents=True, exist_ok=True) 