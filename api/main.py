"""FastAPI application for Flickd AI Engine"""
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import logging
import tempfile
import os
from pathlib import Path
import shutil
import asyncio
from concurrent.futures import ThreadPoolExecutor
import uuid

from models.video_pipeline import VideoAnalysisPipeline
from config import VIDEO_UPLOAD_MAX_SIZE_MB, SUPPORTED_VIDEO_FORMATS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Flickd AI Smart Tagging & Vibe Classification Engine",
    description="Automatically detect fashion items, match products, and classify vibes from videos",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline (done once at startup)
pipeline = None
executor = ThreadPoolExecutor(max_workers=4)

# Response models
class ProductMatch(BaseModel):
    type: str
    color: str
    match_type: str
    matched_product_id: str
    matched_product_name: str
    confidence: float
    similarity: float
    occurrences: int
    frames: List[int]

class VideoAnalysisResponse(BaseModel):
    video_id: str
    vibes: List[str]
    products: List[ProductMatch]
    metadata: Dict
    frames_processed: int
    processing_time: float

class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: bool

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize models and resources on startup"""
    global pipeline
    
    logger.info("Starting Flickd AI Engine...")
    
    try:
        # Initialize pipeline with CPU (no GPU)
        pipeline = VideoAnalysisPipeline(use_gpu=False, use_transformer_nlp=False)
        logger.info("Pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        # Continue without pipeline for basic endpoints
        pipeline = None

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    executor.shutdown(wait=True)
    logger.info("Flickd AI Engine shutdown complete")

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the service is healthy"""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        models_loaded=pipeline is not None
    )

# Main video processing endpoint
@app.post("/process-video", response_model=VideoAnalysisResponse)
async def process_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    caption: Optional[str] = Form(None),
    transcript: Optional[str] = Form(None)
):
    """
    Process a video file and return detected products and vibes
    
    Args:
        video: Video file (MP4, MOV, AVI, etc.)
        caption: Optional caption/hashtags
        transcript: Optional audio transcript
        
    Returns:
        Analysis results with detected products and vibes
    """
    # Validate file type
    file_extension = Path(video.filename).suffix.lower()
    if file_extension not in SUPPORTED_VIDEO_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format. Supported formats: {', '.join(SUPPORTED_VIDEO_FORMATS)}"
        )
    
    # Check file size
    video.file.seek(0, 2)  # Seek to end
    file_size = video.file.tell()
    video.file.seek(0)  # Reset to beginning
    
    if file_size > VIDEO_UPLOAD_MAX_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"Video file too large. Maximum size: {VIDEO_UPLOAD_MAX_SIZE_MB}MB"
        )
    
    # Save uploaded file temporarily
    temp_dir = tempfile.mkdtemp()
    temp_video_path = os.path.join(temp_dir, video.filename)
    
    try:
        # Save uploaded file
        with open(temp_video_path, "wb") as f:
            shutil.copyfileobj(video.file, f)
        
        # Process video in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            executor,
            pipeline.process_video,
            temp_video_path,
            caption,
            transcript,
            False  # Don't save visualizations for API
        )
        
        # Schedule cleanup
        background_tasks.add_task(cleanup_temp_files, temp_dir)
        
        # Convert to response model
        response = VideoAnalysisResponse(
            video_id=results['video_id'],
            vibes=results['vibes'],
            products=[ProductMatch(**p) for p in results['products']],
            metadata=results['metadata'],
            frames_processed=results['frames_processed'],
            processing_time=results['processing_time']
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        # Cleanup on error
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise HTTPException(status_code=500, detail=str(e))

# Batch processing endpoint
@app.post("/process-batch")
async def process_batch(
    videos: List[UploadFile] = File(...),
    captions: Optional[List[str]] = Form(None),
    transcripts: Optional[List[str]] = Form(None)
):
    """Process multiple videos in batch"""
    if len(videos) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 videos allowed per batch"
        )
    
    results = []
    for i, video in enumerate(videos):
        caption = captions[i] if captions and i < len(captions) else None
        transcript = transcripts[i] if transcripts and i < len(transcripts) else None
        
        try:
            result = await process_video(
                BackgroundTasks(),
                video,
                caption,
                transcript
            )
            results.append(result.dict())
        except Exception as e:
            results.append({
                "video_id": video.filename,
                "error": str(e)
            })
    
    return {"results": results}

# Update catalog endpoint
@app.post("/update-catalog")
async def update_catalog(
    catalog_file: UploadFile = File(...)
):
    """Update the product catalog CSV file"""
    if not catalog_file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=400,
            detail="Catalog file must be a CSV"
        )
    
    temp_path = f"/tmp/{uuid.uuid4()}.csv"
    
    try:
        # Save uploaded file
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(catalog_file.file, f)
        
        # Update catalog in pipeline
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            executor,
            pipeline.product_matcher.update_catalog,
            Path(temp_path)
        )
        
        # Cleanup
        os.remove(temp_path)
        
        return {"message": "Catalog updated successfully"}
        
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))

# Add products endpoint
@app.post("/add-products")
async def add_products(products: List[Dict]):
    """Add new products to the catalog"""
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            executor,
            pipeline.product_matcher.add_products,
            products
        )
        return {"message": f"Added {len(products)} products successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Get supported vibes endpoint
@app.get("/vibes")
async def get_supported_vibes():
    """Get list of supported fashion vibes"""
    return {
        "vibes": pipeline.vibe_classifier.supported_vibes,
        "total": len(pipeline.vibe_classifier.supported_vibes)
    }

# Cleanup function
def cleanup_temp_files(temp_dir: str):
    """Clean up temporary files"""
    try:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")
    except Exception as e:
        logger.error(f"Error cleaning up {temp_dir}: {e}")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 