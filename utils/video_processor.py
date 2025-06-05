"""Video processing utilities for frame extraction"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import logging
from tqdm import tqdm
import hashlib
import os

from config import FRAMES_DIR, MAX_FRAMES_PER_VIDEO, FRAME_EXTRACTION_INTERVAL

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Handles video frame extraction and preprocessing"""
    
    def __init__(self, frames_dir: Path = FRAMES_DIR):
        self.frames_dir = frames_dir
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_video_id(self, video_path: str) -> str:
        """Generate unique ID for video based on content hash"""
        hasher = hashlib.md5()
        with open(video_path, 'rb') as f:
            # Read in chunks to handle large files
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()[:12]
    
    def extract_frames(
        self, 
        video_path: str, 
        interval: float = FRAME_EXTRACTION_INTERVAL,
        max_frames: int = MAX_FRAMES_PER_VIDEO
    ) -> Tuple[str, List[Tuple[int, np.ndarray]]]:
        """
        Extract frames from video at specified interval
        
        Args:
            video_path: Path to video file
            interval: Time interval between frames in seconds
            max_frames: Maximum number of frames to extract
            
        Returns:
            Tuple of (video_id, list of (frame_number, frame_array))
        """
        video_id = self.generate_video_id(video_path)
        video_frames_dir = self.frames_dir / video_id
        video_frames_dir.mkdir(exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = int(fps * interval)
        
        frames = []
        frame_count = 0
        extracted_count = 0
        
        logger.info(f"Extracting frames from video {video_id} (FPS: {fps}, Total frames: {total_frames})")
        
        with tqdm(total=min(max_frames, total_frames // frame_interval), desc="Extracting frames") as pbar:
            while cap.isOpened() and extracted_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_count % frame_interval == 0:
                    # Save frame to disk
                    frame_path = video_frames_dir / f"frame_{frame_count:06d}.jpg"
                    cv2.imwrite(str(frame_path), frame)
                    
                    # Convert BGR to RGB for processing
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append((frame_count, frame_rgb))
                    
                    extracted_count += 1
                    pbar.update(1)
                    
                frame_count += 1
                
        cap.release()
        
        logger.info(f"Extracted {len(frames)} frames from video {video_id}")
        return video_id, frames
    
    def get_video_metadata(self, video_path: str) -> dict:
        """Extract metadata from video file"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")
            
        metadata = {
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "duration": cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        }
        
        cap.release()
        return metadata
    
    def clean_frames(self, video_id: str):
        """Clean up extracted frames for a video"""
        video_frames_dir = self.frames_dir / video_id
        if video_frames_dir.exists():
            import shutil
            shutil.rmtree(video_frames_dir)
            logger.info(f"Cleaned up frames for video {video_id}")
    
    def preprocess_frame(self, frame: np.ndarray, target_size: Tuple[int, int] = (640, 640)) -> np.ndarray:
        """
        Preprocess frame for model input
        
        Args:
            frame: Input frame array
            target_size: Target size for resizing
            
        Returns:
            Preprocessed frame
        """
        # Resize while maintaining aspect ratio
        h, w = frame.shape[:2]
        scale = min(target_size[0] / w, target_size[1] / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = cv2.resize(frame, (new_w, new_h))
        
        # Pad to target size
        pad_w = (target_size[0] - new_w) // 2
        pad_h = (target_size[1] - new_h) // 2
        
        padded = cv2.copyMakeBorder(
            resized,
            pad_h, target_size[1] - new_h - pad_h,
            pad_w, target_size[0] - new_w - pad_w,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114)  # Gray padding
        )
        
        return padded 