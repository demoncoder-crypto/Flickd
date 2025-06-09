#!/usr/bin/env python3
"""
Flickd AI Demo with Bounding Box Visualization
Shows real-time fashion detection with visual bounding boxes
"""

import os
import sys
import cv2
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models.video_pipeline import VideoAnalysisPipeline
from models.custom_fashion_detector import EnhancedFashionDetector
from utils.video_processor import VideoProcessor
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FlickdVisualizationDemo:
    """Demo class that processes videos and creates bounding box visualizations"""
    
    def __init__(self):
        self.output_dir = Path("frontend/demo_outputs")
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "frames").mkdir(exist_ok=True)
        (self.output_dir / "visualizations").mkdir(exist_ok=True)
        (self.output_dir / "results").mkdir(exist_ok=True)
        
        # Initialize components
        self.detector = EnhancedFashionDetector()
        self.video_processor = VideoProcessor()
        
    def process_video_with_visualization(self, video_path: str) -> Dict:
        """Process a single video and create visualizations"""
        
        video_path = Path(video_path)
        video_name = video_path.stem
        
        logger.info(f"🎬 Processing {video_name} with visualization...")
        
        # Extract frames - returns (video_id, frames)
        video_id, frames = self.video_processor.extract_frames(str(video_path))
        
        results = {
            'video_id': video_name,
            'detections_per_frame': [],
            'total_detections': 0,
            'visualization_paths': []
        }
        
        # Process each frame
        for i, (frame_num, frame) in enumerate(frames):
            logger.info(f"Processing frame {i+1}/{len(frames)} (frame #{frame_num})")
            
            # Detect fashion items with visualization
            detections, validation_metrics = self.detector.detect_with_validation(
                frame, 
                save_visualization=True,
                output_path=str(self.output_dir / "visualizations" / f"{video_name}_frame_{frame_num:04d}.jpg")
            )
            
            # Save original frame
            frame_path = self.output_dir / "frames" / f"{video_name}_frame_{frame_num:04d}.jpg"
            cv2.imwrite(str(frame_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
            # Store results
            frame_result = {
                'frame_number': frame_num,
                'detections': detections,
                'detection_count': len(detections),
                'validation_metrics': validation_metrics,
                'original_frame_path': str(frame_path),
                'visualization_path': str(self.output_dir / "visualizations" / f"{video_name}_frame_{frame_num:04d}.jpg")
            }
            
            results['detections_per_frame'].append(frame_result)
            results['total_detections'] += len(detections)
            
            if detections:
                results['visualization_paths'].append(frame_result['visualization_path'])
        
        # Save results
        results_path = self.output_dir / "results" / f"{video_name}_analysis.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"✅ Completed {video_name}: {results['total_detections']} total detections")
        return results
    
    def create_detection_summary_image(self, video_results: Dict) -> str:
        """Create a summary image showing all detections"""
        
        video_name = video_results['video_id']
        
        # Find frames with most detections
        frames_with_detections = [
            frame for frame in video_results['detections_per_frame'] 
            if frame['detection_count'] > 0
        ]
        
        if not frames_with_detections:
            logger.warning(f"No detections found for {video_name}")
            return None
        
        # Sort by detection count and take top 4
        top_frames = sorted(frames_with_detections, 
                          key=lambda x: x['detection_count'], 
                          reverse=True)[:4]
        
        # Create grid layout
        grid_size = 2  # 2x2 grid
        cell_width, cell_height = 400, 300
        
        summary_image = np.zeros((grid_size * cell_height, grid_size * cell_width, 3), dtype=np.uint8)
        
        for i, frame_data in enumerate(top_frames):
            if i >= 4:  # Only show top 4
                break
                
            # Load visualization image
            viz_path = frame_data['visualization_path']
            if Path(viz_path).exists():
                viz_img = cv2.imread(viz_path)
                if viz_img is not None:
                    # Resize to fit grid cell
                    viz_img = cv2.resize(viz_img, (cell_width, cell_height))
                    
                    # Calculate position in grid
                    row = i // grid_size
                    col = i % grid_size
                    
                    y_start = row * cell_height
                    y_end = y_start + cell_height
                    x_start = col * cell_width
                    x_end = x_start + cell_width
                    
                    summary_image[y_start:y_end, x_start:x_end] = viz_img
                    
                    # Add frame info
                    info_text = f"Frame {frame_data['frame_number']}: {frame_data['detection_count']} items"
                    cv2.putText(summary_image, info_text, 
                              (x_start + 10, y_start + 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Save summary image
        summary_path = self.output_dir / "visualizations" / f"{video_name}_summary.jpg"
        cv2.imwrite(str(summary_path), summary_image)
        
        logger.info(f"✅ Created summary visualization: {summary_path}")
        return str(summary_path)
    
    def run_demo(self, video_dir: str = "videos") -> Dict:
        """Run the complete demo with visualizations"""
        
        video_dir = Path(video_dir)
        video_files = list(video_dir.glob("*.mp4"))
        
        if not video_files:
            logger.error(f"No video files found in {video_dir}")
            return {}
        
        logger.info(f"🚀 Starting Flickd AI Visualization Demo")
        logger.info(f"📁 Processing {len(video_files)} videos from {video_dir}")
        logger.info(f"💾 Outputs will be saved to {self.output_dir}")
        
        demo_results = {
            'total_videos': len(video_files),
            'processed_videos': [],
            'total_detections': 0,
            'summary_visualizations': []
        }
        
        for i, video_path in enumerate(video_files, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"🎬 Processing video {i}/{len(video_files)}: {video_path.name}")
            
            try:
                # Process video
                video_results = self.process_video_with_visualization(str(video_path))
                
                # Create summary visualization
                summary_path = self.create_detection_summary_image(video_results)
                if summary_path:
                    demo_results['summary_visualizations'].append(summary_path)
                
                demo_results['processed_videos'].append({
                    'video_name': video_path.name,
                    'detections': video_results['total_detections'],
                    'frames_processed': len(video_results['detections_per_frame']),
                    'summary_visualization': summary_path
                })
                
                demo_results['total_detections'] += video_results['total_detections']
                
            except Exception as e:
                logger.error(f"❌ Failed to process {video_path.name}: {e}")
                continue
        
        # Save overall demo results
        demo_summary_path = self.output_dir / "demo_summary.json"
        with open(demo_summary_path, 'w') as f:
            json.dump(demo_results, f, indent=2, default=str)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"🎉 Demo Complete!")
        logger.info(f"📊 Total videos processed: {len(demo_results['processed_videos'])}")
        logger.info(f"🔍 Total detections: {demo_results['total_detections']}")
        logger.info(f"📁 Visualizations saved to: {self.output_dir}")
        
        return demo_results

def main():
    """Main demo function"""
    
    print("🚀 FLICKD AI - BOUNDING BOX VISUALIZATION DEMO")
    print("=" * 60)
    
    # Create demo instance
    demo = FlickdVisualizationDemo()
    
    # Process a video with actual detections (not the first one which has 0)
    video_files = list(Path("videos").glob("*.mp4"))
    if len(video_files) > 1:
        # Use the second video which should have detections
        results = demo.process_video_with_visualization(str(video_files[1]))
        print(f"✅ Demo completed! Check frontend/demo_outputs/ for visualizations")
        print(f"📊 Processed {results['total_detections']} detections")
    elif video_files:
        results = demo.process_video_with_visualization(str(video_files[0]))
        print(f"✅ Demo completed! Check frontend/demo_outputs/ for visualizations")
        print(f"📊 Processed {results['total_detections']} detections")
    else:
        print("❌ No video files found in videos/ directory")

if __name__ == "__main__":
    main() 