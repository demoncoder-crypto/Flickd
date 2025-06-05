#!/usr/bin/env python3
"""
Hackathon Video Analysis Script
Process all videos and generate JSON outputs in required format
"""
import json
import logging
from pathlib import Path
from typing import Dict, List
import time

from models.video_pipeline import VideoAnalysisPipeline
from models.vibe_classifier import VibeClassifier

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HackathonVideoProcessor:
    """Process videos for hackathon submission"""
    
    def __init__(self):
        """Initialize the processor"""
        logger.info("Initializing Hackathon Video Processor...")
        
        # Initialize pipeline (CPU mode for compatibility)
        self.pipeline = VideoAnalysisPipeline(
            use_gpu=False, 
            use_transformer_nlp=False
        )
        
        # Paths
        self.videos_dir = Path("videos")
        self.outputs_dir = Path("outputs")
        self.outputs_dir.mkdir(exist_ok=True)
        
        # Hackathon requirements
        self.min_similarity_threshold = 0.75  # As per requirements
        
    def process_single_video(self, video_path: Path) -> Dict:
        """Process a single video and return hackathon format"""
        
        logger.info(f"Processing: {video_path.name}")
        
        try:
            # Get video ID from filename (remove extension)
            video_id = video_path.stem
            
            # Process video through pipeline
            results = self.pipeline.process_video(
                str(video_path),
                caption="",  # No caption provided
                transcript="",  # No transcript
                save_visualizations=False
            )
            
            # Convert to hackathon format
            hackathon_output = self._convert_to_hackathon_format(
                video_id, 
                results
            )
            
            return hackathon_output
            
        except Exception as e:
            logger.error(f"Error processing {video_path.name}: {e}")
            return {
                "video_id": video_path.stem,
                "vibes": [],
                "products": [],
                "error": str(e)
            }
    
    def _convert_to_hackathon_format(self, video_id: str, results: Dict) -> Dict:
        """Convert pipeline results to hackathon format"""
        
        # Extract vibes (limit to 1-3 as per requirements)
        vibes = results.get('vibes', [])[:3]
        
        # Convert products to hackathon format
        hackathon_products = []
        
        for product in results.get('products', []):
            # Only include products above similarity threshold
            similarity = product.get('similarity', 0)
            if similarity >= self.min_similarity_threshold:
                
                hackathon_product = {
                    "type": product.get('type', 'unknown'),
                    "color": product.get('color', 'unknown'),
                    "matched_product_id": product.get('matched_product_id', ''),
                    "match_type": product.get('match_type', 'similar'),
                    "confidence": round(product.get('confidence', similarity), 2)
                }
                
                hackathon_products.append(hackathon_product)
        
        # Limit to 2-4 products as suggested
        hackathon_products = hackathon_products[:4]
        
        return {
            "video_id": video_id,
            "vibes": vibes,
            "products": hackathon_products
        }
    
    def process_all_videos(self) -> Dict:
        """Process all videos in the videos directory"""
        
        logger.info(f"🎬 Starting video analysis...")
        logger.info(f"Videos directory: {self.videos_dir}")
        logger.info(f"Outputs directory: {self.outputs_dir}")
        
        # Find all MP4 files
        video_files = list(self.videos_dir.glob("*.mp4"))
        
        if not video_files:
            logger.error("No MP4 files found in videos/ directory!")
            return {"processed": 0, "results": []}
        
        logger.info(f"Found {len(video_files)} videos to process")
        
        results = []
        processed_count = 0
        
        for video_path in video_files:
            logger.info(f"\n📹 Processing {processed_count + 1}/{len(video_files)}: {video_path.name}")
            
            # Process video
            result = self.process_single_video(video_path)
            results.append(result)
            
            # Save individual JSON file
            output_file = self.outputs_dir / f"{result['video_id']}.json"
            
            try:
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2)
                
                logger.info(f"✅ Saved: {output_file}")
                logger.info(f"   Vibes: {result.get('vibes', [])}")
                logger.info(f"   Products: {len(result.get('products', []))}")
                
                processed_count += 1
                
            except Exception as e:
                logger.error(f"Error saving {output_file}: {e}")
        
        # Save summary
        summary = {
            "processed": processed_count,
            "total_videos": len(video_files),
            "results": results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        summary_file = self.outputs_dir / "processing_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"\n🎯 Processing Complete!")
        logger.info(f"✅ Processed: {processed_count}/{len(video_files)} videos")
        logger.info(f"📂 Results saved in: {self.outputs_dir}")
        logger.info(f"📊 Summary: {summary_file}")
        
        return summary

def main():
    """Main function"""
    
    logger.info("🚀 HACKATHON VIDEO ANALYSIS")
    logger.info("=" * 50)
    
    # Create processor
    processor = HackathonVideoProcessor()
    
    # Process all videos
    summary = processor.process_all_videos()
    
    # Show final summary
    logger.info("\n" + "=" * 50)
    logger.info("📊 FINAL SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Videos processed: {summary['processed']}")
    logger.info(f"Output files generated: {summary['processed']}")
    logger.info(f"Results directory: outputs/")
    
    # List output files
    outputs_dir = Path("outputs")
    json_files = list(outputs_dir.glob("*.json"))
    logger.info(f"\nGenerated files:")
    for file in json_files:
        logger.info(f"  - {file.name}")
    
    logger.info("\n🎉 Ready for hackathon submission!")

if __name__ == "__main__":
    main() 