"""Main video processing pipeline that combines all components"""
import time
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import json
import numpy as np

from utils.video_processor import VideoProcessor
from models.object_detector import FashionDetector
from models.custom_fashion_detector import load_best_available_model
from models.product_matcher import ProductMatcher
from models.vibe_classifier import VibeClassifier

logger = logging.getLogger(__name__)


class VideoAnalysisPipeline:
    """Main pipeline for video analysis"""
    
    def __init__(
        self,
        use_gpu: bool = True,
        use_transformer_nlp: bool = True
    ):
        self.use_gpu = use_gpu
        device = 'cuda' if use_gpu else 'cpu'
        
        # Initialize components
        logger.info("Initializing video analysis pipeline...")
        
        self.video_processor = VideoProcessor()
        
        # Try to load custom trained model, fallback to original detector
        try:
            self.fashion_detector = load_best_available_model()
            logger.info("Using custom trained fashion detector")
        except Exception as e:
            logger.warning(f"Failed to load custom model, using original detector: {e}")
        self.fashion_detector = FashionDetector(device=device)
            
        self.product_matcher = ProductMatcher(device=device)
        self.vibe_classifier = VibeClassifier(use_transformer=use_transformer_nlp)
        
        logger.info("Pipeline initialized successfully")
        
    def process_video(
        self,
        video_path: str,
        caption: Optional[str] = None,
        transcript: Optional[str] = None,
        save_visualizations: bool = False
    ) -> Dict:
        """
        Process a video and return analysis results
        
        Args:
            video_path: Path to video file
            caption: Optional caption/hashtags
            transcript: Optional audio transcript
            save_visualizations: Whether to save detection visualizations
            
        Returns:
            Dictionary with analysis results
        """
        start_time = time.time()
        
        try:
            # Extract frames from video
            logger.info(f"Processing video: {video_path}")
            video_id, frames = self.video_processor.extract_frames(video_path)
            
            # Get video metadata
            metadata = self.video_processor.get_video_metadata(video_path)
            
            # Process each frame
            all_detections = []
            all_products = []
            
            for frame_num, frame in frames:
                logger.info(f"Processing frame {frame_num}")
                
                # Detect fashion items
                detections = self.fashion_detector.detect(frame)
                
                # Extract regions and match products
                if detections:
                    regions = self.fashion_detector.extract_fashion_regions(frame, detections)
                    
                    for cropped_region, detection_info in regions:
                        # Match to products
                        matches = self.product_matcher.match_product(cropped_region, top_k=3)
                        
                        # Add frame number to detection
                        detection_info['frame_number'] = frame_num
                        
                        # Get best match
                        if matches and matches[0]['match_type'] in ['exact', 'similar']:
                            best_match = matches[0]
                            product_info = {
                                'type': detection_info['class'],
                                'color': best_match.get('color', 'unknown'),
                                'match_type': best_match['match_type'],
                                'matched_product_id': best_match['product_id'],
                                'matched_product_name': best_match['product_name'],
                                'confidence': detection_info['confidence'],
                                'similarity': best_match['similarity'],
                                'frame_number': frame_num,
                                'bounding_box': detection_info['bbox']
                            }
                            all_products.append(product_info)
                            
                    all_detections.extend(detections)
                    
                    # Save visualization if requested
                    if save_visualizations:
                        vis_path = self.video_processor.frames_dir / video_id / f"detection_{frame_num:06d}.jpg"
                        self.fashion_detector.visualize_detections(frame, detections, str(vis_path))
                        
            # Classify vibes from text
            vibes = []
            if caption or transcript:
                combined_text = f"{caption or ''} {transcript or ''}".strip()
                vibe_results = self.vibe_classifier.classify_vibes(combined_text)
                vibes = [vibe for vibe, _ in vibe_results]
            else:
                # Use visual analysis when no text is available
                frame_vibes = []
                for frame_num, frame in frames[:5]:  # Analyze first 5 frames
                    detected_vibes = self.vibe_classifier.classify_vibes_from_image(frame)
                    frame_vibes.extend(detected_vibes)
                
                # Aggregate vibes from all frames
                if frame_vibes:
                    from collections import Counter
                    vibe_counts = Counter(frame_vibes)
                    # Get most common vibes (up to 3)
                    vibes = [vibe for vibe, count in vibe_counts.most_common(3)]
                
            # Aggregate results
            processing_time = time.time() - start_time
            
            results = {
                'video_id': video_id,
                'metadata': metadata,
                'vibes': vibes,
                'products': self._aggregate_products(all_products),
                'raw_detections': len(all_detections),
                'frames_processed': len(frames),
                'processing_time': round(processing_time, 2)
            }
            
            # Clean up frames if not needed
            if not save_visualizations:
                self.video_processor.clean_frames(video_id)
                
            logger.info(f"Video processing completed in {processing_time:.2f} seconds")
            return results
            
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            raise
            
    def _aggregate_products(self, products: List[Dict]) -> List[Dict]:
        """Aggregate detected products to remove duplicates"""
        # Group by product ID
        product_groups = {}
        
        for product in products:
            product_id = product['matched_product_id']
            if product_id not in product_groups:
                product_groups[product_id] = []
            product_groups[product_id].append(product)
            
        # Aggregate each group
        aggregated = []
        
        for product_id, group in product_groups.items():
            # Calculate average confidence and similarity
            avg_confidence = np.mean([p['confidence'] for p in group])
            avg_similarity = np.mean([p['similarity'] for p in group])
            
            # Get most common attributes
            types = [p['type'] for p in group]
            most_common_type = max(set(types), key=types.count)
            
            colors = [p['color'] for p in group]
            most_common_color = max(set(colors), key=colors.count)
            
            # Determine overall match type
            match_types = [p['match_type'] for p in group]
            if 'exact' in match_types:
                match_type = 'exact'
            elif 'similar' in match_types:
                match_type = 'similar'
            else:
                match_type = 'no_match'
                
            aggregated_product = {
                'type': most_common_type,
                'color': most_common_color,
                'match_type': match_type,
                'matched_product_id': product_id,
                'matched_product_name': group[0]['matched_product_name'],
                'confidence': round(float(avg_confidence), 3),
                'similarity': round(float(avg_similarity), 3),
                'occurrences': len(group),
                'frames': [p['frame_number'] for p in group]
            }
            
            aggregated.append(aggregated_product)
            
        # Sort by confidence
        aggregated.sort(key=lambda x: x['confidence'], reverse=True)
        
        return aggregated
    
    def process_batch(
        self,
        video_paths: List[str],
        captions: Optional[List[str]] = None,
        transcripts: Optional[List[str]] = None
    ) -> List[Dict]:
        """Process multiple videos in batch"""
        results = []
        
        if captions is None:
            captions = [None] * len(video_paths)
        if transcripts is None:
            transcripts = [None] * len(video_paths)
            
        for video_path, caption, transcript in zip(video_paths, captions, transcripts):
            try:
                result = self.process_video(video_path, caption, transcript)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {video_path}: {e}")
                results.append({
                    'video_id': Path(video_path).stem,
                    'error': str(e)
                })
                
        return results
    
    def save_results(self, results: Dict, output_path: str):
        """Save results to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_path}")
        
    def load_results(self, input_path: str) -> Dict:
        """Load results from JSON file"""
        with open(input_path, 'r') as f:
            return json.load(f) 