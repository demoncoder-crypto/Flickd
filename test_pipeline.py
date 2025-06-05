"""CLI script to test the video processing pipeline"""
import argparse
import json
import logging
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from models.video_pipeline import VideoAnalysisPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Test Flickd AI video processing pipeline')
    parser.add_argument('video', type=str, help='Path to video file')
    parser.add_argument('--caption', type=str, help='Caption/hashtags for the video')
    parser.add_argument('--transcript', type=str, help='Audio transcript')
    parser.add_argument('--output', type=str, default='results.json', help='Output JSON file')
    parser.add_argument('--visualize', action='store_true', help='Save detection visualizations')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU usage')
    parser.add_argument('--no-transformer', action='store_true', help='Disable transformer NLP')
    
    args = parser.parse_args()
    
    # Check if video exists
    if not Path(args.video).exists():
        logger.error(f"Video file not found: {args.video}")
        return 1
    
    try:
        # Initialize pipeline
        logger.info("Initializing pipeline...")
        pipeline = VideoAnalysisPipeline(
            use_gpu=not args.no_gpu,
            use_transformer_nlp=not args.no_transformer
        )
        
        # Process video
        logger.info(f"Processing video: {args.video}")
        results = pipeline.process_video(
            args.video,
            caption=args.caption,
            transcript=args.transcript,
            save_visualizations=args.visualize
        )
        
        # Save results
        pipeline.save_results(results, args.output)
        
        # Print summary
        print("\n" + "="*50)
        print("ANALYSIS RESULTS")
        print("="*50)
        print(f"Video ID: {results['video_id']}")
        print(f"Duration: {results['metadata']['duration']:.2f} seconds")
        print(f"Frames processed: {results['frames_processed']}")
        print(f"Processing time: {results['processing_time']:.2f} seconds")
        
        print(f"\nVibes detected: {', '.join(results['vibes']) if results['vibes'] else 'None'}")
        
        print(f"\nProducts detected: {len(results['products'])}")
        for i, product in enumerate(results['products'][:5]):  # Show top 5
            print(f"\n  {i+1}. {product['matched_product_name']}")
            print(f"     Type: {product['type']}")
            print(f"     Color: {product['color']}")
            print(f"     Match: {product['match_type']} (similarity: {product['similarity']:.3f})")
            print(f"     Confidence: {product['confidence']:.3f}")
            print(f"     Seen in {product['occurrences']} frames")
        
        if len(results['products']) > 5:
            print(f"\n  ... and {len(results['products']) - 5} more products")
        
        print("\n" + "="*50)
        print(f"Full results saved to: {args.output}")
        
        if args.visualize:
            print(f"Visualizations saved to: frames/{results['video_id']}/")
        
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 