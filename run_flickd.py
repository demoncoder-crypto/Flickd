#!/usr/bin/env python3
"""
Flickd AI Smart Tagging & Vibe Classification Engine
Single command to run the complete system
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_system():
    """Setup and verify system requirements"""
    print("🚀 FLICKD AI ENGINE - HACKATHON SUBMISSION")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version.split()[0]}")
    
    # Check required packages with correct import names
    required_packages = [
        ('torch', 'torch'),
        ('ultralytics', 'ultralytics'), 
        ('transformers', 'transformers'),
        ('opencv-python', 'cv2'),
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('pillow', 'PIL'),
        ('fastapi', 'fastapi'),
        ('uvicorn', 'uvicorn')
    ]
    
    missing_packages = []
    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
            print(f"✅ {package_name}")
        except ImportError:
            missing_packages.append(package_name)
            print(f"❌ {package_name}")
    
    if missing_packages:
        print(f"\n⚠️  Install missing packages:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def process_videos(video_dir: str = "videos", output_dir: str = "outputs"):
    """Process all videos in the directory"""
    from models.video_pipeline import VideoAnalysisPipeline
    
    video_path = Path(video_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    if not video_path.exists():
        print(f"❌ Video directory not found: {video_dir}")
        return []
    
    # Find video files
    video_files = []
    for ext in ['.mp4', '.mov', '.avi', '.mkv']:
        video_files.extend(video_path.glob(f"*{ext}"))
    
    if not video_files:
        print(f"❌ No video files found in {video_dir}")
        return []
    
    print(f"\n📹 Processing {len(video_files)} videos...")
    
    # Initialize pipeline
    pipeline = VideoAnalysisPipeline()
    results = []
    
    for i, video_file in enumerate(video_files, 1):
        print(f"\n🎬 Processing {i}/{len(video_files)}: {video_file.name}")
        
        try:
            # Process video
            result = pipeline.process_video(str(video_file))
            
            # Save result
            output_file = output_path / f"{video_file.stem}.json"
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            print(f"✅ Saved: {output_file}")
            print(f"   Vibes: {', '.join(result.get('vibes', []))}")
            print(f"   Products: {len(result.get('products', []))}")
            
            results.append(result)
            
        except Exception as e:
            print(f"❌ Error processing {video_file.name}: {e}")
            logger.error(f"Error processing {video_file}: {e}")
    
    return results

def start_api_server():
    """Start the FastAPI server"""
    import uvicorn
    from api.main import app
    
    print("\n🌐 Starting API Server...")
    print(f"   URL: http://localhost:8000")
    print(f"   Docs: http://localhost:8000/docs")
    print(f"   Health: http://localhost:8000/health")
    
    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")

def create_demo_data():
    """Create demo data for testing"""
    print("\n📊 Creating demo data...")
    
    # Create sample product catalog
    products_data = [
        {
            'product_id': 'prod_001',
            'product_name': 'Black Evening Dress',
            'image_url': 'https://example.com/dress1.jpg',
            'category': 'dress',
            'color': 'black'
        },
        {
            'product_id': 'prod_002', 
            'product_name': 'White Cotton Blouse',
            'image_url': 'https://example.com/top1.jpg',
            'category': 'top',
            'color': 'white'
        },
        {
            'product_id': 'prod_003',
            'product_name': 'Blue Denim Jeans',
            'image_url': 'https://example.com/jeans1.jpg',
            'category': 'bottom',
            'color': 'blue'
        },
        {
            'product_id': 'prod_004',
            'product_name': 'Brown Leather Handbag',
            'image_url': 'https://example.com/bag1.jpg',
            'category': 'bag',
            'color': 'brown'
        },
        {
            'product_id': 'prod_005',
            'product_name': 'Gold Chain Necklace',
            'image_url': 'https://example.com/necklace1.jpg',
            'category': 'accessories',
            'color': 'gold'
        }
    ]
    
    # Save catalog
    import pandas as pd
    df = pd.DataFrame(products_data)
    catalog_path = Path("data/products.csv")
    catalog_path.parent.mkdir(exist_ok=True)
    df.to_csv(catalog_path, index=False)
    
    print(f"✅ Created product catalog: {catalog_path}")
    print(f"   Products: {len(products_data)}")

def show_results(output_dir: str = "outputs"):
    """Show processing results summary"""
    output_path = Path(output_dir)
    
    if not output_path.exists():
        print(f"❌ No results found in {output_dir}")
        return
    
    json_files = list(output_path.glob("*.json"))
    
    if not json_files:
        print(f"❌ No result files found in {output_dir}")
        return
    
    print(f"\n📊 RESULTS SUMMARY")
    print("=" * 50)
    
    total_vibes = set()
    total_products = 0
    
    for json_file in json_files:
        try:
            with open(json_file) as f:
                result = json.load(f)
            
            vibes = result.get('vibes', [])
            products = result.get('products', [])
            
            print(f"\n🎬 {json_file.stem}")
            print(f"   Vibes: {', '.join(vibes) if vibes else 'None'}")
            print(f"   Products: {len(products)}")
            
            if products:
                for product in products[:3]:  # Show first 3
                    print(f"     - {product.get('matched_product_name', 'Unknown')}")
                    print(f"       Type: {product.get('type', 'Unknown')}")
                    print(f"       Similarity: {product.get('similarity', 0):.3f}")
            
            total_vibes.update(vibes)
            total_products += len(products)
            
        except Exception as e:
            print(f"❌ Error reading {json_file}: {e}")
    
    print(f"\n🎯 TOTALS:")
    print(f"   Videos processed: {len(json_files)}")
    print(f"   Unique vibes detected: {len(total_vibes)}")
    print(f"   Total products matched: {total_products}")
    print(f"   All vibes: {', '.join(sorted(total_vibes))}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Flickd AI Engine")
    parser.add_argument('--mode', choices=['process', 'api', 'demo', 'results'], 
                       default='process', help='Run mode')
    parser.add_argument('--videos', default='videos', help='Video directory')
    parser.add_argument('--output', default='outputs', help='Output directory')
    
    args = parser.parse_args()
    
    # Setup system
    if not setup_system():
        sys.exit(1)
    
    if args.mode == 'demo':
        create_demo_data()
        
    elif args.mode == 'process':
        create_demo_data()  # Ensure we have demo data
        results = process_videos(args.videos, args.output)
        
        if results:
            print(f"\n🎉 Processing complete!")
            show_results(args.output)
        else:
            print(f"\n⚠️  No videos processed. Add video files to '{args.videos}' directory")
            
    elif args.mode == 'api':
        create_demo_data()  # Ensure we have demo data
        start_api_server()
        
    elif args.mode == 'results':
        show_results(args.output)
    
    print(f"\n✨ Flickd AI Engine ready for hackathon submission!")

if __name__ == "__main__":
    main() 