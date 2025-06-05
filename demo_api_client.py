"""Demo script showing how to use the Flickd AI Engine API"""
import requests
import json
from pathlib import Path
import time

# API base URL
BASE_URL = "http://localhost:8000"

def check_health():
    """Check if the API is healthy"""
    response = requests.get(f"{BASE_URL}/health")
    if response.status_code == 200:
        data = response.json()
        print("✅ API Health Check:")
        print(f"   Status: {data['status']}")
        print(f"   Version: {data['version']}")
        print(f"   Models Loaded: {data['models_loaded']}")
        return True
    else:
        print("❌ API is not responding")
        return False

def get_supported_vibes():
    """Get list of supported vibes"""
    response = requests.get(f"{BASE_URL}/vibes")
    if response.status_code == 200:
        data = response.json()
        print("\n📋 Supported Vibes:")
        for vibe in data['vibes']:
            print(f"   - {vibe}")
        print(f"   Total: {data['total']} vibes")
    else:
        print("❌ Failed to get vibes")

def process_video(video_path: str, caption: str = None):
    """Process a video file"""
    print(f"\n🎥 Processing video: {video_path}")
    
    # Check if file exists
    if not Path(video_path).exists():
        print(f"❌ Video file not found: {video_path}")
        return
    
    # Prepare the request
    files = {
        'video': open(video_path, 'rb')
    }
    
    data = {}
    if caption:
        data['caption'] = caption
    
    # Send request
    print("⏳ Uploading and processing video...")
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{BASE_URL}/process-video",
            files=files,
            data=data
        )
        
        if response.status_code == 200:
            result = response.json()
            elapsed = time.time() - start_time
            
            print(f"\n✅ Video processed successfully in {elapsed:.2f} seconds!")
            print(f"\n📊 Results:")
            print(f"   Video ID: {result['video_id']}")
            print(f"   Duration: {result['metadata']['duration']:.2f} seconds")
            print(f"   Resolution: {result['metadata']['width']}x{result['metadata']['height']}")
            print(f"   Frames Processed: {result['frames_processed']}")
            print(f"   Processing Time: {result['processing_time']} seconds")
            
            print(f"\n🎨 Vibes Detected:")
            if result['vibes']:
                for vibe in result['vibes']:
                    print(f"   - {vibe}")
            else:
                print("   No vibes detected")
            
            print(f"\n🛍️ Products Detected: {len(result['products'])}")
            for i, product in enumerate(result['products'][:5]):
                print(f"\n   {i+1}. {product['matched_product_name']}")
                print(f"      Type: {product['type']}")
                print(f"      Color: {product['color']}")
                print(f"      Match: {product['match_type']} (similarity: {product['similarity']:.3f})")
                print(f"      Confidence: {product['confidence']:.3f}")
                print(f"      Seen in {product['occurrences']} frames")
            
            if len(result['products']) > 5:
                print(f"\n   ... and {len(result['products']) - 5} more products")
            
            # Save full results
            output_file = f"demo_results_{result['video_id']}.json"
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\n💾 Full results saved to: {output_file}")
            
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"   {response.json()}")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
    finally:
        files['video'].close()

def main():
    """Main demo function"""
    print("🚀 Flickd AI Engine API Demo")
    print("="*50)
    
    # Check health
    if not check_health():
        print("\n⚠️  Please make sure the API server is running:")
        print("   python run_server.py")
        return
    
    # Get supported vibes
    get_supported_vibes()
    
    # Demo video processing
    print("\n" + "="*50)
    print("📹 Video Processing Demo")
    print("="*50)
    
    # You can replace this with an actual video file
    demo_video = "sample_video.mp4"
    demo_caption = "Check out my #coquette #fashion look! Pink dress with bows 🎀"
    
    print(f"\n⚠️  To test video processing, you need a video file.")
    print(f"   Expected: {demo_video}")
    print(f"   Caption: {demo_caption}")
    
    # Uncomment to test with actual video
    # process_video(demo_video, demo_caption)
    
    print("\n✨ Demo complete!")

if __name__ == "__main__":
    main() 