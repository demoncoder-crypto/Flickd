# 🎬 Flickd AI Smart Tagging & Vibe Classification Engine

**Hackathon Submission - Fashion Video Analysis System**

## 🚀 Quick Start (One Command)

```bash
python run_flickd.py
```

That's it! The system will:
- ✅ Check dependencies
- ✅ Create demo data
- ✅ Process videos in `videos/` folder
- ✅ Generate results in `outputs/` folder
- ✅ Show summary

## 📋 Requirements

- Python 3.8+
- Required packages (auto-checked):
  ```
  torch ultralytics transformers opencv-python
  numpy pandas pillow fastapi uvicorn
  ```

## 🎯 What It Does

### 1. **Object Detection (YOLO)**
- Detects fashion items: tops, bottoms, dresses, bags, accessories
- Uses YOLOv8 with enhanced person-based detection
- Confidence thresholds optimized for fashion

### 2. **Product Matching (CLIP + FAISS)**
- Matches detected items to product catalog
- Similarity scores: Exact (>0.85), Similar (0.65-0.85), No Match (<0.65)
- Returns product IDs, names, and confidence scores

### 3. **Vibe Classification (NLP)**
- Classifies videos into fashion vibes:
  - Coquette, Clean Girl, Cottagecore, Streetcore, Y2K, Boho, Party Glam
- Uses keyword analysis and context understanding

### 4. **JSON Output**
```json
{
  "video_id": "fashion_video_001",
  "vibes": ["Coquette", "Clean Girl"],
  "products": [
    {
      "type": "dress",
      "color": "black",
      "match_type": "exact",
      "matched_product_id": "prod_001",
      "matched_product_name": "Black Evening Dress",
      "confidence": 0.92,
      "similarity": 0.94
    }
  ]
}
```

## 🎮 Usage Modes

### Process Videos (Default)
```bash
python run_flickd.py --mode process
```

### Start API Server
```bash
python run_flickd.py --mode api
```
- API: http://localhost:8000
- Docs: http://localhost:8000/docs

### View Results
```bash
python run_flickd.py --mode results
```

### Create Demo Data
```bash
python run_flickd.py --mode demo
```

## 📁 Project Structure

```
flickd-submission/
├── run_flickd.py          # Main script (ONE COMMAND)
├── config.py              # Configuration
├── models/                # AI Models
│   ├── object_detector.py # YOLO fashion detection
│   ├── product_matcher.py # CLIP + FAISS matching
│   ├── vibe_classifier.py # NLP vibe classification
│   └── video_pipeline.py  # Main processing pipeline
├── api/                   # FastAPI server
├── videos/                # Input videos (add your videos here)
├── outputs/               # Results (JSON files)
└── data/                  # Product catalog
```

## 🎯 For Reviewers

1. **Add video files** to `videos/` folder
2. **Run**: `python run_flickd.py`
3. **Check results** in `outputs/` folder

The system automatically:
- Downloads YOLO models if needed
- Creates sample product catalog
- Processes all videos
- Shows detailed results

## 🏆 Key Features

- **High Accuracy**: Optimized YOLO detection for fashion items
- **Real Similarity Scores**: CLIP-based product matching with realistic scores
- **Complete Vibe Coverage**: All 7 required fashion vibes supported
- **Production Ready**: FastAPI server with full documentation
- **One Command Setup**: No complex configuration needed

## 📊 Expected Output

For each video, you'll get:
- **Detected vibes** (1-3 per video)
- **Matched products** with similarity scores
- **Bounding boxes** and confidence levels
- **Processing metadata**

## 🔧 Technical Details

- **YOLO**: YOLOv8-medium for fashion detection
- **CLIP**: OpenAI CLIP for image embeddings
- **FAISS**: Fast similarity search
- **NLP**: Keyword + transformer-based vibe classification
- **API**: FastAPI with automatic documentation

---

**Ready for hackathon submission! 🎉**
