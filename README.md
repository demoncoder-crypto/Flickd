# 🎬 Flickd AI Smart Tagging & Vibe Classification Engine

## 🏆 Hackathon Submission - AI Fashion Video Analysis

**Automatically detect fashion items, match products, and classify vibes from short-form video content.**

![Flickd AI Demo](https://img.shields.io/badge/Status-Ready%20for%20Submission-brightgreen)
![Python](https://img.shields.io/badge/Python-3.13+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🎯 **What This Does**

Flickd AI Engine processes fashion videos to:
- 🔍 **Detect fashion items** using YOLOv8 object detection
- 🛍️ **Match products** from your catalog using CLIP embeddings + FAISS
- 🌟 **Classify fashion vibes** (Coquette, Clean Girl, Cottagecore, Y2K, etc.)
- 📱 **Generate JSON outputs** in hackathon-required format

---

## 🚀 **Quick Start**

### Prerequisites
- Python 3.13+ (recommended) or 3.8+
- 4GB+ RAM
- Windows/macOS/Linux

### 1. Install Dependencies
```bash
# Clone the repository
git clone https://github.com/demoncoder-crypto/Flickd.git
cd Flickd

# Create virtual environment
python -m venv venv

# Activate environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements-py313.txt
```

### 2. Setup Data
```bash
# Process your dataset
python process_dataset.py

# Your data structure should be:
# data/catalog.csv          - Product catalog
# data/vibeslist.json      - Supported vibes
# videos/*.mp4             - Input videos
```

### 3. Run Analysis
```bash
# Process all videos
python run_video_analysis.py

# Start API server
python run_server.py
```

### 4. View Results
- **JSON Outputs:** Check `outputs/` folder
- **API Interface:** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/health

---

## 📁 **Project Structure**

```
Flickd/
├── 📂 api/                  # FastAPI server & endpoints
├── 📂 models/               # ML models (YOLO, CLIP, NLP)
├── 📂 data/                 # Product catalog & vibes
├── 📂 outputs/              # Generated evaluation JSONs
├── 📂 videos/               # Sample input videos
├── 📂 utils/                # Helper functions
├── 🐍 run_video_analysis.py # Main processing script
├── 🐍 run_server.py         # API server launcher
├── 📋 requirements.txt      # Dependencies
└── 📖 README.md             # This file
```

---

## 🎬 **Video Analysis Output Format**

Each video generates a JSON file in the required hackathon format:

```json
{
  "video_id": "reel_001",
  "vibes": ["Coquette", "Party Glam"],
  "products": [
    {
      "type": "top",
      "color": "white", 
      "matched_product_id": "prod_002",
      "match_type": "exact",
      "confidence": 0.93
    }
  ]
}
```

---

## 🔧 **Technical Architecture**

### Core Components
1. **YOLOv8 Object Detection** - Identifies people and fashion items
2. **CLIP + FAISS Matching** - Matches detected items to product catalog  
3. **NLP Vibe Classification** - Analyzes text/hashtags for fashion vibes
4. **Video Processing Pipeline** - Orchestrates end-to-end analysis

### Supported Fashion Vibes
- 🎀 **Coquette** - Feminine, romantic, pink/pastel aesthetics
- ✨ **Clean Girl** - Minimal, effortless, natural look
- 🌸 **Cottagecore** - Rural, vintage, floral patterns
- 🏙️ **Streetcore** - Urban, edgy, grunge style
- 💫 **Y2K** - 2000s inspired, metallic, cyber elements
- 🌿 **Boho** - Free-spirited, ethnic patterns, flowing
- 💎 **Party Glam** - Sparkly, elegant evening wear

---

## 🌐 **API Endpoints**

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/process-video` | Upload & analyze video |
| GET | `/health` | Service health check |
| GET | `/vibes` | List supported vibes |
| POST | `/update-catalog` | Update product catalog |
| GET | `/docs` | Interactive API documentation |

---

## 📊 **Performance & Requirements**

- **Processing Speed:** ~10-15 seconds per video
- **Accuracy:** >75% similarity threshold for product matching
- **Memory Usage:** ~2-4GB during processing
- **Supported Formats:** MP4, MOV, AVI (720p recommended)

---

## 🔧 **Configuration**

Key settings in `config.py`:
- `MATCH_THRESHOLD_EXACT = 0.85` - Exact product match threshold
- `MATCH_THRESHOLD_SIMILAR = 0.75` - Similar product match threshold  
- `SUPPORTED_VIBES` - List of fashion vibes to detect
- `VIDEO_UPLOAD_MAX_SIZE_MB = 100` - Max video file size

---

## 🚨 **Troubleshooting**

### Common Issues

**1. CUDA/GPU Errors**
```bash
# Solution: Force CPU mode
export USE_GPU=false
python run_video_analysis.py
```

**2. Missing Dependencies**
```bash
# Install missing packages
pip install transformers torch ultralytics faiss-cpu
```

**3. Memory Issues**
```bash
# Reduce batch size in config.py
FRAMES_TO_EXTRACT = 10  # Reduce from 30
```

---

## 📈 **Model Versions Used**

- **Object Detection:** YOLOv8m (medium)
- **Image Embedding:** CLIP ViT-B/32
- **Similarity Search:** FAISS IndexFlatIP
- **NLP Processing:** spaCy en_core_web_sm
- **Framework:** FastAPI + Uvicorn

---

## 🎥 **Demo & Examples**

### 🎬 Loom Demo Video
**[Watch 5-Minute Demo](LOOM_LINK_HERE)**

### 📱 API Testing
Import the Postman collection: `postman_collection.json`

### 🧪 Sample Results
Check `outputs/` folder for example JSON outputs from processed videos.

---

## 🤝 **Contributing**

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open Pull Request

---

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🏆 **Hackathon Submission Checklist**

- ✅ Complete video processing pipeline
- ✅ Hackathon-format JSON outputs
- ✅ REST API with documentation
- ✅ Product catalog integration
- ✅ Fashion vibe classification
- ✅ Comprehensive README
- ✅ Sample videos & results
- ✅ Requirements & setup guide

---

## 👨‍💻 **Built With ❤️ for Flickd Hackathon**

*Revolutionizing fashion discovery through AI-powered video analysis*

**Questions?** Open an issue or contact the development team!
