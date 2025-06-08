# 🎬 Flickd AI Smart Tagging & Vibe Classification Engine

**Complete Full-Stack Fashion Video Analysis System**

## 🚀 Quick Start Options

### Option 1: Full-Stack Web App (Recommended)
```bash
python start_demo.py
```
- 🌐 **Frontend**: http://localhost:3000 (Beautiful web interface)
- 🔧 **API**: http://localhost:8000 (Backend server)
- 📚 **Docs**: http://localhost:8000/docs (API documentation)

### Option 2: Command Line Processing
```bash
python run_flickd.py
```

### Option 3: API Server Only
```bash
python run_flickd.py --mode api
```

## 🌐 Frontend Features

### **Modern Web Interface**
- 🎨 **Beautiful Gen Z Aesthetic**: Gradients, glassmorphism, modern design
- 📱 **Responsive Design**: Works on desktop, tablet, and mobile
- 🎯 **Drag & Drop Upload**: Simply drag videos to upload
- ⚡ **Real-time Processing**: Live progress indicators
- 📊 **Rich Results Display**: Visual product cards and vibe badges

### **User Experience**
- 🎬 **Demo Mode**: Try with pre-loaded sample results
- 🔄 **Processing Status**: Real-time feedback during analysis
- 📋 **Detailed Results**: Fashion items, products, and vibes
- 💾 **Download Results**: Export JSON data
- 🎨 **Visual Feedback**: Color-coded confidence levels

### **Frontend Usage**

1. **Start the system**:
   ```bash
   python start_demo.py
   ```

2. **Open your browser** to http://localhost:3000

3. **Upload a video**:
   - Click "Choose Video" or drag & drop
   - Supported formats: MP4, AVI, MOV
   - Max size: 100MB

4. **View results**:
   - Fashion vibes with confidence scores
   - Detected products with similarity ratings
   - Visual confidence indicators

5. **Try demo mode**:
   - Click "Try Demo" to see sample results
   - Perfect for testing without uploading

## 📋 Requirements

- Python 3.8+
- Required packages (auto-installed):
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
- Matches detected items to **969-product Shopify catalog**
- Real similarity scores: Exact (>0.85), Similar (0.65-0.85), No Match (<0.65)
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

### Full-Stack Web App
```bash
python start_demo.py
```
- Complete web interface with drag & drop
- Real-time processing visualization
- Beautiful results display

### Process Videos (Command Line)
```bash
python run_flickd.py --mode process
```

### Start API Server Only
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
├── start_demo.py          # 🚀 FULL-STACK LAUNCHER
├── run_flickd.py          # Command line interface
├── frontend/              # 🌐 Modern Web Interface
│   ├── index.html         # Beautiful UI with Gen Z aesthetic
│   ├── server.py          # Frontend server
│   └── README.md          # Frontend documentation
├── api/                   # 🔧 FastAPI Backend
│   └── main.py            # REST API with CORS support
├── models/                # 🤖 AI Models
│   ├── object_detector.py # YOLO fashion detection
│   ├── product_matcher.py # CLIP + FAISS matching
│   ├── vibe_classifier.py # NLP vibe classification
│   ├── video_pipeline.py  # Main processing pipeline
│   └── custom_fashion_detector.py # Enhanced fashion detection
├── config.py              # Configuration
├── videos/                # Input videos
├── outputs/               # Results (JSON files)
└── data/                  # 969-product Shopify catalog
```

## 🎯 For Reviewers

### **Quick Demo (Recommended)**
1. **Run**: `python start_demo.py`
2. **Open**: http://localhost:3000
3. **Try Demo**: Click "Try Demo" button
4. **Upload Video**: Drag & drop your own video

### **Command Line Testing**
1. **Add video files** to `videos/` folder
2. **Run**: `python run_flickd.py`
3. **Check results** in `outputs/` folder

## 🏆 Key Features

### **AI Engine**
- **High Accuracy**: Optimized YOLO detection for fashion items
- **Real Product Catalog**: 969 actual Shopify products with working URLs
- **Realistic Similarity Scores**: CLIP-based matching (0.6-0.9 range)
- **Complete Vibe Coverage**: All 7 required fashion vibes supported

### **Full-Stack System**
- **Modern Frontend**: Beautiful Gen Z aesthetic with glassmorphism
- **REST API**: FastAPI with comprehensive documentation
- **Real-time Processing**: Live progress indicators
- **Production Ready**: Clean, documented, deployable code

### **User Experience**
- **One Command Setup**: No complex configuration needed
- **Drag & Drop Upload**: Intuitive file handling
- **Visual Results**: Rich product cards and vibe displays
- **Demo Mode**: Try without uploading files

## 📊 Expected Output

For each video, you'll get:
- **Detected vibes** (1-3 per video) with confidence scores
- **Matched products** from real Shopify catalog
- **Similarity scores** ranging 0.6-0.9 (realistic values)
- **Visual confidence indicators** in the web interface

## 🔧 Technical Stack

### **Backend**
- **YOLO**: YOLOv8-medium for fashion detection
- **CLIP**: OpenAI CLIP for image embeddings
- **FAISS**: Fast similarity search with 969 products
- **NLP**: Keyword + transformer-based vibe classification
- **API**: FastAPI with automatic documentation and CORS

### **Frontend**
- **HTML5**: Modern semantic markup
- **CSS3**: Gradients, glassmorphism, responsive design
- **JavaScript**: Async file upload, real-time updates
- **Design**: Gen Z aesthetic with beautiful animations

## 🌟 Demo Experience

The web interface provides:
- **Instant Demo**: See results without uploading
- **Beautiful Visualizations**: Product cards with images
- **Real-time Feedback**: Processing status updates
- **Professional Results**: Clean, organized output
- **Mobile Friendly**: Works on all devices

Perfect for hackathon demonstrations! 🚀

