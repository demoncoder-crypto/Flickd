# 🌐 Flickd AI Frontend

Beautiful, modern web interface for the Flickd AI Smart Tagging & Vibe Classification Engine.

## ✨ Features

- **Drag & Drop Video Upload** - Intuitive file upload with visual feedback
- **Real-time Processing** - Live updates during AI analysis
- **Beautiful Results Display** - Modern cards showing detected vibes and matched products
- **Responsive Design** - Works perfectly on desktop and mobile
- **API Integration** - Seamless connection to FastAPI backend
- **Demo Mode** - Show real results without uploading videos

## 🚀 Quick Start

### Option 1: Using the main script
```bash
python run_flickd.py --mode frontend
```

### Option 2: Direct server
```bash
cd frontend
python server.py
```

The frontend will open automatically at: **http://localhost:3000**

## 🎨 Design Features

- **Gen Z Aesthetic** - Modern gradients, glassmorphism effects
- **Interactive Elements** - Smooth animations and hover effects  
- **Real-time Stats** - Live display of system capabilities
- **Product Cards** - Beautiful display of matched fashion items
- **Vibe Tags** - Colorful tags for detected fashion vibes

## 🔧 Technical Stack

- **Pure HTML/CSS/JavaScript** - No frameworks, fast loading
- **Modern CSS** - Flexbox, Grid, CSS Variables
- **Fetch API** - Modern HTTP requests to backend
- **Responsive Design** - Mobile-first approach
- **CORS Enabled** - Cross-origin requests supported

## 📱 Screenshots

### Main Interface
- Clean upload area with drag & drop
- Real-time processing indicators
- Beautiful gradient backgrounds

### Results Display
- Detected vibes as colorful tags
- Product matches with similarity scores
- Video preview integration

### API Documentation
- Live endpoint information
- Direct links to FastAPI docs
- Interactive testing capabilities

## 🎯 Usage

1. **Start the backend API**:
   ```bash
   python run_flickd.py --mode api
   ```

2. **Start the frontend**:
   ```bash
   python run_flickd.py --mode frontend
   ```

3. **Upload a video** or click "Show Demo Results"

4. **View AI analysis** with detected vibes and matched products

## 🌟 Demo Mode

Click "Show Demo Results" to see real data from the system:
- **Vibes**: Streetcore, Clean Girl, Y2K
- **Products**: Dakota Co-Ord Set (80.7%), Avery Cami Top (74.9%), etc.
- **Similarity Scores**: Real scores from 0.6-0.9 range

## 🔗 API Endpoints

The frontend connects to these backend endpoints:

- `POST /process-video` - Upload and analyze videos
- `GET /health` - System health check
- `GET /vibes` - List supported fashion vibes
- `GET /docs` - Interactive API documentation

## 🎨 Customization

### Colors
- Primary: `#4ecdc4` (Teal)
- Secondary: `#45b7d1` (Blue)
- Accent: `#ff6b6b` (Coral)
- Background: Linear gradient purple to blue

### Fonts
- **Inter** - Modern, clean typography
- **Font Awesome** - Beautiful icons throughout

### Layout
- **Grid System** - Responsive 2-column layout
- **Card Design** - Glassmorphism effects
- **Mobile First** - Optimized for all screen sizes

## 🚀 Perfect for Demos

This frontend is designed to impress:
- **Professional appearance** for hackathon presentations
- **Real-time processing** shows AI capabilities
- **Beautiful results** highlight system accuracy
- **Easy to use** for non-technical reviewers

## 🎬 Demo Script Integration

Perfect companion to your Loom demo:
1. Show the beautiful interface
2. Upload a video or use demo mode
3. Highlight the real similarity scores
4. Show the API documentation
5. Emphasize the production-ready design

**This frontend showcases your Flickd AI system as a complete, production-ready solution! 🚀** 