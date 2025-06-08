#!/usr/bin/env python3
"""
Start both API server and frontend for complete demo
"""

import subprocess
import sys
import time
import webbrowser
from pathlib import Path
import threading
import signal
import os

def start_api_server():
    """Start the API server"""
    print("🚀 Starting API Server on localhost:8000...")
    try:
        subprocess.run([
            sys.executable, "run_flickd.py", "--mode", "api"
        ], check=True)
    except KeyboardInterrupt:
        print("\n🛑 API Server stopped")
    except Exception as e:
        print(f"❌ API Server error: {e}")

def start_frontend_server():
    """Start the frontend server"""
    print("🌐 Starting Frontend Server on localhost:3000...")
    try:
        frontend_path = Path(__file__).parent / "frontend" / "server.py"
        subprocess.run([sys.executable, str(frontend_path)], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Frontend Server stopped")
    except Exception as e:
        print(f"❌ Frontend Server error: {e}")

def main():
    """Start both servers"""
    print("🎬 FLICKD AI DEMO - Starting Complete System")
    print("=" * 50)
    
    # Start API server in background thread
    api_thread = threading.Thread(target=start_api_server, daemon=True)
    api_thread.start()
    
    # Wait a moment for API to start
    print("⏳ Waiting for API server to initialize...")
    time.sleep(3)
    
    # Open browser to frontend
    print("🌐 Opening frontend in browser...")
    try:
        webbrowser.open('http://localhost:3000')
    except:
        pass
    
    print("\n✨ DEMO READY!")
    print("   Frontend: http://localhost:3000")
    print("   API Docs: http://localhost:8000/docs")
    print("   API Health: http://localhost:8000/health")
    print("\n💡 Click 'Show Demo Results' to see real AI analysis!")
    print("   Press Ctrl+C to stop both servers")
    
    try:
        # Start frontend server (this will block)
        start_frontend_server()
    except KeyboardInterrupt:
        print("\n🛑 Demo stopped")
        print("   Both servers have been shut down")

if __name__ == "__main__":
    main() 