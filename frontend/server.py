#!/usr/bin/env python3
"""
Simple HTTP server to serve the Flickd AI frontend
"""

import http.server
import socketserver
import webbrowser
import os
from pathlib import Path

PORT = 3000

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Add CORS headers to allow API calls
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

def main():
    # Change to frontend directory
    frontend_dir = Path(__file__).parent
    os.chdir(frontend_dir)
    
    # Create server
    with socketserver.TCPServer(("", PORT), CustomHTTPRequestHandler) as httpd:
        print(f"🌐 Flickd AI Frontend Server")
        print(f"   URL: http://localhost:{PORT}")
        print(f"   Directory: {frontend_dir}")
        print(f"   Press Ctrl+C to stop")
        
        # Open browser
        try:
            webbrowser.open(f'http://localhost:{PORT}')
        except:
            pass
        
        # Start server
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print(f"\n🛑 Server stopped")

if __name__ == "__main__":
    main() 