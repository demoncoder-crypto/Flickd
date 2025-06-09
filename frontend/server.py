#!/usr/bin/env python3
"""
Simple HTTP server to serve the Flickd AI frontend
"""

import http.server
import socketserver
import webbrowser
import os
from pathlib import Path
import urllib.parse

PORT = 3000

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Add CORS headers to allow API calls
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()
    
    def do_GET(self):
        # Handle demo_outputs requests by serving from the correct directory
        if self.path.startswith('/demo_outputs/'):
            # Remove leading slash and serve from parent directory
            file_path = self.path[1:]  # Remove leading /
            full_path = Path(__file__).parent / file_path
            
            if full_path.exists() and full_path.is_file():
                self.send_response(200)
                if file_path.endswith('.jpg') or file_path.endswith('.jpeg'):
                    self.send_header('Content-Type', 'image/jpeg')
                elif file_path.endswith('.png'):
                    self.send_header('Content-Type', 'image/png')
                else:
                    self.send_header('Content-Type', 'application/octet-stream')
                self.end_headers()
                
                with open(full_path, 'rb') as f:
                    self.wfile.write(f.read())
                return
            else:
                self.send_error(404, f"File not found: {file_path}")
                return
        
        # Default handling for other requests
        super().do_GET()

def main():
    # Change to frontend directory
    frontend_dir = Path(__file__).parent
    os.chdir(frontend_dir)
    
    # Create server
    with socketserver.TCPServer(("", PORT), CustomHTTPRequestHandler) as httpd:
        print(f"🌐 Flickd AI Frontend Server")
        print(f"   URL: http://localhost:{PORT}")
        print(f"   Directory: {frontend_dir}")
        print(f"   Demo outputs: {frontend_dir / 'demo_outputs'}")
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