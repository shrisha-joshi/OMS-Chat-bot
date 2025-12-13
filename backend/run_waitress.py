"""
Run backend with waitress server (Windows-stable alternative to uvicorn)
"""
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(__file__))

from waitress import serve
from app.main import app

if __name__ == "__main__":
    print("=" * 80)
    print("🚀 Starting OMS Chat Bot Backend with Waitress")
    print("=" * 80)
    print("Server: http://0.0.0.0:8000")
    print("Process will run in background...")
    print("=" * 80)
    
    # Start waitress server
    serve(app, host='0.0.0.0', port=8000, threads=4)
