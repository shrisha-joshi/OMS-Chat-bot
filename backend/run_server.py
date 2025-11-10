"""
Direct server runner that bypasses uvicorn CLI to avoid Python 3.13 lifespan issues.
"""
import asyncio
import signal
import sys
import uvicorn

# Disable KeyboardInterrupt during import and startup
def signal_handler(sig, frame):
    """Ignore SIGINT during critical sections"""
    pass

# Temporarily ignore Ctrl+C during imports
original_handler = signal.signal(signal.SIGINT, signal_handler)

try:
    from app.main import app
finally:
    # Restore original handler after import
    signal.signal(signal.SIGINT, original_handler)

if __name__ == "__main__":
    print("=" * 80)
    print("Starting OMS Chatbot Backend Server")
    print("=" * 80)
    print(f"Server: http://127.0.0.1:8000")
    print(f"API Docs: http://127.0.0.1:8000/docs")
    print(f"Press Ctrl+C to stop")
    print("=" * 80)
    print()
    
    try:
        # Run server directly without uvicorn CLI
        config = uvicorn.Config(
            app=app,
            host="127.0.0.1",
            port=8000,
            reload=False,  # Disable reload in Python 3.13 to avoid issues
            log_level="info"
        )
        server = uvicorn.Server(config)
        asyncio.run(server.serve())
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
        sys.exit(0)
