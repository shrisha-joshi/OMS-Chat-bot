import uvicorn
import os
import sys

if __name__ == "__main__":
    # Add the current directory to sys.path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    print("🚀 Starting OMS Chat Bot Backend...")
    print("   Host: 127.0.0.1")
    print("   Port: 8000")
    print("   Reload: True")
    
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )
