#!/usr/bin/env python3

import uvicorn
from pathlib import Path
import sys

def main():
    # Get the absolute path to the backend directory
    backend_dir = Path(__file__).parent / 'backend'
    
    # Add backend directory to Python path
    sys.path.append(str(backend_dir))
    
    # Run the FastAPI application
    uvicorn.run(
        "backend.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        reload_dirs=[str(backend_dir)]
    )

if __name__ == "__main__":
    main() 