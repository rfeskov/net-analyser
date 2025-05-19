#!/usr/bin/env python3

import uvicorn
from pathlib import Path

def main():
    # Get the absolute path to the backend directory
    backend_dir = Path(__file__).parent / 'backend'
    
    # Run the FastAPI application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[str(backend_dir)]
    )

if __name__ == "__main__":
    main() 