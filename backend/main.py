from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pathlib import Path
import json
from typing import List, Dict, Optional
from datetime import datetime, timedelta

app = FastAPI(title="Wi-Fi Monitoring and Prediction System")

# Mount static files
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

# Templates
templates = Jinja2Templates(directory="frontend/templates")

# Mock data for initial development
MOCK_POINTS = [
    {"id": 1, "name": "AP-1", "type": "real", "location": "Floor 1"},
    {"id": 2, "name": "AP-2", "type": "real", "location": "Floor 2"},
    {"id": 3, "name": "Virtual-1", "type": "virtual", "location": "Floor 1"}
]

MOCK_PREDICTIONS = {
    "points": [
        {
            "id": 1,
            "name": "AP-1",
            "predictions": [
                {"timestamp": "2024-01-01T00:00:00", "channel": 1, "load": 0.3},
                {"timestamp": "2024-01-01T01:00:00", "channel": 6, "load": 0.4},
                {"timestamp": "2024-01-01T02:00:00", "channel": 11, "load": 0.5}
            ]
        }
    ]
}

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the main dashboard page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/points")
async def get_points():
    """Get list of available Wi-Fi access points."""
    return MOCK_POINTS

@app.get("/api/predictions")
async def get_predictions(point_id: Optional[int] = None):
    """Get predictions for a specific point or all points."""
    if point_id:
        # Filter predictions for specific point
        point_predictions = next(
            (p for p in MOCK_PREDICTIONS["points"] if p["id"] == point_id),
            None
        )
        if not point_predictions:
            raise HTTPException(status_code=404, detail="Point not found")
        return point_predictions
    
    return MOCK_PREDICTIONS

@app.get("/api/settings")
async def get_settings():
    """Get current system settings."""
    return {
        "prediction_period": 24,  # hours
        "virtual_points_enabled": True,
        "update_interval": 5  # minutes
    }

@app.post("/api/settings")
async def update_settings(settings: Dict):
    """Update system settings."""
    # In a real implementation, this would save settings to a file or database
    return {"status": "success", "message": "Settings updated"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 