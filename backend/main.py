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

# Load analysis results
def load_analysis_results():
    try:
        with open("analysis_results.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Analysis results file not found")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid JSON in analysis results file")

# Get points from analysis results
def get_points_from_analysis():
    analysis_data = load_analysis_results()
    points = []
    for point_id, point_data in analysis_data["point_analyses"].items():
        points.append({
            "id": point_id,
            "name": point_id,
            "type": "real",
            "location": "Unknown"  # You might want to add location data to your analysis results
        })
    return points

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the main dashboard page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/points")
async def get_points():
    """Get list of available Wi-Fi access points."""
    return get_points_from_analysis()

@app.get("/api/predictions")
async def get_predictions(point_id: Optional[str] = None):
    """Get predictions for a specific point or all points."""
    analysis_data = load_analysis_results()
    
    if point_id:
        if point_id not in analysis_data["point_analyses"]:
            raise HTTPException(status_code=404, detail="Point not found")
        
        point_data = analysis_data["point_analyses"][point_id]
        return {
            "id": point_id,
            "name": point_id,
            "time_periods": point_data["time_periods"],
            "recommendations": analysis_data["final_recommendations"][point_id]
        }
    
    # Return all points data
    return {
        "points": [
            {
                "id": point_id,
                "name": point_id,
                "time_periods": point_data["time_periods"],
                "recommendations": analysis_data["final_recommendations"][point_id]
            }
            for point_id, point_data in analysis_data["point_analyses"].items()
        ]
    }

@app.get("/api/conflicts")
async def get_conflicts():
    """Get list of channel conflicts between points."""
    analysis_data = load_analysis_results()
    return analysis_data["conflicts"]

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