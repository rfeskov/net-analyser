from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pathlib import Path
import json
from typing import List, Dict, Optional
from datetime import datetime, timedelta

app = FastAPI(title="Wi-Fi Network Analysis System")

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

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the main dashboard page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/points")
async def get_points():
    """Get list of available Wi-Fi access points with their basic information."""
    analysis_data = load_analysis_results()
    points = []
    
    for point_id, point_data in analysis_data["point_analyses"].items():
        # Get the first time period to extract basic info
        first_period = point_data["time_periods"][0]
        points.append({
            "id": point_id,
            "name": point_id,
            "band": first_period["band"],
            "channel": first_period["channel"],
            "load_score": first_period["load_score"],
            "stability": first_period["stability"]
        })
    
    return points

@app.get("/api/points/{point_id}")
async def get_point_details(point_id: str):
    """Get detailed information for a specific point."""
    analysis_data = load_analysis_results()
    
    if point_id not in analysis_data["point_analyses"]:
        raise HTTPException(status_code=404, detail="Point not found")
    
    point_data = analysis_data["point_analyses"][point_id]
    recommendations = analysis_data["final_recommendations"][point_id]
    
    return {
        "id": point_id,
        "time_periods": point_data["time_periods"],
        "recommendations": recommendations
    }

@app.get("/api/points/{point_id}/metrics")
async def get_point_metrics(point_id: str, band: Optional[str] = None):
    """Get metrics for a specific point, optionally filtered by band."""
    analysis_data = load_analysis_results()
    
    if point_id not in analysis_data["point_analyses"]:
        raise HTTPException(status_code=404, detail="Point not found")
    
    point_data = analysis_data["point_analyses"][point_id]
    time_periods = point_data["time_periods"]
    
    if band:
        time_periods = [period for period in time_periods if period["band"] == band]
    
    return {
        "id": point_id,
        "band": band,
        "time_periods": time_periods
    }

@app.get("/api/conflicts")
async def get_conflicts():
    """Get list of channel conflicts between points."""
    analysis_data = load_analysis_results()
    return analysis_data["conflicts"]

@app.get("/api/recommendations")
async def get_recommendations():
    """Get channel recommendations for all points."""
    analysis_data = load_analysis_results()
    return analysis_data["final_recommendations"]

@app.get("/api/recommendations/{point_id}")
async def get_point_recommendations(point_id: str):
    """Get channel recommendations for a specific point."""
    analysis_data = load_analysis_results()
    
    if point_id not in analysis_data["final_recommendations"]:
        raise HTTPException(status_code=404, detail="Point not found")
    
    return analysis_data["final_recommendations"][point_id]

@app.get("/api/summary")
async def get_summary():
    """Get a summary of the network analysis."""
    analysis_data = load_analysis_results()
    
    summary = {
        "total_points": len(analysis_data["point_analyses"]),
        "total_conflicts": len(analysis_data["conflicts"]),
        "bands": {
            "2.4 GHz": 0,
            "5 GHz": 0
        },
        "points": []
    }
    
    for point_id, point_data in analysis_data["point_analyses"].items():
        point_summary = {
            "id": point_id,
            "bands": set(),
            "channels": set(),
            "avg_load_score": 0,
            "avg_stability": 0
        }
        
        total_load_score = 0
        total_stability = 0
        period_count = 0
        
        for period in point_data["time_periods"]:
            point_summary["bands"].add(period["band"])
            point_summary["channels"].add(period["channel"])
            total_load_score += period["load_score"]
            total_stability += period["stability"]
            period_count += 1
            
            summary["bands"][period["band"]] += 1
        
        point_summary["avg_load_score"] = total_load_score / period_count
        point_summary["avg_stability"] = total_stability / period_count
        point_summary["bands"] = list(point_summary["bands"])
        point_summary["channels"] = list(point_summary["channels"])
        
        summary["points"].append(point_summary)
    
    return summary

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 