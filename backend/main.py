from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pathlib import Path
import json
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import pandas as pd
import os
from backend.routes import points, settings

app = FastAPI(title="Wi-Fi Network Analysis System")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

# Templates
templates = Jinja2Templates(directory="frontend/templates")

# Load analysis results
with open("analysis_results.json", "r") as f:
    analysis_results = json.load(f)

def load_csv_data(point_id: str) -> pd.DataFrame:
    """Load and process CSV data for a specific point."""
    csv_path = Path(__file__).parent / f"{point_id}.csv"
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail=f"CSV file not found for point {point_id}")
    
    df = pd.read_csv(csv_path)
    return df

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the main dashboard page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/points", response_class=HTMLResponse)
async def points_page(request: Request):
    """Serve the points page."""
    return templates.TemplateResponse("points.html", {"request": request})

@app.get("/api/points")
async def get_points():
    """Get list of available points."""
    points = []
    
    # Add dynamic points from analysis
    for point_id, data in analysis_results["point_analyses"].items():
        # Get the latest time period to extract current info
        latest_period = data["time_periods"][-1]
        
        # Get channel load data to extract current metrics
        try:
            df = load_csv_data(point_id)
            latest_data = df.iloc[-1]  # Get the latest record
            
            points.append({
                "id": point_id,
                "name": point_id,
                "band": "2.4 GHz & 5 GHz",  # Both bands
                "is_online": True,
                "clients_count": int(latest_data["total_client_count"]),
                "channel": str(latest_data["channel"]),
                "channel_24": str(latest_data["channel"]),
                "channel_5": "Авто",
                "power": "100",
                "signal_strength": int(float(latest_data["avg_signal_strength"]))  # Round to integer
            })
        except Exception as e:
            points.append({
                "id": point_id,
                "name": point_id,
                "band": "2.4 GHz & 5 GHz",
                "is_online": True,
                "clients_count": 0,
                "channel": latest_period["channel"],
                "channel_24": latest_period["channel"],
                "channel_5": "Авто",
                "power": "100",
                "signal_strength": None
            })
    
    return points

@app.get("/api/points/{point_id}")
async def get_point_details(point_id: str):
    """Get detailed information for a specific point."""
    analysis_data = analysis_results
    
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
async def get_point_metrics(point_id: str):
    """Get metrics for a specific point."""
    if point_id not in analysis_results["point_analyses"]:
        raise HTTPException(status_code=404, detail="Point not found")
    return analysis_results["point_analyses"][point_id]

@app.get("/api/points/{point_id}/channel_load")
async def get_channel_load(point_id: str):
    """Get channel load data from CSV file."""
    try:
        df = load_csv_data(point_id)
        
        # Group by band and channel, then calculate average metrics
        result = {}
        for band in df['band'].unique():
            band_data = df[df['band'] == band]
            channels = {}
            
            for channel in band_data['channel'].unique():
                channel_data = band_data[band_data['channel'] == channel]
                channels[str(channel)] = {
                    "times": channel_data['minutes_since_midnight'].tolist(),
                    "airtime": channel_data['avg_airtime'].tolist(),
                    "signal": channel_data['avg_signal_strength'].tolist(),
                    "clients": channel_data['total_client_count'].tolist(),
                    "retransmissions": channel_data['avg_retransmission_count'].tolist(),
                    "lost_packets": channel_data['avg_lost_packets'].tolist()
                }
            
            result[band] = channels
            
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/conflicts")
async def get_conflicts():
    """Get list of channel conflicts between points."""
    return analysis_results["conflicts"]

@app.get("/api/recommendations")
async def get_recommendations():
    """Get channel recommendations for all points."""
    return analysis_results["final_recommendations"]

@app.get("/api/recommendations/{point_id}")
async def get_point_recommendations(point_id: str):
    """Get channel recommendations for a specific point."""
    analysis_data = analysis_results
    
    if point_id not in analysis_data["final_recommendations"]:
        raise HTTPException(status_code=404, detail="Point not found")
    
    return analysis_data["final_recommendations"][point_id]

@app.get("/api/summary")
async def get_summary():
    """Get network analysis summary."""
    total_points = len(analysis_results["point_analyses"])
    total_conflicts = len(analysis_results["conflicts"])
    
    # Count points by band
    bands = {"2.4 GHz": 0, "5 GHz": 0}
    for point_data in analysis_results["point_analyses"].values():
        first_period = point_data["time_periods"][0]
        bands[first_period["band"]] += 1
    
    return {
        "total_points": total_points,
        "total_conflicts": total_conflicts,
        "bands": bands
    }

# Include routers
app.include_router(points.router)
app.include_router(settings.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 