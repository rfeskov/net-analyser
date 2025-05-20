import json
import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

router = APIRouter()

SETTINGS_FILE = "frontend/static/data/point_settings.json"

class SettingsData(BaseModel):
    pointId: str
    settings: Dict[str, Any]

def ensure_settings_file():
    if not os.path.exists(SETTINGS_FILE):
        os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
        with open(SETTINGS_FILE, 'w') as f:
            json.dump({"points": {}}, f)

@router.post("/api/settings/save")
async def save_settings(data: SettingsData):
    try:
        ensure_settings_file()
        
        # Read current settings
        with open(SETTINGS_FILE, 'r') as f:
            settings = json.load(f)
        
        # Update settings for the point
        settings["points"][data.pointId] = data.settings
        
        # Save updated settings
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=4)
        
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/settings/load")
async def load_settings():
    try:
        ensure_settings_file()
        
        # Read settings
        with open(SETTINGS_FILE, 'r') as f:
            settings = json.load(f)
        
        return settings
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 