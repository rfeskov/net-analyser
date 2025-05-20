from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import json

router = APIRouter()

@router.get("/api/points/{point_id}/settings")
async def get_point_settings(point_id: str):
    """Get settings for a specific point."""
    try:
        with open("frontend/static/data/point_settings.json", 'r') as f:
            settings = json.load(f)
        return settings["points"].get(point_id, {})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 