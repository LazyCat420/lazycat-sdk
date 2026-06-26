import time
from fastapi import APIRouter

router = APIRouter()

# Capture start time when this module is imported
_START_TIME = time.time()

# Services can override these values if they want
SERVICE_NAME = "unknown_service"
SERVICE_VERSION = "0.0.0"

def configure_health(name: str, version: str):
    global SERVICE_NAME, SERVICE_VERSION
    SERVICE_NAME = name
    SERVICE_VERSION = version

@router.get("/health", tags=["Health"])
async def health_check():
    """Standardized health check endpoint for all LazyCat services."""
    uptime_seconds = int(time.time() - _START_TIME)
    
    return {
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "status": "healthy",
        "uptime_seconds": uptime_seconds,
        "uptime_display": f"{uptime_seconds // 3600}h {(uptime_seconds % 3600) // 60}m {uptime_seconds % 60}s"
    }
