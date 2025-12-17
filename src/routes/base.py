from fastapi import APIRouter
from utils import get_settings, Settings
from fastapi import Depends

router = APIRouter(
    prefix="/api/v1",
    tags=["api_v1"],
)


@router.get("/")
async def welcome_message(settings: Settings = Depends(get_settings)):
    return {
        "app_name": settings.APP_NAME,
        "status": "running",
        "version": settings.APP_VERSION,
    }



