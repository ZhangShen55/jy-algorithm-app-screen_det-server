from __future__ import annotations

from fastapi import APIRouter

from app.core.config import get_settings, reload_settings


router = APIRouter(tags=["config"])


def _screen_detection_dict() -> dict:
    cfg = get_settings().screen_detection
    return {
        **cfg.__dict__,
        "allowed_class_ids": list(cfg.allowed_class_ids),
    }


@router.get("/config")
async def get_runtime_config() -> dict:
    settings = get_settings()
    return {
        "app": settings.app.__dict__,
        "server": settings.server.__dict__,
        "gpu": settings.gpu.__dict__,
        "detection": settings.detection.__dict__,
        "screen_detection": _screen_detection_dict(),
        "runtime": settings.runtime.__dict__,
    }


@router.post("/config/reload")
async def reload_runtime_config() -> dict:
    settings = reload_settings()
    return {
        "code": 200,
        "msg": "Config reloaded",
        "detection": settings.detection.__dict__,
        "screen_detection": {
            **settings.screen_detection.__dict__,
            "allowed_class_ids": list(settings.screen_detection.allowed_class_ids),
        },
    }
