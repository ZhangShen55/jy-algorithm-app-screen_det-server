from __future__ import annotations

from fastapi import APIRouter

from app.core.config import get_settings, reload_settings
from app.services.occlusion_detector import reset_occlusion_yolo_model_cache


router = APIRouter(tags=["config"])


def _screen_detection_dict() -> dict:
    cfg = get_settings().screen_detection
    return {
        **cfg.__dict__,
        "allowed_class_ids": list(cfg.allowed_class_ids),
    }


def _quality_abnormal_detection_dict() -> dict:
    return get_settings().quality_abnormal_detection.__dict__


def _occlusion_detection_dict() -> dict:
    return get_settings().occlusion_detection.__dict__


@router.get("/config")
async def get_runtime_config() -> dict:
    settings = get_settings()
    return {
        "app": settings.app.__dict__,
        "server": settings.server.__dict__,
        "gpu": settings.gpu.__dict__,
        "detection": settings.detection.__dict__,
        "screen_detection": _screen_detection_dict(),
        "quality_abnormal_detection": _quality_abnormal_detection_dict(),
        "occlusion_detection": _occlusion_detection_dict(),
        "runtime": settings.runtime.__dict__,
    }


@router.post("/config/reload")
async def reload_runtime_config() -> dict:
    settings = reload_settings()
    reset_occlusion_yolo_model_cache()
    return {
        "code": 200,
        "msg": "Config reloaded",
        "detection": settings.detection.__dict__,
        "screen_detection": {
            **settings.screen_detection.__dict__,
            "allowed_class_ids": list(settings.screen_detection.allowed_class_ids),
        },
        "quality_abnormal_detection": settings.quality_abnormal_detection.__dict__,
        "occlusion_detection": settings.occlusion_detection.__dict__,
    }
