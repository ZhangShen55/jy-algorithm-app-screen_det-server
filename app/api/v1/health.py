from __future__ import annotations

import time

import psutil
from fastapi import APIRouter, Response

from app.api.v1.common import format_elapsed
from app.core.config import get_settings
from app.core.state import app_state
from app.services.screen_detector import _holder as screen_model_holder
from app.services.screen_detector import is_screen_model_ready


router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check(response: Response) -> dict:
    settings = get_settings()
    elapsed = time.time() - app_state.start_time
    process = psutil.Process()
    ready = is_screen_model_ready() or not settings.screen_detection.preload_at_startup
    screen_model = screen_model_holder.status

    if settings.screen_detection.preload_at_startup and not ready:
        response.status_code = 503

    return {
        "status": "success" if ready else "not_ready",
        "ready": ready,
        "elapsed_time": format_elapsed(elapsed),
        "total_requests": app_state.request_count,
        "memory_mb": round(process.memory_info().rss / 1024 / 1024, 2),
        "gpu": {
            **settings.gpu.__dict__,
            "tilt_inference_device": "cpu",
            "device_id_config": settings.gpu.device_id,
            "yolo_device_resolved": (
                screen_model_holder.device
                if screen_model["loaded"]
                else screen_model_holder.resolve_device(settings.gpu)
            ),
            "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "screen_model": screen_model,
    }
