from __future__ import annotations

import logging
import time
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from app.api.v1.router import router as v1_router
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.services.screen_detector import ensure_screen_model_loaded


settings = get_settings()
setup_logging(settings.logging)

logger = logging.getLogger(__name__)
access_logger = logging.getLogger("tilt.access")

app = FastAPI(
    title=settings.app.name,
    version=settings.app.version,
    debug=settings.app.debug,
)


@app.on_event("startup")
async def preload_screen_model() -> None:
    settings = get_settings()
    if not settings.screen_detection.preload_at_startup:
        logger.info("Screen YOLO startup preload disabled by config")
        return

    import torch

    cuda_visible = __import__("os").environ.get("CUDA_VISIBLE_DEVICES", "(not set)")
    logger.info(
        "Screen YOLO startup preload begin config_device_id=%s cuda_visible=%s cuda_count=%s",
        settings.gpu.device_id,
        cuda_visible,
        torch.cuda.device_count() if torch.cuda.is_available() else 0,
    )
    status = await run_in_threadpool(ensure_screen_model_loaded)
    if not status.get("loaded") or (status.get("device") != "cpu" and not status.get("warmed_up")):
        raise RuntimeError(f"Screen YOLO startup preload failed: {status}")
    logger.info("Screen YOLO startup preload complete: %s", status)


@app.middleware("http")
async def request_log_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid4()))
    start_time = time.time()
    try:
        response = await call_next(request)
    except Exception as exc:
        cost_ms = round((time.time() - start_time) * 1000, 2)
        access_logger.exception(
            "request_id=%s method=%s path=%s status=500 cost_ms=%s error=%s",
            request_id,
            request.method,
            request.url.path,
            cost_ms,
            exc,
        )
        raise

    cost_ms = round((time.time() - start_time) * 1000, 2)
    response.headers["X-Request-ID"] = request_id
    access_logger.info(
        "request_id=%s method=%s path=%s status=%s cost_ms=%s client=%s",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        cost_ms,
        request.client.host if request.client else "-",
    )
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error for %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"code": 500, "msg": "Internal server error"},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    if isinstance(exc.detail, dict) and {"code", "msg"} <= set(exc.detail):
        content = exc.detail
    else:
        content = {"code": exc.status_code, "msg": str(exc.detail)}
    return JSONResponse(status_code=exc.status_code, content=content)


app.include_router(v1_router, prefix=settings.app.api_prefix)
app.include_router(v1_router)


@app.get("/")
async def root() -> dict:
    return {
        "service": settings.app.name,
        "version": settings.app.version,
        "health": f"{settings.app.api_prefix}/health",
        "detect_tilt": f"{settings.app.api_prefix}/detect_tilt",
        "detect_screen": f"{settings.app.api_prefix}/detect_screen",
        "detect_inspect": f"{settings.app.api_prefix}/detect_inspect",
        "detect_all": f"{settings.app.api_prefix}/detect_all",
        "detect_quality_abnormal": f"{settings.app.api_prefix}/detect_quality_abnormal",
        "detect_occlusion": f"{settings.app.api_prefix}/detect_occlusion",
    }
