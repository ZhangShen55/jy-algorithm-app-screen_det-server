from __future__ import annotations

from fastapi import APIRouter

from app.api.v1 import config, health, screen, tilt

router = APIRouter()
router.include_router(tilt.router)
router.include_router(screen.router)
router.include_router(health.router)
router.include_router(config.router)
