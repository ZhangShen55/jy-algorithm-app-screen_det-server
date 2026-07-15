from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.core.config import BASE_DIR, GpuConfig, ScreenDetectionConfig, get_settings
from app.services.tilt_detector import decode_base64_image
from app.services.yolo_compat import patch_legacy_aattn


logger = logging.getLogger(__name__)

ALLOWED_LABELS = frozenset({0, 1, 2, 3})


@dataclass(frozen=True)
class ScreenBoxResult:
    label: int
    confidence: float
    box: list[float]


@dataclass(frozen=True)
class ScreenImageDetectResult:
    index: int
    cost_ms: float
    primary: ScreenBoxResult | None
    detections: list[ScreenBoxResult]


class ScreenModelHolder:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model = None
        self._device: str | int = "cpu"
        self._weights_path: Path | None = None
        self._aattn_patched = 0
        self._warmed_up = False
        self._gpu_memory_mb: float | None = None

    def resolve_device(self, gpu: GpuConfig) -> str | int:
        if not gpu.enabled:
            return "cpu"

        import torch

        if not torch.cuda.is_available():
            if gpu.require_gpu:
                raise RuntimeError("GPU is enabled but CUDA is not available in this container/process")
            return "cpu"

        visible = torch.cuda.device_count()
        if visible <= 0:
            if gpu.require_gpu:
                raise RuntimeError("GPU is enabled but no CUDA device is visible")
            return "cpu"

        try:
            requested = int(gpu.device_id)
        except ValueError:
            return gpu.device_id

        # docker run --gpus device=N 时容器内仅 1 张卡，逻辑编号恒为 cuda:0
        if visible == 1:
            if requested != 0:
                logger.warning(
                    "config device_id=%s but container exposes 1 GPU; using cuda:0",
                    gpu.device_id,
                )
            return 0

        if requested < 0 or requested >= visible:
            raise RuntimeError(
                f"GPU device_id={requested} is out of range, visible device count={visible}"
            )
        return requested

    def _warmup_on_device(self, model, device: str | int) -> None:
        import torch

        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        logger.info("Screen YOLO GPU warmup begin device=%s", device)
        model.predict(source=dummy, device=device, verbose=False)
        self._warmed_up = True
        if isinstance(device, int) and torch.cuda.is_available():
            self._gpu_memory_mb = round(torch.cuda.memory_allocated(device) / 1024 / 1024, 2)
            logger.info(
                "Screen YOLO GPU warmup done device=%s gpu_memory_mb=%s",
                device,
                self._gpu_memory_mb,
            )

    def load(self, force: bool = False) -> None:
        settings = get_settings()
        weights = Path(settings.screen_detection.weights_path)
        if not weights.is_absolute():
            weights = BASE_DIR / weights
        device = self.resolve_device(settings.gpu)

        if not weights.is_file():
            raise FileNotFoundError(f"Screen YOLO weights not found: {weights}")

        with self._lock:
            if (
                not force
                and self._model is not None
                and self._weights_path == weights
                and self._device == device
                and self._warmed_up
            ):
                return

            from ultralytics import YOLO

            logger.info("Loading screen YOLO weights=%s device=%s", weights, device)
            model = YOLO(str(weights))
            patched = patch_legacy_aattn(model)
            if device != "cpu":
                self._warmup_on_device(model, device)
            self._model = model
            self._device = device
            self._weights_path = weights
            self._aattn_patched = patched
            logger.info("Screen YOLO loaded, AAttn patched=%s", patched)

    @property
    def is_ready(self) -> bool:
        settings = get_settings()
        if not settings.screen_detection.preload_at_startup:
            return True
        return self._model is not None and (self._device == "cpu" or self._warmed_up)

    @property
    def model(self):
        if self._model is None:
            self.load()
        return self._model

    @property
    def device(self) -> str | int:
        if self._model is None:
            self.load()
        return self._device

    @property
    def status(self) -> dict:
        import torch

        device_name = None
        if self._model is not None and isinstance(self._device, int) and torch.cuda.is_available():
            try:
                device_name = torch.cuda.get_device_name(self._device)
            except Exception:
                device_name = None

        return {
            "loaded": self._model is not None,
            "warmed_up": self._warmed_up,
            "weights": str(self._weights_path) if self._weights_path else None,
            "device": self._device,
            "device_name": device_name,
            "gpu_memory_mb": self._gpu_memory_mb,
            "aattn_patched": self._aattn_patched,
        }


_holder = ScreenModelHolder()


def ensure_screen_model_loaded() -> dict:
    settings = get_settings()
    if not settings.screen_detection.preload_at_startup:
        return _holder.status

    _holder.load()
    status = _holder.status
    if not status["loaded"]:
        raise RuntimeError("Screen YOLO preload finished but model is not loaded")
    if status["device"] != "cpu" and not status.get("warmed_up"):
        raise RuntimeError(f"Screen YOLO preload finished but GPU warmup incomplete: {status}")
    return status


def is_screen_model_ready() -> bool:
    return _holder.is_ready


def _parse_boxes(result, allowed: frozenset[int]) -> list[ScreenBoxResult]:
    if result.boxes is None or len(result.boxes) == 0:
        return []
    items: list[ScreenBoxResult] = []
    boxes = result.boxes
    for i in range(len(boxes)):
        label = int(boxes.cls[i].item())
        if label not in allowed:
            continue
        items.append(
            ScreenBoxResult(
                label=label,
                confidence=round(float(boxes.conf[i].item()), 4),
                box=[round(float(v), 1) for v in boxes.xyxy[i].tolist()],
            )
        )
    items.sort(key=lambda item: item.confidence, reverse=True)
    return items


def detect_screen_from_base64_list(
    images_base64: list[str],
    conf: float | None = None,
    iou: float | None = None,
) -> tuple[list[ScreenImageDetectResult], float, float]:
    settings = get_settings()
    screen_cfg = settings.screen_detection
    runtime = settings.runtime
    conf_used = screen_cfg.conf if conf is None else conf
    iou_used = screen_cfg.iou if iou is None else iou
    allowed = frozenset(screen_cfg.allowed_class_ids)

    if len(images_base64) > screen_cfg.max_batch_size:
        raise ValueError(
            f"Too many images: {len(images_base64)} > max_batch_size={screen_cfg.max_batch_size}"
        )

    decoded: list[np.ndarray] = []
    for index, image_base64 in enumerate(images_base64):
        try:
            decoded.append(decode_base64_image(image_base64, runtime.max_image_bytes))
        except ValueError as exc:
            raise ValueError(f"images[{index}]: {exc}") from exc

    return detect_screen_from_arrays(decoded, conf=conf_used, iou=iou_used)


def detect_screen_from_arrays(
    images: list[np.ndarray],
    conf: float,
    iou: float,
    device: str | int | None = None,
) -> tuple[list[ScreenImageDetectResult], float, float]:
    settings = get_settings()
    screen_cfg = settings.screen_detection
    allowed = frozenset(screen_cfg.allowed_class_ids)

    if len(images) > screen_cfg.max_batch_size:
        raise ValueError(
            f"Too many images: {len(images)} > max_batch_size={screen_cfg.max_batch_size}"
        )

    _holder.load()
    model = _holder.model
    device_used = _holder.device if device is None else device

    results: list[ScreenImageDetectResult] = []
    if len(images) == 1:
        predict_inputs: list[np.ndarray] | np.ndarray = images[0]
    else:
        predict_inputs = images

    start = time.time()
    yolo_results = model.predict(
        predict_inputs,
        conf=conf,
        iou=iou,
        device=device_used,
        verbose=False,
    )
    batch_elapsed_ms = (time.time() - start) * 1000

    if not isinstance(yolo_results, list):
        yolo_results = [yolo_results]

    per_image_ms = round(batch_elapsed_ms / max(len(yolo_results), 1), 2)
    for index, yolo_result in enumerate(yolo_results):
        detections = _parse_boxes(yolo_result, allowed)
        primary = detections[0] if detections else None
        results.append(
            ScreenImageDetectResult(
                index=index,
                cost_ms=per_image_ms,
                primary=primary,
                detections=detections,
            )
        )
        if primary:
            logger.info(
                "screen_detect index=%s label=%s conf=%.3f boxes=%s",
                index,
                primary.label,
                primary.confidence,
                len(detections),
            )
        else:
            logger.info("screen_detect index=%s no allowed detections", index)

    return results, conf, iou
