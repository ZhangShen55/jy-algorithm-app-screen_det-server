from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.core.config import BASE_DIR, OcclusionDetectionConfig, get_settings
from app.services.image_preprocess import clamp01, prepare_image
from app.services.yolo_compat import patch_legacy_aattn


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OcclusionDetectResult:
    is_occluded: bool
    occlusion_area_ratio: float
    score: float
    threshold: float
    area_ratio: float
    message: str


class YoloOcclusionModelHolder:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model = None
        self._weights_path: Path | None = None
        self._device: str | int | None = None

    def _resolve_weights(self, config: OcclusionDetectionConfig) -> Path:
        weights = Path(config.yolo_seg_weights_path)
        if not weights.is_absolute():
            weights = BASE_DIR / weights
        return weights

    def _resolve_device(self, config: OcclusionDetectionConfig) -> str | int:
        device = config.yolo_device
        try:
            return int(device)
        except (TypeError, ValueError):
            return str(device)

    def load(self) -> None:
        config = get_settings().occlusion_detection
        weights = self._resolve_weights(config)
        device = self._resolve_device(config)
        if not weights.is_file():
            raise FileNotFoundError(f"YOLO occlusion weights not found: {weights}")

        with self._lock:
            if (
                self._model is not None
                and self._weights_path == weights
                and self._device == device
            ):
                return

            try:
                from ultralytics import YOLO
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "YOLO occlusion dependency missing: install ultralytics and torch"
                ) from exc

            logger.info("Loading occlusion YOLO-seg weights=%s device=%s", weights, device)
            model = YOLO(str(weights))
            patch_legacy_aattn(model)
            self._model = model
            self._weights_path = weights
            self._device = device

    def reset(self) -> None:
        with self._lock:
            self._model = None
            self._weights_path = None
            self._device = None

    @property
    def model(self):
        if self._model is None:
            self.load()
        return self._model

    @property
    def device(self) -> str | int:
        if self._model is None:
            self.load()
        return "cpu" if self._device is None else self._device


_yolo_holder = YoloOcclusionModelHolder()


def _as_numpy_1d(values) -> np.ndarray:
    if values is None:
        return np.array([], dtype=np.float32)
    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "numpy"):
        values = values.numpy()
    return np.asarray(values, dtype=np.float32).reshape(-1)


def _as_mask_array(values) -> np.ndarray:
    if values is None:
        return np.zeros((0, 0, 0), dtype=np.float32)
    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "numpy"):
        values = values.numpy()
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 2:
        array = array[None, :, :]
    if array.ndim != 3:
        return np.zeros((0, 0, 0), dtype=np.float32)
    return array


def _detect_yolo_seg(
    bgr: np.ndarray,
    config: OcclusionDetectionConfig,
    threshold: float,
    area_threshold: float,
) -> OcclusionDetectResult:
    _yolo_holder.load()
    model = _yolo_holder.model
    device = _yolo_holder.device
    yolo_results = model.predict(
        source=bgr,
        imgsz=config.yolo_imgsz,
        conf=threshold,
        device=device,
        verbose=False,
        retina_masks=config.yolo_retina_masks,
    )
    if not isinstance(yolo_results, list):
        yolo_results = [yolo_results]
    if not yolo_results:
        return OcclusionDetectResult(
            False,
            0.0,
            0.0,
            round(threshold, 4),
            round(area_threshold, 4),
            "未检测到镜头遮挡",
        )

    result = yolo_results[0]
    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        return OcclusionDetectResult(
            False,
            0.0,
            0.0,
            round(threshold, 4),
            round(area_threshold, 4),
            "未检测到镜头遮挡",
        )

    masks = _as_mask_array(result.masks.data)
    confs = _as_numpy_1d(result.boxes.conf)
    if masks.size == 0 or confs.size == 0:
        return OcclusionDetectResult(
            False,
            0.0,
            0.0,
            round(threshold, 4),
            round(area_threshold, 4),
            "未检测到镜头遮挡",
        )

    count = min(masks.shape[0], confs.shape[0])
    union: np.ndarray | None = None
    max_conf = 0.0
    for idx in range(count):
        conf = float(confs[idx])
        if conf < threshold:
            continue
        mask = masks[idx] > 0.5
        if mask.size == 0:
            continue
        union = mask if union is None else (union | mask)
        max_conf = max(max_conf, conf)

    if union is None:
        return OcclusionDetectResult(
            False,
            0.0,
            0.0,
            round(threshold, 4),
            round(area_threshold, 4),
            "未检测到镜头遮挡",
        )

    predicted_area_ratio = clamp01(float(union.sum()) / float(union.size))
    is_occluded = predicted_area_ratio >= area_threshold
    if not is_occluded:
        return OcclusionDetectResult(
            False,
            0.0,
            0.0,
            round(threshold, 4),
            round(area_threshold, 4),
            "未检测到镜头遮挡",
        )

    score = clamp01(max_conf)
    return OcclusionDetectResult(
        True,
        round(predicted_area_ratio, 4),
        round(score, 4),
        round(threshold, 4),
        round(area_threshold, 4),
        "检测到镜头遮挡",
    )


def reset_occlusion_yolo_model_cache() -> None:
    _yolo_holder.reset()


def detect_occlusion_from_base64(
    image_base64: str,
    threshold: float | None = None,
    area_ratio: float | None = None,
) -> OcclusionDetectResult:
    settings = get_settings()
    config = settings.occlusion_detection
    threshold_used = config.threshold if threshold is None else threshold
    area_ratio_used = config.area_ratio if area_ratio is None else area_ratio
    if not config.enabled:
        return OcclusionDetectResult(
            False,
            0.0,
            0.0,
            round(threshold_used, 4),
            round(area_ratio_used, 4),
            "镜头遮挡检测未启用",
        )
    if not 0 <= threshold_used <= 1:
        raise ValueError("threshold must be between 0 and 1")
    if not 0 <= area_ratio_used <= 1:
        raise ValueError("area_ratio must be between 0 and 1")

    image = prepare_image(
        image_base64,
        settings.runtime.max_image_bytes,
        config.analyze_max_side,
    )
    return _detect_yolo_seg(image.bgr, config, threshold_used, area_ratio_used)
