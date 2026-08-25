"""CV-based identification of microalgae in bright-field images."""

from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np

@dataclass(frozen=True)
class DetectionParams:

    adaptive_block_size: int
    adaptive_c: int
    diameter_min: float
    diameter_max: float
    major_min: float
    major_max: float
    minor_min: float
    minor_max: float
    ratio_max: float
    area_min: float
    area_max: float
    morph_close_size: int = 0
    morph_open_size: int = 0

@dataclass(frozen=True)
class Detection:

    x: float
    y: float
    diameter: float
    major: float
    minor: float
    angle: float
    contour_area: float
    source: str
    circularity: float | None = None
    fill_ratio: float | None = None
    local_contrast: float | None = None
    contour: tuple[tuple[int, int], ...] | None = None

@dataclass(frozen=True)
class SplitParams:

    peak_threshold_ratio: float = 0.70
    min_peak_radius: float = 3.0
    min_center_distance: float = 6.0
    pad: int = 3

@dataclass(frozen=True)
class FaintRecoveryParams:

    percentile: float = 99.4
    merge_distance: float = 12.0
    candidate_area_min: float = 45.0
    candidate_area_max: float = 1250.0
    candidate_diameter_min: float = 10.0
    candidate_diameter_max: float = 48.0
    candidate_major_min: float = 10.0
    candidate_major_max: float = 48.0
    candidate_minor_min: float = 6.0
    candidate_minor_max: float = 37.0
    candidate_ratio_max: float = 3.1
    candidate_circularity_min: float = 0.28
    final_area_min: float = 65.0
    final_diameter_min: float = 12.0
    final_diameter_max: float = 45.5
    faint_major_min: float = 20.0
    background_area_min: float = 350.0
    background_area_max: float = 380.0
    background_contrast_min: float = 14.0
    background_contrast_max: float = 19.0
    background_circularity_max: float = 0.70
    right_border_margin: float = 100.0
    right_border_fill_min: float = 0.82
    split_circularity_min: float = 0.50
    split_fill_min: float = 0.75
    split_local_contrast_min: float = 6.0

DUNALIELLA_IDENTIFICATION_PARAMS = DetectionParams(
    adaptive_block_size=251,
    adaptive_c=5,
    diameter_min=7.5,
    diameter_max=45.5,
    major_min=7.0,
    major_max=45.0,
    minor_min=5.0,
    minor_max=37.0,
    ratio_max=5.0,
    area_min=280.0,
    area_max=1150.0,
    morph_close_size=0,
    morph_open_size=0,
)

DUNALIELLA_FAINT_RECOVERY_PARAMS = FaintRecoveryParams()

def _validate_params(params: DetectionParams) -> None:
    if params.adaptive_block_size <= 1 or params.adaptive_block_size % 2 == 0:
        raise ValueError("adaptive_block_size must be an odd integer greater than 1")
    if params.diameter_min >= params.diameter_max:
        raise ValueError("diameter_min must be smaller than diameter_max")
    if params.major_min >= params.major_max:
        raise ValueError("major_min must be smaller than major_max")
    if params.minor_min >= params.minor_max:
        raise ValueError("minor_min must be smaller than minor_max")
    if params.area_min >= params.area_max:
        raise ValueError("area_min must be smaller than area_max")

def prepare_image(image: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:

    if image is None or image.size == 0:
        raise ValueError("image must be a non-empty NumPy array")

    if image.dtype != np.uint8:
        finite = np.asarray(image, dtype=np.float32)
        finite = np.nan_to_num(finite, nan=0.0, posinf=255.0, neginf=0.0)
        minimum = float(finite.min())
        maximum = float(finite.max())
        if maximum > minimum:
            image = np.clip((finite - minimum) * 255.0 / (maximum - minimum), 0, 255).astype(np.uint8)
        else:
            image = np.zeros(finite.shape, dtype=np.uint8)

    if image.ndim == 2:
        return image, None
    if image.ndim != 3:
        raise ValueError("image must be a 2-D grayscale or 3-D BGR array")

    if image.shape[2] == 4:
        color = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif image.shape[2] == 3:
        color = image
    elif image.shape[2] == 1:
        return image[:, :, 0], None
    else:
        raise ValueError("color image must have 1, 3, or 4 channels")

    return cv2.cvtColor(color, cv2.COLOR_BGR2GRAY), color

def build_cv_binary(gray: np.ndarray, params: DetectionParams) -> np.ndarray:

    _validate_params(params)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        params.adaptive_block_size,
        params.adaptive_c,
    )

    if params.morph_close_size > 0:
        size = int(params.morph_close_size)
        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)),
        )

    if params.morph_open_size > 0:
        size = int(params.morph_open_size)
        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)),
        )

    return binary

def _contour_to_points(contour: np.ndarray) -> tuple[tuple[int, int], ...]:
    points = contour.reshape(-1, 2)
    return tuple((int(x), int(y)) for x, y in points)

def classify_contour(
    contour: np.ndarray,
    params: DetectionParams,
    source: str = "cv",
) -> tuple[Detection | None, str]:

    if len(contour) < 5:
        return None, "too_few_points"

    (_, _), radius = cv2.minEnclosingCircle(contour)
    diameter = 2.0 * radius
    if diameter <= params.diameter_min or diameter > params.diameter_max:
        return None, "diameter"

    (x, y), (axis_a, axis_b), angle = cv2.fitEllipse(contour)
    if axis_a >= axis_b:
        major_axis = axis_a
        minor_axis = axis_b
    else:
        major_axis = axis_b
        minor_axis = axis_a
        angle = (angle + 90.0) % 180.0

    if minor_axis <= 0:
        return None, "axis"

    ratio = major_axis / minor_axis
    if ratio >= params.ratio_max:
        return None, "ratio"

    if major_axis < params.major_min or major_axis > params.major_max:
        return None, "axis"
    if minor_axis < params.minor_min or minor_axis > params.minor_max:
        return None, "axis"

    contour_area = float(cv2.contourArea(contour))
    if contour_area < params.area_min or contour_area > params.area_max:
        return None, "area"

    return (
        Detection(
            x=float(x),
            y=float(y),
            diameter=float(diameter),
            major=float(major_axis),
            minor=float(minor_axis),
            angle=float(angle),
            contour_area=contour_area,
            source=source,
            contour=_contour_to_points(contour),
        ),
        "accepted",
    )

def _mask_object_features(
    gray: np.ndarray,
    contour: np.ndarray,
    detection: Detection,
    annulus_kernel_size: int = 9,
) -> dict[str, float]:
    perimeter = cv2.arcLength(contour, True)
    circularity = (
        0.0
        if perimeter == 0
        else 4.0 * np.pi * detection.contour_area / (perimeter * perimeter)
    )

    ellipse_area = np.pi * (detection.major / 2.0) * (detection.minor / 2.0)
    fill_ratio = 0.0 if ellipse_area == 0 else detection.contour_area / ellipse_area

    x, y, width, height = cv2.boundingRect(contour)
    pad = max(annulus_kernel_size, 4)
    image_height, image_width = gray.shape[:2]
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(image_width, x + width + pad)
    y1 = min(image_height, y + height + pad)

    roi = gray[y0:y1, x0:x1]
    local_contour = contour.copy()
    local_contour[:, :, 0] -= x0
    local_contour[:, :, 1] -= y0

    object_mask = np.zeros(roi.shape, dtype=np.uint8)
    cv2.drawContours(object_mask, [local_contour], -1, 255, -1)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (annulus_kernel_size, annulus_kernel_size),
    )
    outer_mask = cv2.subtract(cv2.dilate(object_mask, kernel, iterations=1), object_mask)

    object_pixels = roi[object_mask > 0]
    outer_pixels = roi[outer_mask > 0]
    object_mean = float(object_pixels.mean()) if object_pixels.size else 0.0
    object_std = float(object_pixels.std()) if object_pixels.size else 0.0
    outer_mean = float(outer_pixels.mean()) if outer_pixels.size else object_mean

    return {
        "contour_area": float(detection.contour_area),
        "circularity": float(circularity),
        "fill_ratio": float(fill_ratio),
        "local_contrast": float(outer_mean - object_mean),
        "object_std": object_std,
    }

def _has_color_information(color: np.ndarray | None) -> bool:
    if color is None or color.ndim != 3:
        return False
    channel_range = np.max(color, axis=2).astype(np.int16) - np.min(color, axis=2).astype(np.int16)
    return float(channel_range.mean()) > 1.0

def _color_saturation_channel(color: np.ndarray | None) -> np.ndarray | None:
    if color is None or color.ndim != 3:
        return None
    return cv2.cvtColor(color, cv2.COLOR_BGR2HSV)[:, :, 1]

def _contour_mean_saturation_from_channel(
    saturation_channel: np.ndarray,
    contour: np.ndarray,
) -> float:
    x, y, width, height = cv2.boundingRect(contour)
    if width <= 0 or height <= 0:
        return 0.0
    roi = saturation_channel[y:y + height, x:x + width]
    local_contour = contour.copy()
    local_contour[:, :, 0] -= x
    local_contour[:, :, 1] -= y
    mask = np.zeros(roi.shape, dtype=np.uint8)
    cv2.drawContours(mask, [local_contour], -1, 255, -1)
    values = roi[mask > 0]
    return float(values.mean()) if values.size else 0.0

def _contour_mean_saturation(color: np.ndarray, contour: np.ndarray) -> float:
    saturation_channel = _color_saturation_channel(color)
    if saturation_channel is None:
        return 0.0
    return _contour_mean_saturation_from_channel(saturation_channel, contour)

def _is_large_rgb_debris_like(
    detection: Detection,
    features: dict[str, float],
    saturation: float,
    params: DetectionParams,
) -> bool:

    scale = params.major_max / 19.0
    aspect_ratio = detection.major / max(detection.minor, 1e-6)
    return (
        detection.contour_area <= 78.8 * scale * scale
        and detection.major < 13.4 * scale
        and detection.diameter < 13.4 * scale
        and detection.minor < 8.5 * scale
        and aspect_ratio >= 1.55
        and features["circularity"] < 0.80
        and features["local_contrast"] < 48
        and saturation < 70
        and features["object_std"] < 34
    )

def _is_cv_artifact(
    gray: np.ndarray,
    color: np.ndarray | None,
    contour: np.ndarray,
    detection: Detection,
    params: DetectionParams,
    has_color: bool,
    saturation_channel: np.ndarray | None,
) -> bool:
    features = _mask_object_features(gray, contour, detection)
    saturation = (
        _contour_mean_saturation_from_channel(saturation_channel, contour)
        if has_color and saturation_channel is not None
        else 255.0
    )

    if _is_large_rgb_debris_like(detection, features, saturation, params):
        return True
    if has_color and features["local_contrast"] < 12 and saturation < 25:
        return True
    if features["circularity"] < 0.55 and features["local_contrast"] < 8:
        return True
    if features["circularity"] < 0.35 and features["fill_ratio"] < 0.78:
        return True
    return False

def _normalize_u8(image: np.ndarray) -> np.ndarray:
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def _detect_faint_round_candidates(
    gray: np.ndarray,
    params: FaintRecoveryParams,
) -> tuple[list[Detection], dict[str, int]]:

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    background = cv2.GaussianBlur(clahe, (0, 0), 15)
    bright_local = cv2.subtract(clahe, background)
    dark_local = cv2.subtract(background, clahe)
    grad_x = cv2.Sobel(clahe, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(clahe, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(grad_x, grad_y)
    local_ring = np.maximum(_normalize_u8(bright_local), _normalize_u8(dark_local))
    score = (
        0.55 * local_ring.astype(np.float32)
        + 0.45 * _normalize_u8(gradient).astype(np.float32)
    ).astype(np.uint8)

    threshold = float(np.percentile(score, params.percentile))
    mask = (score >= threshold).astype(np.uint8) * 255
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    height, width = gray.shape[:2]
    detections: list[Detection] = []
    reject_counts: dict[str, int] = {}

    for contour in contours:
        if len(contour) < 5:
            reject_counts["too_few_points"] = reject_counts.get("too_few_points", 0) + 1
            continue

        area = float(cv2.contourArea(contour))
        if not (params.candidate_area_min <= area <= params.candidate_area_max):
            reject_counts["area"] = reject_counts.get("area", 0) + 1
            continue

        (_, _), radius = cv2.minEnclosingCircle(contour)
        diameter = float(2.0 * radius)
        if not (
            params.candidate_diameter_min
            <= diameter
            <= params.candidate_diameter_max
        ):
            reject_counts["diameter"] = reject_counts.get("diameter", 0) + 1
            continue

        (x, y), (axis_a, axis_b), angle = cv2.fitEllipse(contour)
        major_axis = float(max(axis_a, axis_b))
        minor_axis = float(min(axis_a, axis_b))
        if axis_b > axis_a:
            angle = (angle + 90.0) % 180.0
        aspect_ratio = major_axis / max(minor_axis, 1e-6)
        if not (
            params.candidate_major_min
            <= major_axis
            <= params.candidate_major_max
            and params.candidate_minor_min
            <= minor_axis
            <= params.candidate_minor_max
            and aspect_ratio <= params.candidate_ratio_max
        ):
            reject_counts["shape"] = reject_counts.get("shape", 0) + 1
            continue

        perimeter = cv2.arcLength(contour, True)
        circularity = float(4.0 * np.pi * area / max(perimeter * perimeter, 1.0))
        if circularity < params.candidate_circularity_min:
            reject_counts["circularity"] = reject_counts.get("circularity", 0) + 1
            continue

        object_mask = np.zeros_like(gray)
        cv2.drawContours(object_mask, [contour], -1, 255, -1)
        x0, y0, box_width, box_height = cv2.boundingRect(contour)
        pad = 10
        x1 = max(0, x0 - pad)
        y1 = max(0, y0 - pad)
        x2 = min(width, x0 + box_width + pad)
        y2 = min(height, y0 + box_height + pad)
        ring_mask = np.zeros_like(gray)
        cv2.rectangle(ring_mask, (x1, y1), (x2 - 1, y2 - 1), 255, -1)
        ring_mask[object_mask > 0] = 0
        object_values = score[object_mask > 0]
        ring_values = score[ring_mask > 0]
        if object_values.size == 0 or ring_values.size == 0:
            reject_counts["empty"] = reject_counts.get("empty", 0) + 1
            continue
        object_score = float(np.percentile(object_values, 75))
        ring_score = float(np.percentile(ring_values, 75))
        if object_score < threshold + 2 or object_score - ring_score < 5:
            reject_counts["weak"] = reject_counts.get("weak", 0) + 1
            continue

        density_pad = 28
        density_x1 = max(0, x0 - density_pad)
        density_y1 = max(0, y0 - density_pad)
        density_x2 = min(width, x0 + box_width + density_pad)
        density_y2 = min(height, y0 + box_height + density_pad)
        density_mask = np.zeros_like(gray)
        cv2.rectangle(
            density_mask,
            (density_x1, density_y1),
            (density_x2 - 1, density_y2 - 1),
            255,
            -1,
        )
        density_mask[object_mask > 0] = 0
        if np.any(density_mask > 0):
            local_density = float(np.mean(score[density_mask > 0] >= threshold))
            if local_density > 0.04:
                reject_counts["dense_background"] = (
                    reject_counts.get("dense_background", 0) + 1
                )
                continue

        detections.append(
            Detection(
                x=float(x),
                y=float(y),
                diameter=diameter,
                major=major_axis,
                minor=minor_axis,
                angle=float(angle),
                contour_area=area,
                source="cv_faint_recovered",
                circularity=circularity,
                contour=_contour_to_points(contour),
            )
        )

    return detections, reject_counts

def _merge_recovered_detections(
    detections: list[Detection],
    recovered: list[Detection],
    min_distance: float,
) -> tuple[list[Detection], list[Detection]]:
    merged = list(detections)
    added: list[Detection] = []
    min_distance_sq = min_distance * min_distance

    for candidate in recovered:
        if any(
            (candidate.x - existing.x) ** 2 + (candidate.y - existing.y) ** 2
            < min_distance_sq
            for existing in merged
        ):
            continue
        merged.append(candidate)
        added.append(candidate)

    return merged, added

def _enrich_detection_features(
    gray: np.ndarray,
    detections: list[Detection],
) -> list[Detection]:
    enriched: list[Detection] = []
    for detection in detections:
        if detection.contour is None or len(detection.contour) < 3:
            enriched.append(detection)
            continue
        contour = np.array(detection.contour, dtype=np.int32).reshape(-1, 1, 2)
        features = _mask_object_features(gray, contour, detection)
        enriched.append(
            replace(
                detection,
                circularity=features["circularity"],
                fill_ratio=features["fill_ratio"],
                local_contrast=features["local_contrast"],
            )
        )
    return enriched

def _feature(detection: Detection, name: str, default: float = 0.0) -> float:
    value = getattr(detection, name, None)
    return default if value is None else float(value)

def _is_faint_background(
    detection: Detection,
    image_width: int,
    params: FaintRecoveryParams,
) -> bool:
    if detection.source != "cv_faint_recovered":
        return False

    low_contrast_irregular = (
        params.background_area_min
        <= detection.contour_area
        <= params.background_area_max
        and params.background_contrast_min
        <= _feature(detection, "local_contrast")
        <= params.background_contrast_max
        and _feature(detection, "circularity") < params.background_circularity_max
    )
    right_border_artifact = (
        float(image_width) - detection.x < params.right_border_margin
        and _feature(detection, "fill_ratio") >= params.right_border_fill_min
    )
    return low_contrast_irregular or right_border_artifact

def _is_low_quality_split(
    detection: Detection,
    params: FaintRecoveryParams,
) -> bool:
    return detection.source == "cv_split_added" and (
        _feature(detection, "circularity") < params.split_circularity_min
        or _feature(detection, "fill_ratio") < params.split_fill_min
        or _feature(detection, "local_contrast") < params.split_local_contrast_min
    )

def _find_two_distance_peaks(
    contour: np.ndarray,
    split_params: SplitParams,
) -> list[tuple[float, float, float]]:
    x, y, width, height = cv2.boundingRect(contour)
    pad = split_params.pad
    local_contour = contour.copy()
    local_contour[:, :, 0] -= x - pad
    local_contour[:, :, 1] -= y - pad

    mask = np.zeros((height + 2 * pad, width + 2 * pad), dtype=np.uint8)
    cv2.drawContours(mask, [local_contour], -1, 255, -1)
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    max_distance = float(distance.max())
    if max_distance < split_params.min_peak_radius:
        return []

    dilated = cv2.dilate(
        distance,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    peaks = (
        (distance >= dilated - 1e-6)
        & (distance >= max_distance * split_params.peak_threshold_ratio)
        & (distance >= split_params.min_peak_radius)
    )
    component_count, labels = cv2.connectedComponents(peaks.astype(np.uint8), 8)
    candidates: list[tuple[float, float, float]] = []

    for component_index in range(1, component_count):
        ys, xs = np.where(labels == component_index)
        if len(xs) == 0:
            continue
        values = distance[ys, xs]
        best_index = int(np.argmax(values))
        candidates.append(
            (
                float(xs[best_index] + x - pad),
                float(ys[best_index] + y - pad),
                float(values[best_index]),
            )
        )

    candidates.sort(key=lambda item: item[2], reverse=True)
    selected: list[tuple[float, float, float]] = []
    min_distance_sq = split_params.min_center_distance ** 2
    for peak_x, peak_y, value in candidates:
        if all(
            (peak_x - old_x) ** 2 + (peak_y - old_y) ** 2 >= min_distance_sq
            for old_x, old_y, _ in selected
        ):
            selected.append((peak_x, peak_y, value))
        if len(selected) == 2:
            break
    return selected

def _resolve_oversplit_contour(
    contour: np.ndarray,
    detections: list[Detection],
    params: DetectionParams,
    gray: np.ndarray,
    color: np.ndarray | None,
) -> list[Detection] | None:
    if len(detections) != 2:
        return None

    first, second = detections
    center_distance = float(np.hypot(first.x - second.x, first.y - second.y))
    mean_diameter = max(1.0, (first.diameter + second.diameter) / 2.0)
    if center_distance / mean_diameter >= 0.78:
        return None

    relaxed_params = replace(
        params,
        diameter_max=max(params.diameter_max * 1.45, params.diameter_max + 10),
        major_min=0.0,
        major_max=max(params.major_max * 1.6, params.major_max + 10),
        minor_max=max(params.minor_max * 2.0, params.minor_max + 20),
        area_max=params.area_max * 4.0,
    )
    merged, _ = classify_contour(contour, relaxed_params, source="cv_split_merged")
    if merged is None:
        return None

    feature_kernel_size = max(9, int(round(mean_diameter * 0.55)))
    if feature_kernel_size % 2 == 0:
        feature_kernel_size += 1
    features = _mask_object_features(
        gray,
        contour,
        merged,
        annulus_kernel_size=feature_kernel_size,
    )
    saturation = _contour_mean_saturation(color, contour) if _has_color_information(color) else 255.0

    if (
        features["fill_ratio"] >= 0.85
        and features["local_contrast"] >= 2
        and saturation >= 35
    ):
        return [
            replace(
                merged,
                circularity=features["circularity"],
                fill_ratio=features["fill_ratio"],
                local_contrast=features["local_contrast"],
            )
        ]

    if (
        features["fill_ratio"] < 0.76
        or features["circularity"] < 0.20
        or (saturation < 35 and features["local_contrast"] < 24)
    ):
        return []
    return None

def _split_dumbbell_contour(
    contour: np.ndarray,
    params: DetectionParams,
    split_params: SplitParams,
    gray: np.ndarray,
    color: np.ndarray | None,
) -> list[Detection]:
    peaks = _find_two_distance_peaks(contour, split_params)
    if len(peaks) != 2:
        return []

    x, y, width, height = cv2.boundingRect(contour)
    pad = split_params.pad
    local_contour = contour.copy()
    local_contour[:, :, 0] -= x - pad
    local_contour[:, :, 1] -= y - pad

    mask = np.zeros((height + 2 * pad, width + 2 * pad), dtype=np.uint8)
    cv2.drawContours(mask, [local_contour], -1, 255, -1)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return []

    centers = np.array(
        [
            [peaks[0][0] - (x - pad), peaks[0][1] - (y - pad)],
            [peaks[1][0] - (x - pad), peaks[1][1] - (y - pad)],
        ],
        dtype=np.float32,
    )
    distance_0 = (xs - centers[0, 0]) ** 2 + (ys - centers[0, 1]) ** 2
    distance_1 = (xs - centers[1, 0]) ** 2 + (ys - centers[1, 1]) ** 2
    labels = (distance_1 < distance_0).astype(np.uint8)

    detections: list[Detection] = []
    for label_value in (0, 1):
        part_mask = np.zeros_like(mask)
        part_mask[ys[labels == label_value], xs[labels == label_value]] = 255
        part_mask = cv2.morphologyEx(
            part_mask,
            cv2.MORPH_OPEN,
            np.ones((2, 2), dtype=np.uint8),
        )
        part_contours, _ = cv2.findContours(
            part_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if not part_contours:
            continue

        part = max(part_contours, key=cv2.contourArea)
        part[:, :, 0] += x - pad
        part[:, :, 1] += y - pad
        detection, _ = classify_contour(part, params, source="cv_split_added")
        if detection is not None:
            detections.append(detection)

    if len(detections) < 2:
        return []

    oversplit_decision = _resolve_oversplit_contour(
        contour,
        detections,
        params,
        gray,
        color,
    )
    return detections if oversplit_decision is None else oversplit_decision

def identify_base_cv(
    image: np.ndarray,
    params: DetectionParams = DUNALIELLA_IDENTIFICATION_PARAMS,
    split_params: SplitParams | None = None,
) -> tuple[list[Detection], list[Detection], dict[str, int]]:

    split_params = split_params or SplitParams()
    gray, color = prepare_image(image)
    binary = build_cv_binary(gray, params)
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    has_color = _has_color_information(color)
    saturation_channel = _color_saturation_channel(color) if has_color else None

    detections: list[Detection] = []
    split_detections: list[Detection] = []
    counts: dict[str, int] = {}

    for contour in contours:
        detection, reason = classify_contour(contour, params, source="cv")
        if detection is not None and _is_cv_artifact(
            gray,
            color,
            contour,
            detection,
            params,
            has_color,
            saturation_channel,
        ):
            detection = None
            reason = "artifact"

        counts[reason] = counts.get(reason, 0) + 1
        if detection is not None:
            detections.append(detection)
            continue
        if reason != "diameter":
            continue

        (_, _), radius = cv2.minEnclosingCircle(contour)
        if 2.0 * radius <= params.diameter_max:
            counts["split_small_diameter_skipped"] = (
                counts.get("split_small_diameter_skipped", 0) + 1
            )
            continue

        counts["split_source_contours"] = counts.get("split_source_contours", 0) + 1
        parts = _split_dumbbell_contour(
            contour,
            params,
            split_params,
            gray,
            color,
        )
        if not parts:
            counts["split_no_recovery"] = counts.get("split_no_recovery", 0) + 1
            continue

        counts["split_recovered_contours"] = (
            counts.get("split_recovered_contours", 0) + 1
        )
        counts["split_added"] = counts.get("split_added", 0) + len(parts)
        split_detections.extend(parts)

    combined = [*detections, *split_detections]
    counts["cv_with_split"] = len(combined)
    counts["cv_split_added"] = len(split_detections)
    return combined, split_detections, counts

def identify_microalgae(
    image: np.ndarray,
    params: DetectionParams = DUNALIELLA_IDENTIFICATION_PARAMS,
    split_params: SplitParams | None = None,
    faint_recovery: bool = True,
    faint_params: FaintRecoveryParams = DUNALIELLA_FAINT_RECOVERY_PARAMS,
) -> tuple[list[Detection], list[Detection], dict[str, int]]:

    gray, _ = prepare_image(image)
    detections, split_detections, counts = identify_base_cv(
        image,
        params=params,
        split_params=split_params,
    )
    counts = dict(counts)

    if not faint_recovery:
        counts["cv_faint_candidates"] = 0
        counts["cv_faint_recovered"] = 0
        return detections, split_detections, counts

    faint_candidates, faint_reject_counts = _detect_faint_round_candidates(
        gray,
        faint_params,
    )
    detections, faint_added = _merge_recovered_detections(
        detections,
        faint_candidates,
        faint_params.merge_distance,
    )
    detections = _enrich_detection_features(gray, detections)
    split_detections = _enrich_detection_features(gray, split_detections)

    counts["cv_faint_candidates"] = len(faint_candidates)
    counts["cv_faint_recovered"] = len(faint_added)
    for key, value in faint_reject_counts.items():
        counts[f"cv_faint_reject_{key}"] = int(value)

    final_gate = [
        detection
        for detection in detections
        if detection.contour_area >= faint_params.final_area_min
        and faint_params.final_diameter_min
        <= detection.diameter
        <= faint_params.final_diameter_max
    ]
    final_split = [
        detection
        for detection in split_detections
        if detection.contour_area >= faint_params.final_area_min
        and faint_params.final_diameter_min
        <= detection.diameter
        <= faint_params.final_diameter_max
    ]
    counts["cv_final_gate_rejected"] = len(detections) - len(final_gate)

    major_gate = [
        detection
        for detection in final_gate
        if detection.source != "cv_faint_recovered"
        or detection.major >= faint_params.faint_major_min
    ]
    counts["cv_faint_major_rejected"] = len(final_gate) - len(major_gate)

    accepted = [
        detection
        for detection in major_gate
        if not _is_faint_background(detection, gray.shape[1], faint_params)
        and not _is_low_quality_split(detection, faint_params)
    ]
    accepted_split = [
        detection
        for detection in final_split
        if not _is_low_quality_split(detection, faint_params)
    ]
    counts["cv_faint_background_rejected"] = sum(
        _is_faint_background(detection, gray.shape[1], faint_params)
        for detection in major_gate
    )
    counts["cv_low_quality_split_rejected"] = sum(
        _is_low_quality_split(detection, faint_params)
        for detection in major_gate
    )
    counts["cv_total_after_artifact_gates"] = len(accepted)
    return accepted, accepted_split, counts

__all__ = [
    "DUNALIELLA_FAINT_RECOVERY_PARAMS",
    "DUNALIELLA_IDENTIFICATION_PARAMS",
    "Detection",
    "DetectionParams",
    "FaintRecoveryParams",
    "SplitParams",
    "build_cv_binary",
    "classify_contour",
    "identify_base_cv",
    "identify_microalgae",
    "prepare_image",
]
