"""Visualization of microalgal identification and tracking results."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import tifffile

@dataclass(frozen=True)
class OverlayStyle:

    contour_color: tuple[int, int, int] = (0, 0, 0)
    circle_color: tuple[int, int, int] = (255, 0, 0)
    ellipse_color: tuple[int, int, int] = (0, 0, 255)
    trajectory_color: tuple[int, int, int] = (0, 0, 0)
    label_color: tuple[int, int, int] = (0, 0, 0)
    header_color: tuple[int, int, int] = (0, 0, 255)
    line_thickness: int = 1
    label_font_scale: float = 0.34
    label_thickness: int = 1
    header_font_scale: float = 1.0
    header_thickness: int = 2
    trace_points: int = 20
    draw_contours: bool = True
    draw_circles: bool = True
    draw_ellipses: bool = True
    draw_trajectories: bool = True
    draw_labels: bool = True
    draw_header: bool = True

def _to_bgr_uint8(image: np.ndarray) -> np.ndarray:

    if image is None or image.size == 0:
        raise ValueError("image must be a non-empty NumPy array")

    converted = np.asarray(image)
    if converted.dtype != np.uint8:
        finite = np.asarray(converted, dtype=np.float32)
        finite = np.nan_to_num(finite, nan=0.0, posinf=0.0, neginf=0.0)
        minimum = float(finite.min())
        maximum = float(finite.max())
        if maximum > minimum:
            converted = np.clip(
                (finite - minimum) * 255.0 / (maximum - minimum),
                0,
                255,
            ).astype(np.uint8)
        else:
            converted = np.zeros(finite.shape, dtype=np.uint8)

    if converted.ndim == 2:
        return cv2.cvtColor(converted, cv2.COLOR_GRAY2BGR)
    if converted.ndim != 3:
        raise ValueError("image must be a 2-D grayscale or 3-D color array")
    if converted.shape[2] == 1:
        return cv2.cvtColor(converted[:, :, 0], cv2.COLOR_GRAY2BGR)
    if converted.shape[2] == 3:
        return converted.copy()
    if converted.shape[2] == 4:
        return cv2.cvtColor(converted, cv2.COLOR_BGRA2BGR)
    raise ValueError("color image must have 1, 3, or 4 channels")

def _contour_array(detection: object) -> np.ndarray | None:
    contour = getattr(detection, "contour", None)
    if contour is None or len(contour) < 2:
        return None
    return np.asarray(contour, dtype=np.int32).reshape(-1, 1, 2)

def _label_position(
    detection: object,
    label: str,
    image_shape: tuple[int, ...],
    font_scale: float,
    thickness: int,
    offset: int = 2,
) -> tuple[int, int]:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(
        label,
        font,
        font_scale,
        thickness,
    )
    image_height, image_width = image_shape[:2]
    contour = _contour_array(detection)

    if contour is not None:
        points = contour.reshape(-1, 2)
        x_min = int(points[:, 0].min())
        x_max = int(points[:, 0].max())
        y_min = int(points[:, 1].min())
    else:
        x_center = float(getattr(detection, "x"))
        y_center = float(getattr(detection, "y"))
        diameter = float(getattr(detection, "diameter"))
        major = float(getattr(detection, "major"))
        minor = float(getattr(detection, "minor"))
        half_major = max(major, diameter) / 2.0
        half_minor = max(minor, 1.0) / 2.0
        x_min = int(round(x_center - half_major))
        x_max = int(round(x_center + half_major))
        y_min = int(round(y_center - half_minor))

    x = x_max + offset
    y = y_min + text_height + offset
    if x + text_width >= image_width:
        x = max(x_min - text_width - offset, offset)
    if y - text_height < 0:
        y = text_height + offset
    x = max(offset, min(x, image_width - text_width - offset))
    y = max(text_height + offset, min(y, image_height - baseline - offset))
    return int(x), int(y)

def draw_tracking_overlay(
    image: np.ndarray,
    tracked: list[object],
    tracks: dict[int, object],
    frame_number: int | None = None,
    style: OverlayStyle | None = None,
    output_path: str | Path | None = None,
) -> np.ndarray:

    style = style or OverlayStyle()
    canvas = _to_bgr_uint8(image)

    for item in tracked:
        detection = item.detection
        contour = _contour_array(detection)

        if style.draw_contours and contour is not None:
            cv2.drawContours(
                canvas,
                [contour],
                -1,
                style.contour_color,
                style.line_thickness,
                cv2.LINE_AA,
            )

        if style.draw_circles and contour is not None:
            (circle_x, circle_y), radius = cv2.minEnclosingCircle(contour)
            cv2.circle(
                canvas,
                (int(round(circle_x)), int(round(circle_y))),
                max(1, int(round(radius))),
                style.circle_color,
                style.line_thickness,
                cv2.LINE_AA,
            )

        if style.draw_ellipses:
            ellipse = (
                (float(detection.x), float(detection.y)),
                (float(detection.major), float(detection.minor)),
                float(detection.angle),
            )
            cv2.ellipse(
                canvas,
                ellipse,
                style.ellipse_color,
                style.line_thickness,
                cv2.LINE_AA,
            )

        track = tracks.get(item.track_id)
        if style.draw_trajectories and track is not None:
            history = track.history[-max(2, style.trace_points) :]
            for start, end in zip(history, history[1:]):
                start_frame, x0, y0 = start
                end_frame, x1, y1 = end
                if end_frame - start_frame != 1:
                    continue
                cv2.line(
                    canvas,
                    (int(round(x0)), int(round(y0))),
                    (int(round(x1)), int(round(y1))),
                    style.trajectory_color,
                    style.line_thickness,
                    cv2.LINE_AA,
                )

        if style.draw_labels:
            label = str(item.track_id)
            cv2.putText(
                canvas,
                label,
                _label_position(
                    detection,
                    label,
                    canvas.shape,
                    style.label_font_scale,
                    style.label_thickness,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                style.label_font_scale,
                style.label_color,
                style.label_thickness,
                cv2.LINE_AA,
            )

    if style.draw_header and frame_number is not None:
        cv2.putText(
            canvas,
            f"Frame {frame_number} | objects {len(tracked)}",
            (20, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            style.header_font_scale,
            style.header_color,
            style.header_thickness,
            cv2.LINE_AA,
        )

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), canvas, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
            raise OSError(f"Could not write overlay image: {path}")
    return canvas

def write_imagej_tiff_stack(
    output_path: str | Path,
    bgr_frames: Iterable[np.ndarray],
    frame_interval: float | None = None,
) -> Path:

    rgb_frames = [
        cv2.cvtColor(_to_bgr_uint8(frame), cv2.COLOR_BGR2RGB)
        for frame in bgr_frames
    ]
    if not rgb_frames:
        raise ValueError("at least one frame is required")
    first_shape = rgb_frames[0].shape
    if any(frame.shape != first_shape for frame in rgb_frames[1:]):
        raise ValueError("all frames must have the same dimensions")

    stack = np.stack(rgb_frames, axis=0)
    metadata: dict[str, object] = {"axes": "TYXS"}
    if frame_interval is not None:
        if frame_interval <= 0:
            raise ValueError("frame_interval must be positive")
        metadata["finterval"] = float(frame_interval)
        metadata["tunit"] = "s"

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        path,
        stack,
        imagej=True,
        photometric="rgb",
        metadata=metadata,
    )
    return path

__all__ = [
    "OverlayStyle",
    "draw_tracking_overlay",
    "write_imagej_tiff_stack",
]
