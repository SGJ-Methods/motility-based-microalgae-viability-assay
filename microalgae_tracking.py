"""Centroid-based tracking of CV-identified microalgae."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path

try:
    from .microalgae_identification import (
        DUNALIELLA_IDENTIFICATION_PARAMS,
        Detection,
        DetectionParams,
        SplitParams,
        identify_microalgae,
    )
except ImportError:
    from microalgae_identification import (
        DUNALIELLA_IDENTIFICATION_PARAMS,
        Detection,
        DetectionParams,
        SplitParams,
        identify_microalgae,
    )

@dataclass(frozen=True)
class TrackingParams:

    max_distance: float = 36.839
    max_missed: int = 2
    time_interval: float = 0.1
    relink_max_gap: int = 8
    relink_max_distance: float = 97.515
    relink_min_diameter_ratio: float = 0.55

@dataclass
class Track:
    track_id: int
    x: float
    y: float
    last_frame: int
    missed: int = 0
    history: list[tuple[int, float, float]] = field(default_factory=list)
    diameter: float = 0.0
    major: float = 0.0
    minor: float = 0.0
    source: str = ""

@dataclass(frozen=True)
class TrackedDetection:
    detection: Detection
    track_id: int
    displacement: float
    velocity: float

class CentroidTracker:

    def __init__(self, params: TrackingParams | None = None) -> None:
        self.params = params or TrackingParams()
        self.max_distance = float(self.params.max_distance)
        self.max_distance_sq = self.max_distance * self.max_distance
        self.max_missed = int(self.params.max_missed)
        self.time_interval = float(self.params.time_interval)
        self.relink_max_gap = int(self.params.relink_max_gap)
        self.relink_max_distance = float(self.params.relink_max_distance)
        self.relink_min_diameter_ratio = float(self.params.relink_min_diameter_ratio)
        self.next_id = 0
        self.active: dict[int, Track] = {}
        self.all_tracks: dict[int, Track] = {}
        self.last_counts: dict[str, int] = {}

    def update(
        self,
        detections: list[Detection],
        frame_number: int,
    ) -> list[TrackedDetection]:

        if not self.active:
            tracked = [self._start_track(detection, frame_number) for detection in detections]
            self.last_counts = {
                "track_relinked": 0,
                "tracks_started": len(tracked),
                "tracks_lost": 0,
            }
            return tracked

        detection_cells = self._build_detection_cells(detections)
        pairs: list[tuple[float, int, int]] = []
        cell_size = self.max_distance

        for track_id, track in self.active.items():
            cell_x = int(track.x // cell_size)
            cell_y = int(track.y // cell_size)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for detection_index in detection_cells.get((cell_x + dx, cell_y + dy), []):
                        detection = detections[detection_index]
                        distance_sq = (track.x - detection.x) ** 2 + (track.y - detection.y) ** 2
                        if distance_sq <= self.max_distance_sq:
                            pairs.append((distance_sq, track_id, detection_index))

        pairs.sort(key=lambda item: item[0])
        assigned_tracks: set[int] = set()
        assigned_detections: set[int] = set()
        tracked: list[TrackedDetection] = []

        for distance_sq, track_id, detection_index in pairs:
            if track_id in assigned_tracks or detection_index in assigned_detections:
                continue
            detection = detections[detection_index]
            tracked.append(
                self._update_track(
                    track_id,
                    detection,
                    frame_number,
                    math.sqrt(distance_sq),
                )
            )
            assigned_tracks.add(track_id)
            assigned_detections.add(detection_index)

        relinked = 0
        if self.relink_max_gap > 0:
            relink_pairs = self._build_relink_pairs(
                detections,
                frame_number,
                assigned_tracks,
                assigned_detections,
            )
            relink_pairs.sort(key=lambda item: item[0])
            for _, distance, track_id, detection_index in relink_pairs:
                if track_id in assigned_tracks or detection_index in assigned_detections:
                    continue
                detection = detections[detection_index]
                tracked.append(
                    self._update_track(track_id, detection, frame_number, distance)
                )
                assigned_tracks.add(track_id)
                assigned_detections.add(detection_index)
                relinked += 1

        started = 0
        for detection_index, detection in enumerate(detections):
            if detection_index in assigned_detections:
                continue
            tracked_detection = self._start_track(detection, frame_number)
            tracked.append(tracked_detection)
            assigned_tracks.add(tracked_detection.track_id)
            started += 1

        lost = 0
        for track_id in list(self.active):
            if track_id in assigned_tracks:
                continue
            track = self.active[track_id]
            track.missed += 1
            if track.missed > self.max_missed:
                del self.active[track_id]
                lost += 1

        self.last_counts = {
            "track_relinked": relinked,
            "tracks_started": started,
            "tracks_lost": lost,
        }
        return tracked

    def _build_detection_cells(
        self,
        detections: list[Detection],
        indices: list[int] | None = None,
        cell_size: float | None = None,
    ) -> dict[tuple[int, int], list[int]]:
        cells: dict[tuple[int, int], list[int]] = {}
        cell_size = max(1.0, float(cell_size or self.max_distance))
        iterable = indices if indices is not None else range(len(detections))
        for index in iterable:
            detection = detections[index]
            key = (int(detection.x // cell_size), int(detection.y // cell_size))
            cells.setdefault(key, []).append(index)
        return cells

    def _build_relink_pairs(
        self,
        detections: list[Detection],
        frame_number: int,
        assigned_tracks: set[int],
        assigned_detections: set[int],
    ) -> list[tuple[float, float, int, int]]:
        unmatched_indices = [
            index for index in range(len(detections)) if index not in assigned_detections
        ]
        if not unmatched_indices:
            return []

        cell_size = max(self.max_distance, self.relink_max_distance)
        detection_cells = self._build_detection_cells(
            detections,
            indices=unmatched_indices,
            cell_size=cell_size,
        )
        pairs: list[tuple[float, float, int, int]] = []

        for track_id, track in self.all_tracks.items():
            if track_id in assigned_tracks or track.last_frame >= frame_number:
                continue
            frame_gap = frame_number - track.last_frame
            if frame_gap <= 1 or frame_gap > self.relink_max_gap:
                continue

            allowed_distance = min(
                self.relink_max_distance,
                self.max_distance * (1.0 + 0.75 * frame_gap),
            )
            allowed_distance_sq = allowed_distance * allowed_distance
            cell_x = int(track.x // cell_size)
            cell_y = int(track.y // cell_size)
            search_radius = max(1, int(math.ceil(allowed_distance / cell_size)))

            for dx in range(-search_radius, search_radius + 1):
                for dy in range(-search_radius, search_radius + 1):
                    for detection_index in detection_cells.get((cell_x + dx, cell_y + dy), []):
                        detection = detections[detection_index]
                        diameter_score = self._diameter_similarity_score(track, detection)
                        if diameter_score is None:
                            continue
                        distance_sq = (track.x - detection.x) ** 2 + (track.y - detection.y) ** 2
                        if distance_sq > allowed_distance_sq:
                            continue
                        distance = math.sqrt(distance_sq)
                        score = (
                            distance_sq / max(1.0, allowed_distance_sq)
                            + 0.08 * frame_gap
                            + 0.30 * diameter_score
                        )
                        pairs.append((score, distance, track_id, detection_index))
        return pairs

    def _diameter_similarity_score(
        self,
        track: Track,
        detection: Detection,
    ) -> float | None:
        if track.diameter <= 0 or detection.diameter <= 0:
            return 0.0
        ratio = min(track.diameter, detection.diameter) / max(track.diameter, detection.diameter)
        if ratio < self.relink_min_diameter_ratio:
            return None
        return 1.0 - ratio

    def _start_track(
        self,
        detection: Detection,
        frame_number: int,
    ) -> TrackedDetection:
        track_id = self.next_id
        self.next_id += 1
        track = Track(
            track_id=track_id,
            x=detection.x,
            y=detection.y,
            last_frame=frame_number,
            history=[(frame_number, detection.x, detection.y)],
            diameter=detection.diameter,
            major=detection.major,
            minor=detection.minor,
            source=detection.source,
        )
        self.active[track_id] = track
        self.all_tracks[track_id] = track
        return TrackedDetection(detection, track_id, 0.0, 0.0)

    def _update_track(
        self,
        track_id: int,
        detection: Detection,
        frame_number: int,
        distance: float,
    ) -> TrackedDetection:
        track = self.all_tracks[track_id]
        frame_gap = max(1, frame_number - track.last_frame)
        velocity = distance / (frame_gap * self.time_interval)
        track.x = detection.x
        track.y = detection.y
        track.last_frame = frame_number
        track.missed = 0
        track.diameter = detection.diameter
        track.major = detection.major
        track.minor = detection.minor
        track.source = detection.source
        track.history.append((frame_number, detection.x, detection.y))
        self.active[track_id] = track
        return TrackedDetection(detection, track_id, distance, velocity)

def identify_and_track_frame(
    image: np.ndarray,
    frame_number: int,
    tracker: CentroidTracker,
    detection_params: DetectionParams = DUNALIELLA_IDENTIFICATION_PARAMS,
    split_params: SplitParams | None = None,
) -> tuple[list[TrackedDetection], list[Detection], dict[str, int]]:

    detections, split_detections, counts = identify_microalgae(
        image,
        params=detection_params,
        split_params=split_params,
    )
    tracked = tracker.update(detections, frame_number)
    return tracked, split_detections, counts

def tracking_rows(
    image_name: str,
    frame_number: int,
    tracked: list[TrackedDetection],
    time_interval: float = 0.1,
) -> list[dict[str, object]]:

    rows: list[dict[str, object]] = []
    for item in tracked:
        detection = item.detection
        rows.append(
            {
                "image_name": image_name,
                "frame": frame_number,
                "time": frame_number * time_interval,
                "track_id": item.track_id,
                "x": detection.x,
                "y": detection.y,
                "diameter": detection.diameter,
                "major": detection.major,
                "minor": detection.minor,
                "angle": detection.angle,
                "contour_area": detection.contour_area,
                "source": detection.source,
                "displacement": item.displacement,
                "velocity": item.velocity,
            }
        )
    return rows

def write_tracking_csv(path: str | Path, rows: list[dict[str, object]]) -> None:

    fieldnames = [
        "image_name",
        "frame",
        "time",
        "track_id",
        "x",
        "y",
        "diameter",
        "major",
        "minor",
        "angle",
        "contour_area",
        "source",
        "displacement",
        "velocity",
    ]
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})

__all__ = [
    "CentroidTracker",
    "Track",
    "TrackedDetection",
    "TrackingParams",
    "identify_and_track_frame",
    "tracking_rows",
    "write_tracking_csv",
]
