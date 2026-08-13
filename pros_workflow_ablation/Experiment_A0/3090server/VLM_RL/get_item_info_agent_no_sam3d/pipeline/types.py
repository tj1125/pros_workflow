from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> np.ndarray:
        return np.array([(self.x1 + self.x2) * 0.5, (self.y1 + self.y2) * 0.5], dtype=np.float32)

    @property
    def width(self) -> float:
        return float(self.x2 - self.x1)

    @property
    def height(self) -> float:
        return float(self.y2 - self.y1)


@dataclass(frozen=True)
class TopicObservation:
    camera_id: str
    bbox: BoundingBox


@dataclass(frozen=True)
class WorldPositionObject:
    label: str
    item_id: int
    center_world_unity: np.ndarray
    observations: dict[str, TopicObservation]
    topic_key: str


@dataclass(frozen=True)
class MapInfo:
    resolution: float
    origin: tuple[float, float, float]
    width: int
    height: int
