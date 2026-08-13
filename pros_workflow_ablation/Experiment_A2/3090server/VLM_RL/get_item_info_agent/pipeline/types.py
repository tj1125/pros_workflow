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
    def top_left(self) -> tuple[float, float]:
        return (self.x1, self.y1)

    @property
    def bottom_left(self) -> tuple[float, float]:
        return (self.x1, self.y2)

    @property
    def top_right(self) -> tuple[float, float]:
        return (self.x2, self.y1)

    @property
    def bottom_right(self) -> tuple[float, float]:
        return (self.x2, self.y2)

    @property
    def top_center(self) -> np.ndarray:
        return np.array([(self.x1 + self.x2) * 0.5, self.y1], dtype=float)

    @property
    def bottom_center(self) -> np.ndarray:
        return np.array([(self.x1 + self.x2) * 0.5, self.y2], dtype=float)

    @property
    def center(self) -> np.ndarray:
        return np.array([(self.x1 + self.x2) * 0.5, (self.y1 + self.y2) * 0.5], dtype=float)

    @property
    def width(self) -> float:
        return float(self.x2 - self.x1)

    @property
    def height(self) -> float:
        return float(self.y2 - self.y1)


@dataclass(frozen=True)
class Grasp:
    name: str
    position: np.ndarray
    rotation: np.ndarray
    confidence: float
    group: int


@dataclass(frozen=True)
class MapInfo:
    resolution: float
    origin: tuple[float, float, float]
    width: int
    height: int
