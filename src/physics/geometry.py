import math
from dataclasses import dataclass
from enum import Enum

Vector2 = tuple[float, float]


class Region(Enum):
    A = "A"  # x > 0, y > 0
    B = "B"  # x < 0, y > 0
    C = "C"  # x < 0, y < 0
    D = "D"  # x > 0, y < 0
    GAP = "gap"
    OUTSIDE = "outside"


@dataclass(frozen=True)
class CollectorGeometry:
    gap_width: float
    radius: float

    def __post_init__(self) -> None:
        if self.gap_width < 0.0:
            raise ValueError("gap_width cannot be negative.")

        if self.radius <= 0.0:
            raise ValueError("radius must be positive.")

    def is_inside_collector(self, coords: Vector2) -> bool:
        x, y = coords
        return x**2 + y**2 <= self.radius**2

    def is_inside_gap(self, coords: Vector2) -> bool:
        x, y = coords
        half_gap = self.gap_width / 2.0

        return abs(x) <= half_gap or abs(y) <= half_gap

    def region_at(self, coords: Vector2) -> Region:
        x, y = coords

        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError(f"Coordinates must be finite, received {coords}.")

        if not self.is_inside_collector(coords):
            return Region.OUTSIDE

        if self.is_inside_gap(coords):
            return Region.GAP

        if x > 0.0 and y > 0.0:
            return Region.A

        if x < 0.0 and y > 0.0:
            return Region.B

        if x < 0.0 and y < 0.0:
            return Region.C

        if x > 0.0 and y < 0.0:
            return Region.D

        # The axes should already have been caught by is_inside_gap.
        raise RuntimeError(f"Could not classify collector coordinates {coords}.")
