import math
from dataclasses import dataclass

import numpy as np

from .geometry import CollectorGeometry, Region

Vector2 = tuple[float, float]


@dataclass(frozen=True)
class ProtonImpact:
    entry_position: Vector2
    collector_position: Vector2
    alpha: float
    source_region: Region

    @property
    def hit_valid_region(self) -> bool:
        return self.source_region in {
            Region.A,
            Region.B,
            Region.C,
            Region.D,
        }


def sample_uniform_disk(rng: np.random.Generator, radius: float) -> Vector2:
    if radius <= 0.0:
        raise ValueError("radius must be positive.")

    r = radius * math.sqrt(float(rng.uniform(0.0, 1.0)))
    theta = float(rng.uniform(0.0, 2 * math.pi))

    x = r * math.cos(theta)
    y = r * math.sin(theta)

    return x, y


def collector_impact_from_entry(
    entry_position: Vector2,
    alpha: float,
    grid_height: float,
) -> Vector2:
    x_entry, y_entry = entry_position

    x = x_entry + grid_height * math.tan(alpha)
    y = y_entry

    return x, y


def sample_proton_impact(
    rng: np.random.Generator,
    geometry: CollectorGeometry,
    alpha: float,
    beam_radius: float,
    grid_height: float,
) -> ProtonImpact:
    entry_position = sample_uniform_disk(rng=rng, radius=beam_radius)
    collector_position = collector_impact_from_entry(
        entry_position=entry_position,
        alpha=alpha,
        grid_height=grid_height,
    )
    source_region = geometry.region_at(collector_position)

    return ProtonImpact(
        entry_position=entry_position,
        collector_position=collector_position,
        alpha=alpha,
        source_region=source_region,
    )
