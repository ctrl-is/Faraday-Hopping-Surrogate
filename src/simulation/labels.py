from dataclasses import dataclass
from enum import Enum

from src.physics.emission import EmittedElectron
from src.physics.geometry import CollectorGeometry, Region
from src.physics.trajectory import TrajectoryResult, TrajectoryStatus


QUADRANT_REGIONS = {
    Region.A,
    Region.B,
    Region.C,
    Region.D,
}


class LandingOutcome(Enum):
    SAME_QUADRANT = "same_quadrant"
    DIFFERENT_QUADRANT = "different_quadrant"
    GAP = "gap"
    OUTSIDE = "outside"
    DID_NOT_RETURN = "did_not_return"
    INVALID_SOURCE = "invalid_source"


@dataclass(frozen=True)
class LandingResult:
    outcome: LandingOutcome
    source_region: Region
    final_region: Region | None

    @property
    def hopped(self) -> bool:
        return self.outcome == LandingOutcome.DIFFERENT_QUADRANT


def classify_landing(
    electron: EmittedElectron,
    trajectory_result: TrajectoryResult,
    geometry: CollectorGeometry,
) -> LandingResult:
    """
    Classify where an emitted electron lands relative to its source quadrant.

    An electron is a plate-hopper when it starts on one quadrant and returns
    to a different quadrant.
    """
    source_region = geometry.region_at((electron.x0, electron.y0))

    # A valid emitted electron must originate from a collector quadrant.
    if source_region not in QUADRANT_REGIONS:
        return LandingResult(
            outcome=LandingOutcome.INVALID_SOURCE,
            source_region=source_region,
            final_region=None,
        )

    # The electron may hit the grid, leave the domain, time out,
    # or encounter a solver failure before reaching the collector.
    if trajectory_result.status != TrajectoryStatus.HIT_COLLECTOR:
        return LandingResult(
            outcome=LandingOutcome.DID_NOT_RETURN,
            source_region=source_region,
            final_region=None,
        )

    xf, yf, _ = trajectory_result.final_position

    final_region = geometry.region_at((xf, yf))

    if final_region == Region.GAP:
        outcome = LandingOutcome.GAP

    elif final_region == Region.OUTSIDE:
        outcome = LandingOutcome.OUTSIDE

    elif final_region == source_region:
        outcome = LandingOutcome.SAME_QUADRANT

    elif final_region in QUADRANT_REGIONS:
        outcome = LandingOutcome.DIFFERENT_QUADRANT

    else:
        raise RuntimeError(f"Unexpected final collector region: {final_region}")

    return LandingResult(
        outcome=outcome,
        source_region=source_region,
        final_region=final_region,
    )