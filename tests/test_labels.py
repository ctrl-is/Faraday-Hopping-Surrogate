import pytest

from src.physics.emission import EmittedElectron
from src.physics.geometry import (
    CollectorGeometry,
    Region,
)
from src.physics.trajectory import (
    TrajectoryResult,
    TrajectoryStatus,
)
from src.simulation.labels import (
    LandingOutcome,
    classify_landing,
)


@pytest.fixture
def geometry() -> CollectorGeometry:
    return CollectorGeometry(gap_width=1e-4, radius=0.06)


def make_electron(x0: float, y0: float) -> EmittedElectron:
    return EmittedElectron(
        x0=x0,
        y0=y0,
        energy_eV=5.0,
        theta=0.0,
        psi=0.0,
    )


def make_result(
    status: TrajectoryStatus, final_position: tuple[float, float, float]
) -> TrajectoryResult:
    return TrajectoryResult(
        status=status,
        final_position=final_position,
        final_velocity=(0.0, 0.0, -1.0),
        event_time=1e-9,
        solution=None,
    )


@pytest.mark.parametrize(
    (
        "source",
        "destination",
        "expected_outcome",
    ),
    [
        (
            (0.01, 0.01),
            (0.02, 0.01, 0.0),
            LandingOutcome.SAME_QUADRANT,
        ),
        (
            (0.01, 0.01),
            (-0.01, 0.01, 0.0),
            LandingOutcome.DIFFERENT_QUADRANT,
        ),
        (
            (0.01, 0.01),
            (0.0, 0.01, 0.0),
            LandingOutcome.GAP,
        ),
        (
            (0.01, 0.01),
            (0.10, 0.10, 0.0),
            LandingOutcome.OUTSIDE,
        ),
    ],
)
def test_landing_classification(
    geometry,
    source,
    destination,
    expected_outcome,
) -> None:
    electron = make_electron(*source)

    trajectory_result = make_result(TrajectoryStatus.HIT_COLLECTOR, destination)

    landing = classify_landing(
        electron=electron,
        trajectory_result=trajectory_result,
        geometry=geometry,
    )

    assert landing.outcome == expected_outcome


def test_invalid_source(geometry) -> None:
    electron = make_electron(
        x0=0.0,
        y0=0.01,
    )

    trajectory_result = make_result(TrajectoryStatus.HIT_COLLECTOR, (0.01, 0.01, 0.0))

    landing = classify_landing(electron, trajectory_result, geometry)

    assert landing.outcome == LandingOutcome.INVALID_SOURCE


@pytest.mark.parametrize(
    "status",
    [
        TrajectoryStatus.HIT_GRID,
        TrajectoryStatus.LEFT_RADIAL_DOMAIN,
        TrajectoryStatus.TIMEOUT,
        TrajectoryStatus.SOLVER_FAILURE,
    ],
)
def test_electron_that_does_not_return(geometry, status) -> None:
    electron = make_electron(x0=0.01, y0=0.01)

    trajectory_result = make_result(status, (0.01, 0.01, 1e-3))

    landing = classify_landing(electron, trajectory_result, geometry)

    assert landing.outcome == LandingOutcome.DID_NOT_RETURN


def test_hopped_property(geometry) -> None:
    electron = make_electron(x0=0.01, y0=0.01)

    trajectory_result = make_result(TrajectoryStatus.HIT_COLLECTOR, (-0.01, 0.01, 0.0))

    landing = classify_landing(electron, trajectory_result, geometry)

    assert landing.hopped is True
    assert landing.source_region == Region.A
    assert landing.final_region == Region.B
