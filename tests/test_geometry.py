import pytest

from src.physics.geometry import CollectorGeometry, Region


@pytest.fixture
def geometry() -> CollectorGeometry:
    return CollectorGeometry(
        gap_width=1e-4,  # 0.1 mm
        radius=0.06,  # 60 mm
    )


@pytest.mark.parametrize(
    ("coords", "expected_region"),
    [
        ((0.01, 0.01), Region.A),
        ((-0.01, 0.01), Region.B),
        ((-0.01, -0.01), Region.C),
        ((0.01, -0.01), Region.D),
        ((0.0, 0.01), Region.GAP),
        ((0.01, 0.0), Region.GAP),
        ((0.10, 0.10), Region.OUTSIDE),
    ],
)
def test_region_at(
    geometry: CollectorGeometry, coords: tuple[float, float], expected_region: Region
) -> None:
    assert geometry.region_at(coords) == expected_region


def test_gap_boundary_is_gap(geometry: CollectorGeometry) -> None:
    half_gap = geometry.gap_width / 2.0

    assert geometry.region_at((half_gap, 0.01)) == Region.GAP


def test_collector_boundary_is_inside(geometry: CollectorGeometry) -> None:
    assert geometry.is_inside_collector((geometry.radius, 0.0))


def test_invalid_geometry() -> None:
    with pytest.raises(ValueError):
        CollectorGeometry(gap_width=-1e-4, radius=0.06)


def test_nonfinite_coordinate_raises(geometry: CollectorGeometry) -> None:
    with pytest.raises(ValueError):
        geometry.region_at((float("nan"), 0.0))
