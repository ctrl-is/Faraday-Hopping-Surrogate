import math

import pytest

from src.physics.constants import ELECTRON_CHARGE, ELECTRON_MASS
from src.physics.emission import EmittedElectron
from src.physics.fields import (
    Vector3,
    make_uniform_suppressor_field,
    solve_parallel_plate_field,
    zero_magnetic_field,
)
from src.physics.trajectory import (
    Trajectory,
    TrajectoryStatus,
    initial_velocity_from_emission,
)

GRID_SPACING = 3.5e-3
GRID_HEIGHT = 3.9e-3
WIRE_RADIUS = 2.5e-4


def zero_electric_field(position: Vector3) -> Vector3:
    return 0.0, 0.0, 0.0


def analytic_uniform_field_result(
    electron: EmittedElectron,
    initial_velocity: Vector3,
    voltage: float,
    grid_height: float,
    z0: float,
) -> tuple[float, float, float, float]:
    vx0, vy0, vz0 = initial_velocity

    electric_field_z = -voltage / grid_height
    acceleration_z = ELECTRON_CHARGE * electric_field_z / ELECTRON_MASS

    discriminant = vz0**2 - 2.0 * acceleration_z * z0
    return_time = (-vz0 - math.sqrt(discriminant)) / acceleration_z

    final_x = electron.x0 + vx0 * return_time
    final_y = electron.y0 + vy0 * return_time
    final_vz = -math.sqrt(discriminant)

    return return_time, final_x, final_y, final_vz


def test_uniform_field_matches_analytic_solution() -> None:
    voltage = -55.0
    grid_height = 3.9e-3
    z0 = 1e-6

    electron = EmittedElectron(
        x0=0.01,
        y0=0.01,
        energy_eV=5.0,
        theta=math.radians(45.0),
        psi=math.radians(30.0),
    )

    initial_velocity = initial_velocity_from_emission(electron)

    efield = make_uniform_suppressor_field(
        suppressor_voltage=voltage, grid_height=grid_height
    )

    trajectory = Trajectory(
        efield=efield,
        bfield=zero_magnetic_field,
        initial_position=(electron.x0, electron.y0, z0),
        initial_velocity=initial_velocity,
    )

    result = trajectory.solve(t_max=1e-8, max_step=1e-12, grid_height=grid_height)

    assert result.status == TrajectoryStatus.HIT_COLLECTOR

    (
        analytic_return_time,
        analytic_xf,
        analytic_yf,
        analytic_final_vz,
    ) = analytic_uniform_field_result(
        electron=electron,
        initial_velocity=initial_velocity,
        voltage=voltage,
        grid_height=grid_height,
        z0=z0,
    )

    xf, yf, zf = result.final_position

    assert result.return_time == pytest.approx(analytic_return_time, rel=1e-5)

    assert xf == pytest.approx(analytic_xf, rel=1e-5)
    assert yf == pytest.approx(analytic_yf, rel=1e-5)
    assert zf == pytest.approx(0.0, abs=1e-9)

    assert result.final_velocity[2] == pytest.approx(analytic_final_vz, rel=1e-5)


def test_numerical_parallel_plate_field_matches_analytic_trajectory() -> None:
    voltage = -55.0
    collector_voltage = 0.0
    grid_height = 3.9e-3
    grid_spacing = 3.5e-3
    z0 = 1e-6

    electron = EmittedElectron(
        x0=0.0,
        y0=0.0,
        energy_eV=5.0,
        theta=0.0,
        psi=0.0,
    )

    initial_velocity = initial_velocity_from_emission(electron)

    efield = solve_parallel_plate_field(
        grid_spacing=grid_spacing,
        grid_height=grid_height,
        collector_voltage=collector_voltage,
        suppressor_voltage=voltage,
        nx=11,
        ny=11,
        nz=11,
        tolerance=1e-10,
        max_iterations=100,
    )

    trajectory = Trajectory(
        efield=efield,
        bfield=zero_magnetic_field,
        initial_position=(electron.x0, electron.y0, z0),
        initial_velocity=initial_velocity,
    )

    result = trajectory.solve(t_max=1e-8, max_step=1e-12, grid_height=grid_height)

    assert result.status == TrajectoryStatus.HIT_COLLECTOR

    (
        analytic_return_time,
        analytic_xf,
        analytic_yf,
        analytic_final_vz,
    ) = analytic_uniform_field_result(
        electron=electron,
        initial_velocity=initial_velocity,
        voltage=voltage,
        grid_height=grid_height,
        z0=z0,
    )

    xf, yf, zf = result.final_position

    assert result.return_time == pytest.approx(analytic_return_time, rel=1e-4)

    assert xf == pytest.approx(analytic_xf, abs=1e-10)
    assert yf == pytest.approx(analytic_yf, abs=1e-10)
    assert zf == pytest.approx(0.0, abs=1e-9)

    assert result.final_velocity[2] == pytest.approx(analytic_final_vz, rel=1e-4)


def test_electron_can_reach_grid_without_wire_geometry() -> None:
    trajectory = Trajectory(
        efield=zero_electric_field,
        bfield=zero_magnetic_field,
        initial_position=(0.0, 0.0, 1e-6),
        initial_velocity=(0.0, 0.0, 1e6),
    )

    result = trajectory.solve(t_max=1e-7, max_step=1e-11, grid_height=1e-3)

    assert result.status == TrajectoryStatus.HIT_GRID
    assert result.final_position[2] == pytest.approx(1e-3, abs=1e-10)


def test_electron_hits_wire_when_grid_crossing_is_on_wire_axis() -> None:
    trajectory = Trajectory(
        efield=zero_electric_field,
        bfield=zero_magnetic_field,
        initial_position=(0.0, GRID_SPACING / 4.0, 1e-6),
        initial_velocity=(0.0, 0.0, 1e6),
    )

    result = trajectory.solve(
        t_max=1e-7,
        max_step=1e-11,
        grid_height=GRID_HEIGHT,
        wire_grid_spacing=GRID_SPACING,
        wire_radius=WIRE_RADIUS,
    )

    assert result.status == TrajectoryStatus.HIT_WIRE
    assert result.final_position[2] == pytest.approx(GRID_HEIGHT, abs=1e-10)


def test_electron_passes_through_grid_opening() -> None:
    trajectory = Trajectory(
        efield=zero_electric_field,
        bfield=zero_magnetic_field,
        initial_position=(GRID_SPACING / 4.0, GRID_SPACING / 4.0, 1e-6),
        initial_velocity=(0.0, 0.0, 1e6),
    )

    result = trajectory.solve(
        t_max=1e-7,
        max_step=1e-11,
        grid_height=GRID_HEIGHT,
        wire_grid_spacing=GRID_SPACING,
        wire_radius=WIRE_RADIUS,
    )

    assert result.status == TrajectoryStatus.PASSED_GRID_OPENING
    assert result.final_position[2] == pytest.approx(GRID_HEIGHT, abs=1e-10)


def test_wire_grid_spacing_and_wire_radius_must_be_provided_together() -> None:
    trajectory = Trajectory(
        efield=zero_electric_field,
        bfield=zero_magnetic_field,
        initial_position=(0.0, 0.0, 1e-6),
        initial_velocity=(0.0, 0.0, 1e6),
    )

    with pytest.raises(ValueError, match="wire_grid_spacing and wire_radius"):
        trajectory.solve(
            t_max=1e-7,
            max_step=1e-11,
            grid_height=GRID_HEIGHT,
            wire_grid_spacing=GRID_SPACING,
        )

    with pytest.raises(ValueError, match="wire_grid_spacing and wire_radius"):
        trajectory.solve(
            t_max=1e-7,
            max_step=1e-11,
            grid_height=GRID_HEIGHT,
            wire_radius=WIRE_RADIUS,
        )


def test_wire_geometry_requires_grid_height() -> None:
    trajectory = Trajectory(
        efield=zero_electric_field,
        bfield=zero_magnetic_field,
        initial_position=(0.0, 0.0, 1e-6),
        initial_velocity=(0.0, 0.0, 1e6),
    )

    with pytest.raises(ValueError, match="grid_height is required"):
        trajectory.solve(
            t_max=1e-7,
            max_step=1e-11,
            wire_grid_spacing=GRID_SPACING,
            wire_radius=WIRE_RADIUS,
        )


def test_electron_leaves_radial_domain() -> None:
    trajectory = Trajectory(
        efield=zero_electric_field,
        bfield=zero_magnetic_field,
        initial_position=(0.0, 0.0, 1e-3),
        initial_velocity=(1e6, 0.0, 0.0),
    )

    result = trajectory.solve(t_max=1e-7, max_step=1e-11, radial_limit=1e-2)

    assert result.status == TrajectoryStatus.LEFT_RADIAL_DOMAIN

    x, y, _ = result.final_position

    assert math.hypot(x, y) == pytest.approx(1e-2, rel=1e-6)


def test_stationary_electron_times_out() -> None:
    trajectory = Trajectory(
        efield=zero_electric_field,
        bfield=zero_magnetic_field,
        initial_position=(0.0, 0.0, 1e-3),
        initial_velocity=(0.0, 0.0, 0.0),
    )

    result = trajectory.solve(t_max=1e-9, max_step=1e-10)

    assert result.status == TrajectoryStatus.TIMEOUT
    assert result.return_time is None
