import math

import numpy as np
import pytest

from src.physics.emission import EmittedElectron
from src.physics.fields import solve_wire_grid_field, zero_magnetic_field
from src.physics.trajectory import (
    Trajectory,
    TrajectoryStatus,
    initial_velocity_from_emission,
)


GRID_SPACING = 3.5e-3
GRID_HEIGHT = 3.9e-3
WIRE_RADIUS = 2.5e-4
COLLECTOR_VOLTAGE = 0.0
SUPPRESSOR_VOLTAGE = -55.0


def test_low_energy_electron_returns_to_collector_in_wire_grid_field() -> None:
    electron = EmittedElectron(
        x0=GRID_SPACING / 4.0,
        y0=GRID_SPACING / 4.0,
        energy_eV=2.0,
        theta=0.0,
        psi=0.0,
    )

    electric_field = solve_wire_grid_field(
        grid_spacing=GRID_SPACING,
        grid_height=GRID_HEIGHT,
        wire_radius=WIRE_RADIUS,
        collector_voltage=COLLECTOR_VOLTAGE,
        suppressor_voltage=SUPPRESSOR_VOLTAGE,
        nx=9,
        ny=9,
        nz=9,
        tolerance=1e-5,
        max_iterations=10_000,
    )

    trajectory = Trajectory(
        efield=electric_field,
        bfield=zero_magnetic_field,
        initial_position=(electron.x0, electron.y0, 1e-6),
        initial_velocity=initial_velocity_from_emission(electron),
    )

    result = trajectory.solve(t_max=1e-8, max_step=1e-12, grid_height=GRID_HEIGHT)

    assert result.status == TrajectoryStatus.HIT_COLLECTOR
    assert result.return_time is not None

    final_x, final_y, final_z = result.final_position

    assert np.isfinite(final_x)
    assert np.isfinite(final_y)
    assert final_z == pytest.approx(0.0, abs=1e-9)


def test_high_energy_electron_can_reach_grid_region_in_wire_grid_field() -> None:
    electron = EmittedElectron(
        x0=GRID_SPACING / 4.0,
        y0=GRID_SPACING / 4.0,
        energy_eV=80.0,
        theta=0.0,
        psi=0.0,
    )

    electric_field = solve_wire_grid_field(
        grid_spacing=GRID_SPACING,
        grid_height=GRID_HEIGHT,
        wire_radius=WIRE_RADIUS,
        collector_voltage=COLLECTOR_VOLTAGE,
        suppressor_voltage=SUPPRESSOR_VOLTAGE,
        nx=9,
        ny=9,
        nz=9,
        tolerance=1e-5,
        max_iterations=10_000,
    )

    trajectory = Trajectory(
        efield=electric_field,
        bfield=zero_magnetic_field,
        initial_position=(electron.x0, electron.y0, 1e-6),
        initial_velocity=initial_velocity_from_emission(electron),
    )

    result = trajectory.solve(t_max=5e-8, max_step=1e-12, grid_height=GRID_HEIGHT)

    assert result.status in {TrajectoryStatus.HIT_GRID, TrajectoryStatus.HIT_COLLECTOR}

    final_x, final_y, final_z = result.final_position

    assert math.isfinite(final_x)
    assert math.isfinite(final_y)
    assert math.isfinite(final_z)