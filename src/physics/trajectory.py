import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

from src.physics.fields import (
    FieldFunction,
    Vector3,
    is_inside_wire,
    zero_magnetic_field,
)

from .constants import E_CHARGE, ELECTRON_CHARGE, ELECTRON_MASS
from .emission import EmittedElectron


def initial_velocity_from_emission(electron: EmittedElectron) -> Vector3:
    """
    Convert electron energy and emission angles into velocity components.

    theta is measured from the +z surface normal.
    psi is measured in the x-y plane.
    """
    energy_joules = electron.energy_eV * E_CHARGE

    speed = math.sqrt(2.0 * energy_joules / ELECTRON_MASS)

    vx = speed * math.sin(electron.theta) * math.cos(electron.psi)

    vy = speed * math.sin(electron.theta) * math.sin(electron.psi)

    vz = speed * math.cos(electron.theta)

    return vx, vy, vz


class TrajectoryStatus(Enum):
    HIT_COLLECTOR = "hit_collector"
    HIT_GRID = "hit_grid"
    HIT_WIRE = "hit_wire"
    PASSED_GRID_OPENING = "passed_grid_opening"
    LEFT_RADIAL_DOMAIN = "left_radial_domain"
    TIMEOUT = "timeout"
    SOLVER_FAILURE = "solver_failure"


@dataclass(frozen=True)
class TrajectoryResult:
    status: TrajectoryStatus
    final_position: Vector3
    final_velocity: Vector3
    event_time: float | None
    solution: Any

    @property
    def hit_collector(self) -> bool:
        return self.status == TrajectoryStatus.HIT_COLLECTOR

    @property
    def return_time(self) -> float | None:
        if self.hit_collector:
            return self.event_time

        return None


class Trajectory:
    def __init__(
        self,
        efield: FieldFunction,
        initial_position: Vector3,
        initial_velocity: Vector3,
        bfield: FieldFunction = zero_magnetic_field,
    ):
        self.efield = efield
        self.bfield = bfield
        self.initial_position = initial_position
        self.initial_velocity = initial_velocity

    @classmethod
    def from_emitted_electron(
        cls,
        electron: EmittedElectron,
        efield: FieldFunction,
        bfield: FieldFunction = zero_magnetic_field,
    ) -> "Trajectory":
        initial_velocity = initial_velocity_from_emission(electron)

        return cls(
            efield=efield,
            bfield=bfield,
            initial_position=(
                electron.x0,
                electron.y0,
                0.0,
            ),
            initial_velocity=initial_velocity,
        )

    def equations_of_motion(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        State:
            [x, y, z, vx, vy, vz]

        Lorentz-force equation:
            m a = q(E + v × B)
        """
        x, y, z, vx, vy, vz = state

        position = (x, y, z)

        velocity = np.array([vx, vy, vz], dtype=float)

        electric_field = np.asarray(self.efield(position), dtype=float)
        magnetic_field = np.asarray(self.bfield(position), dtype=float)

        if electric_field.shape != (3,):
            raise ValueError("Electric field must return three components.")

        if magnetic_field.shape != (3,):
            raise ValueError("Magnetic field must return three components.")

        lorentz_force = ELECTRON_CHARGE * (
            electric_field + np.cross(velocity, magnetic_field)
        )

        acceleration = lorentz_force / ELECTRON_MASS

        return np.array(
            [
                vx,
                vy,
                vz,
                acceleration[0],
                acceleration[1],
                acceleration[2],
            ],
            dtype=float,
        )

    def solve(
        self,
        t_max: float = 1e-7,
        max_step: float = 1e-10,
        grid_height: float | None = None,
        radial_limit: float | None = None,
        wire_grid_spacing: float | None = None,
        wire_radius: float | None = None,
    ) -> TrajectoryResult:
        if t_max <= 0.0:
            raise ValueError("t_max must be positive.")

        if max_step <= 0.0:
            raise ValueError("max_step must be positive.")

        if (wire_grid_spacing is None) != (wire_radius is None):
            raise ValueError(
                "wire_grid_spacing and wire_radius must either both be provided or both be omitted."
            )

        if wire_grid_spacing is not None and grid_height is None:
            raise ValueError("grid_height is required when wire geometry is provided.")

        x0, y0, z0 = self.initial_position
        vx0, vy0, vz0 = self.initial_velocity

        # Start slightly above the surface so z = 0 can be detected cleanly.
        z0 = max(z0, 1e-12)

        initial_state = np.array(
            [x0, y0, z0, vx0, vy0, vz0],
            dtype=float,
        )

        def collector_event(t: float, state: np.ndarray) -> float:
            return float(state[2])

        collector_event.terminal = True
        collector_event.direction = -1

        events = [collector_event]
        event_statuses = [TrajectoryStatus.HIT_COLLECTOR]

        if grid_height is not None:
            if grid_height <= 0.0:
                raise ValueError("grid_height must be positive.")

            def grid_event(t: float, state: np.ndarray) -> float:
                return float(state[2] - grid_height)

            grid_event.terminal = True
            grid_event.direction = 1

            events.append(grid_event)
            event_statuses.append(TrajectoryStatus.HIT_GRID)

        if radial_limit is not None:
            if radial_limit <= 0.0:
                raise ValueError("radial_limit must be positive.")

            def radial_event(t: float, state: np.ndarray) -> float:
                x, y = state[0], state[1]
                return float(math.sqrt(x**2 + y**2) - radial_limit)

            radial_event.terminal = True
            radial_event.direction = 1

            events.append(radial_event)
            event_statuses.append(TrajectoryStatus.LEFT_RADIAL_DOMAIN)

        absolute_tolerance = np.array(
            [
                1e-12,  # x
                1e-12,  # y
                1e-12,  # z
                1e-3,  # vx
                1e-3,  # vy
                1e-3,  # vz
            ],
            dtype=float,
        )

        solution = solve_ivp(
            fun=self.equations_of_motion,
            t_span=(0.0, t_max),
            y0=initial_state,
            events=events,
            rtol=1e-8,
            atol=absolute_tolerance,
            max_step=max_step,
        )

        final_state = solution.y[:, -1]
        event_time = None

        if not solution.success:
            status = TrajectoryStatus.SOLVER_FAILURE
        else:
            status = TrajectoryStatus.TIMEOUT

            for index, event_times in enumerate(solution.t_events):
                if len(event_times) == 0:
                    continue

                status = event_statuses[index]
                event_time = float(event_times[0])
                final_state = solution.y_events[index][0]
                break

        if (
            status == TrajectoryStatus.HIT_GRID
            and wire_grid_spacing is not None
            and wire_radius is not None
        ):
            if grid_height is None:
                raise RuntimeError("grid_height unexpectedly missing.")

            final_x = float(final_state[0])
            final_y = float(final_state[1])
            final_z = float(final_state[2])

            if is_inside_wire(
                x=final_x,
                y=final_y,
                z=final_z,
                grid_spacing=wire_grid_spacing,
                grid_height=grid_height,
                wire_radius=wire_radius,
            ):
                status = TrajectoryStatus.HIT_WIRE
            else:
                status = TrajectoryStatus.PASSED_GRID_OPENING

        return TrajectoryResult(
            status=status,
            final_position=(
                float(final_state[0]),
                float(final_state[1]),
                float(final_state[2]),
            ),
            final_velocity=(
                float(final_state[3]),
                float(final_state[4]),
                float(final_state[5]),
            ),
            event_time=event_time,
            solution=solution,
        )
