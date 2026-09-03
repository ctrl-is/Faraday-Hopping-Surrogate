"""
Run a small end-to-end Monte Carlo smoke test for the Faraday cup
secondary-electron hopping simulator.

This script builds one baseline collector/grid configuration, runs the
physics-based Monte Carlo pipeline, and prints the resulting hopping and
return probabilities.
"""

import math

import numpy as np

from src.physics.constants import (
    DEFAULT_BEAM_RADIUS,
    DEFAULT_COLLECTOR_RADIUS,
    DEFAULT_GAP_WIDTH,
    DEFAULT_GRID_HEIGHT,
    DEFAULT_GRID_SPACING,
    DEFAULT_SUPPRESSOR_VOLTAGE,
    DEFAULT_WIRE_RADIUS,
)
from src.physics.fields import solve_wire_grid_field
from src.physics.geometry import CollectorGeometry
from src.simulation.monte_carlo import (
    MonteCarloConfig,
    estimate_hopping_probability,
)


def main() -> None:
    rng = np.random.default_rng(42)

    geometry = CollectorGeometry(
        gap_width=DEFAULT_GAP_WIDTH,
        radius=DEFAULT_COLLECTOR_RADIUS,
    )

    electric_field = solve_wire_grid_field(
        grid_spacing=DEFAULT_GRID_SPACING,
        grid_height=DEFAULT_GRID_HEIGHT,
        wire_radius=DEFAULT_WIRE_RADIUS,
        collector_voltage=0.0,
        suppressor_voltage=DEFAULT_SUPPRESSOR_VOLTAGE,
        nx=9,
        ny=9,
        nz=9,
        tolerance=1e-5,
        max_iterations=10_000,
    )

    config = MonteCarloConfig(
        num_protons=1_000,
        proton_energy_eV=800.0,
        modulation_frequency_hz=1024.0,
        incidence_angle_rad=math.radians(0.0),
        beam_radius=DEFAULT_BEAM_RADIUS,
        grid_spacing=DEFAULT_GRID_SPACING,
        grid_height=DEFAULT_GRID_HEIGHT,
        wire_radius=DEFAULT_WIRE_RADIUS,
        collector_voltage=0.0,
        suppressor_voltage=DEFAULT_SUPPRESSOR_VOLTAGE,
        secondary_energy_model="exponential",
        secondary_direction_model="cosine_weighted",
        trajectory_t_max=1e-7,
        trajectory_max_step=1e-11,
    )

    summary = estimate_hopping_probability(
        rng=rng,
        config=config,
        geometry=geometry,
        electric_field=electric_field,
    )

    print(summary)
    print(f"Ion hopping probability: {summary.ion_hopping_probability:.6f}")
    print(f"Electron hopping probability: {summary.electron_hopping_probability:.6f}")
    print(f"Return probability: {summary.return_probability:.6f}")


if __name__ == "__main__":
    main()
