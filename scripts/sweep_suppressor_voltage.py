"""
Sweep suppressor voltage for the Faraday cup secondary-electron hopping
simulator.

This script runs the Monte Carlo pipeline across several suppressor voltages
and random seeds, then saves one CSV row per voltage/seed result. The output
is an early dataset format for studying how hopping probability changes with
design parameters.
"""

import csv
import math
from pathlib import Path

import numpy as np

from src.physics.constants import (
    DEFAULT_BEAM_RADIUS,
    DEFAULT_COLLECTOR_RADIUS,
    DEFAULT_GAP_WIDTH,
    DEFAULT_GRID_HEIGHT,
    DEFAULT_GRID_SPACING,
    DEFAULT_WIRE_RADIUS,
)
from src.physics.fields import solve_wire_grid_field
from src.physics.geometry import CollectorGeometry
from src.simulation.monte_carlo import (
    MonteCarloConfig,
    MonteCarloSummary,
    estimate_hopping_probability,
)

OUTPUT_PATH = Path("data/processed/suppressor_voltage_sweep.csv")


def summary_to_row(
    summary: MonteCarloSummary,
    seed: int,
    suppressor_voltage: float,
    config: MonteCarloConfig,
) -> dict[str, int | float | str]:
    return {
        "seed": seed,
        "num_protons": summary.num_protons,
        "proton_energy_eV": config.proton_energy_eV,
        "modulation_frequency_hz": config.modulation_frequency_hz,
        "incidence_angle_rad": config.incidence_angle_rad,
        "beam_radius": config.beam_radius,
        "gap_width": DEFAULT_GAP_WIDTH,
        "grid_spacing": config.grid_spacing,
        "grid_height": config.grid_height,
        "wire_radius": config.wire_radius,
        "collector_voltage": config.collector_voltage,
        "suppressor_voltage": suppressor_voltage,
        "secondary_energy_model": config.secondary_energy_model,
        "secondary_direction_model": config.secondary_direction_model,
        "valid_proton_impacts": summary.valid_proton_impacts,
        "total_emitted_electrons": summary.total_emitted_electrons,
        "returned_to_collector_count": summary.returned_to_collector_count,
        "ions_with_hopper_count": summary.ions_with_hopper_count,
        "same_quadrant_count": summary.same_quadrant_count,
        "different_quadrant_count": summary.different_quadrant_count,
        "gap_count": summary.gap_count,
        "outside_count": summary.outside_count,
        "hit_wire_count": summary.hit_wire_count,
        "passed_grid_opening_count": summary.passed_grid_opening_count,
        "did_not_return_count": summary.did_not_return_count,
        "solver_failure_count": summary.solver_failure_count,
        "ion_hopping_probability": summary.ion_hopping_probability,
        "electron_hopping_probability": summary.electron_hopping_probability,
        "return_probability": summary.return_probability,
    }


def main() -> None:
    suppressor_voltages = [-20.0, -35.0, -55.0, -75.0, -100.0]
    seeds = range(3)

    geometry = CollectorGeometry(
        gap_width=DEFAULT_GAP_WIDTH,
        radius=DEFAULT_COLLECTOR_RADIUS,
    )

    rows = []

    for suppressor_voltage in suppressor_voltages:
        electric_field = solve_wire_grid_field(
            grid_spacing=DEFAULT_GRID_SPACING,
            grid_height=DEFAULT_GRID_HEIGHT,
            wire_radius=DEFAULT_WIRE_RADIUS,
            collector_voltage=0.0,
            suppressor_voltage=suppressor_voltage,
            nx=9,
            ny=9,
            nz=9,
            tolerance=1e-5,
            max_iterations=10_000,
        )

        for seed in seeds:
            rng = np.random.default_rng(seed)

            config = MonteCarloConfig(
                num_protons=10_000,
                proton_energy_eV=800.0,
                modulation_frequency_hz=1024.0,
                incidence_angle_rad=math.radians(0.0),
                beam_radius=DEFAULT_BEAM_RADIUS,
                grid_spacing=DEFAULT_GRID_SPACING,
                grid_height=DEFAULT_GRID_HEIGHT,
                wire_radius=DEFAULT_WIRE_RADIUS,
                collector_voltage=0.0,
                suppressor_voltage=suppressor_voltage,
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

            rows.append(
                summary_to_row(
                    summary=summary,
                    seed=seed,
                    suppressor_voltage=suppressor_voltage,
                    config=config,
                )
            )

            print(summary)
            print(f"Seed #: {seed}")
            print(f"Suppressor Voltage: {suppressor_voltage}")
            print(f"Ion hopping probability: {summary.ion_hopping_probability:.6f}")
            print(
                "Electron hopping probability: "
                f"{summary.electron_hopping_probability:.6f}"
            )
            print(f"Return probability: {summary.return_probability:.6f}")
            print()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
