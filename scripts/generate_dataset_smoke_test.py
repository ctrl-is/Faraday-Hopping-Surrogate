"""
Generates a Monte Carlo surrogate-model dataset smoke test for the Faraday cup
secondary-electron hopping simulator.
"""

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.physics.constants import (
    DEFAULT_BEAM_RADIUS,
    DEFAULT_COLLECTOR_RADIUS,
    MM,
)
from src.physics.fields import solve_wire_grid_field
from src.physics.geometry import CollectorGeometry
from src.simulation.monte_carlo import (
    MonteCarloConfig,
    MonteCarloSummary,
    estimate_hopping_probability,
)


OUTPUT_PATH = Path("data/processed/surrogate_dataset.csv")

NUM_DESIGNS = 12
NUM_PROTONS_PER_SEED = 3_000
SEEDS_PER_DESIGN = 2
DESIGN_SEED = 123

FIELD_NX = 9
FIELD_NY = 9
FIELD_NZ = 9
FIELD_TOLERANCE = 1e-5
FIELD_MAX_ITERATIONS = 10_000


@dataclass(frozen=True)
class DesignParameters:
    gap_width: float
    proton_energy_eV: float
    incidence_angle_deg: float
    grid_spacing: float
    grid_height: float
    wire_radius: float
    suppressor_voltage: float


def sample_design_parameters(rng: np.random.Generator) -> DesignParameters:
    return DesignParameters(
        gap_width=float(rng.uniform(0.05, 0.30) * MM),
        proton_energy_eV=float(rng.uniform(500.0, 5_000.0)),
        incidence_angle_deg=float(rng.uniform(-60.0, 60.0)),
        grid_spacing=float(rng.uniform(2.5, 5.0) * MM),
        grid_height=float(rng.uniform(2.0, 6.0) * MM),
        wire_radius=float(rng.uniform(0.10, 0.35) * MM),
        suppressor_voltage=float(-rng.uniform(20.0, 100.0)),
    )


def build_config(parameters: DesignParameters, num_protons: int) -> MonteCarloConfig:
    return MonteCarloConfig(
        num_protons=num_protons,
        proton_energy_eV=parameters.proton_energy_eV,
        modulation_frequency_hz=1024.0,
        incidence_angle_rad=math.radians(parameters.incidence_angle_deg),
        beam_radius=DEFAULT_BEAM_RADIUS,
        grid_spacing=parameters.grid_spacing,
        grid_height=parameters.grid_height,
        wire_radius=parameters.wire_radius,
        collector_voltage=0.0,
        suppressor_voltage=parameters.suppressor_voltage,
        secondary_energy_model="exponential",
        secondary_direction_model="cosine_weighted",
        trajectory_t_max=1e-7,
        trajectory_max_step=1e-11,
    )


def safe_divide(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0

    return numerator / denominator


def summarize_pooled_results(summaries: list[MonteCarloSummary]) -> dict[str, int | float]:
    num_protons = sum(summary.num_protons for summary in summaries)
    valid_proton_impacts = sum(
        summary.valid_proton_impacts for summary in summaries
    )
    total_emitted_electrons = sum(
        summary.total_emitted_electrons for summary in summaries
    )
    returned_to_collector_count = sum(
        summary.returned_to_collector_count for summary in summaries
    )
    ions_with_hopper_count = sum(
        summary.ions_with_hopper_count for summary in summaries
    )
    same_quadrant_count = sum(
        summary.same_quadrant_count for summary in summaries
    )
    different_quadrant_count = sum(
        summary.different_quadrant_count for summary in summaries
    )
    gap_count = sum(summary.gap_count for summary in summaries)
    outside_count = sum(summary.outside_count for summary in summaries)
    hit_wire_count = sum(summary.hit_wire_count for summary in summaries)
    passed_grid_opening_count = sum(
        summary.passed_grid_opening_count for summary in summaries
    )
    did_not_return_count = sum(
        summary.did_not_return_count for summary in summaries
    )
    solver_failure_count = sum(
        summary.solver_failure_count for summary in summaries
    )

    return {
        "num_protons": num_protons,
        "valid_proton_impacts": valid_proton_impacts,
        "total_emitted_electrons": total_emitted_electrons,
        "returned_to_collector_count": returned_to_collector_count,
        "ions_with_hopper_count": ions_with_hopper_count,
        "same_quadrant_count": same_quadrant_count,
        "different_quadrant_count": different_quadrant_count,
        "gap_count": gap_count,
        "outside_count": outside_count,
        "hit_wire_count": hit_wire_count,
        "passed_grid_opening_count": passed_grid_opening_count,
        "did_not_return_count": did_not_return_count,
        "solver_failure_count": solver_failure_count,
        "ion_hopping_probability": safe_divide(
            ions_with_hopper_count,
            valid_proton_impacts,
        ),
        "electron_hopping_probability": safe_divide(
            different_quadrant_count,
            total_emitted_electrons,
        ),
        "return_probability": safe_divide(
            returned_to_collector_count,
            total_emitted_electrons,
        ),
    }


def design_to_row(
    design_id: int, parameters: DesignParameters, pooled_summary: dict[str, int | float],
) -> dict[str, int | float | str]:
    return {
        "design_id": design_id,
        "num_seeds": SEEDS_PER_DESIGN,
        "num_protons_per_seed": NUM_PROTONS_PER_SEED,
        "collector_radius_mm": DEFAULT_COLLECTOR_RADIUS / MM,
        "beam_radius_mm": DEFAULT_BEAM_RADIUS / MM,
        "gap_width_mm": parameters.gap_width / MM,
        "proton_energy_eV": parameters.proton_energy_eV,
        "incidence_angle_deg": parameters.incidence_angle_deg,
        "grid_spacing_mm": parameters.grid_spacing / MM,
        "grid_height_mm": parameters.grid_height / MM,
        "wire_radius_mm": parameters.wire_radius / MM,
        "collector_voltage": 0.0,
        "suppressor_voltage": parameters.suppressor_voltage,
        "secondary_energy_model": "exponential",
        "secondary_direction_model": "cosine_weighted",
        **pooled_summary,
    }


def run_design(
    design_id: int, parameters: DesignParameters,
) -> dict[str, int | float | str]:
    geometry = CollectorGeometry(
        gap_width=parameters.gap_width, radius=DEFAULT_COLLECTOR_RADIUS,
    )

    electric_field = solve_wire_grid_field(
        grid_spacing=parameters.grid_spacing,
        grid_height=parameters.grid_height,
        wire_radius=parameters.wire_radius,
        collector_voltage=0.0,
        suppressor_voltage=parameters.suppressor_voltage,
        nx=FIELD_NX,
        ny=FIELD_NY,
        nz=FIELD_NZ,
        tolerance=FIELD_TOLERANCE,
        max_iterations=FIELD_MAX_ITERATIONS,
    )

    config = build_config(parameters=parameters, num_protons=NUM_PROTONS_PER_SEED)

    summaries = []

    for seed in range(SEEDS_PER_DESIGN):
        rng_seed = DESIGN_SEED + design_id * 10_000 + seed
        rng = np.random.default_rng(rng_seed)

        summary = estimate_hopping_probability(
            rng=rng,
            config=config,
            geometry=geometry,
            electric_field=electric_field,
        )

        summaries.append(summary)

        print(
            f"Design {design_id}, seed {seed}: "
            f"ion_hopping={summary.ion_hopping_probability:.6f}, "
            f"electron_hopping={summary.electron_hopping_probability:.6f}, "
            f"return={summary.return_probability:.6f}"
        )

    pooled_summary = summarize_pooled_results(summaries=summaries)

    return design_to_row(
        design_id=design_id,
        parameters=parameters,
        pooled_summary=pooled_summary,
    )


def main() -> None:
    design_rng = np.random.default_rng(DESIGN_SEED)
    rows = []

    for design_id in range(NUM_DESIGNS):
        parameters = sample_design_parameters(rng=design_rng)

        print()
        print(f"Running design {design_id}")
        print(parameters)

        row = run_design(design_id=design_id, parameters=parameters)

        rows.append(row)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"Saved dataset to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()