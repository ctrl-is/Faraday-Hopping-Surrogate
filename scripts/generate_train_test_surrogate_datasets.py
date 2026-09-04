"""
Generate train and test datasets for the Faraday cup secondary-electron
plate-hopping surrogate model.

Each row represents one physical design/operating configuration. For each
configuration, this script builds the electrostatic wire-grid suppressor field,
adds a uniform magnetic field, runs the Monte Carlo secondary-electron trajectory
simulation across multiple random seeds, pools the resulting counts, and saves
the hopping/return probabilities as supervised-learning labels.

The train/test split is done by design configuration, not by random seed, so the
same physical setup does not leak into both datasets.
"""

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from src.physics.constants import (
    DEFAULT_BEAM_RADIUS,
    DEFAULT_COLLECTOR_RADIUS,
    MM,
)
from src.physics.fields import (
    make_uniform_magnetic_field,
    solve_wire_grid_field,
)
from src.physics.geometry import CollectorGeometry
from src.simulation.monte_carlo import (
    MonteCarloConfig,
    MonteCarloSummary,
    estimate_hopping_probability,
)

TRAIN_OUTPUT_PATH = Path("data/processed/surrogate_train.csv")
TEST_OUTPUT_PATH = Path("data/processed/surrogate_test.csv")

TRAIN_DESIGNS = 1000
TEST_DESIGNS = 200

NUM_PROTONS_PER_SEED = 10000
SEEDS_PER_DESIGN = 5
DESIGN_SEED = 123

FIELD_NX = 9
FIELD_NY = 9
FIELD_NZ = 9
FIELD_TOLERANCE = 1e-5
FIELD_MAX_ITERATIONS = 10000

NANOTESLA_TO_TESLA = 1e-9


@dataclass(frozen=True)
class DesignParameters:
    gap_width_mm: float
    proton_energy_eV: float
    incidence_angle_deg: float
    grid_spacing_mm: float
    grid_height_mm: float
    wire_radius_mm: float
    suppressor_voltage: float
    magnetic_field_x_nT: float
    magnetic_field_y_nT: float
    magnetic_field_z_nT: float


def sample_designs(num_designs: int, seed: int) -> list[DesignParameters]:
    sampler = qmc.LatinHypercube(d=10, seed=seed)

    unit_samples = sampler.random(n=num_designs)

    lower_bounds = np.array(
        [
            0.05,
            500.0,
            -60.0,
            2.5,
            2.0,
            0.10,
            -120.0,
            -500.0,
            -500.0,
            -500.0,
        ]
    )

    upper_bounds = np.array(
        [
            0.30,
            5_000.0,
            60.0,
            5.0,
            6.0,
            0.35,
            -20.0,
            500.0,
            500.0,
            500.0,
        ]
    )

    scaled_samples = qmc.scale(
        sample=unit_samples,
        l_bounds=lower_bounds,
        u_bounds=upper_bounds,
    )

    designs = []

    for sample in scaled_samples:
        designs.append(
            DesignParameters(
                gap_width_mm=float(sample[0]),
                proton_energy_eV=float(sample[1]),
                incidence_angle_deg=float(sample[2]),
                grid_spacing_mm=float(sample[3]),
                grid_height_mm=float(sample[4]),
                wire_radius_mm=float(sample[5]),
                suppressor_voltage=float(sample[6]),
                magnetic_field_x_nT=float(sample[7]),
                magnetic_field_y_nT=float(sample[8]),
                magnetic_field_z_nT=float(sample[9]),
            )
        )

    return designs


def build_config(parameters: DesignParameters, num_protons: int) -> MonteCarloConfig:
    return MonteCarloConfig(
        num_protons=num_protons,
        proton_energy_eV=parameters.proton_energy_eV,
        modulation_frequency_hz=1024.0,
        incidence_angle_rad=math.radians(parameters.incidence_angle_deg),
        beam_radius=DEFAULT_BEAM_RADIUS,
        grid_spacing=parameters.grid_spacing_mm * MM,
        grid_height=parameters.grid_height_mm * MM,
        wire_radius=parameters.wire_radius_mm * MM,
        collector_voltage=0.0,
        suppressor_voltage=parameters.suppressor_voltage,
        secondary_energy_model="exponential",
        secondary_direction_model="cosine_weighted",
        trajectory_t_max=1e-7,
        trajectory_max_step=1e-11,
    )


def build_magnetic_field(parameters: DesignParameters):
    return make_uniform_magnetic_field(
        bx=parameters.magnetic_field_x_nT * NANOTESLA_TO_TESLA,
        by=parameters.magnetic_field_y_nT * NANOTESLA_TO_TESLA,
        bz=parameters.magnetic_field_z_nT * NANOTESLA_TO_TESLA,
    )


def safe_divide(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0

    return numerator / denominator


def summarize_pooled_results(
    summaries: list[MonteCarloSummary],
) -> dict[str, int | float]:
    num_protons = sum(summary.num_protons for summary in summaries)

    valid_proton_impacts = sum(summary.valid_proton_impacts for summary in summaries)

    total_emitted_electrons = sum(
        summary.total_emitted_electrons for summary in summaries
    )

    returned_to_collector_count = sum(
        summary.returned_to_collector_count for summary in summaries
    )

    ions_with_hopper_count = sum(
        summary.ions_with_hopper_count for summary in summaries
    )

    same_quadrant_count = sum(summary.same_quadrant_count for summary in summaries)

    different_quadrant_count = sum(
        summary.different_quadrant_count for summary in summaries
    )

    gap_count = sum(summary.gap_count for summary in summaries)
    outside_count = sum(summary.outside_count for summary in summaries)
    hit_wire_count = sum(summary.hit_wire_count for summary in summaries)

    passed_grid_opening_count = sum(
        summary.passed_grid_opening_count for summary in summaries
    )

    did_not_return_count = sum(summary.did_not_return_count for summary in summaries)

    solver_failure_count = sum(summary.solver_failure_count for summary in summaries)

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
            numerator=ions_with_hopper_count,
            denominator=valid_proton_impacts,
        ),
        "electron_hopping_probability": safe_divide(
            numerator=different_quadrant_count,
            denominator=total_emitted_electrons,
        ),
        "return_probability": safe_divide(
            numerator=returned_to_collector_count,
            denominator=total_emitted_electrons,
        ),
    }


def design_to_row(
    split: str,
    design_id: int,
    parameters: DesignParameters,
    pooled_summary: dict[str, int | float],
) -> dict[str, int | float | str]:
    magnetic_field_magnitude_nT = math.sqrt(
        parameters.magnetic_field_x_nT**2
        + parameters.magnetic_field_y_nT**2
        + parameters.magnetic_field_z_nT**2
    )

    return {
        "split": split,
        "design_id": design_id,
        "num_seeds": SEEDS_PER_DESIGN,
        "num_protons_per_seed": NUM_PROTONS_PER_SEED,
        "collector_radius_mm": DEFAULT_COLLECTOR_RADIUS / MM,
        "beam_radius_mm": DEFAULT_BEAM_RADIUS / MM,
        "gap_width_mm": parameters.gap_width_mm,
        "proton_energy_eV": parameters.proton_energy_eV,
        "incidence_angle_deg": parameters.incidence_angle_deg,
        "grid_spacing_mm": parameters.grid_spacing_mm,
        "grid_height_mm": parameters.grid_height_mm,
        "wire_radius_mm": parameters.wire_radius_mm,
        "collector_voltage": 0.0,
        "suppressor_voltage": parameters.suppressor_voltage,
        "magnetic_field_x_nT": parameters.magnetic_field_x_nT,
        "magnetic_field_y_nT": parameters.magnetic_field_y_nT,
        "magnetic_field_z_nT": parameters.magnetic_field_z_nT,
        "magnetic_field_magnitude_nT": magnetic_field_magnitude_nT,
        "secondary_energy_model": "exponential",
        "secondary_direction_model": "cosine_weighted",
        "grid_spacing_over_height": (
            parameters.grid_spacing_mm / parameters.grid_height_mm
        ),
        "wire_radius_over_spacing": (
            parameters.wire_radius_mm / parameters.grid_spacing_mm
        ),
        "gap_width_over_grid_spacing": (
            parameters.gap_width_mm / parameters.grid_spacing_mm
        ),
        "field_strength_proxy": (
            abs(parameters.suppressor_voltage) / parameters.grid_height_mm
        ),
        **pooled_summary,
    }


def run_design(
    split: str,
    design_id: int,
    parameters: DesignParameters,
) -> dict[str, int | float | str]:
    geometry = CollectorGeometry(
        gap_width=parameters.gap_width_mm * MM,
        radius=DEFAULT_COLLECTOR_RADIUS,
    )

    electric_field = solve_wire_grid_field(
        grid_spacing=parameters.grid_spacing_mm * MM,
        grid_height=parameters.grid_height_mm * MM,
        wire_radius=parameters.wire_radius_mm * MM,
        collector_voltage=0.0,
        suppressor_voltage=parameters.suppressor_voltage,
        nx=FIELD_NX,
        ny=FIELD_NY,
        nz=FIELD_NZ,
        tolerance=FIELD_TOLERANCE,
        max_iterations=FIELD_MAX_ITERATIONS,
    )

    magnetic_field = build_magnetic_field(parameters=parameters)

    config = build_config(
        parameters=parameters,
        num_protons=NUM_PROTONS_PER_SEED,
    )

    summaries = []

    for seed_index in range(SEEDS_PER_DESIGN):
        rng_seed = DESIGN_SEED + design_id * 10_000 + seed_index
        rng = np.random.default_rng(rng_seed)

        summary = estimate_hopping_probability(
            rng=rng,
            config=config,
            geometry=geometry,
            electric_field=electric_field,
            magnetic_field=magnetic_field,
        )

        summaries.append(summary)

        print(
            f"{split} design {design_id}, seed {seed_index}: "
            f"ion_hopping={summary.ion_hopping_probability:.6f}, "
            f"electron_hopping={summary.electron_hopping_probability:.6f}, "
            f"return={summary.return_probability:.6f}"
        )

    pooled_summary = summarize_pooled_results(summaries=summaries)

    return design_to_row(
        split=split,
        design_id=design_id,
        parameters=parameters,
        pooled_summary=pooled_summary,
    )


def write_rows(
    output_path: Path,
    rows: list[dict[str, int | float | str]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)


def generate_split(
    split: str,
    num_designs: int,
    seed: int,
    output_path: Path,
) -> None:
    designs = sample_designs(num_designs=num_designs, seed=seed)

    rows = []

    for design_id, parameters in enumerate(designs):
        print()
        print(f"Running {split} design {design_id}")
        print(parameters)

        row = run_design(
            split=split,
            design_id=design_id,
            parameters=parameters,
        )

        rows.append(row)

        write_rows(output_path=output_path, rows=rows)

    print()
    print(f"Saved {split} dataset to {output_path}")


def main() -> None:
    generate_split(
        split="train",
        num_designs=TRAIN_DESIGNS,
        seed=DESIGN_SEED,
        output_path=TRAIN_OUTPUT_PATH,
    )

    generate_split(
        split="test",
        num_designs=TEST_DESIGNS,
        seed=DESIGN_SEED + 1,
        output_path=TEST_OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
