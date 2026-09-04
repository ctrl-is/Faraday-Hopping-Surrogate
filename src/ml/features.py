"""
Defines the canonical input features, target labels, metadata columns, and
leakage-exclusion rules for the Faraday cup surrogate model.
"""

from collections.abc import Sequence

import pandas as pd

GEOMETRY_FEATURE_COLUMNS = [
    "gap_width_mm",
    "grid_spacing_mm",
    "grid_height_mm",
    "wire_radius_mm",
]

OPERATING_FEATURE_COLUMNS = [
    "proton_energy_eV",
    "incidence_angle_deg",
    "suppressor_voltage",
]

MAGNETIC_FIELD_FEATURE_COLUMNS = [
    "magnetic_field_x_nT",
    "magnetic_field_y_nT",
    "magnetic_field_z_nT",
    "magnetic_field_magnitude_nT",
]

ENGINEERED_FEATURE_COLUMNS = [
    "grid_spacing_over_height",
    "wire_radius_over_spacing",
    "gap_width_over_grid_spacing",
    "field_strength_proxy",
]

DEFAULT_FEATURE_COLUMNS = (
    GEOMETRY_FEATURE_COLUMNS
    + OPERATING_FEATURE_COLUMNS
    + MAGNETIC_FIELD_FEATURE_COLUMNS
    + ENGINEERED_FEATURE_COLUMNS
)

PRIMARY_TARGET_COLUMN = "ion_hopping_probability"

TARGET_COLUMNS = [
    "ion_hopping_probability",
    "electron_hopping_probability",
    "return_probability",
]

LEAKAGE_COLUMNS = [
    "num_protons",
    "valid_proton_impacts",
    "total_emitted_electrons",
    "returned_to_collector_count",
    "ions_with_hopper_count",
    "same_quadrant_count",
    "different_quadrant_count",
    "gap_count",
    "outside_count",
    "hit_wire_count",
    "passed_grid_opening_count",
    "did_not_return_count",
    "solver_failure_count",
    "ion_hopping_probability",
    "electron_hopping_probability",
    "return_probability",
]

METADATA_COLUMNS = [
    "split",
    "design_id",
    "num_seeds",
    "num_protons_per_seed",
    "collector_radius_mm",
    "beam_radius_mm",
    "collector_voltage",
    "secondary_energy_model",
    "secondary_direction_model",
]


def get_feature_columns() -> list[str]:
    return list(DEFAULT_FEATURE_COLUMNS)


def get_target_column() -> str:
    return PRIMARY_TARGET_COLUMN


def validate_columns_present(dataframe: pd.DataFrame, columns: Sequence[str]) -> None:
    missing_columns = [column for column in columns if column not in dataframe.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def validate_no_leakage(feature_columns: Sequence[str]) -> None:
    leaked_columns = [column for column in feature_columns if column in LEAKAGE_COLUMNS]

    if leaked_columns:
        raise ValueError(f"Feature columns include leakage columns: {leaked_columns}")


def split_features_and_target(
    dataframe: pd.DataFrame,
    target_column: str = PRIMARY_TARGET_COLUMN,
    feature_columns: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    if feature_columns is None:
        feature_columns = DEFAULT_FEATURE_COLUMNS

    validate_no_leakage(feature_columns=feature_columns)
    validate_columns_present(dataframe=dataframe, columns=feature_columns)
    validate_columns_present(dataframe=dataframe, columns=[target_column])

    features = dataframe.loc[:, list(feature_columns)]
    target = dataframe.loc[:, target_column]

    return features, target
