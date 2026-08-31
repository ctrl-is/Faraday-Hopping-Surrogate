import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_YIELD_CSV = PROJECT_ROOT / "data" / "processed" / "eder1997_fig4.csv"


@lru_cache(maxsize=4)
def load_yield_curve(
    csv_path: str | Path = DEFAULT_YIELD_CSV,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load digitized H+ on Au secondary-electron yield measurements.

    Returns:
        energies_eV:
            Sorted proton impact energies in eV.

        gammas:
            Mean emitted electrons per incident proton.
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Yield CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required_columns = {
        "energy_eV",
        "gamma_electrons_per_ion",
        "target",
        "projectile",
    }

    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(f"Yield CSV is missing columns: {sorted(missing_columns)}")

    df = df[(df["target"] == "Au") & (df["projectile"] == "H+")].copy()

    df = df.dropna(subset=["energy_eV", "gamma_electrons_per_ion"])

    if df.empty:
        raise ValueError("No H+ on Au yield data were found.")

    # Exact duplicate energies are averaged so np.interp receives
    # strictly increasing x-values.
    df = df.groupby("energy_eV", as_index=False, sort=True)[
        "gamma_electrons_per_ion"
    ].mean()

    energies_eV = df["energy_eV"].to_numpy(dtype=float)
    gammas = df["gamma_electrons_per_ion"].to_numpy(dtype=float)

    if len(energies_eV) < 2:
        raise ValueError("At least two yield data points are required.")

    if np.any(np.diff(energies_eV) <= 0.0):
        raise ValueError("Yield energies must be strictly increasing.")

    if np.any(gammas < 0.0):
        raise ValueError("Secondary-electron yields cannot be negative.")

    return energies_eV, gammas


def gamma_func(U_eV: float, csv_path: str | Path = DEFAULT_YIELD_CSV) -> float:
    """
    Estimate secondary-electron yield for H+ impact on Au.

    gamma is the expected number of emitted electrons per proton.
    """
    if U_eV < 0.0:
        raise ValueError("Proton energy cannot be negative.")

    energies_eV, gammas = load_yield_curve(csv_path)

    if U_eV < energies_eV[0]:
        # Current baseline assumption below the digitized range.
        return 0.0

    if U_eV > energies_eV[-1]:
        # Avoid uncontrolled extrapolation.
        return float(gammas[-1])

    return float(np.interp(U_eV, energies_eV, gammas))


def sample_num_emitted_electrons(
    rng: np.random.Generator,
    U_eV: float,
    csv_path: str | Path = DEFAULT_YIELD_CSV,
) -> int:
    """
    Sample the number of electrons emitted by one proton.
    """
    gamma = gamma_func(U_eV, csv_path)

    return int(rng.poisson(gamma))


def sample_beam_angle(
    rng: np.random.Generator,
    mean_deg: float = 0.0,
    std_deg: float = 20.0,
    min_deg: float = -60.0,
    max_deg: float = 60.0,
) -> float:
    """
    Sample the proton beam angle from a truncated normal distribution.

    Returns:
        Beam angle in radians.
    """
    if std_deg <= 0.0:
        raise ValueError("std_deg must be positive.")

    if min_deg >= max_deg:
        raise ValueError("min_deg must be less than max_deg.")

    while True:
        alpha_deg = rng.normal(loc=mean_deg, scale=std_deg)

        if min_deg <= alpha_deg <= max_deg:
            return math.radians(float(alpha_deg))


def sample_emission_direction(
    rng: np.random.Generator,
    model: str = "cosine_weighted",
    fixed_theta_deg: float = 45.0,
    fixed_psi_deg: float = 0.0,
) -> tuple[float, float]:
    """
    Sample a post-emission electron direction.

    theta:
        Polar angle from the +z surface normal, in radians.

    psi:
        Azimuthal angle in the x-y plane, in radians.
    """
    if model == "fixed":
        theta = math.radians(fixed_theta_deg)
        psi = math.radians(fixed_psi_deg)

        if not 0.0 <= theta <= math.pi / 2.0:
            raise ValueError("fixed_theta_deg must describe an upward direction.")

        return theta, psi

    psi = float(rng.uniform(0.0, 2.0 * math.pi))

    if model == "isotropic":
        cos_theta = float(rng.uniform(0.0, 1.0))
        theta = math.acos(cos_theta)

        return theta, psi

    if model == "cosine_weighted":
        cos_theta = math.sqrt(float(rng.uniform(0.0, 1.0)))
        theta = math.acos(cos_theta)

        return theta, psi

    raise ValueError(f"Unknown emission direction model: {model}")


def sample_secondary_energy_eV(
    rng: np.random.Generator,
    model: str = "fixed",
    mean_energy_eV: float = 5.0,
    max_energy_eV: float = 50.0,
) -> float:
    """
    Sample the electron's post-escape kinetic energy in eV.
    """
    if mean_energy_eV <= 0.0:
        raise ValueError("mean_energy_eV must be positive.")

    if max_energy_eV <= 0.0:
        raise ValueError("max_energy_eV must be positive.")

    if mean_energy_eV > max_energy_eV and model == "fixed":
        raise ValueError("Fixed energy cannot exceed max_energy_eV.")

    if model == "fixed":
        return float(mean_energy_eV)

    if model == "exponential":
        while True:
            energy_eV = float(rng.exponential(scale=mean_energy_eV))

            if energy_eV <= max_energy_eV:
                return energy_eV

    if model == "maxwellian":
        shape = 1.5
        scale = mean_energy_eV / shape

        while True:
            energy_eV = float(rng.gamma(shape=shape, scale=scale))

            if energy_eV <= max_energy_eV:
                return energy_eV

    raise ValueError(f"Unknown secondary energy model: {model}")


@dataclass(frozen=True)
class EmittedElectron:
    x0: float
    y0: float
    energy_eV: float
    theta: float
    psi: float

    def __post_init__(self) -> None:
        if self.energy_eV < 0.0:
            raise ValueError("Electron energy cannot be negative.")

        if not 0.0 <= self.theta <= math.pi / 2.0:
            raise ValueError("theta must describe an upward emission direction.")
