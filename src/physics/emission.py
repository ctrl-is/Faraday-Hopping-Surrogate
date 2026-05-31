from pathlib import Path
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

def load_yield_curve(csv_path: str | Path = "data/processed/eder1997_fig4.csv"):
    df = pd.read_csv(csv_path)

    df = df[(df["target"] == "Au") & (df["projectile"] == "H+")].copy()
    df = df.sort_values("energy_eV")

    energies_eV = df["energy_eV"].to_numpy()
    gammas = df["gamma_electrons_per_ion"].to_numpy()
    return energies_eV, gammas

def gamma_func(
    U_eV: float, 
    csv_path: str | Path = "data/processed/eder1997_fig4.csv") -> float:
    energies_eV, gammas = load_yield_curve(csv_path)

    if U_eV < energies_eV[0]:
        return 0.0
    
    if U_eV > energies_eV[-1]:
        return float(gammas[-1])
    
    return float(np.interp(U_eV, energies_eV, gammas))

def sampled_num_emitted_electrons(
    rng: np.random.Generator,
    U_eV: float, 
    csv_path: str | Path = "data/processed/eder1997_fig4.csv") -> int:
    gamma = gamma_func(U_eV, csv_path)
    return int(rng.poisson(gamma))

def sample_beam_angle(
    rng: np.random.Generator,
    mean_deg: float, 
    std_deg: float,
    min_deg: float,
    max_deg: float) -> float:
    while True:
        alpha = rng.normal(loc=mean_deg, scale=std_deg)

        if alpha >= min_deg and alpha <= max_deg:
            return np.radians(alpha)

def sample_emission_direction(
    rng: np.random.Generator,
    model: str,
    fixed_theta_deg: float,
    fixed_psi_deg: float) -> tuple[float, float]:
    """
    theta: polar angle from +z surface normal, radians
    psi: azimuth angle in x-y plane, radians
    """
    if model == "fixed":
        theta = math.radians(fixed_theta_deg)
        psi = math.radians(fixed_psi_deg)
        return theta, psi
    
    if model == "isotropic":
        psi = rng.uniform(0.0, 2.0 * math.pi)
        cos_theta = rng.uniform(0.0, 1.0)
        theta = math.acos(cos_theta)
        return theta, psi
    
    if model == "cos_weighted":
        psi = rng.uniform(0.0, 2.0 * math.pi)
        cos_theta = math.sqrt(rng.uniform(0.0, 1.0))
        theta = math.acos(cos_theta)
        return theta, psi

    raise ValueError(f"Unknown emission direciton model: {model}")

def sample_secondary_energy_eV(
    rng: np.random.Generator,
    model: str,
    mean_energy_eV: float,
    max_energy_eV: float) -> float:
    if model == "fixed":
        return float(mean_energy_eV)

    if model == "exponential":
        while True:
            energy_eV = rng.exponential(scale=mean_energy_eV)
            if 0.0 <= energy_eV <= max_energy_eV:
                return float(energy_eV)

    if model == "maxwellian":
        shape = 1.5
        scale = mean_energy_eV / shape

        while True:
            energy_eV = rng.gamma(shape=shape, scale=scale)
            if 0.0 <= energy_eV <= max_energy_eV:
                return float(energy_eV)

    raise ValueError(f"Unknown secondary energy model: {model}")

@dataclass
class EmittedElectron:
    x0: float
    y0: float
    energy_eV: float
    theta: float  # polar angle from +z, radians
    psi: float    # azimuth angle in x-y plane, radians