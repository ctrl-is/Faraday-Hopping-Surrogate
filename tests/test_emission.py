import math

import numpy as np
import pandas as pd
import pytest

from src.physics.emission import (
    EmittedElectron,
    gamma_func,
    load_yield_curve,
    sample_emission_direction,
    sample_secondary_energy_eV,
)


@pytest.fixture
def yield_csv(tmp_path):
    path = tmp_path / "yield.csv"

    df = pd.DataFrame(
        {
            "energy_eV": [
                1000.0,
                2000.0,
                3000.0,
                1000.0,
            ],
            "gamma_electrons_per_ion": [
                0.10,
                0.20,
                0.30,
                99.0,
            ],
            "target": [
                "Au",
                "Au",
                "Au",
                "Al",
            ],
            "projectile": [
                "H+",
                "H+",
                "H+",
                "H+",
            ],
        }
    )

    df.to_csv(path, index=False)
    return path


def test_load_yield_curve_filters_material(yield_csv) -> None:
    energies, gammas = load_yield_curve(yield_csv)

    assert np.allclose(
        energies,
        [1000.0, 2000.0, 3000.0],
    )

    assert np.allclose(
        gammas,
        [0.10, 0.20, 0.30],
    )


def test_gamma_interpolation(yield_csv) -> None:
    gamma = gamma_func(U_eV=1500.0, csv_path=yield_csv)

    assert gamma == pytest.approx(0.15)


def test_gamma_below_range(yield_csv) -> None:
    assert gamma_func(500.0, yield_csv) == pytest.approx(0.0)


def test_gamma_above_range(yield_csv) -> None:
    assert gamma_func(4000.0, yield_csv) == pytest.approx(0.30)


@pytest.mark.parametrize(
    "model",
    ["isotropic", "cosine_weighted"],
)
def test_sampled_direction_is_upward(model) -> None:
    rng = np.random.default_rng(123)

    for _ in range(100):
        theta, psi = sample_emission_direction(rng=rng, model=model)

        assert 0.0 <= theta <= math.pi / 2.0
        assert 0.0 <= psi <= 2.0 * math.pi


@pytest.mark.parametrize(
    "model",
    ["fixed", "exponential", "maxwellian"],
)
def test_sampled_energy_is_valid(model) -> None:
    rng = np.random.default_rng(123)

    for _ in range(100):
        energy = sample_secondary_energy_eV(
            rng=rng,
            model=model,
            mean_energy_eV=5.0,
            max_energy_eV=50.0,
        )

        assert 0.0 <= energy <= 50.0


def test_invalid_direction_model() -> None:
    rng = np.random.default_rng(123)

    with pytest.raises(ValueError):
        sample_emission_direction(
            rng,
            model="unknown",
        )


def test_negative_electron_energy_rejected() -> None:
    with pytest.raises(ValueError):
        EmittedElectron(
            x0=0.01,
            y0=0.01,
            energy_eV=-1.0,
            theta=0.0,
            psi=0.0,
        )