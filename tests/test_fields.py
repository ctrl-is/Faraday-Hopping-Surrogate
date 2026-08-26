import numpy as np
import pytest

from src.physics.fields import (
    build_parallel_plate_boundary_conditions,
    compute_electric_field_from_potential,
    initialize_potential_grid,
    make_grid_axes,
    make_interpolated_electric_field,
    solve_laplace,
    solve_parallel_plate_field,
)


def test_make_grid_axes_uses_grid_spacing_for_x_and_y() -> None:
    grid_spacing = 3.5e-3
    grid_height = 3.9e-3

    x_values, y_values, z_values = make_grid_axes(
        grid_spacing=grid_spacing,
        grid_height=grid_height,
        nx=11,
        ny=13,
        nz=15,
    )

    assert x_values[0] == pytest.approx(-grid_spacing / 2.0)
    assert x_values[-1] == pytest.approx(grid_spacing / 2.0)

    assert y_values[0] == pytest.approx(-grid_spacing / 2.0)
    assert y_values[-1] == pytest.approx(grid_spacing / 2.0)

    assert z_values[0] == pytest.approx(0.0)
    assert z_values[-1] == pytest.approx(grid_height)

    assert len(x_values) == 11
    assert len(y_values) == 13
    assert len(z_values) == 15


def test_initialize_potential_grid_linear_in_z() -> None:
    x_values, y_values, z_values = make_grid_axes(
        grid_spacing=3.5e-3,
        grid_height=3.9e-3,
        nx=5,
        ny=7,
        nz=9,
    )

    phi = initialize_potential_grid(
        x_values=x_values,
        y_values=y_values,
        z_values=z_values,
        collector_voltage=0.0,
        suppressor_voltage=-55.0,
    )

    assert phi.shape == (
        len(x_values),
        len(y_values),
        len(z_values),
    )

    assert np.allclose(phi[:, :, 0], 0.0)
    assert np.allclose(phi[:, :, -1], -55.0)

    middle_index = len(z_values) // 2
    expected_middle_voltage = -55.0 * (
        z_values[middle_index] / z_values[-1]
    )

    assert np.allclose(
        phi[:, :, middle_index],
        expected_middle_voltage,
    )


def test_parallel_plate_boundary_conditions_mark_expected_points() -> None:
    x_values, y_values, z_values = make_grid_axes(
        grid_spacing=3.5e-3,
        grid_height=3.9e-3,
        nx=5,
        ny=5,
        nz=5,
    )

    fixed_mask, fixed_values = build_parallel_plate_boundary_conditions(
        x_values=x_values,
        y_values=y_values,
        z_values=z_values,
        collector_voltage=0.0,
        suppressor_voltage=-55.0,
    )

    assert fixed_mask.shape == fixed_values.shape
    assert fixed_mask.shape == (5, 5, 5)

    assert np.all(fixed_mask[:, :, 0])
    assert np.all(fixed_mask[:, :, -1])

    assert np.all(fixed_values[:, :, 0] == pytest.approx(0.0))
    assert np.all(fixed_values[:, :, -1] == pytest.approx(-55.0))

    assert fixed_mask[2, 2, 2] is np.False_ or fixed_mask[2, 2, 2] == False


def test_solve_laplace_keeps_parallel_plate_linear_solution() -> None:
    x_values, y_values, z_values = make_grid_axes(
        grid_spacing=3.5e-3,
        grid_height=3.9e-3,
        nx=9,
        ny=9,
        nz=9,
    )

    phi = initialize_potential_grid(
        x_values=x_values,
        y_values=y_values,
        z_values=z_values,
        collector_voltage=0.0,
        suppressor_voltage=-55.0,
    )

    fixed_mask, fixed_values = build_parallel_plate_boundary_conditions(
        x_values=x_values,
        y_values=y_values,
        z_values=z_values,
        collector_voltage=0.0,
        suppressor_voltage=-55.0,
    )

    solved_phi = solve_laplace(
        phi=phi,
        fixed_mask=fixed_mask,
        fixed_values=fixed_values,
        x_values=x_values,
        y_values=y_values,
        z_values=z_values,
        tolerance=1e-10,
        max_iterations=100,
    )

    assert np.allclose(solved_phi, phi, atol=1e-10)


def test_compute_electric_field_from_parallel_plate_potential() -> None:
    grid_height = 3.9e-3
    collector_voltage = 0.0
    suppressor_voltage = -55.0

    x_values, y_values, z_values = make_grid_axes(
        grid_spacing=3.5e-3,
        grid_height=grid_height,
        nx=11,
        ny=11,
        nz=11,
    )

    phi = initialize_potential_grid(
        x_values=x_values,
        y_values=y_values,
        z_values=z_values,
        collector_voltage=collector_voltage,
        suppressor_voltage=suppressor_voltage,
    )

    Ex, Ey, Ez = compute_electric_field_from_potential(
        phi=phi,
        x_values=x_values,
        y_values=y_values,
        z_values=z_values,
    )

    expected_Ez = -(
        suppressor_voltage - collector_voltage
    ) / grid_height

    assert Ex.shape == phi.shape
    assert Ey.shape == phi.shape
    assert Ez.shape == phi.shape

    assert np.allclose(Ex, 0.0, atol=1e-8)
    assert np.allclose(Ey, 0.0, atol=1e-8)
    assert np.allclose(Ez, expected_Ez, rtol=1e-6)


def test_make_interpolated_electric_field_returns_expected_values() -> None:
    x_values, y_values, z_values = make_grid_axes(
        grid_spacing=3.5e-3,
        grid_height=3.9e-3,
        nx=7,
        ny=7,
        nz=7,
    )

    shape = (
        len(x_values),
        len(y_values),
        len(z_values),
    )

    Ex = np.zeros(shape)
    Ey = np.zeros(shape)
    Ez = np.ones(shape) * 123.0

    electric_field = make_interpolated_electric_field(
        x_values=x_values,
        y_values=y_values,
        z_values=z_values,
        Ex=Ex,
        Ey=Ey,
        Ez=Ez,
    )

    field = electric_field((0.0, 0.0, 1.0e-3))

    assert field[0] == pytest.approx(0.0)
    assert field[1] == pytest.approx(0.0)
    assert field[2] == pytest.approx(123.0)


def test_solve_parallel_plate_field_matches_expected_uniform_field() -> None:
    grid_height = 3.9e-3
    collector_voltage = 0.0
    suppressor_voltage = -55.0

    electric_field = solve_parallel_plate_field(
        grid_spacing=3.5e-3,
        grid_height=grid_height,
        collector_voltage=collector_voltage,
        suppressor_voltage=suppressor_voltage,
        nx=11,
        ny=11,
        nz=11,
        tolerance=1e-10,
        max_iterations=100,
    )

    Ex, Ey, Ez = electric_field((0.0, 0.0, grid_height / 2.0))

    expected_Ez = -(
        suppressor_voltage - collector_voltage
    ) / grid_height

    assert Ex == pytest.approx(0.0, abs=1e-8)
    assert Ey == pytest.approx(0.0, abs=1e-8)
    assert Ez == pytest.approx(expected_Ez, rel=1e-6)