import numpy as np
import pytest

from src.physics.fields import (
    build_parallel_plate_boundary_conditions,
    build_wire_grid_boundary_conditions,
    build_wire_mask,
    compute_electric_field_from_potential,
    initialize_potential_grid,
    is_inside_wire,
    make_grid_axes,
    make_interpolated_electric_field,
    solve_laplace,
    solve_parallel_plate_field,
    solve_wire_grid_field,
)

GRID_SPACING = 3.5e-3
GRID_HEIGHT = 3.9e-3
WIRE_RADIUS = 2.5e-4
COLLECTOR_VOLTAGE = 0.0
SUPPRESSOR_VOLTAGE = -55.0


def test_make_grid_axes_uses_grid_spacing_for_x_and_y() -> None:
    x_values, y_values, z_values = make_grid_axes(
        grid_spacing=GRID_SPACING,
        grid_height=GRID_HEIGHT,
        nx=11,
        ny=13,
        nz=15,
    )

    assert x_values[0] == pytest.approx(-GRID_SPACING / 2.0)
    assert x_values[-1] == pytest.approx(GRID_SPACING / 2.0)

    assert y_values[0] == pytest.approx(-GRID_SPACING / 2.0)
    assert y_values[-1] == pytest.approx(GRID_SPACING / 2.0)

    assert z_values[0] == pytest.approx(0.0)
    assert z_values[-1] == pytest.approx(GRID_HEIGHT)

    assert len(x_values) == 11
    assert len(y_values) == 13
    assert len(z_values) == 15


def test_make_grid_axes_can_extend_above_wire_plane() -> None:
    x_values, y_values, z_values = make_grid_axes(
        grid_spacing=GRID_SPACING,
        grid_height=GRID_HEIGHT,
        nx=5,
        ny=5,
        nz=5,
        domain_height=2.0 * GRID_HEIGHT,
    )

    assert x_values[0] == pytest.approx(-GRID_SPACING / 2.0)
    assert x_values[-1] == pytest.approx(GRID_SPACING / 2.0)

    assert y_values[0] == pytest.approx(-GRID_SPACING / 2.0)
    assert y_values[-1] == pytest.approx(GRID_SPACING / 2.0)

    assert z_values[0] == pytest.approx(0.0)
    assert z_values[2] == pytest.approx(GRID_HEIGHT)
    assert z_values[-1] == pytest.approx(2.0 * GRID_HEIGHT)


def test_initialize_potential_grid_linear_in_z() -> None:
    x_values, y_values, z_values = make_grid_axes(
        grid_spacing=GRID_SPACING,
        grid_height=GRID_HEIGHT,
        nx=5,
        ny=7,
        nz=9,
    )

    phi = initialize_potential_grid(
        x_values=x_values,
        y_values=y_values,
        z_values=z_values,
        collector_voltage=COLLECTOR_VOLTAGE,
        suppressor_voltage=SUPPRESSOR_VOLTAGE,
    )

    assert phi.shape == (len(x_values), len(y_values), len(z_values))
    assert np.allclose(phi[:, :, 0], COLLECTOR_VOLTAGE)
    assert np.allclose(phi[:, :, -1], SUPPRESSOR_VOLTAGE)

    middle_index = len(z_values) // 2
    expected_middle_voltage = SUPPRESSOR_VOLTAGE * (
        z_values[middle_index] / z_values[-1]
    )

    assert np.allclose(phi[:, :, middle_index], expected_middle_voltage)


def test_parallel_plate_boundary_conditions_mark_expected_points() -> None:
    x_values, y_values, z_values = make_grid_axes(
        grid_spacing=GRID_SPACING,
        grid_height=GRID_HEIGHT,
        nx=5,
        ny=5,
        nz=5,
    )

    fixed_mask, fixed_values = build_parallel_plate_boundary_conditions(
        x_values=x_values,
        y_values=y_values,
        z_values=z_values,
        collector_voltage=COLLECTOR_VOLTAGE,
        suppressor_voltage=SUPPRESSOR_VOLTAGE,
    )

    assert fixed_mask.shape == fixed_values.shape
    assert fixed_mask.shape == (5, 5, 5)

    assert np.all(fixed_mask[:, :, 0])
    assert np.all(fixed_mask[:, :, -1])

    assert np.allclose(fixed_values[:, :, 0], COLLECTOR_VOLTAGE)
    assert np.allclose(fixed_values[:, :, -1], SUPPRESSOR_VOLTAGE)

    assert not fixed_mask[2, 2, 2]


def test_solve_laplace_keeps_parallel_plate_linear_solution() -> None:
    x_values, y_values, z_values = make_grid_axes(
        grid_spacing=GRID_SPACING,
        grid_height=GRID_HEIGHT,
        nx=9,
        ny=9,
        nz=9,
    )

    phi = initialize_potential_grid(
        x_values=x_values,
        y_values=y_values,
        z_values=z_values,
        collector_voltage=COLLECTOR_VOLTAGE,
        suppressor_voltage=SUPPRESSOR_VOLTAGE,
    )

    fixed_mask, fixed_values = build_parallel_plate_boundary_conditions(
        x_values=x_values,
        y_values=y_values,
        z_values=z_values,
        collector_voltage=COLLECTOR_VOLTAGE,
        suppressor_voltage=SUPPRESSOR_VOLTAGE,
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
    x_values, y_values, z_values = make_grid_axes(
        grid_spacing=GRID_SPACING,
        grid_height=GRID_HEIGHT,
        nx=11,
        ny=11,
        nz=11,
    )

    phi = initialize_potential_grid(
        x_values=x_values,
        y_values=y_values,
        z_values=z_values,
        collector_voltage=COLLECTOR_VOLTAGE,
        suppressor_voltage=SUPPRESSOR_VOLTAGE,
    )

    Ex, Ey, Ez = compute_electric_field_from_potential(
        phi=phi,
        x_values=x_values,
        y_values=y_values,
        z_values=z_values,
    )

    expected_Ez = -(SUPPRESSOR_VOLTAGE - COLLECTOR_VOLTAGE) / GRID_HEIGHT

    assert Ex.shape == phi.shape
    assert Ey.shape == phi.shape
    assert Ez.shape == phi.shape

    assert np.allclose(Ex, 0.0, atol=1e-8)
    assert np.allclose(Ey, 0.0, atol=1e-8)
    assert np.allclose(Ez, expected_Ez, rtol=1e-6)


def test_make_interpolated_electric_field_returns_expected_values() -> None:
    x_values, y_values, z_values = make_grid_axes(
        grid_spacing=GRID_SPACING,
        grid_height=GRID_HEIGHT,
        nx=7,
        ny=7,
        nz=7,
    )

    shape = (len(x_values), len(y_values), len(z_values))

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
    electric_field = solve_parallel_plate_field(
        grid_spacing=GRID_SPACING,
        grid_height=GRID_HEIGHT,
        collector_voltage=COLLECTOR_VOLTAGE,
        suppressor_voltage=SUPPRESSOR_VOLTAGE,
        nx=11,
        ny=11,
        nz=11,
        tolerance=1e-10,
        max_iterations=100,
    )

    Ex, Ey, Ez = electric_field((0.0, 0.0, GRID_HEIGHT / 2.0))

    expected_Ez = -(SUPPRESSOR_VOLTAGE - COLLECTOR_VOLTAGE) / GRID_HEIGHT

    assert Ex == pytest.approx(0.0, abs=1e-8)
    assert Ey == pytest.approx(0.0, abs=1e-8)
    assert Ez == pytest.approx(expected_Ez, rel=1e-6)


def test_is_inside_wire_detects_x_directed_wire() -> None:
    assert is_inside_wire(
        x=GRID_SPACING / 4.0,
        y=0.0,
        z=GRID_HEIGHT,
        grid_spacing=GRID_SPACING,
        grid_height=GRID_HEIGHT,
        wire_radius=WIRE_RADIUS,
    )


def test_is_inside_wire_detects_y_directed_wire() -> None:
    assert is_inside_wire(
        x=0.0,
        y=GRID_SPACING / 4.0,
        z=GRID_HEIGHT,
        grid_spacing=GRID_SPACING,
        grid_height=GRID_HEIGHT,
        wire_radius=WIRE_RADIUS,
    )


def test_is_inside_wire_rejects_mesh_opening() -> None:
    assert not is_inside_wire(
        x=GRID_SPACING / 4.0,
        y=GRID_SPACING / 4.0,
        z=GRID_HEIGHT,
        grid_spacing=GRID_SPACING,
        grid_height=GRID_HEIGHT,
        wire_radius=WIRE_RADIUS,
    )


def test_is_inside_wire_validates_geometry_inputs() -> None:
    with pytest.raises(ValueError, match="grid_spacing must be positive"):
        is_inside_wire(
            x=0.0,
            y=0.0,
            z=GRID_HEIGHT,
            grid_spacing=0.0,
            grid_height=GRID_HEIGHT,
            wire_radius=WIRE_RADIUS,
        )

    with pytest.raises(ValueError, match="grid_height must be positive"):
        is_inside_wire(
            x=0.0,
            y=0.0,
            z=GRID_HEIGHT,
            grid_spacing=GRID_SPACING,
            grid_height=0.0,
            wire_radius=WIRE_RADIUS,
        )

    with pytest.raises(ValueError, match="wire_radius must be positive"):
        is_inside_wire(
            x=0.0,
            y=0.0,
            z=GRID_HEIGHT,
            grid_spacing=GRID_SPACING,
            grid_height=GRID_HEIGHT,
            wire_radius=0.0,
        )

    with pytest.raises(ValueError, match="wire_radius must be less than half"):
        is_inside_wire(
            x=0.0,
            y=0.0,
            z=GRID_HEIGHT,
            grid_spacing=GRID_SPACING,
            grid_height=GRID_HEIGHT,
            wire_radius=GRID_SPACING / 2.0,
        )


def test_build_wire_mask_marks_cross_shape_at_wire_height() -> None:
    x_values, y_values, z_values = make_grid_axes(
        grid_spacing=GRID_SPACING,
        grid_height=GRID_HEIGHT,
        nx=5,
        ny=5,
        nz=5,
        domain_height=2.0 * GRID_HEIGHT,
    )

    wire_mask = build_wire_mask(
        x_values=x_values,
        y_values=y_values,
        z_values=z_values,
        grid_spacing=GRID_SPACING,
        grid_height=GRID_HEIGHT,
        wire_radius=WIRE_RADIUS,
    )

    center_x_index = 2
    center_y_index = 2
    wire_z_index = 2

    assert wire_mask[:, center_y_index, wire_z_index].all()
    assert wire_mask[center_x_index, :, wire_z_index].all()

    assert not wire_mask[1, 1, wire_z_index]
    assert not wire_mask[3, 3, wire_z_index]


def test_wire_grid_boundary_conditions_fix_expected_points() -> None:
    x_values, y_values, z_values = make_grid_axes(
        grid_spacing=GRID_SPACING,
        grid_height=GRID_HEIGHT,
        nx=5,
        ny=5,
        nz=5,
        domain_height=2.0 * GRID_HEIGHT,
    )

    fixed_mask, fixed_values = build_wire_grid_boundary_conditions(
        x_values=x_values,
        y_values=y_values,
        z_values=z_values,
        collector_voltage=COLLECTOR_VOLTAGE,
        suppressor_voltage=SUPPRESSOR_VOLTAGE,
        grid_spacing=GRID_SPACING,
        grid_height=GRID_HEIGHT,
        wire_radius=WIRE_RADIUS,
    )

    center_x_index = 2
    center_y_index = 2
    wire_z_index = 2

    assert fixed_mask[:, :, 0].all()
    assert np.allclose(fixed_values[:, :, 0], COLLECTOR_VOLTAGE)

    assert fixed_mask[:, :, -1].all()
    assert np.allclose(fixed_values[:, :, -1], SUPPRESSOR_VOLTAGE)

    assert fixed_mask[0, :, :].all()
    assert fixed_mask[-1, :, :].all()
    assert fixed_mask[:, 0, :].all()
    assert fixed_mask[:, -1, :].all()

    assert fixed_mask[:, center_y_index, wire_z_index].all()
    assert fixed_mask[center_x_index, :, wire_z_index].all()

    assert fixed_values[center_x_index, center_y_index, wire_z_index] == pytest.approx(
        SUPPRESSOR_VOLTAGE
    )


def test_wire_grid_boundary_conditions_leave_mesh_opening_unfixed() -> None:
    x_values, y_values, z_values = make_grid_axes(
        grid_spacing=GRID_SPACING,
        grid_height=GRID_HEIGHT,
        nx=5,
        ny=5,
        nz=5,
        domain_height=2.0 * GRID_HEIGHT,
    )

    fixed_mask, _ = build_wire_grid_boundary_conditions(
        x_values=x_values,
        y_values=y_values,
        z_values=z_values,
        collector_voltage=COLLECTOR_VOLTAGE,
        suppressor_voltage=SUPPRESSOR_VOLTAGE,
        grid_spacing=GRID_SPACING,
        grid_height=GRID_HEIGHT,
        wire_radius=WIRE_RADIUS,
    )

    wire_z_index = 2

    assert not fixed_mask[1, 1, wire_z_index]
    assert not fixed_mask[3, 3, wire_z_index]


def test_solve_wire_grid_field_returns_finite_values() -> None:
    electric_field = solve_wire_grid_field(
        grid_spacing=GRID_SPACING,
        grid_height=GRID_HEIGHT,
        wire_radius=WIRE_RADIUS,
        collector_voltage=COLLECTOR_VOLTAGE,
        suppressor_voltage=SUPPRESSOR_VOLTAGE,
        nx=7,
        ny=7,
        nz=7,
        tolerance=1e-5,
        max_iterations=10_000,
    )

    Ex, Ey, Ez = electric_field(
        (GRID_SPACING / 4.0, GRID_SPACING / 4.0, GRID_HEIGHT / 2.0)
    )

    assert np.isfinite(Ex)
    assert np.isfinite(Ey)
    assert np.isfinite(Ez)
