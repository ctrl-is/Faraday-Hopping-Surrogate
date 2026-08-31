import numpy as np
import pytest

from src.physics.fields import (
    build_wire_grid_boundary_conditions,
    build_wire_mask,
    is_inside_wire,
    make_grid_axes,
    solve_wire_grid_field,
)

GRID_SPACING = 3.5e-3
GRID_HEIGHT = 3.9e-3
WIRE_RADIUS = 2.5e-4
COLLECTOR_VOLTAGE = 0.0
SUPPRESSOR_VOLTAGE = -55.0


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


def test_is_inside_wire_rejects_point_outside_wire_radius() -> None:
    assert not is_inside_wire(
        x=GRID_SPACING / 4.0,
        y=WIRE_RADIUS + 1e-6,
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


def test_wire_grid_boundary_conditions_fix_collector_top_sides_and_wires() -> None:
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
        (
            GRID_SPACING / 4.0,
            GRID_SPACING / 4.0,
            GRID_HEIGHT / 2.0,
        )
    )

    assert np.isfinite(Ex)
    assert np.isfinite(Ey)
    assert np.isfinite(Ez)
