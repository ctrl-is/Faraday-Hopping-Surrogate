from collections.abc import Callable
import math

import numpy as np
from scipy.interpolate import RegularGridInterpolator


Vector3 = tuple[float, float, float]
FieldFunction = Callable[[Vector3], Vector3]


def zero_magnetic_field(position: Vector3) -> Vector3:
    return 0.0, 0.0, 0.0


def make_uniform_magnetic_field(field: Vector3) -> FieldFunction:
    bx, by, bz = field

    def magnetic_field(position: Vector3) -> Vector3:
        return bx, by, bz

    return magnetic_field


def make_uniform_suppressor_field(suppressor_voltage: float, grid_height: float) -> FieldFunction:
    if grid_height <= 0.0:
        raise ValueError("grid_height must be positive.")

    electric_field_z = -suppressor_voltage / grid_height

    def electric_field(position: Vector3) -> Vector3:
        return 0.0, 0.0, electric_field_z

    return electric_field

# Helper Functions
# =============================================================================
def _validate_axis(values: np.ndarray, name: str) -> None:
    if len(values) < 2:
        raise ValueError(f"{name} must contain at least two values.")

    if not np.all(np.diff(values) > 0.0):
        raise ValueError(f"{name} must be strictly increasing.")


def _validate_even_spacing(values: np.ndarray, name: str) -> float:
    _validate_axis(values, name)

    spacing = float(values[1] - values[0])

    if not np.allclose(np.diff(values), spacing):
        raise ValueError(f"{name} must be evenly spaced.")

    return spacing


def _validate_field_shape(field: np.ndarray, expected_shape: tuple[int, int, int], name: str) -> None:
    if field.shape != expected_shape:
        raise ValueError(f"{name} must be {expected_shape}, got {field.shape}.")


def _validate_wire_parameters(grid_spacing: float, grid_height: float, wire_radius: float) -> None:
    if grid_spacing <= 0.0:
        raise ValueError("grid_spacing must be positive.")

    if grid_height <= 0.0:
        raise ValueError("grid_height must be positive.")

    if wire_radius <= 0.0:
        raise ValueError("wire_radius must be positive.")

    if wire_radius >= grid_spacing / 2.0:
        raise ValueError("wire_radius must be less than half the grid spacing.")


def _get_wire_domain_height(grid_height: float, domain_height: float | None) -> float:
    if domain_height is None:
        return 2.0 * grid_height

    if domain_height <= grid_height:
        raise ValueError("domain_height must be greater than grid_height.")

    return domain_height


def _validate_wire_domain(z_values: np.ndarray, grid_height: float) -> None:
    _validate_axis(z_values, "z_values")

    if z_values[0] > 0.0:
        raise ValueError("z_values must start at or below the collector plane.")

    if z_values[-1] <= grid_height:
        raise ValueError("z_values must extend above grid_height for the wire-grid model.")

    has_wire_slice = np.any(
        np.isclose(
            z_values,
            grid_height,
            rtol=0.0,
            atol=1e-12,
        ),
    )

    if not has_wire_slice:
        raise ValueError("z_values must contain grid_height for the wire-grid model.")
# =============================================================================

def make_grid_axes(
    grid_spacing: float,
    grid_height: float,
    nx: int,
    ny: int,
    nz: int,
    domain_height: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if grid_spacing <= 0.0:
        raise ValueError("grid_spacing must be positive.")

    if grid_height <= 0.0:
        raise ValueError("grid_height must be positive.")

    if nx < 3 or ny < 3 or nz < 3:
        raise ValueError("nx, ny, and nz must each be at least 3.")

    z_max = grid_height

    if domain_height is not None:
        z_max = _get_wire_domain_height(grid_height, domain_height)

    x_axis = np.linspace(-grid_spacing / 2.0, grid_spacing / 2.0, nx)
    y_axis = np.linspace(-grid_spacing / 2.0, grid_spacing / 2.0, ny)
    z_axis = np.linspace(0.0, z_max, nz)

    return x_axis, y_axis, z_axis


def initialize_potential_grid(
    x_values: np.ndarray,
    y_values: np.ndarray,
    z_values: np.ndarray,
    collector_voltage: float,
    suppressor_voltage: float,
) -> np.ndarray:
    _validate_axis(x_values, "x_values")
    _validate_axis(y_values, "y_values")
    _validate_axis(z_values, "z_values")

    z_min = z_values[0]
    z_max = z_values[-1]

    z_fraction = (z_values - z_min) / (z_max - z_min)

    potential_z = collector_voltage + z_fraction * (
        suppressor_voltage - collector_voltage
    )

    shape = (len(x_values), len(y_values), len(z_values))
    phi = np.broadcast_to(potential_z, shape).copy()

    return phi


def build_parallel_plate_boundary_conditions(
    x_values: np.ndarray,
    y_values: np.ndarray,
    z_values: np.ndarray,
    collector_voltage: float,
    suppressor_voltage: float,
) -> tuple[np.ndarray, np.ndarray]:
    fixed_values = initialize_potential_grid(
        x_values=x_values,
        y_values=y_values,
        z_values=z_values,
        collector_voltage=collector_voltage,
        suppressor_voltage=suppressor_voltage,
    )

    fixed_mask = np.zeros_like(fixed_values, dtype=bool)

    fixed_mask[:, :, 0] = True
    fixed_mask[:, :, -1] = True

    fixed_mask[0, :, :] = True
    fixed_mask[-1, :, :] = True
    fixed_mask[:, 0, :] = True
    fixed_mask[:, -1, :] = True

    return fixed_mask, fixed_values


def solve_laplace(
    phi: np.ndarray,
    fixed_mask: np.ndarray,
    fixed_values: np.ndarray,
    x_values: np.ndarray,
    y_values: np.ndarray,
    z_values: np.ndarray,
    tolerance: float = 1e-8,
    max_iterations: int = 10_000,
) -> np.ndarray:
    expected_shape = (len(x_values), len(y_values), len(z_values))

    _validate_field_shape(phi, expected_shape, "phi")
    _validate_field_shape(fixed_mask, expected_shape, "fixed_mask")
    _validate_field_shape(fixed_values, expected_shape, "fixed_values")

    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive.")

    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive.")

    dx = _validate_even_spacing(x_values, "x_values")
    dy = _validate_even_spacing(y_values, "y_values")
    dz = _validate_even_spacing(z_values, "z_values")

    inv_dx2 = 1.0 / dx**2
    inv_dy2 = 1.0 / dy**2
    inv_dz2 = 1.0 / dz**2

    denominator = 2.0 * (inv_dx2 + inv_dy2 + inv_dz2)

    current_phi = phi.copy()
    current_phi[fixed_mask] = fixed_values[fixed_mask]

    for _ in range(max_iterations):
        previous_phi = current_phi.copy()
        next_phi = previous_phi.copy()

        next_phi[1:-1, 1:-1, 1:-1] = (
            inv_dx2
            * (previous_phi[2:, 1:-1, 1:-1]+ previous_phi[:-2, 1:-1, 1:-1])
            + inv_dy2
            * (previous_phi[1:-1, 2:, 1:-1] + previous_phi[1:-1, :-2, 1:-1])
            + inv_dz2
            * (previous_phi[1:-1, 1:-1, 2:] + previous_phi[1:-1, 1:-1, :-2])
        ) / denominator

        next_phi[fixed_mask] = fixed_values[fixed_mask]

        max_delta = float(np.max(np.abs(next_phi - previous_phi)))
        current_phi = next_phi

        if max_delta < tolerance:
            return current_phi

    raise RuntimeError("Laplace solver did not converge.")


def compute_electric_field_from_potential(
    phi: np.ndarray,
    x_values: np.ndarray,
    y_values: np.ndarray,
    z_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    expected_shape = (len(x_values), len(y_values), len(z_values))

    _validate_field_shape(phi, expected_shape, "phi")

    _validate_axis(x_values, "x_values")
    _validate_axis(y_values, "y_values")
    _validate_axis(z_values, "z_values")

    dphi_dx, dphi_dy, dphi_dz = np.gradient(
        phi,
        x_values,
        y_values,
        z_values,
    )

    Ex = -dphi_dx
    Ey = -dphi_dy
    Ez = -dphi_dz

    return Ex, Ey, Ez


def make_interpolated_electric_field(
    x_values: np.ndarray,
    y_values: np.ndarray,
    z_values: np.ndarray,
    Ex: np.ndarray,
    Ey: np.ndarray,
    Ez: np.ndarray,
) -> FieldFunction:
    expected_shape = (len(x_values), len(y_values), len(z_values))

    _validate_field_shape(Ex, expected_shape, "Ex")
    _validate_field_shape(Ey, expected_shape, "Ey")
    _validate_field_shape(Ez, expected_shape, "Ez")

    _validate_axis(x_values, "x_values")
    _validate_axis(y_values, "y_values")
    _validate_axis(z_values, "z_values")

    points = (x_values, y_values, z_values)

    Ex_interpolator = RegularGridInterpolator(
        points,
        Ex,
        bounds_error=False,
        fill_value=None,
    )

    Ey_interpolator = RegularGridInterpolator(
        points,
        Ey,
        bounds_error=False,
        fill_value=None,
    )

    Ez_interpolator = RegularGridInterpolator(
        points,
        Ez,
        bounds_error=False,
        fill_value=None,
    )

    def electric_field(position: Vector3) -> Vector3:
        x, y, z = position
        point = np.array([[x, y, z]])

        Ex_value = float(Ex_interpolator(point)[0])
        Ey_value = float(Ey_interpolator(point)[0])
        Ez_value = float(Ez_interpolator(point)[0])

        return Ex_value, Ey_value, Ez_value

    return electric_field


def solve_parallel_plate_field(
    grid_spacing: float,
    grid_height: float,
    collector_voltage: float,
    suppressor_voltage: float,
    nx: int,
    ny: int,
    nz: int,
    tolerance: float = 1e-8,
    max_iterations: int = 10_000,
) -> FieldFunction:
    x_values, y_values, z_values = make_grid_axes(
        grid_spacing=grid_spacing,
        grid_height=grid_height,
        nx=nx,
        ny=ny,
        nz=nz,
    )

    phi = initialize_potential_grid(
        x_values=x_values,
        y_values=y_values,
        z_values=z_values,
        collector_voltage=collector_voltage,
        suppressor_voltage=suppressor_voltage,
    )

    fixed_mask, fixed_values = build_parallel_plate_boundary_conditions(
        x_values=x_values,
        y_values=y_values,
        z_values=z_values,
        collector_voltage=collector_voltage,
        suppressor_voltage=suppressor_voltage,
    )

    solved_phi = solve_laplace(
        phi=phi,
        fixed_mask=fixed_mask,
        fixed_values=fixed_values,
        x_values=x_values,
        y_values=y_values,
        z_values=z_values,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )

    Ex, Ey, Ez = compute_electric_field_from_potential(
        phi=solved_phi,
        x_values=x_values,
        y_values=y_values,
        z_values=z_values,
    )

    return make_interpolated_electric_field(
        x_values=x_values,
        y_values=y_values,
        z_values=z_values,
        Ex=Ex,
        Ey=Ey,
        Ez=Ez,
    )


def is_inside_wire(
    x: float,
    y: float,
    z: float,
    grid_spacing: float,
    grid_height: float,
    wire_radius: float,
) -> bool:
    """
    First wire-grid approximation.

    Coordinate convention:
        - one x-directed wire centered at y = 0, z = grid_height
        - one y-directed wire centered at x = 0, z = grid_height
    """
    if not math.isfinite(x):
        raise ValueError("x must be finite.")

    if not math.isfinite(y):
        raise ValueError("y must be finite.")

    if not math.isfinite(z):
        raise ValueError("z must be finite.")

    _validate_wire_parameters(grid_spacing, grid_height, wire_radius)

    distance_to_x_wire = math.sqrt(y**2 + (z - grid_height) ** 2)
    distance_to_y_wire = math.sqrt(x**2 + (z - grid_height) ** 2)

    return (
        distance_to_x_wire <= wire_radius
        or distance_to_y_wire <= wire_radius
    )


def build_wire_mask(
    x_values: np.ndarray,
    y_values: np.ndarray,
    z_values: np.ndarray,
    grid_spacing: float,
    grid_height: float,
    wire_radius: float,
) -> np.ndarray:
    _validate_axis(x_values, "x_values")
    _validate_axis(y_values, "y_values")
    _validate_axis(z_values, "z_values")

    _validate_wire_parameters(grid_spacing, grid_height, wire_radius)

    X, Y, Z = np.meshgrid(
        x_values,
        y_values,
        z_values,
        indexing="ij",
    )

    distance_to_x_wire = np.sqrt(Y**2 + (Z - grid_height) ** 2)
    distance_to_y_wire = np.sqrt(X**2 + (Z - grid_height) ** 2)

    wire_mask = (
        (distance_to_x_wire <= wire_radius)
        | (distance_to_y_wire <= wire_radius)
    )

    return wire_mask


def build_wire_grid_boundary_conditions(
    x_values: np.ndarray,
    y_values: np.ndarray,
    z_values: np.ndarray,
    collector_voltage: float,
    suppressor_voltage: float,
    grid_spacing: float,
    grid_height: float,
    wire_radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    _validate_wire_domain(z_values, grid_height)

    fixed_values = initialize_potential_grid(
        x_values=x_values,
        y_values=y_values,
        z_values=z_values,
        collector_voltage=collector_voltage,
        suppressor_voltage=suppressor_voltage,
    )

    fixed_mask = np.zeros_like(fixed_values, dtype=bool)

    fixed_mask[:, :, 0] = True
    fixed_values[:, :, 0] = collector_voltage

    fixed_mask[:, :, -1] = True
    fixed_values[:, :, -1] = suppressor_voltage

    fixed_mask[0, :, :] = True
    fixed_mask[-1, :, :] = True
    fixed_mask[:, 0, :] = True
    fixed_mask[:, -1, :] = True

    wire_mask = build_wire_mask(
        x_values=x_values,
        y_values=y_values,
        z_values=z_values,
        grid_spacing=grid_spacing,
        grid_height=grid_height,
        wire_radius=wire_radius,
    )

    fixed_mask[wire_mask] = True
    fixed_values[wire_mask] = suppressor_voltage

    return fixed_mask, fixed_values


def solve_wire_grid_field(
    grid_spacing: float,
    grid_height: float,
    wire_radius: float,
    collector_voltage: float,
    suppressor_voltage: float,
    nx: int,
    ny: int,
    nz: int,
    domain_height: float | None = None,
    tolerance: float = 1e-8,
    max_iterations: int = 10_000,
) -> FieldFunction:
    if nx % 2 == 0 or ny % 2 == 0:
        raise ValueError("nx and ny must be odd so the wire axes land on the grid.")

    if domain_height is None and nz % 2 == 0:
        raise ValueError("nz must be odd for the default wire-grid domain.")

    wire_domain_height = _get_wire_domain_height(grid_height, domain_height)

    x_values, y_values, z_values = make_grid_axes(
        grid_spacing=grid_spacing,
        grid_height=grid_height,
        nx=nx,
        ny=ny,
        nz=nz,
        domain_height=wire_domain_height,
    )

    _validate_wire_domain(z_values, grid_height)

    phi = initialize_potential_grid(
        x_values=x_values,
        y_values=y_values,
        z_values=z_values,
        collector_voltage=collector_voltage,
        suppressor_voltage=suppressor_voltage,
    )

    fixed_mask, fixed_values = build_wire_grid_boundary_conditions(
        x_values=x_values,
        y_values=y_values,
        z_values=z_values,
        collector_voltage=collector_voltage,
        suppressor_voltage=suppressor_voltage,
        grid_spacing=grid_spacing,
        grid_height=grid_height,
        wire_radius=wire_radius,
    )

    solved_phi = solve_laplace(
        phi=phi,
        fixed_mask=fixed_mask,
        fixed_values=fixed_values,
        x_values=x_values,
        y_values=y_values,
        z_values=z_values,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )

    Ex, Ey, Ez = compute_electric_field_from_potential(
        phi=solved_phi,
        x_values=x_values,
        y_values=y_values,
        z_values=z_values,
    )

    return make_interpolated_electric_field(
        x_values=x_values,
        y_values=y_values,
        z_values=z_values,
        Ex=Ex,
        Ey=Ey,
        Ez=Ez,
    )