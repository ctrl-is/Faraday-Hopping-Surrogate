from collections.abc import Callable
import numpy as np
from scipy.interpolate import RegularGridInterpolator

Vector3 = tuple[float, float, float]
FieldFunction = Callable[[Vector3], Vector3]


def zero_magnetic_field(position: Vector3) -> Vector3:
    return 0.0, 0.0, 0.0


def make_uniform_magnetic_field(
    field: Vector3,
) -> FieldFunction:
    bx, by, bz = field

    def magnetic_field(position: Vector3) -> Vector3:
        return bx, by, bz

    return magnetic_field


def make_uniform_suppressor_field(
    suppressor_voltage: float,
    grid_height: float,
) -> FieldFunction:
    if grid_height <= 0.0:
        raise ValueError("grid_height must be positive.")

    electric_field_z = -suppressor_voltage / grid_height

    def electric_field(position: Vector3) -> Vector3:
        return 0.0, 0.0, electric_field_z

    return electric_field


def make_grid_axes(
    grid_spacing: float,
    grid_height: float,
    nx: int,
    ny: int,
    nz: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_axis = np.linspace(-grid_spacing / 2, grid_spacing / 2, nx)
    y_axis = np.linspace(-grid_height / 2, grid_height / 2, ny)
    z_axis = np.linspace(0, grid_height, nz)

    return x_axis, y_axis, z_axis


def initialize_potential_grid(
    x_values,
    y_values,
    z_values,
    collector_voltage: float,
    suppressor_voltage: float,
) -> np.ndarray:
    if not np.all(np.diff(z_values) > 0):
        raise ValueError("z_values must be strictly increasing.")

    z_min = z_values[0]
    z_max = z_values[-1]
    z_frac = (z_values - z_min) / (z_max - z_min)

    potential_z = collector_voltage + z_frac * (suppressor_voltage - collector_voltage)

    phi = np.broadcast_to(
        potential_z,
        (len(x_values), len(y_values), len(z_values)),
    ).copy()

    return phi 


def build_parallel_plate_boundary_conditions(
    x_values,
    y_values,
    z_values,
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

    # Physical plates
    fixed_mask[:, :, 0] = True
    fixed_mask[:, :, -1] = True

    # Outer simulation-box boundaries for validation
    fixed_mask[0, :, :] = True
    fixed_mask[-1, :, :] = True
    fixed_mask[:, 0, :] = True
    fixed_mask[:, -1, :] = True

    return fixed_mask, fixed_values


def solve_laplace(
    phi,
    fixed_mask,
    fixed_values,
    x_values,
    y_values,
    z_values,
    tolerance: float = 1e-8,
    max_iterations: int = 10_000,
) -> np.ndarray:
    if phi.shape != fixed_mask.shape:
        raise ValueError("phi and fixed_mask must have the same shape.")

    if phi.shape != fixed_values.shape:
        raise ValueError("phi and fixed_values must have the same shape.")

    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive.")

    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive.")

    if not np.all(np.diff(x_values) > 0):
        raise ValueError("x_values must be strictly increasing.")

    if not np.all(np.diff(y_values) > 0):
        raise ValueError("y_values must be strictly increasing.")

    if not np.all(np.diff(z_values) > 0):
        raise ValueError("z_values must be strictly increasing.")

    dx = float(x_values[1] - x_values[0])
    dy = float(y_values[1] - y_values[0])
    dz = float(z_values[1] - z_values[0])

    if not np.allclose(np.diff(x_values), dx):
        raise ValueError("x_values must be evenly spaced.")

    if not np.allclose(np.diff(y_values), dy):
        raise ValueError("y_values must be evenly spaced.")

    if not np.allclose(np.diff(z_values), dz):
        raise ValueError("z_values must be evenly spaced.")

    inv_dx2 = 1.0 / dx**2
    inv_dy2 = 1.0 / dy**2
    inv_dz2 = 1.0 / dz**2

    denominator = 2.0 * (inv_dx2 + inv_dy2 + inv_dz2)

    curr_phi = phi.copy()
    curr_phi[fixed_mask] = fixed_values[fixed_mask]

    for iteration in range(max_iterations):
        prev_phi = curr_phi.copy()
        next_phi = prev_phi.copy()

        next_phi[1:-1, 1:-1, 1:-1] = (
            inv_dx2
            * (prev_phi[2:, 1:-1, 1:-1] + prev_phi[:-2, 1:-1, 1:-1])
            + inv_dy2
            * (prev_phi[1:-1, 2:, 1:-1] + prev_phi[1:-1, :-2, 1:-1])
            + inv_dz2
            * (prev_phi[1:-1, 1:-1, 2:] + prev_phi[1:-1, 1:-1, :-2])
        ) / denominator

        next_phi[fixed_mask] = fixed_values[fixed_mask]

        max_delta = float(np.max(np.abs(next_phi - prev_phi)))

        curr_phi = next_phi

        if max_delta < tolerance:
            return curr_phi

    raise RuntimeError("Laplace solver did not converge.")


def compute_electric_field_from_potential(
    phi,
    x_values,
    y_values,
    z_values,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    expected_shape = (
        len(x_values),
        len(y_values),
        len(z_values),
    )

    if phi.shape != expected_shape:
        raise ValueError(f"phi must be {expected_shape}, got {phi.shape()}.")

    if not np.all(np.diff(x_values) > 0):
        raise ValueError("x_values must be strictly increasing.")

    if not np.all(np.diff(y_values) > 0):
        raise ValueError("y_values must be strictly increasing.")

    if not np.all(np.diff(z_values) > 0):
        raise ValueError("z_values must be strictly increasing.")

    dphi_dx, dphi_dy, dphi_dz = np.gradient(
        phi,
        x_values,
        y_values,
        z_values,
    )

    E_x = -dphi_dx
    E_y = -dphi_dy
    E_z = -dphi_dz

    return E_x, E_y, E_z


def make_interpolated_electric_field(
    x_values,
    y_values,
    z_values,
    Ex,
    Ey,
    Ez,
) -> FieldFunction:
    expected_shape = (
        len(x_values),
        len(y_values),
        len(z_values),
    )

    if Ex.shape != expected_shape:
        raise ValueError(f"Ex must be {expected_shape}, got {Ex.shape()}")

    if Ey.shape != expected_shape:
            raise ValueError(f"Ey must be {expected_shape}, got {Ey.shape()}")

    if Ez.shape != expected_shape:
            raise ValueError(f"Ez must be {expected_shape}, got {Ez.shape()}")

    if not np.all(np.diff(x_values) > 0):
        raise ValueError("x_values must be strictly increasing.")

    if not np.all(np.diff(y_values) > 0):
        raise ValueError("y_values must be strictly increasing.")

    if not np.all(np.diff(z_values) > 0):
        raise ValueError("z_values must be strictly increasing.")

    Ex_interpolator = RegularGridInterpolator(
        (x_values, y_values, z_values),
        Ex,
    )

    Ey_interpolator = RegularGridInterpolator(
        (x_values, y_values, z_values),
        Ey,
    )

    Ez_interpolator = RegularGridInterpolator(
        (x_values, y_values, z_values),
        Ez,
    )

    def electric_field(position):
        x, y, z = position

        point = np.array([[x, y, z]])

        Ex_val = float(Ex_interpolator(point)[0])
        Ey_val = float(Ey_interpolator(point)[0])
        Ez_val = float(Ez_interpolator(point)[0])

        return Ex_val, Ey_val, Ez_val

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
        grid_spacing,
        grid_height,
        nx,
        ny,
        nz,
    )

    phi = initialize_potential_grid(
        x_values,
        y_values,
        z_values,
        collector_voltage,
        suppressor_voltage,
    )

    fixed_mask, fixed_values = build_parallel_plate_boundary_conditions(
        x_values,
        y_values,
        z_values,
        collector_voltage,
        suppressor_voltage,
    )

    solved_phi = solve_laplace(
        phi,
        fixed_mask,
        fixed_values,
        x_values,
        y_values,
        z_values,
        tolerance,
        max_iterations,
    )

    Ex, Ey, Ez = compute_electric_field_from_potential(
        solved_phi,
        x_values,
        y_values,
        z_values,
    )

    electric_field = make_interpolated_electric_field(
        x_values,
        y_values,
        z_values,
        Ex,
        Ey,
        Ez,
    )

    return electric_field