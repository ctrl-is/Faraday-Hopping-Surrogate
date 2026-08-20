from collections.abc import Callable


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