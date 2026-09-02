from collections import Counter
from dataclasses import dataclass
from enum import Enum

import numpy as np

from src.physics.emission import (
    EmittedElectron,
    sample_emission_direction,
    sample_num_emitted_electrons,
    sample_secondary_energy_eV,
)
from src.physics.fields import FieldFunction, zero_magnetic_field
from src.physics.geometry import CollectorGeometry
from src.physics.impact import ProtonImpact, sample_proton_impact
from src.physics.trajectory import (
    Trajectory,
    TrajectoryResult,
    TrajectoryStatus,
    initial_velocity_from_emission,
)
from src.simulation.labels import (
    LandingOutcome,
    LandingResult,
    classify_landing,
)


class ElectronEventOutcome(str, Enum):
    SAME_QUADRANT = "same_quadrant"
    DIFFERENT_QUADRANT = "different_quadrant"
    GAP = "gap"
    OUTSIDE = "outside"
    HIT_WIRE = "hit_wire"
    PASSED_GRID_OPENING = "passed_grid_opening"
    DID_NOT_RETURN = "did_not_return"
    SOLVER_FAILURE = "solver_failure"


@dataclass(frozen=True)
class MonteCarloConfig:
    num_protons: int
    proton_energy_eV: float
    modulation_frequency_hz: float
    incidence_angle_rad: float
    beam_radius: float
    grid_spacing: float
    grid_height: float
    wire_radius: float
    collector_voltage: float
    suppressor_voltage: float
    secondary_energy_model: str
    secondary_direction_model: str
    trajectory_t_max: float
    trajectory_max_step: float


@dataclass(frozen=True)
class SecondaryElectronEvent:
    electron: EmittedElectron
    trajectory_result: TrajectoryResult
    landing_result: LandingResult | None
    outcome: ElectronEventOutcome


LANDING_OUTCOME_TO_EVENT_OUTCOME = {
    LandingOutcome.SAME_QUADRANT: ElectronEventOutcome.SAME_QUADRANT,
    LandingOutcome.DIFFERENT_QUADRANT: ElectronEventOutcome.DIFFERENT_QUADRANT,
    LandingOutcome.GAP: ElectronEventOutcome.GAP,
    LandingOutcome.OUTSIDE: ElectronEventOutcome.OUTSIDE,
    LandingOutcome.DID_NOT_RETURN: ElectronEventOutcome.DID_NOT_RETURN,
}

TRAJECTORY_STATUS_TO_EVENT_OUTCOME = {
    TrajectoryStatus.HIT_WIRE: ElectronEventOutcome.HIT_WIRE,
    TrajectoryStatus.PASSED_GRID_OPENING: ElectronEventOutcome.PASSED_GRID_OPENING,
    TrajectoryStatus.LEFT_RADIAL_DOMAIN: ElectronEventOutcome.OUTSIDE,
    TrajectoryStatus.TIMEOUT: ElectronEventOutcome.DID_NOT_RETURN,
    TrajectoryStatus.SOLVER_FAILURE: ElectronEventOutcome.SOLVER_FAILURE,
}


@dataclass(frozen=True)
class ProtonImpactEvent:
    impact: ProtonImpact
    emitted_electron_count: int
    secondary_events: list[SecondaryElectronEvent]


@dataclass(frozen=True)
class MonteCarloSummary:
    num_protons: int
    valid_proton_impacts: int
    total_emitted_electrons: int
    returned_to_collector_count: int
    ions_with_hopper_count: int

    same_quadrant_count: int
    different_quadrant_count: int
    gap_count: int
    outside_count: int
    hit_wire_count: int
    passed_grid_opening_count: int
    did_not_return_count: int
    solver_failure_count: int

    @property
    def ion_hopping_probability(self) -> float:
        if self.valid_proton_impacts == 0:
            return 0.0

        return self.ions_with_hopper_count / self.valid_proton_impacts

    @property
    def electron_hopping_probability(self) -> float:
        if self.total_emitted_electrons == 0:
            return 0.0

        return self.different_quadrant_count / self.total_emitted_electrons

    @property
    def return_probability(self) -> float:
        if self.total_emitted_electrons == 0:
            return 0.0

        return self.returned_to_collector_count / self.total_emitted_electrons


def sample_secondary_electron(
    rng: np.random.Generator,
    impact: ProtonImpact,
    secondary_energy_model: str,
    secondary_direction_model: str,
) -> EmittedElectron:
    x0, y0 = impact.collector_position
    energy_eV = sample_secondary_energy_eV(
        rng=rng,
        model=secondary_energy_model,
    )
    theta, psi = sample_emission_direction(
        rng=rng,
        model=secondary_direction_model,
    )

    return EmittedElectron(
        x0=x0,
        y0=y0,
        energy_eV=energy_eV,
        theta=theta,
        psi=psi,
    )


def simulate_secondary_electron(
    electron: EmittedElectron,
    electric_field: FieldFunction,
    magnetic_field: FieldFunction | None,
    geometry: CollectorGeometry,
    grid_spacing: float,
    grid_height: float,
    wire_radius: float,
    trajectory_t_max: float,
    trajectory_max_step: float,
) -> SecondaryElectronEvent:
    initial_velocity = initial_velocity_from_emission(electron)

    if magnetic_field is None:
        magnetic_field = zero_magnetic_field

    trajectory = Trajectory(
        efield=electric_field,
        initial_position=(electron.x0, electron.y0, 1e-12),
        initial_velocity=initial_velocity,
        bfield=magnetic_field,
    )

    res = trajectory.solve(
        t_max=trajectory_t_max,
        max_step=trajectory_max_step,
        grid_height=grid_height,
        radial_limit=geometry.radius,
        wire_grid_spacing=grid_spacing,
        wire_radius=wire_radius,
    )

    landing_result, outcome = classify_trajectory_event(
        electron=electron,
        trajectory_result=res,
        geometry=geometry,
    )

    return SecondaryElectronEvent(
        electron=electron,
        trajectory_result=res,
        landing_result=landing_result,
        outcome=outcome,
    )


def classify_trajectory_event(
    electron: EmittedElectron,
    trajectory_result: TrajectoryResult,
    geometry: CollectorGeometry,
) -> tuple[LandingResult | None, ElectronEventOutcome]:
    if trajectory_result.status == TrajectoryStatus.HIT_COLLECTOR:
        landing_result = classify_landing(
            electron=electron,
            trajectory_result=trajectory_result,
            geometry=geometry,
        )

        outcome = LANDING_OUTCOME_TO_EVENT_OUTCOME.get(
            landing_result.outcome,
            ElectronEventOutcome.DID_NOT_RETURN,
        )

        return landing_result, outcome

    outcome = TRAJECTORY_STATUS_TO_EVENT_OUTCOME.get(
        trajectory_result.status,
        ElectronEventOutcome.DID_NOT_RETURN,
    )

    return None, outcome


def simulate_proton_impact_event(
    rng: np.random.Generator,
    geometry: CollectorGeometry,
    electric_field: FieldFunction,
    magnetic_field: FieldFunction | None,
    proton_energy_eV: float,
    incidence_angle_rad: float,
    beam_radius: float,
    grid_spacing: float,
    grid_height: float,
    wire_radius: float,
    secondary_energy_model: str,
    secondary_direction_model: str,
    trajectory_t_max: float,
    trajectory_max_step: float,
) -> ProtonImpactEvent:
    impact = sample_proton_impact(
        rng=rng,
        geometry=geometry,
        alpha=incidence_angle_rad,
        beam_radius=beam_radius,
        grid_height=grid_height,
    )

    if not impact.hit_valid_region:
        return ProtonImpactEvent(
            impact=impact,
            emitted_electron_count=0,
            secondary_events=[],
        )

    num_electrons = sample_num_emitted_electrons(
        rng=rng,
        U_eV=proton_energy_eV,
    )

    secondary_events = []
    for _ in range(num_electrons):
        electron = sample_secondary_electron(
            rng=rng,
            impact=impact,
            secondary_energy_model=secondary_energy_model,
            secondary_direction_model=secondary_direction_model,
        )

        event = simulate_secondary_electron(
            electron=electron,
            electric_field=electric_field,
            magnetic_field=magnetic_field,
            geometry=geometry,
            grid_spacing=grid_spacing,
            grid_height=grid_height,
            wire_radius=wire_radius,
            trajectory_t_max=trajectory_t_max,
            trajectory_max_step=trajectory_max_step,
        )

        secondary_events.append(event)

    return ProtonImpactEvent(
        impact=impact,
        emitted_electron_count=num_electrons,
        secondary_events=secondary_events,
    )


def run_monte_carlo(
    rng: np.random.Generator,
    config: MonteCarloConfig,
    geometry: CollectorGeometry,
    electric_field: FieldFunction,
    magnetic_field: FieldFunction | None = None,
) -> list[ProtonImpactEvent]:
    if config.num_protons <= 0:
        return []

    events = []
    for _ in range(config.num_protons):
        event = simulate_proton_impact_event(
            rng=rng,
            geometry=geometry,
            electric_field=electric_field,
            magnetic_field=magnetic_field,
            proton_energy_eV=config.proton_energy_eV,
            incidence_angle_rad=config.incidence_angle_rad,
            beam_radius=config.beam_radius,
            grid_spacing=config.grid_spacing,
            grid_height=config.grid_height,
            wire_radius=config.wire_radius,
            secondary_energy_model=config.secondary_energy_model,
            secondary_direction_model=config.secondary_direction_model,
            trajectory_t_max=config.trajectory_t_max,
            trajectory_max_step=config.trajectory_max_step,
        )

        events.append(event)

    return events


def summarize_monte_carlo_events(events: list[ProtonImpactEvent]) -> MonteCarloSummary:
    num_protons = len(events)
    valid_proton_impacts = 0
    total_emitted_electrons = 0
    returned_to_collector_count = 0
    ions_with_hopper_count = 0

    outcome_counts = Counter()

    for proton_event in events:
        if proton_event.impact.hit_valid_region:
            valid_proton_impacts += 1

        total_emitted_electrons += proton_event.emitted_electron_count

        ion_has_hopper = False

        for electron_event in proton_event.secondary_events:
            outcome_counts[electron_event.outcome] += 1

            if (
                electron_event.trajectory_result.status
                == TrajectoryStatus.HIT_COLLECTOR
            ):
                returned_to_collector_count += 1

            if electron_event.outcome == ElectronEventOutcome.DIFFERENT_QUADRANT:
                ion_has_hopper = True

        if ion_has_hopper:
            ions_with_hopper_count += 1

    return MonteCarloSummary(
        num_protons=num_protons,
        valid_proton_impacts=valid_proton_impacts,
        total_emitted_electrons=total_emitted_electrons,
        returned_to_collector_count=returned_to_collector_count,
        ions_with_hopper_count=ions_with_hopper_count,
        same_quadrant_count=outcome_counts[ElectronEventOutcome.SAME_QUADRANT],
        different_quadrant_count=outcome_counts[
            ElectronEventOutcome.DIFFERENT_QUADRANT
        ],
        gap_count=outcome_counts[ElectronEventOutcome.GAP],
        outside_count=outcome_counts[ElectronEventOutcome.OUTSIDE],
        hit_wire_count=outcome_counts[ElectronEventOutcome.HIT_WIRE],
        passed_grid_opening_count=outcome_counts[
            ElectronEventOutcome.PASSED_GRID_OPENING
        ],
        did_not_return_count=outcome_counts[ElectronEventOutcome.DID_NOT_RETURN],
        solver_failure_count=outcome_counts[ElectronEventOutcome.SOLVER_FAILURE],
    )


def estimate_hopping_probability(
    rng: np.random.Generator,
    config: MonteCarloConfig,
    geometry: CollectorGeometry,
    electric_field: FieldFunction,
    magnetic_field: FieldFunction | None = None,
) -> MonteCarloSummary:
    events = run_monte_carlo(
        rng=rng,
        config=config,
        geometry=geometry,
        electric_field=electric_field,
        magnetic_field=magnetic_field,
    )

    return summarize_monte_carlo_events(events)
