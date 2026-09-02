import pytest

from src.physics.emission import EmittedElectron
from src.physics.fields import Vector3
from src.physics.geometry import CollectorGeometry, Region
from src.physics.impact import ProtonImpact
from src.physics.trajectory import TrajectoryResult, TrajectoryStatus
from src.simulation import monte_carlo as mc

GRID_SPACING = 3.5e-3
GRID_HEIGHT = 3.9e-3
WIRE_RADIUS = 2.5e-4


def zero_electric_field(position: Vector3) -> Vector3:
    return 0.0, 0.0, 0.0


def make_geometry() -> CollectorGeometry:
    return CollectorGeometry(
        gap_width=1.0e-4,
        radius=1.0e-2,
    )


def make_electron(x0: float = 1.0e-3, y0: float = 1.0e-3) -> EmittedElectron:
    return EmittedElectron(
        x0=x0,
        y0=y0,
        energy_eV=5.0,
        theta=0.0,
        psi=0.0,
    )


def make_trajectory_result(
    status: TrajectoryStatus,
    final_position: tuple[float, float, float],
) -> TrajectoryResult:
    return TrajectoryResult(
        status=status,
        final_position=final_position,
        final_velocity=(0.0, 0.0, 0.0),
        event_time=1.0e-9,
        solution=None,
    )


def make_valid_impact() -> ProtonImpact:
    return ProtonImpact(
        entry_position=(1.0e-3, 1.0e-3),
        collector_position=(1.0e-3, 1.0e-3),
        alpha=0.0,
        source_region=Region.A,
    )


def make_invalid_impact() -> ProtonImpact:
    return ProtonImpact(
        entry_position=(0.0, 0.0),
        collector_position=(0.0, 0.0),
        alpha=0.0,
        source_region=Region.GAP,
    )


def test_classify_trajectory_event_same_quadrant() -> None:
    geometry = make_geometry()
    electron = make_electron(x0=1.0e-3, y0=1.0e-3)

    trajectory_result = make_trajectory_result(
        status=TrajectoryStatus.HIT_COLLECTOR,
        final_position=(1.0e-3, 1.0e-3, 0.0),
    )

    landing_result, outcome = mc.classify_trajectory_event(
        electron=electron,
        trajectory_result=trajectory_result,
        geometry=geometry,
    )

    assert landing_result is not None
    assert outcome == mc.ElectronEventOutcome.SAME_QUADRANT


def test_classify_trajectory_event_different_quadrant() -> None:
    geometry = make_geometry()
    electron = make_electron(x0=1.0e-3, y0=1.0e-3)

    trajectory_result = make_trajectory_result(
        status=TrajectoryStatus.HIT_COLLECTOR,
        final_position=(-1.0e-3, 1.0e-3, 0.0),
    )

    landing_result, outcome = mc.classify_trajectory_event(
        electron=electron,
        trajectory_result=trajectory_result,
        geometry=geometry,
    )

    assert landing_result is not None
    assert outcome == mc.ElectronEventOutcome.DIFFERENT_QUADRANT


def test_classify_trajectory_event_gap() -> None:
    geometry = make_geometry()
    electron = make_electron(x0=1.0e-3, y0=1.0e-3)

    trajectory_result = make_trajectory_result(
        status=TrajectoryStatus.HIT_COLLECTOR,
        final_position=(0.0, 1.0e-3, 0.0),
    )

    landing_result, outcome = mc.classify_trajectory_event(
        electron=electron,
        trajectory_result=trajectory_result,
        geometry=geometry,
    )

    assert landing_result is not None
    assert outcome == mc.ElectronEventOutcome.GAP


def test_classify_trajectory_event_wire_and_opening_statuses() -> None:
    geometry = make_geometry()
    electron = make_electron()

    wire_result = make_trajectory_result(
        status=TrajectoryStatus.HIT_WIRE,
        final_position=(0.0, 1.0e-3, GRID_HEIGHT),
    )

    opening_result = make_trajectory_result(
        status=TrajectoryStatus.PASSED_GRID_OPENING,
        final_position=(1.0e-3, 1.0e-3, GRID_HEIGHT),
    )

    wire_landing, wire_outcome = mc.classify_trajectory_event(
        electron=electron,
        trajectory_result=wire_result,
        geometry=geometry,
    )

    opening_landing, opening_outcome = mc.classify_trajectory_event(
        electron=electron,
        trajectory_result=opening_result,
        geometry=geometry,
    )

    assert wire_landing is None
    assert wire_outcome == mc.ElectronEventOutcome.HIT_WIRE

    assert opening_landing is None
    assert opening_outcome == mc.ElectronEventOutcome.PASSED_GRID_OPENING


def test_classify_trajectory_event_timeout_and_solver_failure() -> None:
    geometry = make_geometry()
    electron = make_electron()

    timeout_result = make_trajectory_result(
        status=TrajectoryStatus.TIMEOUT,
        final_position=(1.0e-3, 1.0e-3, 1.0e-3),
    )

    failure_result = make_trajectory_result(
        status=TrajectoryStatus.SOLVER_FAILURE,
        final_position=(1.0e-3, 1.0e-3, 1.0e-3),
    )

    _, timeout_outcome = mc.classify_trajectory_event(
        electron=electron,
        trajectory_result=timeout_result,
        geometry=geometry,
    )

    _, failure_outcome = mc.classify_trajectory_event(
        electron=electron,
        trajectory_result=failure_result,
        geometry=geometry,
    )

    assert timeout_outcome == mc.ElectronEventOutcome.DID_NOT_RETURN
    assert failure_outcome == mc.ElectronEventOutcome.SOLVER_FAILURE


def test_simulate_secondary_electron_passes_through_grid_opening() -> None:
    geometry = make_geometry()

    electron = EmittedElectron(
        x0=GRID_SPACING / 4.0,
        y0=GRID_SPACING / 4.0,
        energy_eV=80.0,
        theta=0.0,
        psi=0.0,
    )

    event = mc.simulate_secondary_electron(
        electron=electron,
        electric_field=zero_electric_field,
        magnetic_field=None,
        geometry=geometry,
        grid_spacing=GRID_SPACING,
        grid_height=GRID_HEIGHT,
        wire_radius=WIRE_RADIUS,
        trajectory_t_max=1.0e-7,
        trajectory_max_step=1.0e-11,
    )

    assert event.electron == electron
    assert event.outcome == mc.ElectronEventOutcome.PASSED_GRID_OPENING
    assert event.trajectory_result.status == TrajectoryStatus.PASSED_GRID_OPENING


def test_simulate_proton_impact_event_returns_empty_for_invalid_impact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry = make_geometry()
    invalid_impact = make_invalid_impact()

    def fake_sample_proton_impact(**kwargs) -> ProtonImpact:
        return invalid_impact

    monkeypatch.setattr(mc, "sample_proton_impact", fake_sample_proton_impact)

    event = mc.simulate_proton_impact_event(
        rng=None,
        geometry=geometry,
        electric_field=zero_electric_field,
        magnetic_field=None,
        proton_energy_eV=800.0,
        incidence_angle_rad=0.0,
        beam_radius=1.0e-2,
        grid_spacing=GRID_SPACING,
        grid_height=GRID_HEIGHT,
        wire_radius=WIRE_RADIUS,
        secondary_energy_model="fixed",
        secondary_direction_model="fixed",
        trajectory_t_max=1.0e-7,
        trajectory_max_step=1.0e-11,
    )

    assert event.impact == invalid_impact
    assert event.emitted_electron_count == 0
    assert event.secondary_events == []


def test_run_monte_carlo_returns_requested_number_of_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry = make_geometry()
    proton_event = mc.ProtonImpactEvent(
        impact=make_valid_impact(),
        emitted_electron_count=0,
        secondary_events=[],
    )

    def fake_simulate_proton_impact_event(**kwargs) -> mc.ProtonImpactEvent:
        return proton_event

    monkeypatch.setattr(
        mc,
        "simulate_proton_impact_event",
        fake_simulate_proton_impact_event,
    )

    config = mc.MonteCarloConfig(
        num_protons=5,
        proton_energy_eV=800.0,
        modulation_frequency_hz=1024.0,
        incidence_angle_rad=0.0,
        beam_radius=1.0e-2,
        grid_spacing=GRID_SPACING,
        grid_height=GRID_HEIGHT,
        wire_radius=WIRE_RADIUS,
        collector_voltage=0.0,
        suppressor_voltage=-55.0,
        secondary_energy_model="fixed",
        secondary_direction_model="fixed",
        trajectory_t_max=1.0e-7,
        trajectory_max_step=1.0e-11,
    )

    events = mc.run_monte_carlo(
        rng=None,
        config=config,
        geometry=geometry,
        electric_field=zero_electric_field,
    )

    assert len(events) == 5
    assert all(event == proton_event for event in events)


def test_summarize_monte_carlo_events_counts_outcomes() -> None:
    same_event = mc.SecondaryElectronEvent(
        electron=make_electron(),
        trajectory_result=make_trajectory_result(
            status=TrajectoryStatus.HIT_COLLECTOR,
            final_position=(1.0e-3, 1.0e-3, 0.0),
        ),
        landing_result=None,
        outcome=mc.ElectronEventOutcome.SAME_QUADRANT,
    )

    hopper_event = mc.SecondaryElectronEvent(
        electron=make_electron(),
        trajectory_result=make_trajectory_result(
            status=TrajectoryStatus.HIT_COLLECTOR,
            final_position=(-1.0e-3, 1.0e-3, 0.0),
        ),
        landing_result=None,
        outcome=mc.ElectronEventOutcome.DIFFERENT_QUADRANT,
    )

    wire_event = mc.SecondaryElectronEvent(
        electron=make_electron(),
        trajectory_result=make_trajectory_result(
            status=TrajectoryStatus.HIT_WIRE,
            final_position=(0.0, 1.0e-3, GRID_HEIGHT),
        ),
        landing_result=None,
        outcome=mc.ElectronEventOutcome.HIT_WIRE,
    )

    events = [
        mc.ProtonImpactEvent(
            impact=make_valid_impact(),
            emitted_electron_count=2,
            secondary_events=[same_event, hopper_event],
        ),
        mc.ProtonImpactEvent(
            impact=make_valid_impact(),
            emitted_electron_count=1,
            secondary_events=[wire_event],
        ),
        mc.ProtonImpactEvent(
            impact=make_invalid_impact(),
            emitted_electron_count=0,
            secondary_events=[],
        ),
    ]

    summary = mc.summarize_monte_carlo_events(events)

    assert summary.num_protons == 3
    assert summary.valid_proton_impacts == 2
    assert summary.total_emitted_electrons == 3
    assert summary.returned_to_collector_count == 2
    assert summary.ions_with_hopper_count == 1

    assert summary.same_quadrant_count == 1
    assert summary.different_quadrant_count == 1
    assert summary.hit_wire_count == 1

    assert summary.ion_hopping_probability == pytest.approx(1.0 / 2.0)
    assert summary.electron_hopping_probability == pytest.approx(1.0 / 3.0)
    assert summary.return_probability == pytest.approx(2.0 / 3.0)


def test_estimate_hopping_probability_runs_and_summarizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry = make_geometry()
    events = [
        mc.ProtonImpactEvent(
            impact=make_valid_impact(),
            emitted_electron_count=0,
            secondary_events=[],
        )
    ]

    expected_summary = mc.MonteCarloSummary(
        num_protons=1,
        valid_proton_impacts=1,
        total_emitted_electrons=0,
        returned_to_collector_count=0,
        ions_with_hopper_count=0,
        same_quadrant_count=0,
        different_quadrant_count=0,
        gap_count=0,
        outside_count=0,
        hit_wire_count=0,
        passed_grid_opening_count=0,
        did_not_return_count=0,
        solver_failure_count=0,
    )

    def fake_run_monte_carlo(**kwargs) -> list[mc.ProtonImpactEvent]:
        return events

    def fake_summarize_monte_carlo_events(
        actual_events: list[mc.ProtonImpactEvent],
    ) -> mc.MonteCarloSummary:
        assert actual_events == events
        return expected_summary

    monkeypatch.setattr(mc, "run_monte_carlo", fake_run_monte_carlo)
    monkeypatch.setattr(
        mc,
        "summarize_monte_carlo_events",
        fake_summarize_monte_carlo_events,
    )

    config = mc.MonteCarloConfig(
        num_protons=1,
        proton_energy_eV=800.0,
        modulation_frequency_hz=1024.0,
        incidence_angle_rad=0.0,
        beam_radius=1.0e-2,
        grid_spacing=GRID_SPACING,
        grid_height=GRID_HEIGHT,
        wire_radius=WIRE_RADIUS,
        collector_voltage=0.0,
        suppressor_voltage=-55.0,
        secondary_energy_model="fixed",
        secondary_direction_model="fixed",
        trajectory_t_max=1.0e-7,
        trajectory_max_step=1.0e-11,
    )

    summary = mc.estimate_hopping_probability(
        rng=None,
        config=config,
        geometry=geometry,
        electric_field=zero_electric_field,
    )

    assert summary == expected_summary
