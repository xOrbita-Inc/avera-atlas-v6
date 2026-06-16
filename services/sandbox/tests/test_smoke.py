import numpy as np

from services.sandbox.config import SimConfig, IntegratorConfig
from services.sandbox.scenario import (
    create_hello_world_scenario,
    create_seeded_swarm_and_hosts,
)
from services.sandbox.simulation import run_simulation


def test_hello_world_runs_and_saves_expected_snapshots():
    config = SimConfig(
        integrator=IntegratorConfig(
            dt_seconds=1.0,
            duration_seconds=24.0 * 3600.0,
            save_every_n_steps=3600,
        )
    )

    objects = create_hello_world_scenario(config)
    result = run_simulation(objects, config)

    assert len(result.snapshots) == 25  # t=0 plus hourly saves through 24 hr
    assert "debris_001" in result.snapshots[0].states
    assert "host_001" in result.snapshots[0].states


def test_seeded_generation_is_deterministic():
    config = SimConfig(
        seed=42,
        integrator=IntegratorConfig(
            dt_seconds=10.0,
            duration_seconds=60.0,
            save_every_n_steps=1,
        ),
    )

    objects_a = create_seeded_swarm_and_hosts(config)
    objects_b = create_seeded_swarm_and_hosts(config)

    assert objects_a.keys() == objects_b.keys()

    for object_id in objects_a:
        a = objects_a[object_id]
        b = objects_b[object_id]
        assert (a.r_eci_km == b.r_eci_km).all()
        assert (a.v_eci_km_s == b.v_eci_km_s).all()


def test_propagated_600km_orbit_completes_14_9_orbits_per_day():
    config = SimConfig(
        integrator=IntegratorConfig(
            dt_seconds=1.0,
            duration_seconds=7000.0,
            save_every_n_steps=1,
        ),
    )

    objects = create_hello_world_scenario(config)
    host_only = {"host_001": objects["host_001"]}

    result = run_simulation(host_only, config)

    initial_state = result.snapshots[0].states["host_001"]
    initial_r = initial_state.r_eci_km
    initial_v = initial_state.v_eci_km_s

    radial_hat = initial_r / np.linalg.norm(initial_r)

    transverse_v = initial_v - np.dot(initial_v, radial_hat) * radial_hat
    transverse_hat = transverse_v / np.linalg.norm(transverse_v)

    times = []
    wrapped_phases = []

    for snapshot in result.snapshots:
        state = snapshot.states["host_001"]
        position = state.r_eci_km

        radial_component = float(np.dot(position, radial_hat))
        transverse_component = float(np.dot(position, transverse_hat))

        phase_rad = np.arctan2(transverse_component, radial_component)

        times.append(snapshot.t_seconds)
        wrapped_phases.append(phase_rad)

    unwrapped_phases = np.unwrap(np.asarray(wrapped_phases))
    target_phase = 2.0 * np.pi

    crossing_indices = np.flatnonzero(unwrapped_phases >= target_phase)
    assert crossing_indices.size > 0

    upper_index = int(crossing_indices[0])
    lower_index = upper_index - 1

    lower_phase = unwrapped_phases[lower_index]
    upper_phase = unwrapped_phases[upper_index]
    lower_time = times[lower_index]
    upper_time = times[upper_index]

    fraction = (target_phase - lower_phase) / (upper_phase - lower_phase)
    measured_period_s = lower_time + fraction * (upper_time - lower_time)

    measured_orbits_per_day = 86400.0 / measured_period_s

    assert abs(measured_orbits_per_day - 14.9) / 14.9 < 0.01

def test_seeded_propagated_trajectories_are_bit_identical():
    config = SimConfig(
        seed=42,
        integrator=IntegratorConfig(
            dt_seconds=10.0,
            duration_seconds=600.0,
            save_every_n_steps=10,
        ),
    )

    objects_a = create_seeded_swarm_and_hosts(config)
    objects_b = create_seeded_swarm_and_hosts(config)

    result_a = run_simulation(objects_a, config)
    result_b = run_simulation(objects_b, config)

    assert len(result_a.snapshots) == len(result_b.snapshots)

    for snapshot_a, snapshot_b in zip(result_a.snapshots, result_b.snapshots):
        assert snapshot_a.t_seconds == snapshot_b.t_seconds
        assert snapshot_a.states.keys() == snapshot_b.states.keys()

        for object_id in snapshot_a.states:
            state_a = snapshot_a.states[object_id]
            state_b = snapshot_b.states[object_id]

            assert state_a.t_seconds == state_b.t_seconds
            assert np.array_equal(state_a.r_eci_km, state_b.r_eci_km)
            assert np.array_equal(state_a.v_eci_km_s, state_b.v_eci_km_s)