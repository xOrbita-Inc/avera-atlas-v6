import math

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


def test_600km_orbit_period_sanity():
    # 600 km circular orbit should be about 14.9 orbits/day ±1%
    earth_radius_km = 6378.137
    mu_km3_s2 = 3.986004418e5
    radius_km = earth_radius_km + 600.0

    period_s = 2.0 * math.pi * math.sqrt(radius_km**3 / mu_km3_s2)
    orbits_per_day = 86400.0 / period_s

    assert abs(orbits_per_day - 14.9) / 14.9 < 0.01