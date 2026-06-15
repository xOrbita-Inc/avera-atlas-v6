from services.sandbox.config import DebrisConfig, HostConfig, IntegratorConfig, SimConfig
from services.sandbox.scenario import create_seeded_swarm_and_hosts
from services.sandbox.simulation import run_simulation


def test_seeded_swarm_counts_and_snapshot_count():
    config = SimConfig(
        seed=42,
        debris=DebrisConfig(count=10),
        hosts=HostConfig(count=3),
        integrator=IntegratorConfig(
            dt_seconds=10.0,
            duration_seconds=3600.0,
            save_every_n_steps=60,
        ),
    )

    objects = create_seeded_swarm_and_hosts(config)
    result = run_simulation(objects, config)

    host_count = sum(1 for obj in objects.values() if obj.kind == "host")
    debris_count = sum(1 for obj in objects.values() if obj.kind == "debris")

    assert host_count == 3
    assert debris_count == 10
    assert len(objects) == 13

    # t=0 plus hourly run saved every 60 steps with dt=10s => every 600s
    # 0, 600, 1200, 1800, 2400, 3000, 3600 => 7 snapshots
    assert len(result.snapshots) == 7