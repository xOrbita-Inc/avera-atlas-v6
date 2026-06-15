import numpy as np

from services.sandbox.config import DragConfig
from services.sandbox.models import PhysicalProperties, SimObject
from services.sandbox.propagator import j2_accel_km_s2, rk4_step, two_body_accel_km_s2
from services.sandbox.propagator import (
    j2_accel_km_s2,
    rk4_step,
    rk4_step_batch,
    two_body_accel_km_s2,
)

def make_test_object() -> SimObject:
    return SimObject(
        object_id="test_obj",
        kind="debris",
        r_eci_km=np.array([6978.137, 0.0, 0.0], dtype=np.float64),
        v_eci_km_s=np.array([0.0, 7.5, 1.0], dtype=np.float64),
        physical=PhysicalProperties(
            size_bin="5cm",
            characteristic_size_m=0.05,
            mass_kg=0.05,
            area_m2=0.002,
            cd=2.2,
        ),
    )


def test_two_body_acceleration_points_toward_earth():
    r = np.array([7000.0, 0.0, 0.0], dtype=np.float64)
    a = two_body_accel_km_s2(r)

    assert a[0] < 0
    assert abs(a[1]) < 1e-12
    assert abs(a[2]) < 1e-12


def test_j2_acceleration_is_finite():
    r = np.array([7000.0, 100.0, 200.0], dtype=np.float64)
    a = j2_accel_km_s2(r)

    assert np.isfinite(a).all()


def test_rk4_step_returns_finite_state():
    obj = make_test_object()
    drag_cfg = DragConfig(enabled=True)

    r_next, v_next = rk4_step(
        r_eci_km=obj.r_eci_km,
        v_eci_km_s=obj.v_eci_km_s,
        dt_s=1.0,
        sim_object=obj,
        drag_cfg=drag_cfg,
    )

    assert np.isfinite(r_next).all()
    assert np.isfinite(v_next).all()
    assert r_next.shape == (3,)
    assert v_next.shape == (3,)

def test_batch_rk4_matches_scalar_rk4_for_single_object():
    obj = make_test_object()
    drag_cfg = DragConfig(enabled=True)

    scalar_r, scalar_v = rk4_step(
        r_eci_km=obj.r_eci_km,
        v_eci_km_s=obj.v_eci_km_s,
        dt_s=1.0,
        sim_object=obj,
        drag_cfg=drag_cfg,
    )

    batch_r, batch_v = rk4_step_batch(
        r_eci_km=np.array([obj.r_eci_km], dtype=np.float64),
        v_eci_km_s=np.array([obj.v_eci_km_s], dtype=np.float64),
        dt_s=1.0,
        area_m2=np.array([obj.physical.area_m2], dtype=np.float64),
        mass_kg=np.array([obj.physical.mass_kg], dtype=np.float64),
        cd=np.array([obj.physical.cd], dtype=np.float64),
        drag_cfg=drag_cfg,
    )

    assert np.allclose(batch_r[0], scalar_r)
    assert np.allclose(batch_v[0], scalar_v)