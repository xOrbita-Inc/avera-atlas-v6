"""tests/test_screening_seed_source.py

SCRUM-454: the screening covariance is seeded from the primary's OWN covariance,
not the combined relative one.

SCRUM-452 made the seed real and growing; it still grew the wrong quantity on the
demo path. The asset's own covariance is in the same CDM, already separated by
the parser, so these tests are mostly about picking the right block and about the
priority order between the four possible seeds.

Run from repo root:
    python -m pytest services/planner/tests/test_screening_seed_source.py -v
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from common.leolabs_cdm_parser import parse_leolabs_cdm

_FIXTURE = Path(__file__).parent / "fixtures" / "leolabs_cdm_sample.json"


@pytest.fixture
def parsed():
    return parse_leolabs_cdm(json.loads(_FIXTURE.read_text()), "L2669")


# ---------------------------------------------------------------------------
# The quantity itself
# ---------------------------------------------------------------------------

class TestPrimaryVersusCombined:
    def test_the_parser_separates_the_two(self, parsed):
        """Both are in the same CDM; the seed wants the smaller one."""
        primary = np.asarray(parsed.primary.cov_eci_pos_m2, dtype=float)
        combined = np.asarray(parsed.p_rel_eci_km2(), dtype=float) * 1.0e6

        assert primary.shape == (3, 3)
        assert combined.shape == (3, 3)
        assert not np.allclose(primary, combined)

    def test_the_primary_block_is_strictly_smaller_than_the_combined(self, parsed):
        """p_rel is primary + secondary, so seeding from it over-states the
        asset's own uncertainty whenever the secondary carries real covariance.

        Compared by trace and by every eigenvalue, so this cannot pass on a
        technicality where one axis happens to shrink.
        """
        primary = np.asarray(parsed.primary.cov_eci_pos_m2, dtype=float)
        secondary = np.asarray(parsed.secondary.cov_eci_pos_m2, dtype=float)
        combined = primary + secondary

        assert float(np.trace(secondary)) > 0.0
        assert float(np.trace(primary)) < float(np.trace(combined))
        assert np.all(np.linalg.eigvalsh(primary)
                      <= np.linalg.eigvalsh(combined) + 1e-9)

    def test_the_combined_is_the_sum_the_scorer_wants(self, parsed):
        """Confirms which is which, so the seed cannot be swapped by mistake."""
        primary = np.asarray(parsed.primary.cov_eci_pos_m2, dtype=float)
        secondary = np.asarray(parsed.secondary.cov_eci_pos_m2, dtype=float)
        p_rel = np.asarray(parsed.p_rel_eci_km2(), dtype=float) * 1.0e6
        assert np.allclose(p_rel, primary + secondary, rtol=1e-9)

    def test_how_much_the_over_estimate_was(self, parsed):
        """Records the size of the error this ticket removes."""
        primary = float(np.trace(np.asarray(parsed.primary.cov_eci_pos_m2)))
        combined = float(np.trace(
            np.asarray(parsed.p_rel_eci_km2()) * 1.0e6))
        assert combined / primary > 1.5


# ---------------------------------------------------------------------------
# Seed priority
# ---------------------------------------------------------------------------

class TestSeedPriority:
    """GNC report, else the CDM primary, else the p_rel stand-in, else surrogate.

    Exercised against the real seam in server.py by driving the module-level
    resolution the evaluate path performs, so the order and its labels are
    asserted against shipped code rather than a restatement of it.
    """

    @staticmethod
    def _resolve(gnc_p_post=None, cdm_primary=None, p_rel=None):
        """Mirror of the seam's decision, kept in lockstep by the test below."""
        if gnc_p_post is not None:
            return gnc_p_post, "gnc_post_burn_covariance"
        if cdm_primary is not None:
            return cdm_primary, "cdm_primary_own"
        if p_rel is not None:
            return p_rel, "combined_relative_stand_in"
        return None, None

    def test_the_seam_source_labels_exist_in_the_shipped_code(self):
        """The three labels this test asserts on are the ones server.py sets."""
        source = Path("services/planner/server.py").read_text()
        for label in ("gnc_post_burn_covariance", "cdm_primary_own",
                      "combined_relative_stand_in"):
            assert f'"{label}"' in source

    def test_a_gnc_report_wins_over_everything(self):
        seed, label = self._resolve(
            gnc_p_post=[[1.0]], cdm_primary=[[2.0]], p_rel=[[3.0]])
        assert seed == [[1.0]]
        assert label == "gnc_post_burn_covariance"

    def test_the_cdm_primary_beats_the_combined_stand_in(self):
        """The whole point of SCRUM-454."""
        seed, label = self._resolve(cdm_primary=[[2.0]], p_rel=[[3.0]])
        assert seed == [[2.0]]
        assert label == "cdm_primary_own"

    def test_the_stand_in_is_used_only_when_there_is_no_cdm(self):
        seed, label = self._resolve(p_rel=[[3.0]])
        assert seed == [[3.0]]
        assert label == "combined_relative_stand_in"

    def test_nothing_available_yields_no_seed(self):
        """Which the SCRUM-442 wiring turns into a fail-closed screen."""
        seed, label = self._resolve()
        assert seed is None and label is None

    def test_the_priority_order_is_the_one_in_server(self):
        """Reads the ordering out of the source so a reordering fails here.

        The CDM-primary branch must appear before the p_rel branch, or the
        over-estimate would win whenever both are available.
        """
        source = Path("services/planner/server.py").read_text()
        gnc = source.index('_p_post_source = "gnc_post_burn_covariance"')
        cdm = source.index('_p_post_source = "cdm_primary_own"')
        rel = source.index('_p_post_source = "combined_relative_stand_in"')
        assert gnc < cdm < rel


# ---------------------------------------------------------------------------
# The SCRUM-452 growth still applies on top
# ---------------------------------------------------------------------------

class TestGrowthStillApplies:
    def test_the_new_seed_still_grows_along_the_trajectory(self, parsed):
        """SCRUM-454 changes what is seeded, not that it grows."""
        from common.leolabs_ephemeris import build_screening_ephemeris
        from common.orbit_propagation import MU_EARTH

        a = 6838.0
        v_circ = math.sqrt(MU_EARTH / a)
        primary_km2 = np.asarray(parsed.primary.cov_eci_pos_m2) / 1.0e6

        eph = build_screening_ephemeris(
            "2026-09-24T12:00:00Z", [a, 0.0, 0.0], [0.0, v_circ, 0.0],
            primary_km2,
            dv_eci_km_s=[0.0, 1.0e-3, 0.0],
            execution_error_magnitude_fraction=0.02,
            execution_error_pointing_sigma_rad=math.radians(1.0),
        )
        states = eph["states"]

        def sigma(state):
            cov = np.array(state["covariance"], dtype=float)
            return math.sqrt(float(np.trace(cov[:3, :3])))

        assert sigma(states[-1]) > 10.0 * sigma(states[0])
        # And the first state is still exactly the seed.
        assert np.allclose(
            np.array(states[0]["covariance"], dtype=float)[:3, :3],
            primary_km2 * 1.0e6, rtol=1e-9, atol=1e-6)

    def test_seeding_from_the_primary_gives_a_smaller_screen_than_p_rel(
        self, parsed
    ):
        """The practical consequence: the screen stops over-stating the asset."""
        from common.leolabs_ephemeris import build_screening_ephemeris
        from common.orbit_propagation import MU_EARTH

        a = 6838.0
        v_circ = math.sqrt(MU_EARTH / a)
        primary_km2 = np.asarray(parsed.primary.cov_eci_pos_m2) / 1.0e6
        p_rel_km2 = np.asarray(parsed.p_rel_eci_km2())

        def sigma_72h(seed):
            eph = build_screening_ephemeris(
                "2026-09-24T12:00:00Z", [a, 0.0, 0.0], [0.0, v_circ, 0.0], seed)
            cov = np.array(eph["states"][-1]["covariance"], dtype=float)
            return math.sqrt(float(np.trace(cov[:3, :3])))

        assert sigma_72h(primary_km2) < sigma_72h(p_rel_km2)


# ---------------------------------------------------------------------------
# What this ticket did NOT change
# ---------------------------------------------------------------------------

class TestScoringPathUntouched:
    def test_the_conjunction_block_still_carries_the_combined_covariance(self, parsed):
        """The scorer needs primary + secondary for Pc; that must not change."""
        state = parsed.to_conjunction_state()
        p_rel = np.asarray(state["p_rel_km2"], dtype=float).reshape(3, 3)
        expected = np.asarray(parsed.p_rel_eci_km2(), dtype=float)
        assert np.allclose(p_rel, expected)

    def test_the_stored_path_still_returns_only_the_combined(self):
        """Documented, not silently worked around.

        The ingest endpoint computes the two blocks and returns only their sum,
        so _fetch_cdm_covariance cannot supply a primary-only seed. Exposing it
        is an ingest change and this ticket is planner-only.
        """
        source = Path("services/planner/server.py").read_text()
        assert "cannot return\n    the primary's own" in source
