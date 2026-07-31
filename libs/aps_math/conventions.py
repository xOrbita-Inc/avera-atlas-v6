"""Shared numerical conventions for AVERA-ATLAS.

ADR-010, SCRUM-389.

What belongs here
-----------------
A value belongs in this module when two or more services must agree on it and
disagreeing would produce inconsistent physics. That is the third bucket in
ADR-010:

  - a fact about one object goes on the object (a spacecraft's radius, a CDM's
    covariance)
  - an operator's choice goes in operator policy YAML, loaded per request
  - a system convention goes here, changed by a reviewed commit
  - a fixture goes beside the demo or test it feeds

The distinction that matters most: a value here is NOT an operator setting.
Lowering the hard-body radius or the debris uncertainty would reduce every Pc
the system produces and quietly suppress maneuver recommendations. That is a
safety property wearing the costume of a tuning knob, and configuration is not
reviewed the way a commit is.

Every constant below carries what it represents, why the value was chosen, and
what changing it would affect. A convention with no recorded reasoning becomes
folklore within a sprint, and then nobody can tell a deliberate choice from an
accident.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Hard-body radius
# ---------------------------------------------------------------------------

# Combined hard-body radius, metres, used when a conjunction does not supply
# one and the spacecraft's own radius cannot be combined with a known secondary.
#
# THIS IS A SCREENING CONVENTION, NOT A PHYSICAL RADIUS. Read that again before
# changing it.
#
# For a 6U CubeSat primary the circumscribing sphere radius is 0.221 m stowed
# (the 6U CubeSat Design Specification envelope is 100 x 226.3 x 366 mm) and
# roughly 0.6 to 0.8 m with deployed arrays. A 15 m combined value therefore
# implies a secondary of about 14.8 m. Nothing in the catalogue is that size; a
# spent upper stage is a few metres.
#
# Deliberately inflating the hard-body radius is normal conjunction-assessment
# practice. It absorbs attitude uncertainty, unknown secondary size and the fact
# that neither object is a sphere. But Pc scales as the SQUARE of this value, so
# it is a large lever: at a 250 m covariance and a 300 m miss, 15 m gives
# Pc 8.76e-04 while a physically plausible 2 m gives 1.56e-05, a factor of 56.
# For comparison, the default-mode defect corrected in SCRUM-390 was 4.26x.
#
# The convention is only sound if pc_maneuver_threshold was calibrated against
# the same convention, and there is currently no record of what 1e-4 was
# calibrated against. SCRUM-394 carries that question. Until it is answered this
# value is inherited, not validated. Do not change it on this ticket's authority.
#
# Changing it affects: every Pc the propagator publishes, every Pc the planner
# computes, the demo risk levels, and whether SCRUM-391's tracked-secondary
# presets still reach RED.
DEFAULT_COMBINED_HBR_M: float = 15.0

# Radius of a 6U CubeSat with deployed solar arrays, metres, used as the default
# for SatelliteCapability.radius_m when an operator does not state one.
#
# 0.221 m is the stowed circumscribing sphere. Deployed arrays dominate, and a
# 1.2 m tip-to-tip span gives 0.60 m. The deployed figure is the right one
# because that is the configuration the spacecraft flies in.
#
# This is a default for a fact, not a convention. Any operator who knows their
# spacecraft should state its radius rather than inherit this.
DEFAULT_PRIMARY_RADIUS_M: float = 0.60

# Radius assumed for a secondary object of unknown size, metres.
#
# Upper-stage scale, which is the large end of what a catalogued secondary is
# likely to be. Used only to decide whether a physically large pair exceeds the
# screening convention above; see combined_hbr_m.
DEFAULT_SECONDARY_RADIUS_M: float = 2.0


def combined_hbr_m(primary_radius_m: float) -> tuple:
    """Combined hard-body radius for a conjunction, and where it came from.

    Returns (hbr_m, source).

    The screening convention acts as a FLOOR, not as a replacement for physics.
    DEFAULT_COMBINED_HBR_M is deliberately inflated to absorb attitude
    uncertainty and an unknown secondary, so for a small primary it dominates
    and the physical sum never gets a look in. That is intended.

    But a genuinely large primary must not be screened as though it were small.
    A 14 m-radius spacecraft plus an upper-stage-scale secondary is physically
    larger than the 15 m convention, and using the convention there would
    understate Pc. So the physical sum wins when it exceeds the floor.

    For a 6U CubeSat this returns the convention every time: 0.60 + 2.0 is 2.6 m,
    well under the 15 m floor. The physical branch exists so the behaviour is
    correct for a spacecraft we do not fly yet, rather than correct by accident
    for the one we do.
    """
    physical = float(primary_radius_m) + DEFAULT_SECONDARY_RADIUS_M
    if physical > DEFAULT_COMBINED_HBR_M:
        return physical, "physical_sum"
    return DEFAULT_COMBINED_HBR_M, "screening_convention"


# ---------------------------------------------------------------------------
# Covariance fallbacks
# ---------------------------------------------------------------------------

# 1-sigma position uncertainty, metres, assumed for a secondary object known
# only from a TLE, before scaling by an object's confidence.
#
# A TLE-derived position for a LEO object carries along-track error of order
# kilometres at a screening horizon of a day or two, which is what this
# represents. It is scaled as DEFAULT_DEBRIS_UNCERTAINTY_M / confidence, so it
# can only ever be inflated above 2 km and never reduced. That asymmetry is
# deliberate: an unknown object should not become better known by asserting
# confidence in it.
#
# When a real covariance is available it is used directly and is NOT scaled by
# confidence, because a supplied sigma is how well the object is known and
# confidence is a stand-in for the same thing. See SCRUM-391.
#
# Changing it affects: every Pc computed for an object without a supplied
# covariance, which today is most of them.
DEFAULT_DEBRIS_UNCERTAINTY_M: float = 2000.0

# Ratio of cross-track to along-track 1-sigma for the diagonal covariance built
# by default_covariance_from_uncertainty. Orbit determination error is not
# isotropic; along-track grows fastest.
#
# 0.5 for a secondary and 0.3 for the primary asset are the values the
# propagator has used since the demo path was written. They are not derived from
# a fit and should be treated as placeholders that happen to be the right shape.
DEFAULT_SECONDARY_CROSS_TRACK_FACTOR: float = 0.5
DEFAULT_PRIMARY_CROSS_TRACK_FACTOR: float = 0.3

# 1-sigma position uncertainty, metres, assumed for the primary asset when no
# better figure exists. The asset is the object we know most about, so this is
# tighter than the debris default.
DEFAULT_ASSET_UNCERTAINTY_M: float = 1000.0


# ---------------------------------------------------------------------------
# Maneuver magnitude search
# ---------------------------------------------------------------------------

# Number of delta-v magnitudes evaluated per candidate direction, SCRUM-387.
#
# Before 387 the planner evaluated exactly one magnitude, the effective delta-v
# limit, so the recommended burn was the ceiling by construction and no interior
# optimum could be found even once the utility admitted one.
#
# The grid is GEOMETRIC, not linear. Measured on the RED-001 geometry the
# optimal burn runs from 0.007 m/s at a four-hour lead down to 0.0004 m/s at
# seventy-two hours, against a policy ceiling of order 1 m/s. That is four
# decades. A linear grid would put every sample in the top decade and resolve
# nothing where the answer actually lives.
#
# 24 points across four decades is six per decade, or about one every 1.5x in
# delta-v. Pc varies smoothly with delta-v, so the utility near its maximum is
# flat and a coarser spacing costs little accuracy. Cost is linear in this
# number, 6 directions times this many Pc evaluations per event.
#
# This is a numerical convention rather than an operator choice, per ADR-010.
# An operator who wants to spend less fuel lowers max_dv_per_event_ms; they do
# not tune the resolution of the search.
DV_SEARCH_POINTS: int = 24


# ---------------------------------------------------------------------------
# Screening
# ---------------------------------------------------------------------------

# Separation beyond which an event is not worth computing Pc for, kilometres.
# Purely an optimisation. At 100 km with any covariance this system produces,
# Pc underflows to zero, so nothing is discarded that a threshold would have
# caught.
SCREENING_THRESHOLD_KM: float = 100.0


__all__ = [
    "DEFAULT_COMBINED_HBR_M",
    "DEFAULT_PRIMARY_RADIUS_M",
    "DEFAULT_SECONDARY_RADIUS_M",
    "combined_hbr_m",
    "DEFAULT_DEBRIS_UNCERTAINTY_M",
    "DEFAULT_SECONDARY_CROSS_TRACK_FACTOR",
    "DEFAULT_PRIMARY_CROSS_TRACK_FACTOR",
    "DEFAULT_ASSET_UNCERTAINTY_M",
    "DV_SEARCH_POINTS",
    "SCREENING_THRESHOLD_KM",
]
