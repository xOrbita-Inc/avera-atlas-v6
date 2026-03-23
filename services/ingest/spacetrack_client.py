"""Space-Track.org CDM client using session-cookie authentication."""

from __future__ import annotations

import os

import requests


class SpaceTrackClient:
    """Fetch CCSDS 508.0-B-1 CDM data from Space-Track's public catalogue."""

    BASE_URL = "https://www.space-track.org"

    def __init__(self, username: str | None = None, password: str | None = None) -> None:
        self.username = username or os.environ["SPACETRACK_USER"]
        self.password = password or os.environ["SPACETRACK_PASS"]
        self.session = requests.Session()

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def login(self) -> None:
        """Authenticate and store the session cookie."""
        resp = self.session.post(
            f"{self.BASE_URL}/ajaxauth/login",
            data={"identity": self.username, "password": self.password},
        )
        resp.raise_for_status()
        # Space-Track returns a JSON body; an empty string or error key means failure.
        body = resp.text
        if not body or "failed" in body.lower():
            raise RuntimeError(f"Space-Track login failed: {body[:200]}")

    # ------------------------------------------------------------------
    # CDM queries
    # ------------------------------------------------------------------

    def get_cdms_for_norad(
        self, norad_id: int, days_lookahead: int = 7
    ) -> str:
        """Return raw KVN CDM blocks for a single NORAD catalogue ID.

        Filters to TCAs within *days_lookahead* days from now.
        """
        url = (
            f"{self.BASE_URL}/basicspacedata/query/class/cdm_public"
            f"/NORAD_CAT_ID/{norad_id}"
            f"/TCA/>now-P{days_lookahead}D"
            "/orderby/TCA asc"
            "/format/kvn"
        )
        resp = self.session.get(url)
        resp.raise_for_status()
        return resp.text

    def get_cdms_above_pc(
        self, pc_threshold: float = 1e-5, days_lookahead: int = 7
    ) -> str:
        """Return raw KVN CDM blocks where Pc >= *pc_threshold*."""
        url = (
            f"{self.BASE_URL}/basicspacedata/query/class/cdm_public"
            f"/COLLISION_PROBABILITY/>={pc_threshold}"
            f"/TCA/>now-P{days_lookahead}D"
            "/orderby/TCA asc"
            "/format/kvn"
        )
        resp = self.session.get(url)
        resp.raise_for_status()
        return resp.text
