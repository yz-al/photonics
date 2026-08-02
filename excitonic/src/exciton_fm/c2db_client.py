"""
C2DB client — reads the Computational 2D Materials Database via its public
ASE-db web interface (https://c2db.fysik.dtu.dk/), which is the live front-end
after the DB moved to https://2dhub.org/.

The web app is an htmx ASE-db server. Two things we need are fully public and
require no login or license click-through:

  1. Coverage counts.  GET /table?sid=<sid>&filter=<expr> renders a table
     fragment whose header reads "<N> rows out of <M>".  With `filter=<key>`
     (ASE-db syntax: "material has key named <key>") this yields the exact
     number of materials that carry a non-null value for <key> — i.e. label
     coverage, without scraping a single row.  This is what Phase 0 needs.

  2. Row pull.  The same fragment, paged via &page=<n>, carries the per-row
     values for whatever columns are toggled on.  Used later (Phase 1) to
     assemble the training frame; kept minimal here.

No third-party deps: standard library + `requests`.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Iterable

import requests

BASE = "https://c2db.fysik.dtu.dk"
_ROWS_OUT_OF = re.compile(r"([0-9][0-9,]*)\s*rows?\s*out of\s*([0-9][0-9,]*)", re.I)
_SID = re.compile(r"sid=([0-9]+)")


def _to_int(s: str) -> int:
    return int(s.replace(",", ""))


@dataclass
class C2DBClient:
    """Thin, polite client for the C2DB ASE-db web front-end."""

    base: str = BASE
    timeout: float = 30.0
    pause: float = 0.15  # be gentle to a public academic server
    session: requests.Session = field(default_factory=requests.Session)
    _sid: str | None = None

    # -- session ----------------------------------------------------------
    def sid(self, refresh: bool = False) -> str:
        """Fetch (and cache) a server session id from the landing page."""
        if self._sid is not None and not refresh:
            return self._sid
        r = self.session.get(self.base + "/", timeout=self.timeout)
        r.raise_for_status()
        m = _SID.search(r.text)
        if not m:
            raise RuntimeError("could not obtain an ASE-db session id (sid)")
        self._sid = m.group(1)
        return self._sid

    def _table(self, filter_expr: str, page: int = 0) -> str:
        params = {"sid": self.sid(), "filter": filter_expr, "page": str(page)}
        r = self.session.get(self.base + "/table", params=params, timeout=self.timeout)
        r.raise_for_status()
        time.sleep(self.pause)
        return r.text

    # -- coverage ---------------------------------------------------------
    def count(self, filter_expr: str) -> tuple[int, int]:
        """Return (matching_rows, total_rows) for an ASE-db filter expression.

        A bare key name ("E_B") counts materials that *have* that key.
        Comma-joined keys ("E_B, gap_gw") count materials that have all of them.
        Comparisons ("E_B>0.2") are supported by the server too.
        A key the schema does not know returns (0, total).
        """
        txt = self._table(filter_expr)
        m = _ROWS_OUT_OF.search(txt)
        if not m:
            raise RuntimeError(f"no 'rows out of' header for filter {filter_expr!r}")
        return _to_int(m.group(1)), _to_int(m.group(2))

    def total(self) -> int:
        """Total number of materials currently in C2DB."""
        return self.count("")[1]

    def coverage(self, keys: Iterable[str]) -> dict[str, int]:
        """Map each key -> number of materials carrying a value for it."""
        out: dict[str, int] = {}
        for k in keys:
            out[k] = self.count(k)[0]
        return out
