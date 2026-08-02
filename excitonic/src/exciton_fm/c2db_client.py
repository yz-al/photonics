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
_CELL = re.compile(r"<t[hd][^>]*>(.*?)</t[hd]>", re.S | re.I)
_TAG = re.compile(r"<[^>]+>")

# The ASE-db table renders header + data cells flat, in document order, starting
# at the "Formula" cell. Map the exact human-readable column headers C2DB emits
# (unicode subscripts and all) back onto their schema keys. Covers the default
# columns plus every column this client toggles on.
HEADER_TO_KEY = {
    "Formula": "formula",
    "Energy above hull [eV/atom]": "ehull",
    "Heat of formation [eV/atom]": "hform",
    "Band gap (PBE) [eV]": "gap",
    "Magnetic": "is_magnetic",
    "Layer group (not Space group)": "layergroup",
    "Exciton binding energy (BSE) [eV]": "E_B",
    "Band gap (G₀W₀) [eV]": "gap_gw",
    "Band gap (HSE06) [eV]": "gap_hse",
    "CBM DOS effective mass (PBE)": "emass_cbm",
    "VBM DOS effective mass (PBE)": "emass_vbm",
    "Interband polarizability (x) [Å]": "alphax_el",
    "Interband polarizability (y) [Å]": "alphay_el",
    "Interband polarizability (z) [Å]": "alphaz_el",
    "Plasma frequency (x) [eV Å0.5]": "plasmafrequency_x",
    "Static polarizability (phonons) (x) [Å]": "alphax_lat",
    "Thickness [Å]": "thickness",
    "Unit cell area [Å2]": "area",
    "Number of atoms": "natoms",
    "Number of species": "nspecies",
    "Out-of-plane dipole [e Å/unit cell]": "dipz",
}

# Non-default columns to toggle on for a Phase-1 feature pull. (Formula, ehull,
# hform, is_magnetic, layergroup are already default columns — toggling them
# would switch them OFF, since toggle is a switch.)
FEATURE_TOGGLE_KEYS = (
    "E_B", "gap_gw", "gap_hse", "emass_cbm", "emass_vbm",
    "alphax_el", "alphay_el", "alphaz_el", "plasmafrequency_x", "alphax_lat",
    "thickness", "area", "natoms", "nspecies", "dipz",
)


def _to_int(s: str) -> int:
    return int(s.replace(",", ""))


def _clean(cell_html: str) -> str:
    import html as _html
    return _html.unescape(_TAG.sub("", cell_html)).strip()


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

    def _table(self, filter_expr: str | None = None, page: int | None = None) -> str:
        # The server treats a `filter` param as a *new search* and resets to
        # page 0; to paginate you send `page` alone and rely on the session's
        # stored filter. So never send both together.
        params: dict[str, str] = {"sid": self.sid()}
        if page is not None:
            params["page"] = str(page)
        else:
            params["filter"] = filter_expr or ""
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

    # -- row-level pull ---------------------------------------------------
    def _toggle(self, key: str) -> None:
        params = {"sid": self.sid(), "toggle": key}
        r = self.session.get(self.base + "/table", params=params, timeout=self.timeout)
        r.raise_for_status()
        time.sleep(self.pause)

    @staticmethod
    def _parse_table(text: str) -> tuple[list[str], list[list[str]]]:
        """Return (column_keys, rows) from a table fragment.

        The header is the longest run of known-label cells starting at 'Formula';
        the remainder is chunked into rows of len(header) values.
        """
        cells = [_clean(c) for c in _CELL.findall(text)]
        if "Formula" not in cells:
            return [], []
        cells = cells[cells.index("Formula"):]
        header: list[str] = []
        for c in cells:
            if c in HEADER_TO_KEY:
                header.append(HEADER_TO_KEY[c])
            else:
                break
        ncol = len(header)
        data = cells[ncol:]
        nrows = len(data) // ncol if ncol else 0
        rows = [data[i * ncol:(i + 1) * ncol] for i in range(nrows)]
        return header, rows

    def fetch_rows(
        self,
        filter_expr: str,
        toggle_keys: Iterable[str] = FEATURE_TOGGLE_KEYS,
        max_pages: int = 100,
    ) -> list[dict[str, str]]:
        """Pull every row matching `filter_expr`, as {key: raw_string_value}.

        Empty cells come back as "" (missing value). Values are left as strings;
        numeric parsing is the caller's job (so "Yes"/"No"/formula/number are all
        preserved faithfully). Pages until the returned page repeats or empties.
        """
        # Ensure a fresh session, then toggle the desired feature columns on.
        self.sid(refresh=True)
        for k in toggle_keys:
            self._toggle(k)

        out: list[dict[str, str]] = []
        seen_prev: str | None = None
        for page in range(max_pages):
            # Page 0: set the filter (session-stored). Pages >0: page param only,
            # never the filter (which would reset back to page 0).
            text = self._table(filter_expr) if page == 0 else self._table(page=page)
            header, rows = self._parse_table(text)
            if not rows:
                break
            # Fail loudly if the schema grew a header column we don't map
            # (which would silently truncate ncol and misalign every row).
            if page == 0:
                want = [k for k in toggle_keys if k not in header]
                if want:
                    raise RuntimeError(
                        f"toggled columns missing from parsed header (unmapped "
                        f"header column upstream?): {want}. Header={header}"
                    )
                total = None
                m = _ROWS_OUT_OF.search(text)
                if m:
                    total = _to_int(m.group(1))
            fingerprint = "|".join(rows[0])
            if fingerprint == seen_prev:  # server clamped to last page; stop
                break
            seen_prev = fingerprint
            for r in rows:
                out.append(dict(zip(header, r)))
            if total is not None and len(out) >= total:
                break
        return out
