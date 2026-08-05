"""
BiDB / HetDB client — reads the two vdW-stacking databases that host the
DIPOLAR / interlayer excitons (see reports/u_definition_decision.md), via the
same public ASE-db web front-ends that C2DB uses:

  * BiDB  — van der Waals Bilayer Database        https://bidb.fysik.dtu.dk/
  * HetDB — vdW 2D Heterostructure Database       https://hetdb.fysik.dtu.dk/

Both are htmx ASE-db servers, byte-for-byte the same web app as
c2db.fysik.dtu.dk, so the coverage/row machinery in `c2db_client` transfers
verbatim: get a session id (sid) from the landing page, then
GET /table?sid=<sid>&filter=<expr> renders a fragment whose header reads
"<N> rows out of <M>". `filter=<key>` counts materials that HAVE that key;
comma-joined keys are AND; comparisons (ebind>20, has_inversion_symmetry=False)
work too. Row pulls page via &page=<n> with feature columns toggled on.

The ONLY things that differ between the three databases are (a) the base URL and
(b) the human-readable column-header -> schema-key map (each DB emits its own
column labels). So this module subclasses `C2DBClient`, keeps sid()/_table()/
count()/total()/coverage()/_toggle() unchanged, and overrides just the
header-map, the parse anchor, and the default feature-toggle set.

No third-party deps beyond `requests` (inherited).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

import requests

from .c2db_client import (
    C2DBClient,
    _CELL,
    _ROWS_OUT_OF,
    _clean,
    _make_session,
    _to_int,
)

BIDB_BASE = "https://bidb.fysik.dtu.dk"
HETDB_BASE = "https://hetdb.fysik.dtu.dk"


# --- BiDB (homobilayers) --------------------------------------------------
# Human-readable header (exactly as the server renders it, unicode and all)
# -> schema key.  Taken from https://bidb.fysik.dtu.dk/help .
BIDB_HEADER_TO_KEY = {
    "Formula": "formula",
    "Binding energy [meV/Å2]": "ebind",
    "Slide stability": "slide_stability",
    "Unique ID": "uid",
    "Magnetic": "magnetic",
    "Interlayer distance [Å]": "interlayer_distance",
    "Band gap [eV]": "gap_pbe",
    "Inversion symmetry": "has_inversion_symmetry",
    "Thickness [Å]": "thickness",
    "Unit cell area [Å2]": "area",
    "Number of layers": "number_of_layers",
    "Number of atoms": "natoms",
    "Number of species": "nspecies",
    "2D Bravais type": "bravais_type",
    "Layer group": "layergroup",
    "Layer group number": "lgnum",
    "Monolayer ID": "monolayer_uid",
    "Monolayer (C2DB)": "c2db_uid",
    "Bilayer ID": "bilayer_uid",
    "Reduced formula": "reduced",
    "Stoichiometry": "stoichiometry",
    "Structure origin": "label",
    "Point group": "pointgroup",
}

# Non-default BiDB columns worth toggling on for a dipolar-U feature pull.
# (Formula, ebind, slide_stability, uid, magnetic are already default columns.)
BIDB_FEATURE_TOGGLE_KEYS = (
    "interlayer_distance",
    "has_inversion_symmetry",
    "gap_pbe",
    "thickness",
    "area",
    "number_of_layers",
    "monolayer_uid",
    "c2db_uid",
)


# --- HetDB (heterobilayers) ----------------------------------------------
# From https://hetdb.fysik.dtu.dk/help .
HETDB_HEADER_TO_KEY = {
    "Formula 1": "formula_a",
    "Formula 2": "formula_b",
    "Twist angle [°]": "twist_angle",
    "Number of atoms": "natoms",
    "Number of species": "nspecies",
    "Maximum strain [%]": "maxstrain",
    "Band gap (PBE) [eV]": "pbe_gap_soc",
    "Direct band gap (PBE) [eV]": "pbe_gap_dir_soc",
    "Band gap w/o SOC (PBE) [eV]": "pbe_gap_nosoc",
    "Direct band gap w/o SOC (PBE) [eV]": "pbe_gap_dir_nosoc",
    "Band gap (LAPS) [eV]": "scs_gap_soc",
    "Direct band gap (LAPS) [eV]": "scs_gap_dir_soc",
    "Band gap w/o SOC (LAPS) [eV]": "scs_gap_nosoc",
    "Direct band gap w/o SOC (LAPS) [eV]": "scs_gap_dir_nosoc",
    "Interlayer distance [Å]": "interlayer_distance",
    "Conduction band minimum wrt. vacuum (PBE) [eV]": "pbe_cbm_soc",
    "Valence band maximum wrt. vacuum (PBE) [eV]": "pbe_vbm_soc",
    "Conduction band minimum wrt. vacuum w/o SOC (PBE) [eV]": "pbe_cbm_nosoc",
    "Valence band maximum wrt. vacuum w/o SOC (PBE) [eV]": "pbe_vbm_nosoc",
    "Conduction band minimum wrt. vacuum (LAPS) [eV]": "scs_cbm_soc",
    "Valence band maximum wrt. vacuum (LAPS) [eV]": "scs_vbm_soc",
    "Conduction band minimum wrt. vacuum w/o SOC (LAPS) [eV]": "scs_cbm_nosoc",
    "Valence band maximum wrt. vacuum w/o SOC (LAPS) [eV]": "scs_vbm_nosoc",
    "Fermi level wrt. vacuum (PBE) [eV]": "pbe_efermi_soc",
    "Fermi level wrt. vacuum (LAPS) [eV]": "scs_efermi_soc",
    "Vacuum level (PBE) [eV]": "pbe_evac",
    "Vacuum level (LAPS) [eV]": "scs_evac",
    "Energy [eV]": "energy",
    "Formula": "formula",
    "Reduced formula": "reduced",
    "Stoichiometry": "stoichiometry",
    "Structure origin": "label",
    "Unique ID": "uid",
    "C2DB-uid 1": "uid_a",
    "C2DB-uid 2": "uid_b",
    "Unit cell area [Å2]": "area",
    "Original file-system folder": "folder",
}

# Non-default HetDB columns for a band-alignment / type-II feature pull.  The
# CBM/VBM-wrt-vacuum of each constituent layer are what let type-II (staggered)
# alignment -> spatially-indirect exciton -> permanent dipole be established.
HETDB_FEATURE_TOGGLE_KEYS = (
    "interlayer_distance",
    "pbe_cbm_soc",
    "pbe_vbm_soc",
    "scs_cbm_soc",
    "scs_vbm_soc",
    "uid_a",
    "uid_b",
    "scs_gap_soc",
    "area",
)


@dataclass
class StackDBClient(C2DBClient):
    """ASE-db client for a vdW-stacking database (BiDB or HetDB).

    Inherits every network method from `C2DBClient` unchanged — only the base
    URL, the header->key map, the parse anchor, and the default toggle set are
    database-specific.  `count()`, `total()`, `coverage()` are already fully
    DB-agnostic (they read the "rows out of" header, which every ASE-db server
    emits identically), so no coverage code is duplicated.
    """

    base: str = BIDB_BASE
    header_map: dict[str, str] = field(default_factory=lambda: dict(BIDB_HEADER_TO_KEY))
    feature_toggle_keys: tuple[str, ...] = BIDB_FEATURE_TOGGLE_KEYS
    # First (leftmost, always-present) column header — the parse anchor.
    anchor: str = "Formula"
    timeout: float = 60.0
    pause: float = 0.15
    session: requests.Session = field(default_factory=_make_session)
    _sid: str | None = None

    # -- row parsing (DB-specific header map) -----------------------------
    def _parse_table(self, text: str) -> tuple[list[str], list[list[str]]]:  # type: ignore[override]
        """Return (column_keys, rows) from a table fragment.

        Anchored on this DB's leftmost column; the header is the longest run of
        cells (from the anchor) that map to known schema keys; the remainder is
        chunked into rows of len(header) values.  Empty cells come back as "".
        """
        cells = [_clean(c) for c in _CELL.findall(text)]
        if self.anchor not in cells:
            return [], []
        cells = cells[cells.index(self.anchor):]
        header: list[str] = []
        for c in cells:
            if c in self.header_map:
                header.append(self.header_map[c])
            else:
                break
        ncol = len(header)
        data = cells[ncol:]
        nrows = len(data) // ncol if ncol else 0
        rows = [data[i * ncol:(i + 1) * ncol] for i in range(nrows)]
        return header, rows

    def fetch_rows(  # type: ignore[override]
        self,
        filter_expr: str,
        toggle_keys: Iterable[str] | None = None,
        max_pages: int = 200,
    ) -> list[dict[str, str]]:
        """Pull every row matching `filter_expr`, as {key: raw_string_value}.

        Same pagination contract as C2DBClient.fetch_rows: page 0 sets the
        (session-stored) filter, later pages send `page` alone.  Values are
        left as strings; numeric parsing is the caller's job.
        """
        keys = tuple(self.feature_toggle_keys if toggle_keys is None else toggle_keys)
        self.sid(refresh=True)
        for k in keys:
            self._toggle(k)

        out: list[dict[str, str]] = []
        seen_prev: str | None = None
        total: int | None = None
        for page in range(max_pages):
            text = self._table(filter_expr) if page == 0 else self._table(page=page)
            header, rows = self._parse_table(text)
            if not rows:
                break
            if page == 0:
                want = [k for k in keys if k not in header]
                if want:
                    raise RuntimeError(
                        f"toggled columns missing from parsed header "
                        f"(unmapped header upstream?): {want}. Header={header}"
                    )
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


def bidb_client(**kw) -> StackDBClient:
    """A client bound to the live BiDB (homobilayers)."""
    return StackDBClient(
        base=BIDB_BASE,
        header_map=dict(BIDB_HEADER_TO_KEY),
        feature_toggle_keys=BIDB_FEATURE_TOGGLE_KEYS,
        anchor="Formula",
        **kw,
    )


def hetdb_client(**kw) -> StackDBClient:
    """A client bound to the live HetDB (heterobilayers)."""
    return StackDBClient(
        base=HETDB_BASE,
        header_map=dict(HETDB_HEADER_TO_KEY),
        feature_toggle_keys=HETDB_FEATURE_TOGGLE_KEYS,
        anchor="Formula 1",
        **kw,
    )
