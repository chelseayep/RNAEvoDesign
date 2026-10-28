"""DSSR hydrogen-bond satisfaction energy term.

Rewards sequences whose bases can chemically form the hydrogen bonds that the
native 3-D geometry requires at each DSSR-annotated base-pair contact —
**including the non-canonical (Leontis-Westhof) pairs and base multiplets** that
the secondary-structure terms (``ss_rnafold`` / ``ss`` / ``stacking_context``)
do not see.  Those terms only reason about canonical Watson-Crick/wobble pairing;
this term scores the tertiary atomic H-bond layer.

How it works
------------
The target geometry is fixed during MC, so every contact carries a fixed pair of
interacting **edges** — Watson-Crick (W), Hoogsteen (H) or Sugar (S) — read from
the DSSR LW code (e.g. ``cWW``, ``tHS``, ``cWS``).  Each base presents a known
number of H-bond *donor* and *acceptor* groups on each of its three edges, so the
H-bond *capacity* of a contact is

    cap = min(donors_i^edge, acceptors_j^edge) + min(acceptors_i^edge, donors_j^edge)

i.e. how many donor↔acceptor pairings the two presented edges can satisfy.  A
mutation only changes which atoms a base presents, never the geometry, so ``cap``
is an O(1) table lookup and the term supports fast single-site delta evaluation.

The donor/acceptor counts below are from standard nucleobase chemistry and are a
deliberately simple, training-free default that captures the dominant H-bonding;
they reproduce the 3 H-bonds of a cWW G–C and the 2 of a cWW A–U.  A learned
``log P(nuc_i, nuc_j | LW family)`` table could replace ``hbond_capacity`` later
(cf. the propensity-table terms) without changing the term's wiring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from evo_rna.core.structure import NUC_ALPHABET, NUC_TO_INDEX, Structure
from evo_rna.monte_carlo.energy_terms.evo import EnergyTerm

N_NUC = len(NUC_ALPHABET)
_EDGE_ORDER = {"W": 0, "H": 1, "S": 2}

# (donors, acceptors) presented by each (nucleotide, edge).
# Edges: W = Watson-Crick, H = Hoogsteen / CH, S = Sugar (incl. ribose O2').
_EDGE_HB: dict[tuple[str, str], tuple[int, int]] = {
    # Watson-Crick edge
    ("A", "W"): (1, 1),  # N6 donor; N1 acceptor
    ("U", "W"): (1, 2),  # N3 donor; O4, O2 acceptors
    ("G", "W"): (2, 1),  # N1, N2 donors; O6 acceptor
    ("C", "W"): (1, 2),  # N4 donor; N3, O2 acceptors
    # Hoogsteen (purine) / CH (pyrimidine) edge
    ("A", "H"): (1, 1),  # N6 donor; N7 acceptor
    ("G", "H"): (0, 2),  # O6, N7 acceptors
    ("U", "H"): (0, 1),  # O4 acceptor
    ("C", "H"): (1, 0),  # N4 donor
    # Sugar edge (base sugar-face atoms + ribose O2' donor)
    ("A", "S"): (1, 1),  # O2' donor; N3 acceptor
    ("G", "S"): (2, 1),  # N2, O2' donors; N3 acceptor
    ("U", "S"): (1, 1),  # O2' donor; O2 acceptor
    ("C", "S"): (1, 1),  # O2' donor; O2 acceptor
}


def _norm_edge(ch: str) -> str:
    """Map a single LW edge character to W/H/S (unknown → W)."""
    c = ch.upper()
    return c if c in ("W", "H", "S") else "W"


def edges_from_lw(lw: str) -> tuple[str, str]:
    """Parse a DSSR LW code into (edge_of_nt1, edge_of_nt2).

    LW is ``<orientation><edge1><edge2>`` (e.g. ``cWW``, ``tHS``).  Orientation
    (cis/trans) does not change which atoms an edge presents, so it is ignored
    here.  Malformed/empty codes fall back to the Watson-Crick edge.
    """
    if not lw or len(lw) < 3:
        return ("W", "W")
    return (_norm_edge(lw[1]), _norm_edge(lw[2]))


def hbond_capacity(nuc_i: str, edge_i: str, nuc_j: str, edge_j: str) -> float:
    """Number of donor↔acceptor H-bonds the two presented edges can satisfy.

    Symmetric in the two (nuc, edge) arguments.
    """
    di, ai = _EDGE_HB.get((nuc_i, edge_i), (0, 0))
    dj, aj = _EDGE_HB.get((nuc_j, edge_j), (0, 0))
    return float(min(di, aj) + min(ai, dj))


# ---------------------------------------------------------------------------
# Leontis-Westhof family canonicalisation
# ---------------------------------------------------------------------------
# The 12 geometric families: orientation {c,t} × edge-pair {WW,WH,WS,HH,HS,SS}.
# Edges are ordered W<H<S; for symmetric-edge families (WW/HH/SS) the two bases
# are ordered too, so the same physical contact maps to one canonical cell
# regardless of which partner DSSR listed first.  Used by both the derivation
# script (counting) and the runtime lookup so train/inference agree.

FAMILIES: tuple[str, ...] = tuple(
    f"{o}{a}{b}"
    for o in ("c", "t")
    for ia, a in enumerate("WHS")
    for ib, b in enumerate("WHS")
    if ia <= ib
)


def canonical_pair_key(lw: str, nuc_i: str, nuc_j: str) -> tuple[str, int, int]:
    """Map (LW code, nt1 base, nt2 base) → (family, row, col).

    ``family`` is one of :data:`FAMILIES`; ``row``/``col`` index the canonical
    4×4 base-combo table (row = base on the lower-priority edge).  Returns
    ``row``/``col`` = -1 for non-ACGU bases.
    """
    o = lw[0].lower() if lw and lw[0].lower() in ("c", "t") else "c"
    e1, e2 = edges_from_lw(lw)
    a_idx = NUC_TO_INDEX.get(nuc_i, -1)
    b_idx = NUC_TO_INDEX.get(nuc_j, -1)
    # Order by (edge, base) so symmetric families collapse i<->j duplicates.
    if (_EDGE_ORDER[e1], a_idx) > (_EDGE_ORDER[e2], b_idx):
        e1, e2 = e2, e1
        a_idx, b_idx = b_idx, a_idx
    return f"{o}{e1}{e2}", a_idx, b_idx


@dataclass
class HbondPropensity:
    """``log P(base_i, base_j | LW family)`` tables, one 4×4 per family.

    Replaces the hand-built :func:`hbond_capacity` with data-driven, geometry-
    conditioned base-combo log-odds (derived from the TRAIN split; see
    ``scripts/propensity/derive_hbond_propensity.py``).  Families absent from
    ``tables`` — and non-ACGU bases — fall back to the chemistry capacity, so an
    empty instance reproduces the chemistry term exactly.
    """

    tables: dict[str, np.ndarray] = field(default_factory=dict)

    @classmethod
    def chemistry(cls) -> "HbondPropensity":
        """Empty table → always falls back to :func:`hbond_capacity`."""
        return cls(tables={})

    def score(self, lw: str, nuc_i: str, nuc_j: str) -> float:
        """Favourability of base pair (nt1, nt2) under this contact (higher = better)."""
        family, a, b = canonical_pair_key(lw, nuc_i, nuc_j)
        t = self.tables.get(family)
        if t is None or a < 0 or b < 0:
            ei, ej = edges_from_lw(lw)
            return hbond_capacity(nuc_i, ei, nuc_j, ej)
        return float(t[a, b])

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **{f"fam_{k}": v for k, v in self.tables.items()})

    @classmethod
    def load(cls, path: str | Path) -> "HbondPropensity":
        d = np.load(path)
        return cls(tables={k[len("fam_"):]: d[k] for k in d.files if k.startswith("fam_")})


class DSSRHbondTerm(EnergyTerm):
    """Score = -(1/L) Σ_contacts propensity.score(lw, base_i, base_j).

    ``propensity`` favourability is higher for base combos that satisfy a
    contact's H-bonding; negated because the MC minimises energy.  Defaults to
    :meth:`HbondPropensity.chemistry` (the hand-built donor/acceptor capacity);
    pass a derived :class:`HbondPropensity` to use TRAIN-split log-odds instead.
    With no DSSR contacts the term is a well-defined no-op (0).

    Normalised by sequence length **L**, not by the contact count, so the term
    is *per nucleotide* like its block-mates ``sa`` and ``torsion``.  Dividing by
    the contact count would also be intensive, but it would make the term a mean
    *per contact*, so the fixed simplex weight inside ``tertiary`` would buy a
    different effective magnitude on every target — scaled by that target's
    contact density (contacts/nt), which varies widely between a densely
    multiplet-laced tRNA and a simple hairpin.  With /L the block mix is
    target-independent and H-bond-rich backbones legitimately score stronger.
    """

    def __init__(
        self,
        contacts: list[tuple[int, int, str]],
        propensity: "HbondPropensity | None" = None,
    ) -> None:
        self._prop = propensity if propensity is not None else HbondPropensity.chemistry()
        # ``contacts`` are (i, j, lw) in local (0..L-1) coords with i = nt1
        # (LW edge1), j = nt2.  Adjacency records, per position, the partner, the
        # LW code, and whether this position is nt1 — so the canonical lookup
        # gets its (nt1, nt2) bases in the right order during O(degree) delta.
        self._contacts: list[tuple[int, int, str]] = []
        self._adj: dict[int, list[tuple[int, str, bool]]] = {}
        for i, j, lw in contacts:
            if i == j:
                continue
            self._contacts.append((i, j, lw))
            self._adj.setdefault(i, []).append((j, lw, True))   # i is nt1
            self._adj.setdefault(j, []).append((i, lw, False))  # j is nt2
        self._n = len(self._contacts)

    @classmethod
    def from_structure(
        cls, struct: Structure, propensity: "HbondPropensity | None" = None
    ) -> "DSSRHbondTerm":
        """Build from the DSSR contacts baked onto a :class:`Structure`."""
        return cls(list(struct.contacts), propensity=propensity)

    @property
    def name(self) -> str:
        return "hbond"

    def score(self, struct: Structure) -> float:
        seq = struct.sequence
        L = len(seq)
        if self._n == 0 or L == 0:
            return 0.0
        total = 0.0
        for i, j, lw in self._contacts:
            if i < L and j < L:
                total += self._prop.score(lw, seq[i], seq[j])
        return -total / L

    def delta(self, struct: Structure, pos: int, new_nuc: str) -> float:
        if self._n == 0:
            return 0.0
        nbrs = self._adj.get(pos)
        if not nbrs:
            return 0.0
        seq = struct.sequence
        L = len(seq)
        if L == 0:
            return 0.0
        old_nuc = seq[pos]
        d = 0.0
        for partner, lw, pos_is_nt1 in nbrs:
            if partner >= L:
                continue
            partner_nuc = seq[partner]
            if pos_is_nt1:
                old = self._prop.score(lw, old_nuc, partner_nuc)
                new = self._prop.score(lw, new_nuc, partner_nuc)
            else:
                old = self._prop.score(lw, partner_nuc, old_nuc)
                new = self._prop.score(lw, partner_nuc, new_nuc)
            d += new - old
        return -d / L
