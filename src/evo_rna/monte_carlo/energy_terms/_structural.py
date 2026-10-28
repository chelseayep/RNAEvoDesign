"""Structural energy terms: base-pair validity, SA / torsion / junction propensity.

The propensity terms (SA, torsion, junction/loop-context) consult
``log P(nuc | structural-context class)`` tables; context discretizations and
the table container live in :mod:`evo_rna.monte_carlo.energy_terms.propensities`.
With uniform (no-op) tables these terms contribute 0 until a real propensity
table is derived (see the propensity-table derivation project).

Used via :meth:`EnergyFunction.structural`.
"""

from __future__ import annotations

import numpy as np

from evo_rna.core.structure import NUC_TO_INDEX, Structure
from evo_rna.monte_carlo.energy_terms.evo import EnergyTerm
from evo_rna.monte_carlo.energy_terms.propensities import (
    CHI_INDEX,
    N_ELEM,
    N_NUC,
    N_SA,
    N_TOR,
    chi_to_bin,
    sasa_to_class,
)

# ---------------------------------------------------------------------------
# Secondary-structure (base-pair compatibility) term
# ---------------------------------------------------------------------------

_VALID_PAIRS = {
    frozenset(("A", "U")),
    frozenset(("G", "C")),
    frozenset(("G", "U")),  # wobble
}


def _pair_ok(a: str, b: str) -> bool:
    return frozenset((a, b)) in _VALID_PAIRS


class SSTerm(EnergyTerm):
    """Penalises invalid base pairs at positions the target says should be paired.

    Score = (fraction of target pairs that are not valid WC/wobble) ∈ [0, 1]:
    +1 per invalid pair, divided by the number of pairs.  Normalising by
    n_pairs makes the term intensive (size-independent) so weights transfer
    across structures.  No pairs → 0.
    """

    @property
    def name(self) -> str:
        return "ss"

    def score(self, struct: Structure) -> float:
        n_pairs = len(struct.pair_map) // 2
        if n_pairs == 0:
            return 0.0
        penalty = 0.0
        seen: set = set()
        for i, j in struct.pair_map.items():
            if i < j and (i, j) not in seen:
                seen.add((i, j))
                if not _pair_ok(struct.sequence[i], struct.sequence[j]):
                    penalty += 1.0
        return penalty / n_pairs

    def delta(self, struct: Structure, pos: int, new_nuc: str) -> float:
        partner = struct.paired_with(pos)
        if partner is None:
            return 0.0
        n_pairs = len(struct.pair_map) // 2
        if n_pairs == 0:
            return 0.0
        old_nuc = struct.sequence[pos]
        partner_nuc = struct.sequence[partner]
        old_ok = _pair_ok(old_nuc, partner_nuc)
        new_ok = _pair_ok(new_nuc, partner_nuc)
        if old_ok == new_ok:
            return 0.0
        return (-1.0 if new_ok else 1.0) / n_pairs


# ---------------------------------------------------------------------------
# Solvent-accessibility propensity term
# ---------------------------------------------------------------------------

class SATerm(EnergyTerm):
    """Score = -Σ sa_propensity[sasa_class(sa[i]), nuc_idx(seq[i])].

    ``struct.sa`` holds raw SASA in Å²; it is discretized via
    :func:`sasa_to_class` (the same mapping the derivation script uses), so the
    propensity rows are (buried, intermediate, exposed).  Uniform table → 0.
    """

    def __init__(self, propensity: np.ndarray | None = None) -> None:
        self._prop = propensity if propensity is not None else np.zeros((N_SA, N_NUC))

    @property
    def name(self) -> str:
        return "sa"

    def _class_at(self, struct: Structure, pos: int) -> int:
        if pos >= struct.sa.shape[0]:
            return 1
        return sasa_to_class(float(struct.sa[pos]))

    def _lookup(self, sa_class: int, nuc: str) -> float:
        j = NUC_TO_INDEX.get(nuc, -1)
        if j < 0 or sa_class < 0 or sa_class >= self._prop.shape[0]:
            return 0.0
        return float(self._prop[sa_class, j])

    def score(self, struct: Structure) -> float:
        total = 0.0
        for i, nuc in enumerate(struct.sequence):
            total -= self._lookup(self._class_at(struct, i), nuc)
        L = len(struct.sequence)
        return total / L if L else 0.0

    def delta(self, struct: Structure, pos: int, new_nuc: str) -> float:
        sa_cls = self._class_at(struct, pos)
        old_nuc = struct.sequence[pos]
        L = len(struct.sequence)
        return -(self._lookup(sa_cls, new_nuc) - self._lookup(sa_cls, old_nuc)) / L if L else 0.0


# ---------------------------------------------------------------------------
# Torsion-angle compatibility term
# ---------------------------------------------------------------------------

class TorsionTerm(EnergyTerm):
    """Score = -Σ torsion_propensity[chi_bin(χ_i), nuc_idx(seq[i])].

    The glycosidic χ angle (in DEGREES; ``struct.torsions`` column ``CHI_INDEX``)
    is the most nucleotide-discriminative backbone torsion.  Binned via
    :func:`chi_to_bin`.  Uniform table → 0.
    """

    def __init__(self, propensity: np.ndarray | None = None) -> None:
        self._prop = propensity if propensity is not None else np.zeros((N_TOR, N_NUC))

    @property
    def name(self) -> str:
        return "torsion"

    def _lookup(self, tbin: int, nuc: str) -> float:
        j = NUC_TO_INDEX.get(nuc, -1)
        if j < 0 or tbin < 0 or tbin >= self._prop.shape[0]:
            return 0.0
        return float(self._prop[tbin, j])

    def _bin_at(self, struct: Structure, pos: int) -> int:
        if pos >= struct.torsions.shape[0]:
            return 0
        return chi_to_bin(float(struct.torsions[pos, CHI_INDEX]))

    def score(self, struct: Structure) -> float:
        total = 0.0
        for i, nuc in enumerate(struct.sequence):
            total -= self._lookup(self._bin_at(struct, i), nuc)
        L = len(struct.sequence)
        return total / L if L else 0.0

    def delta(self, struct: Structure, pos: int, new_nuc: str) -> float:
        tbin = self._bin_at(struct, pos)
        old_nuc = struct.sequence[pos]
        L = len(struct.sequence)
        return -(self._lookup(tbin, new_nuc) - self._lookup(tbin, old_nuc)) / L if L else 0.0


# ---------------------------------------------------------------------------
# Junction / loop-context propensity term
# ---------------------------------------------------------------------------

class JunctionTerm(EnergyTerm):
    """Score = -Σ junction_propensity[ss_element[i], nuc_idx(seq[i])].

    Rewards nucleotides consistent with the DSSR secondary-structure element at
    each position (stem / hairpin / internal / bulge / junction), capturing the
    conserved nucleotide preferences of loops and multi-way junctions.  Rows
    follow ``profile.dssr.SS_ELEMENTS`` (the ``struct.ss_element`` codes).
    Requires DSSR-derived ``ss_element``; uniform table → 0.
    """

    def __init__(self, propensity: np.ndarray | None = None) -> None:
        self._prop = propensity if propensity is not None else np.zeros((N_ELEM, N_NUC))

    @property
    def name(self) -> str:
        return "junction"

    def _class_at(self, struct: Structure, pos: int) -> int:
        if pos >= struct.ss_element.shape[0]:
            return 0
        return int(struct.ss_element[pos])

    def _lookup(self, element: int, nuc: str) -> float:
        j = NUC_TO_INDEX.get(nuc, -1)
        if j < 0 or element < 0 or element >= self._prop.shape[0]:
            return 0.0
        return float(self._prop[element, j])

    def score(self, struct: Structure) -> float:
        total = 0.0
        for i, nuc in enumerate(struct.sequence):
            total -= self._lookup(self._class_at(struct, i), nuc)
        L = len(struct.sequence)
        return total / L if L else 0.0

    def delta(self, struct: Structure, pos: int, new_nuc: str) -> float:
        element = self._class_at(struct, pos)
        old_nuc = struct.sequence[pos]
        L = len(struct.sequence)
        return -(self._lookup(element, new_nuc) - self._lookup(element, old_nuc)) / L if L else 0.0
