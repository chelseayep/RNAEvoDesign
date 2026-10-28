"""Ensemble-defect energy term (ViennaRNA partition function).

Thermodynamic-stability term for Monte Carlo RNA sequence design.  Scores the
*normalized ensemble defect* (Zadeh et al., J. Comput. Chem. 2011) of the
candidate sequence with respect to the target secondary structure: the expected
fraction of nucleotides paired *differently* from the target across the
Boltzmann ensemble.

    defect(seq) = (1/L) * Σ_i [ 1 - P(i in its target pairing state | seq) ]

where the target state of position *i* is "paired with *j*" when ``(i, j)`` is in
the target ``pair_map`` and "unpaired" otherwise, and the probabilities come
from the ViennaRNA partition-function base-pair probability matrix.  Lower is
better — 0 means the target structure dominates the entire Boltzmann ensemble,
i.e. the sequence folds to the target *stably and uniquely*.

Unlike the propensity terms (which are O(1) per move) this folds the whole
sequence (partition function, ~O(L^3)), so ``delta`` re-folds the mutant
sequence with a small cache rather than performing an O(1) update`.
It is therefore the most expensive structural term and is opt-in via a non-zero
weight (see :meth:`EnergyFunction.structural`).
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

from evo_rna.core.structure import Structure
from evo_rna.monte_carlo.energy_terms.evo import EnergyTerm

log = logging.getLogger(__name__)

try:  # ViennaRNA python bindings (`pip install ViennaRNA`); ships its own algos.
    import RNA  # type: ignore
except ImportError:  # pragma: no cover - exercised only without the bindings
    RNA = None


def positional_ensemble_defect(
    sequence: str,
    pair_map: Dict[int, int],
    temperature: float = 37.0,
) -> list[float]:
    """Per-position ensemble defect of *sequence* vs. the target *pair_map*.

    ``eps[i] = 1 - P(i is in its target pairing state | sequence)``, where the
    target state of *i* is "paired with *j*" when ``(i, j)`` is in *pair_map*
    and "unpaired" otherwise.  Each entry is in ``[0, 1]``; 0 means the
    ensemble always puts *i* where the target wants it.

    This is the vector the normalized ensemble defect averages over
    (:func:`normalized_ensemble_defect` is ``sum(eps) / L``), and the quantity
    SAMFEO's position sampler is weighted by
    (:class:`~evo_rna.monte_carlo.samfeo_mutations.DefectWeightedPositions`) --
    it says *which* positions the ensemble is getting wrong, not just how many.

    *pair_map* is the target's symmetric ``{i: j}`` map (0-indexed, as on
    :class:`~evo_rna.core.structure.Structure`).
    """
    if RNA is None:
        raise ImportError(
            "positional_ensemble_defect requires the ViennaRNA python bindings "
            "(`import RNA`); install with `pip install ViennaRNA`."
        )
    L = len(sequence)
    if L == 0:
        return []

    md = RNA.md()
    md.temperature = temperature
    fc = RNA.fold_compound(sequence, md)
    fc.pf()                       # partition function (required before bpp)
    bpp = fc.bpp()                # 1-indexed upper triangle: bpp[i][j], i < j

    # Per-position probability of being paired with *anything*.
    p_paired = [0.0] * (L + 1)    # 1-indexed
    for i in range(1, L + 1):
        row = bpp[i]
        for j in range(i + 1, L + 1):
            p = row[j]
            if p:
                p_paired[i] += p
                p_paired[j] += p

    defect: list[float] = []
    for i in range(L):            # 0-indexed positions
        partner = pair_map.get(i)
        if partner is None:       # target says unpaired
            p_correct = 1.0 - p_paired[i + 1]
        else:                     # target says paired with `partner`
            a, b = (i, partner) if i < partner else (partner, i)
            p_correct = bpp[a + 1][b + 1]
        # Clamp: the bpp scan accumulates float error, and a defect outside
        # [0, 1] would give the position sampler a nonsense Boltzmann weight.
        defect.append(min(1.0, max(0.0, 1.0 - p_correct)))
    return defect


def normalized_ensemble_defect(
    sequence: str,
    pair_map: Dict[int, int],
    temperature: float = 37.0,
) -> float:
    """Normalized ensemble defect of *sequence* vs. the target *pair_map*.

    The mean of :func:`positional_ensemble_defect`.  Returns a value in
    ``[0, 1]``; 0 = the target is the entire ensemble.
    """
    defect = positional_ensemble_defect(sequence, pair_map, temperature)
    if not defect:
        return 0.0
    return sum(defect) / len(defect)


class EnsembleDefectTerm(EnergyTerm):
    """Normalized ensemble defect w.r.t. the target secondary structure.

    Score is in ``[0, 1]`` (lower = more thermodynamically committed to the
    target fold).  Requires the ViennaRNA python bindings; folds the full
    sequence so its ``delta`` re-folds the mutant (cached), unlike the O(1)
    propensity terms.
    """

    def __init__(
        self,
        pair_map: Optional[Dict[int, int]] = None,
        temperature: float = 37.0,
    ) -> None:
        self._pair_map = dict(pair_map) if pair_map is not None else None
        self._temperature = temperature
        # Small LRU-ish cache: sequence -> normalized ensemble defect.  Avoids
        # re-folding the same sequence when score() and delta() coincide.
        self._cache: dict[str, float] = {}

    @property
    def name(self) -> str:
        return "ensemble_defect"

    def score(self, struct: Structure) -> float:
        return self._evaluate(struct.sequence, struct)

    def delta(self, struct: Structure, pos: int, new_nuc: str) -> float:
        seq_list = list(struct.sequence)
        seq_list[pos] = new_nuc
        mutant = "".join(seq_list)
        return self._evaluate(mutant, struct) - self._evaluate(struct.sequence, struct)

    # ------------------------------------------------------------------
    def _evaluate(self, sequence: str, struct: Structure) -> float:
        if sequence not in self._cache:
            if len(self._cache) >= 4:
                del self._cache[next(iter(self._cache))]
            pair_map = self._pair_map if self._pair_map is not None else struct.pair_map
            self._cache[sequence] = normalized_ensemble_defect(
                sequence, pair_map, self._temperature
            )
        return self._cache[sequence]
