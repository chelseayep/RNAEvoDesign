"""Stacking-aware base-pair quality energy term.

Adapted from ``get_paring_feature`` in RNAref3D (bp/featurizer.py).
Works entirely from the primary sequence — no 3-D coordinates required.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np

from evo_rna.core.structure import Structure
from evo_rna.monte_carlo.energy_terms.evo import EnergyTerm


_DEFAULT_PAIR_SCORES: Dict[frozenset, float] = {
    frozenset(("G", "C")): 3.0,
    frozenset(("C", "G")): 3.0,
    frozenset(("A", "U")): 2.0,
    frozenset(("U", "A")): 2.0,
    frozenset(("G", "U")): 0.8,
    frozenset(("U", "G")): 0.8,
}


def _build_score_matrix(
    sequence: str,
    pair_scores: Dict[frozenset, float],
) -> np.ndarray:
    """Compute the stacking-aware pair quality matrix for *sequence*.

    Adapted from ``get_paring_feature`` in RNAref3D (bp/featurizer.py).

    Algorithm
    ---------
    1.  Build an (L, L) lower-triangular raw pair matrix ``P`` where
        ``P[i, j]`` is the WC/wobble score for ``(sequence[i], sequence[j])``
        with i > j.
    2.  Propagate along the anti-diagonal with Gaussian decay so that
        consecutive helical stacks amplify each other's score by
        ``exp(-k²)`` for displacement k = 1, 2, …
    3.  Symmetrise.
    """
    L = len(sequence)
    if L == 0:
        return np.zeros((0, 0), dtype=np.float64)

    # Step 1: raw pair scores (lower triangle only, i > j).
    P = np.zeros((L, L), dtype=np.float64)
    for i in range(L):
        for j in range(i):
            P[i, j] = pair_scores.get(frozenset((sequence[i], sequence[j])), 0.0)

    # Step 2: propagate along the anti-diagonal with Gaussian weights.
    pair_map = P.copy()

    # Direction A: shift rows down / cols left  → (i+k, j-k)
    shifted = P.copy()
    active = (P > 0).astype(np.float64)
    for k in range(1, L):
        shifted_new = np.zeros_like(shifted)
        shifted_new[1:, :L - 1] = shifted[:L - 1, 1:]
        shifted = shifted_new
        active_new = np.zeros_like(active)
        active_new[1:, :L - 1] = active[:L - 1, 1:]
        active = active_new
        active *= (shifted > 0)
        if active.sum() == 0:
            break
        pair_map += math.exp(-(k ** 2)) * active * shifted

    # Direction B: shift rows up / cols right  → (i-k, j+k)
    shifted = P.copy()
    active = (P > 0).astype(np.float64)
    for k in range(1, L):
        shifted_new = np.zeros_like(shifted)
        shifted_new[:L - 1, 1:] = shifted[1:, :L - 1]
        shifted = shifted_new
        active_new = np.zeros_like(active)
        active_new[:L - 1, 1:] = active[1:, :L - 1]
        active = active_new
        active *= (shifted > 0)
        if active.sum() == 0:
            break
        pair_map += math.exp(-(k ** 2)) * active * shifted

    # Step 3: symmetrise.
    return pair_map + pair_map.T


class StackingContextTerm(EnergyTerm):
    """Stacking-aware base-pair quality score at target-paired positions.

    Adapted from the ``get_paring_feature`` featurizer in RNAref3D
    (``bp/featurizer.py``).  Works entirely from the primary sequence —
    no 3-D coordinates required.

    Motivation
    ----------
    ``SSTerm`` applies a binary penalty for each target pair that lacks a
    valid WC/wobble combination, but treats all valid pairs equally.
    ``RNAfoldTerm`` captures global thermodynamics but costs an O(N³)
    subprocess call per MC step.  This term fills the gap: it rewards
    sequences where paired positions carry strong WC pairs (GC > AU > GU)
    **and** those pairs form contiguous helical stacks, using Gaussian-
    decayed stacking propagation (``exp(-k²)`` for run length k).

    Parameters
    ----------
    weight_gc, weight_au, weight_gu :
        Override the default pair-type weights (GC=3, AU=2, GU=0.8).
    """

    def __init__(
        self,
        weight_gc: float = 3.0,
        weight_au: float = 2.0,
        weight_gu: float = 0.8,
    ) -> None:
        self._pair_scores: Dict[frozenset, float] = {
            frozenset(("G", "C")): weight_gc,
            frozenset(("C", "G")): weight_gc,
            frozenset(("A", "U")): weight_au,
            frozenset(("U", "A")): weight_au,
            frozenset(("G", "U")): weight_gu,
            frozenset(("U", "G")): weight_gu,
        }
        self._cache: Optional[tuple[str, np.ndarray]] = None

    @property
    def name(self) -> str:
        return "stacking_context"

    def score(self, struct: Structure) -> float:
        n_pairs = len(struct.pair_map) // 2
        if n_pairs == 0:
            return 0.0
        S = self._get_matrix(struct.sequence)
        total = 0.0
        for i, j in struct.pair_map.items():
            if i < j:
                total -= S[i, j]
        return total / n_pairs

    def delta(self, struct: Structure, pos: int, new_nuc: str) -> float:
        n_pairs = len(struct.pair_map) // 2
        if n_pairs == 0:
            return 0.0
        S_old = self._get_matrix(struct.sequence)
        seq_list = list(struct.sequence)
        seq_list[pos] = new_nuc
        S_new = _build_score_matrix("".join(seq_list), self._pair_scores)
        # The stacking-context propagation means mutating one position changes S
        # for the pair at *pos* AND its neighbouring stacked pairs, so the exact
        # Δscore must sum the change over ALL target pairs, not just pos's pair.
        # The O(L^2) matrix rebuild already dominates; the pair loop is cheap.
        # Normalised by n_pairs → mean per-pair quality (intensive, size-free).
        d = 0.0
        for i, j in struct.pair_map.items():
            if i < j:
                d -= float(S_new[i, j] - S_old[i, j])
        return d / n_pairs

    def _get_matrix(self, sequence: str) -> np.ndarray:
        if self._cache is None or self._cache[0] != sequence:
            self._cache = (sequence, _build_score_matrix(sequence, self._pair_scores))
        return self._cache[1]
