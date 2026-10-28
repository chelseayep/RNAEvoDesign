"""Composite energy term: a gauge-fixed weighted sum of sub-terms.

A ``CompositeTerm`` groups related sub-terms (e.g. all secondary-structure
terms) and behaves as a single :class:`EnergyTerm` to the parent
``EnergyFunction``.  Its internal weights are **simplex-normalised** (they sum
to 1), so the composite carries *no magnitude degree of freedom of its own* —
its overall scale is set entirely by the weight it is given in the parent.

This makes the two-level parameterisation **identifiable**: the sub-weights
control only the *mix* inside the block, the parent weight controls the block's
*magnitude*.  Without the simplex gauge, ``w_block · a_j`` would be a pure
reparameterisation and the top-level weight would be redundant with the
sub-weights.
"""

from __future__ import annotations

from typing import Dict, Sequence

from evo_rna.core.structure import Structure
from evo_rna.monte_carlo.energy_terms.evo import EnergyTerm


def _normalise(weights: Sequence[float], name: str) -> list[float]:
    raw = [float(w) for w in weights]
    s = sum(raw)
    if s <= 0:
        raise ValueError(f"CompositeTerm {name!r}: weights must sum to > 0")
    return [w / s for w in raw]


class CompositeTerm(EnergyTerm):
    """A simplex-weighted sum of sub-terms that is itself an ``EnergyTerm``.

    ``score`` / ``delta`` are the normalised-weighted sums of the children's,
    so the composite inherits the *worst* delta complexity among its children:
    O(1) if every child is O(1), but O(L^3) if it contains
    :class:`EnsembleDefectTerm` or :class:`RNAfoldTerm`.  Keep expensive,
    non-O(1) terms in mind when grouping — the MC hot loop calls ``delta``.
    """

    def __init__(
        self,
        name: str,
        terms: Sequence[tuple[EnergyTerm, float]],
        *,
        normalize: bool = True,
    ) -> None:
        if not terms:
            raise ValueError(f"CompositeTerm {name!r} needs at least one sub-term")
        self._name = name
        self._terms = [t for t, _ in terms]
        raw = [float(w) for _, w in terms]
        self._weights = _normalise(raw, name) if normalize else raw

    @property
    def name(self) -> str:
        return self._name

    @property
    def weights(self) -> dict[str, float]:
        """Current (normalised) sub-weights, keyed by sub-term name."""
        return {t.name: w for t, w in zip(self._terms, self._weights)}

    def items(self) -> list[tuple[EnergyTerm, float]]:
        """Sub-terms paired with their current (normalised) weights.

        Used by the energy logger to flatten a composite into per-sub-term
        series so each leaf can be logged (and expensive leaves sampled)
        individually.
        """
        return list(zip(self._terms, self._weights))

    def set_weights(self, weights: Sequence[float], *, normalize: bool = True) -> None:
        """Replace the internal weights — used by the weight-tuning loop.

        Pass raw (e.g. ``softmax``-ed logits) values; with ``normalize=True``
        they are re-projected onto the simplex so the gauge is preserved.
        """
        if len(weights) != len(self._terms):
            raise ValueError(
                f"CompositeTerm {self._name!r}: expected {len(self._terms)} "
                f"weights, got {len(weights)}"
            )
        self._weights = _normalise(weights, self._name) if normalize else list(weights)

    def score(self, struct: Structure) -> float:
        return sum(w * t.score(struct) for t, w in zip(self._terms, self._weights))

    def delta(self, struct: Structure, pos: int, new_nuc: str) -> float:
        return sum(
            w * t.delta(struct, pos, new_nuc)
            for t, w in zip(self._terms, self._weights)
        )

    def breakdown(self, struct: Structure) -> Dict[str, float]:
        """Per-sub-term *normalised-weighted* contributions, for logging."""
        return {t.name: w * t.score(struct) for t, w in zip(self._terms, self._weights)}
