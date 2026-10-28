"""Abstract base for energy terms and the EvoTerm (PSSM log-odds)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from evo_rna.core.structure import NUC_TO_INDEX, Structure


class EnergyTerm(ABC):
    """Interface shared by all energy terms."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def score(self, struct: Structure) -> float:
        """Full-sequence energy (lower = better)."""
        ...

    @abstractmethod
    def delta(self, struct: Structure, pos: int, new_nuc: str) -> float:
        """Energy change if struct.sequence[pos] were mutated to *new_nuc*.
        Must NOT mutate the structure.
        """
        ...


class EvoTerm(EnergyTerm):
    """Score = -(1/L) Σ pssm[i, nuc_idx(seq[i])]  — mean per-position log-odds.

    Negative because pssm values are log-odds (higher = more conserved), but we
    minimise energy.  Normalised by length L so the term is intensive (size-
    independent) and comparable across structures; balance vs. other terms is
    handled purely by the energy-function weights.
    """

    @property
    def name(self) -> str:
        return "evo"

    def score(self, struct: Structure) -> float:
        total = 0.0
        for i, nuc in enumerate(struct.sequence):
            j = NUC_TO_INDEX.get(nuc, -1)
            if j >= 0 and i < struct.pssm.shape[0]:
                total -= struct.pssm[i, j]
        L = len(struct.sequence)
        return total / L if L else 0.0

    def delta(self, struct: Structure, pos: int, new_nuc: str) -> float:
        old_j = NUC_TO_INDEX.get(struct.sequence[pos], -1)
        new_j = NUC_TO_INDEX.get(new_nuc, -1)
        if old_j < 0 or new_j < 0 or pos >= struct.pssm.shape[0]:
            return 0.0
        L = len(struct.sequence)
        return -(struct.pssm[pos, new_j] - struct.pssm[pos, old_j]) / L if L else 0.0
