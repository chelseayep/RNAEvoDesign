"""Learned tertiary potential: one leaf replacing hbond + sa + torsion.

The three binned leaves each look up ``log P(nuc | discretised context)``.  This
term looks up exactly the same *kind* of quantity, but the table is produced
per-target by a small network over continuous geometry rather than by counting
into 2-6 bins (see :mod:`evo_rna.learned`).  The network runs once, at
initialisation; what reaches the MC loop is two arrays:

    residue_logodds  (L, 4)     log P(nuc | geometry_i) - log P(nuc)
    contact_logodds  (C, 4, 4)  log P(nuc_i, nuc_j | geometry_c) - log P(nuc_i, nuc_j)
    contacts         (C, 2)     the (i, j) each contact row refers to

So the hot loop is unchanged in cost: ``score`` is O(L + C) like the terms it
replaces, and ``delta`` is O(1) for the residue part plus O(degree) over the
mutated position's contacts -- the same adjacency trick ``DSSRHbondTerm`` uses.
No torch is imported here, and none is needed at design time.

Why one leaf instead of three
-----------------------------
The objective diagnostic found the tertiary block wrong-signed against refold TM
while its sub-terms disagreed: ``sa`` right-signed, ``hbond`` and ``torsion``
wrong-signed, and the simplex sum landing wrong-signed overall.  A single head
learns the combination internally, so the cancellation cannot happen by
construction.  It also means the block's within-block mix has one element, and
``w_tertiary`` alone carries its magnitude.

The old leaves stay reachable behind the flag that selects this one, for the
whole project: they are the control arm, and deleting them forfeits the answer to
"did the learned potential help, or would reweighting alone have done it?"
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from evo_rna.core.structure import NUC_ALPHABET, NUC_TO_INDEX, Structure

from .evo import EnergyTerm


class LearnedTertiaryTerm(EnergyTerm):
    """Score = -(1/L) [ Σ_i residue[i, nuc_i] + Σ_c contact[c, nuc_i, nuc_j] ].

    Negated because the arrays are log-odds (higher = more native-like) while the
    MC minimises energy, and divided by L for the same reason every other term is:
    the block must be intensive so a 300-nt target is not automatically worse than
    a 30-nt one, and so the weights mean the same thing across the target set.
    """

    def __init__(
        self,
        residue_logodds: np.ndarray,
        contacts: np.ndarray,
        contact_logodds: np.ndarray,
        *,
        source: str = "<memory>",
    ) -> None:
        self._res = np.asarray(residue_logodds, dtype=np.float64)
        self._con = np.asarray(contact_logodds, dtype=np.float64)
        self._contacts = np.asarray(contacts, dtype=np.int64).reshape(-1, 2)
        self._source = source

        n_nuc = len(NUC_ALPHABET)
        if self._res.ndim != 2 or self._res.shape[1] != n_nuc:
            raise ValueError(
                f"{source}: residue_logodds has shape {self._res.shape}, "
                f"expected (L, {n_nuc}) in NUC_ALPHABET order {NUC_ALPHABET}")
        if len(self._contacts) != len(self._con):
            raise ValueError(
                f"{source}: {len(self._contacts)} contacts but "
                f"{len(self._con)} log-odds rows -- the (i, j) indices and the "
                f"table rows must be produced by the same pass over the contacts")
        if len(self._con) and self._con.shape[1:] != (n_nuc, n_nuc):
            raise ValueError(
                f"{source}: contact_logodds has shape {self._con.shape}, "
                f"expected (C, {n_nuc}, {n_nuc})")

        # Per position: (row in the contact table, partner, is this position i?).
        # Mirrors DSSRHbondTerm._adj so a mutation touches only its own contacts.
        self._adj: dict[int, list[tuple[int, int, bool]]] = {}
        for c, (i, j) in enumerate(self._contacts):
            self._adj.setdefault(int(i), []).append((c, int(j), True))
            self._adj.setdefault(int(j), []).append((c, int(i), False))

    # -- construction ------------------------------------------------------

    @classmethod
    def from_npz(cls, path: str | Path) -> "LearnedTertiaryTerm":
        """Load one target's arrays as written by ``export_tables.py``."""
        path = Path(path)
        if not path.exists():
            # Most likely cause: the design job was launched on a whole-entry
            # file (2OIU.cif -> stem "2OIU") while the tables are per design unit
            # ("2OIU_1_P.npz").  Say so rather than surfacing a bare OSError.
            raise FileNotFoundError(
                f"no exported table at {path}. export_tables.py names files by "
                f"design unit (<PDB>_<model>_<chains>.npz); --learned-tertiary "
                f"looks one up by the input file's stem, so run the design on "
                f"the per-unit structure or re-export under this name.")
        with np.load(path, allow_pickle=True) as d:
            alphabet = [str(x) for x in d["nuc_alphabet"]]
            if alphabet != list(NUC_ALPHABET):
                raise ValueError(
                    f"{path}: exported with alphabet {alphabet}, runtime uses "
                    f"{list(NUC_ALPHABET)} -- the columns would be permuted and "
                    f"every lookup silently wrong")
            return cls(d["residue_logodds"], d["contacts"], d["contact_logodds"],
                       source=str(path))

    @classmethod
    def for_structure(cls, struct: Structure, tables_dir: str | Path,
                      target: str) -> "LearnedTertiaryTerm":
        """Load ``<tables_dir>/<target>.npz`` and check it fits *struct*.

        The length check is the important part: a stale table from an earlier
        extraction would index cleanly for a while and then read the wrong
        residue, which is far worse than failing here.
        """
        term = cls.from_npz(Path(tables_dir) / f"{target}.npz")
        L = len(struct.sequence)
        if term._res.shape[0] != L:
            raise ValueError(
                f"{target}: table has {term._res.shape[0]} positions but the "
                f"structure has {L} residues -- re-export this target")
        if len(term._contacts) and int(term._contacts.max()) >= L:
            raise ValueError(
                f"{target}: contact index {int(term._contacts.max())} is outside "
                f"the {L}-residue structure -- table and structure disagree")
        return term

    # -- EnergyTerm --------------------------------------------------------

    @property
    def name(self) -> str:
        return "learned_tertiary"

    def score(self, struct: Structure) -> float:
        seq = struct.sequence
        L = len(seq)
        if L == 0:
            return 0.0
        idx = np.fromiter((NUC_TO_INDEX.get(c, -1) for c in seq), dtype=np.int64,
                          count=L)
        n = min(L, self._res.shape[0])
        ok = idx[:n] >= 0
        total = float(self._res[np.arange(n)[ok], idx[:n][ok]].sum())

        if len(self._con):
            i, j = self._contacts[:, 0], self._contacts[:, 1]
            valid = (i < L) & (j < L) & (idx[i] >= 0) & (idx[j] >= 0)
            if valid.any():
                total += float(
                    self._con[np.nonzero(valid)[0], idx[i][valid], idx[j][valid]].sum())
        return -total / L

    def delta(self, struct: Structure, pos: int, new_nuc: str) -> float:
        seq = struct.sequence
        L = len(seq)
        if L == 0:
            return 0.0
        new = NUC_TO_INDEX.get(new_nuc, -1)
        old = NUC_TO_INDEX.get(seq[pos], -1)
        if new < 0 or old < 0 or new == old:
            return 0.0

        d = 0.0
        if pos < self._res.shape[0]:
            d += float(self._res[pos, new] - self._res[pos, old])

        for c, partner, pos_is_i in self._adj.get(pos, ()):
            if partner >= L:
                continue
            p = NUC_TO_INDEX.get(seq[partner], -1)
            if p < 0:
                continue
            if pos_is_i:
                d += float(self._con[c, new, p] - self._con[c, old, p])
            else:
                d += float(self._con[c, p, new] - self._con[c, p, old])
        return -d / L
