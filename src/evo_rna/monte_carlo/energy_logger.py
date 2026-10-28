"""Lightweight per-term energy logger for MC simulated-annealing runs.

Captures weighted per-term energy at every accepted move.  Expensive terms
(RNAFMTerm) are sampled at a configurable checkpoint interval to bound the
overhead.  All data is serialised to JSON for offline plotting.

Usage
-----
    logger = EnergyLogger(struct, efn, rna_fm_interval=50)
    result = run_sa(struct, energy_fn=efn, config=cfg, callback=logger)

    logger.set_metadata(
        pdb_id      = Path(PDB_PATH).stem,
        best_sequence  = result.best_sequence,
        start_sequence = start_sequence,
        native_sequence = native_sequence,
        pair_map       = pair_map,
        accept_rate    = result.accept_rate,
    )
    logger.save("runs/<pdb>/energy_log.json")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional


# Terms whose .score() is expensive enough to warrant sparse checkpoints
_EXPENSIVE_TERMS = frozenset({"rna_fm", "ensemble_defect"})


class EnergyLogger:
    """Callback for :func:`~evo_rna.monte_carlo.mc.run_sa` that records
    per-term weighted energies at each accepted move.

    Parameters
    ----------
    struct :
        The mutable :class:`~evo_rna.core.structure.Structure` used in the MC
        run, captured by reference so the callback can call
        ``term.score(struct)`` after each accepted move.
    efn :
        The :class:`~evo_rna.monte_carlo.energy.EnergyFunction` whose terms
        will be logged.
    rna_fm_interval :
        Number of **accepted** moves between consecutive RNA-FM checkpoints.
        Ignored when ``"rna_fm"`` is not present in *efn.terms*.
    """

    def __init__(self, struct, efn, rna_fm_interval: int = 50) -> None:
        from evo_rna.monte_carlo.energy_terms.composite import CompositeTerm

        self.struct = struct
        self.efn = efn
        self.rna_fm_interval = rna_fm_interval
        self._accepted = 0

        # ── Flatten the energy function into leaf series ────────────────────
        # Composite blocks (evo/secondary/tertiary) are expanded so each
        # sub-term is logged under "<block>.<sub>" (matching EnergyFunction.score),
        # scaled by block_weight * normalised_sub_weight.  Leaves named in
        # _EXPENSIVE_TERMS (rna_fm, ensemble_defect — O(L) NN / O(L^3) fold) are
        # sampled sparsely; everything else is logged densely.
        #
        # Each leaf: name -> (EnergyTerm, scale).  ``scale`` reproduces exactly
        # that leaf's contribution to EnergyFunction.score (so the dense leaves
        # for a block sum to the block's value minus any expensive leaf).
        self._leaf: Dict[str, tuple] = {}
        for name, (term, weight) in efn.terms.items():
            if isinstance(term, CompositeTerm):
                for sub, sub_w in term.items():
                    self._leaf[f"{name}.{sub.name}"] = (sub, weight * sub_w)
            else:
                self._leaf[name] = (term, weight)

        # ── Dense log (one entry per accepted move) ─────────────────────────
        self._step: List[int] = []
        self._accepted_idx: List[int] = []
        # Running delta-tracked total from the MC loop (includes all terms)
        self._total: List[float] = []
        # Which proposal kind produced each accepted move, and how many
        # positions it rewrote.  Aligned index-for-index with _step / _total,
        # so an analysis can join move type to the energy it bought -- the
        # run-level *_by_kind dicts can only say how often a kind fired, never
        # when in the anneal or how much energy it moved.  Stays empty when
        # run_sa is driven with the older 3-argument callback protocol.
        self._kind: List[str] = []
        self._n_sites: List[int] = []

        self._terms: Dict[str, List[float]] = {
            name: []
            for name, (term, _scale) in self._leaf.items()
            if term.name not in _EXPENSIVE_TERMS
        }

        # ── Sparse log (sampled every rna_fm_interval accepted moves) ───────
        self._sparse: Dict[str, Dict] = {
            name: {"step": [], "accepted": [], "value": []}
            for name, (term, _scale) in self._leaf.items()
            if term.name in _EXPENSIVE_TERMS
        }

        self._meta: dict = {}

    # ─────────────────────────────────────────────────────────────────────────
    # Callback interface
    # ─────────────────────────────────────────────────────────────────────────

    def __call__(
        self,
        step: int,
        total_energy: float,
        sequence: str,
        kind: Optional[str] = None,
        positions: Optional[List[int]] = None,
    ) -> None:
        """Invoked by :func:`run_sa` after every accepted move.

        Parameters
        ----------
        step :
            Total proposed moves elapsed so far (the MC loop's ``total_steps``).
        total_energy :
            Running weighted total energy (delta-tracked by the MC loop).
        sequence :
            Current best sequence (unused here, kept for signature compat).
        kind :
            The proposal kind that produced this accepted move -- the move
            operator's own label ("pair"/"single" for pair_aware, a MotifKind
            for samfeo).  ``run_sa`` passes it only when the callback advertises
            five positional parameters, so a 3-argument callback still works.
        positions :
            Positions the accepted move rewrote; only its length is kept.
        """
        self._accepted += 1
        self._step.append(step)
        self._accepted_idx.append(self._accepted)
        self._total.append(total_energy)
        if kind is not None:
            self._kind.append(kind)
            self._n_sites.append(len(positions) if positions is not None else 0)

        # Score each cheap leaf against the current (already-mutated) struct
        for name in self._terms:
            term, scale = self._leaf[name]
            self._terms[name].append(term.score(self.struct) * scale)

        # Expensive leaves: sample every rna_fm_interval accepted moves only
        if self._sparse and self._accepted % self.rna_fm_interval == 0:
            for name, buf in self._sparse.items():
                term, scale = self._leaf[name]
                buf["step"].append(step)
                buf["accepted"].append(self._accepted)
                buf["value"].append(term.score(self.struct) * scale)

    # ─────────────────────────────────────────────────────────────────────────
    # Metadata
    # ─────────────────────────────────────────────────────────────────────────

    def set_metadata(
        self,
        *,
        pdb_id: str = "",
        best_sequence: str = "",
        start_sequence: str = "",
        native_sequence: str = "",
        pair_map: Optional[dict] = None,
        accept_rate: float = 0.0,
        move_operator: str = "",
        proposed_by_kind: Optional[dict] = None,
        accepted_by_kind: Optional[dict] = None,
        sites_by_kind: Optional[dict] = None,
        position_sampler: Optional[dict] = None,
        position_sampling: str = "",
        by_kind_by_level: Optional[list] = None,
        proposed_sites_by_position: Optional[list] = None,
    ) -> None:
        """Attach run-level metadata.  Call once after :func:`run_sa` returns."""
        self._meta = {
            "pdb_id": pdb_id,
            "best_sequence": best_sequence,
            "start_sequence": start_sequence,
            "native_sequence": native_sequence,
            "accept_rate": accept_rate,
            # Move-operator bookkeeping from DesignResult.  Absent (empty) in
            # logs written before mc.py started counting kinds -- readers must
            # treat {} as "not recorded", not as "zero of every kind".
            "move_operator": move_operator,
            "proposed_by_kind": dict(proposed_by_kind or {}),
            "accepted_by_kind": dict(accepted_by_kind or {}),
            "sites_by_kind": dict(sites_by_kind or {}),
            # Same three counters resolved per temperature level, so the kind
            # mix and per-kind acceptance can be read against annealing
            # progress rather than only as run totals.
            "by_kind_by_level": list(by_kind_by_level or []),
            # Which position sampler ran and how many partition-function folds
            # it paid for.  "" / {} = uniform sampling (or a log written before
            # the sampler existed).
            "position_sampling": position_sampling,
            "position_sampler": dict(position_sampler or {}),
            # Proposal budget per sequence position (accepted or not).
            "proposed_sites_by_position": list(proposed_sites_by_position or []),
            # Store as list-of-[i,j] pairs (i < j) for JSON portability
            "pair_map": (
                [[int(i), int(j)] for i, j in pair_map.items() if i < j]
                if pair_map else []
            ),
            "n_pairs": len(pair_map) // 2 if pair_map else 0,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Serialisation
    # ─────────────────────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            **self._meta,
            "step":     self._step,
            "accepted": self._accepted_idx,
            "total":    self._total,
            # Per-accepted-move proposal kind / site count, aligned with
            # "step", "accepted" and "total".  Empty when the run used the
            # 3-argument callback protocol -- readers must treat [] as "not
            # recorded", never as "no moves".
            "move_kind":  self._kind,
            "move_sites": self._n_sites,
            "terms":    {name: vals for name, vals in self._terms.items()},
            "sparse":   self._sparse,
        }

    def save(self, path: "str | Path") -> None:
        """Write log to *path* as indented JSON.  Creates parent dirs."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        return path
