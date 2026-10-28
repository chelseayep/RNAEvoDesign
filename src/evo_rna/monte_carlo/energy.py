"""RNA energy function for Monte Carlo sequence design.

The EnergyFunction is a weighted sum of energy terms.  The ``default``
configuration groups them into three gauge-fixed composite blocks
(:class:`CompositeTerm`):

    evo          Eevo:       PSSM log-odds from structural alignments
    secondary    Esecondary: ss, stacking_context, ensemble_defect, junction
    tertiary     Etertiary:  hbond, sa, torsion

Each composite's sub-weights are simplex-normalised, so the three top-level
weights carry the block magnitudes and the within-block mixes are separately
parameterised (the two levels are identifiable).


Individual term implementations live in energy_terms/:
    evo              EnergyTerm (ABC), EvoTerm
    stacking         StackingContextTerm
    ss_rnafold       RNAfoldTerm
    rna_fm           RNAFMTerm
    _structural      SSTerm, SATerm, TorsionTerm  (require 3-D context)
"""

from __future__ import annotations

from pathlib import Path

from typing import Dict

from evo_rna.core.structure import Structure
from evo_rna.monte_carlo.energy_terms.evo import EnergyTerm, EvoTerm  # noqa: F401 — re-exported
from evo_rna.monte_carlo.energy_terms.composite import CompositeTerm  # noqa: F401 — re-exported
from evo_rna.monte_carlo.energy_terms.stacking import StackingContextTerm  # noqa: F401 — re-exported
from evo_rna.monte_carlo.energy_terms._structural import (  # noqa: F401 — re-exported
    JunctionTerm,
    SATerm,
    SSTerm,
    TorsionTerm,
)
from evo_rna.monte_carlo.energy_terms.propensities import (  # noqa: F401 — re-exported
    PropensityTables,
    load_default_tables,
)
from evo_rna.monte_carlo.energy_terms.ensemble import EnsembleDefectTerm  # noqa: F401 — re-exported
from evo_rna.monte_carlo.energy_terms.dssr_hbond import (  # noqa: F401 — re-exported
    DSSRHbondTerm,
    HbondPropensity,
)


class EnergyFunction:
    """Weighted sum of energy terms."""

    def __init__(self, terms: dict[str, tuple[EnergyTerm, float]] | None = None) -> None:
        self.terms: dict[str, tuple[EnergyTerm, float]] = terms or {}

    # ------------------------------------------------------------------ #
    # Composite-block builders (gauge-fixed: sub-weights sum to 1)        #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _secondary(
        struct: "Structure",
        tables: "PropensityTables",
        weights: "tuple[float, float, float, float]",
        temperature: float,
    ) -> CompositeTerm:
        """Esecondary: ss, stacking, ensemble_defect, junction.

        ``ss_rnafold`` was dropped from this block: an ablation
        (``scripts/ablate_secondary_terms.py``) showed it is strongly
        redundant with ``ensemble_defect`` (within-target Pearson ~0.8) while
        discriminating no better and costing the same O(L^3) fold — so the
        thermodynamic signal is carried by ``ensemble_defect`` alone.
        ``ensemble_defect`` still makes this block's ``delta`` O(L^3); ``ss``,
        ``stacking_context`` and ``junction`` are O(1)/O(degree).
        """
        return CompositeTerm("secondary", [
            (SSTerm(),                                                       weights[0]),
            (StackingContextTerm(),                                         weights[1]),
            (EnsembleDefectTerm(struct.pair_map, temperature=temperature),  weights[2]),
            (JunctionTerm(tables.junction),                                 weights[3]),
        ])

    @staticmethod
    def _tertiary(
        struct: "Structure",
        tables: "PropensityTables",
        weights: "tuple[float, float, float]",
        hbond_propensity: "HbondPropensity | str | None",
        learned_tables: "str | Path | None" = None,
        target: "str | None" = None,
    ) -> CompositeTerm:
        """Etertiary: hbond, sa, torsion. All O(degree)/O(1) delta.

        With *learned_tables* the block becomes the single learned leaf instead
        (see evo_rna/learned/), which subsumes all three.  The binned
        leaves are kept, not deleted: they are the control arm for "did the
        learned potential help, or would reweighting alone have done it?", and
        that question cannot be answered once they are gone.  ``tertiary_weights``
        is ignored in this mode -- a one-leaf simplex is 1.0 -- while
        ``w_tertiary`` still carries the block magnitude, so saved weight files
        stay valid.
        """
        if learned_tables is not None:
            from .energy_terms.learned_tertiary import LearnedTertiaryTerm
            if target is None:
                raise ValueError(
                    "learned_tertiary requires the target id: the exported tables "
                    "are per-target (<tables_dir>/<target>.npz)")
            return CompositeTerm("tertiary", [
                (LearnedTertiaryTerm.for_structure(struct, learned_tables, target), 1.0),
            ])
        prop = hbond_propensity
        if isinstance(prop, str):
            from .energy_terms.dssr_hbond import HbondPropensity
            prop = HbondPropensity.load(prop)
        return CompositeTerm("tertiary", [
            (DSSRHbondTerm.from_structure(struct, prop), weights[0]),
            (SATerm(tables.sa),                          weights[1]),
            (TorsionTerm(tables.torsion),                weights[2]),
        ])

    @classmethod
    def default(
        cls,
        struct: "Structure",
        tables: "PropensityTables | None" = None,
        *,
        w_evo: float = 1.0,
        w_secondary: float = 1.0,
        w_tertiary: float = 1.0,
        w_composition: float = 0.0,
        secondary_weights: "tuple[float, float, float, float]" = (1.0, 1.0, 1.0, 1.0),
        tertiary_weights: "tuple[float, float, float]" = (1.0, 1.0, 1.0),
        hbond_propensity: "HbondPropensity | str | None" = None,
        gc_anchor: str = "pssm",
        gc_tolerance: float = 0.05,
        gc_min_homologs: int = 10,
        pssm_n_homologs: "float | None" = None,
        native_sequence: "str | None" = None,
        temperature: float = 37.0,
        learned_tertiary: "str | Path | None" = None,
        target: "str | None" = None,
        w_ss_head: float = 0.0,
        ss_head: "object | None" = None,
        ss_head_form: str = "nll",
    ) -> "EnergyFunction":
        """Three-composite energy function: evo + secondary + tertiary.

        The objective is ``w_evo·Eevo + w_secondary·Esecondary + w_tertiary·Etertiary``,
        where each composite is a *simplex-normalised* weighted sum of its
        sub-terms (so the three ``w_*`` carry the block magnitudes and the
        ``*_weights`` carry the within-block mix; the two levels are
        identifiable).  See :class:`CompositeTerm`.

        Blocks
        ------
        Eevo (``w_evo``)
            PSSM log-odds (:class:`EvoTerm`).
        Esecondary (``w_secondary``, mix = ``secondary_weights``)
            ss, stacking_context, ensemble_defect, junction.
        Etertiary (``w_tertiary``, mix = ``tertiary_weights``)
            hbond, sa, torsion -- or, with ``learned_tertiary`` set to a
            directory of exported tables (and ``target`` naming the structure),
            the single learned leaf that replaces all three.
        Ecomposition (``w_composition``, **off by default**)
            GC-content prior (:class:`GCContentTerm`), anchored by
            ``gc_anchor`` to the PSSM's expected GC — derived from structural
            homologs, so it needs no native sequence and leaks nothing.  The
            PSSM anchor degrades to neutral when the profile is flat or rests
            on fewer than ``gc_min_homologs`` homologs (pass the count as
            ``pssm_n_homologs``); a shallow profile's composition reflects
            which few structures cleared the TM cutoff, not the family.
            Opt-in so existing runs reproduce exactly.  Turn it on when the
            energy terms are buying free energy with GC rather than with
            structure — measured on the exp_1 arms, within-target
            ``r(GC, MFE) = -0.963`` while ``r(GC, scMCC) = -0.501`` and
            ``r(GC, TM) = -0.242``: GC bought energy and cost structure.
            Diagnose with ``scripts/analysis/gc_vs_mfe.py``.

            It sits at the top level rather than inside Esecondary on purpose:
            it is a sequence-composition prior, not a secondary-structure term,
            and adding it to that block would change ``secondary_weights`` from
            a 4-tuple to a 5-tuple and invalidate every saved weight file.
        Requires ViennaRNA (ensemble_defect) and the DSSR/FreeSASA
        propensity tables (``tables``; defaults to :func:`load_default_tables`,
        i.e. the configured artifact or uniform no-op tables).
        """
        tables = tables if tables is not None else load_default_tables()
        terms = {
            "evo":       (EvoTerm(), w_evo),
            "secondary": (cls._secondary(struct, tables, secondary_weights, temperature), w_secondary),
            "tertiary":  (cls._tertiary(struct, tables, tertiary_weights,
                                        hbond_propensity, learned_tertiary, target),
                          w_tertiary),
        }
        if w_composition:
            from .energy_terms.composition import GCContentTerm
            terms["composition"] = (
                GCContentTerm.from_structure(
                    struct,
                    tolerance=gc_tolerance,
                    anchor=gc_anchor,
                    native_sequence=native_sequence,
                    n_homologs=pssm_n_homologs,
                    min_homologs=gc_min_homologs,
                ),
                w_composition,
            )
        if w_ss_head:
            from .energy_terms.ss_head import SSHeadTerm
            terms["ss_head"] = (
                SSHeadTerm(struct.pair_map, head=ss_head, form=ss_head_form),
                w_ss_head,
            )
        return cls(terms)

    @classmethod
    def evo_only(cls, struct: "Structure | None" = None, *, w_evo: float = 1.0) -> "EnergyFunction":
        """Single-block energy: Eevo only (PSSM log-odds). ``struct`` unused."""
        return cls({"evo": (EvoTerm(), w_evo)})

    @classmethod
    def structure_only(
        cls,
        struct: "Structure",
        tables: "PropensityTables | None" = None,
        *,
        w_secondary: float = 1.0,
        w_tertiary: float = 1.0,
        secondary_weights: "tuple[float, float, float, float]" = (1.0, 1.0, 1.0, 1.0),
        tertiary_weights: "tuple[float, float, float]" = (1.0, 1.0, 1.0),
        hbond_propensity: "HbondPropensity | str | None" = None,
        temperature: float = 37.0,
    ) -> "EnergyFunction":
        """Structure-only energy: Esecondary + Etertiary (no evolutionary term)."""
        tables = tables if tables is not None else load_default_tables()
        return cls({
            "secondary": (cls._secondary(struct, tables, secondary_weights, temperature), w_secondary),
            "tertiary":  (cls._tertiary(struct, tables, tertiary_weights, hbond_propensity), w_tertiary),
        })

    @classmethod
    def rna_design(
        cls,
        struct: "Structure",
        w_evo: float = 1.0,
        w_ss: float = 1.0,
        temperature: float = 37.0,
    ) -> "EnergyFunction":
        """Lightweight two-term energy function (no RNA-FM).

        Use when torch/multimolecule is unavailable or for fast design
        driven purely by evolutionary conservation and secondary-structure
        fidelity.

        Terms
        -----
        evo  (weight *w_evo*)
            PSSM log-odds from structural alignments.

        ss_rnafold  (weight *w_ss*)
            Base-pair distance between RNAfold-predicted and target structure.
        """
        from .energy_terms.ss_rnafold import RNAfoldTerm

        return cls({
            "evo":        (EvoTerm(),                                      w_evo),
            "ss_rnafold": (RNAfoldTerm.from_structure(struct, temperature), w_ss),
        })

    @classmethod
    def rna_design_stacking(
        cls,
        struct: "Structure",
        w_evo: float = 1.0,
        w_ss: float = 1.0,
        w_stack: float = 1.0,
        temperature: float = 37.0,
    ) -> "EnergyFunction":
        """Three-term energy function for sequence-only RNA design with stacking.

        Extends :meth:`rna_design` with :class:`StackingContextTerm`, which
        adds a stacking-aware base-pair quality signal derived from the primary
        sequence alone.  No 3-D coordinates are required.

        Terms
        -----
        evo  (weight *w_evo*)
            PSSM log-odds from structural alignments.

        ss_rnafold  (weight *w_ss*)
            Base-pair distance between RNAfold-predicted and target structure.

        stacking_context  (weight *w_stack*)
            Rewards sequences where target-paired positions carry strong WC
            pairs (GC > AU > GU wobble) **and** those pairs form contiguous
            helical stacks.  Adapted from the ``get_paring_feature``
            featurizer in RNAref3D (``bp/featurizer.py``).  Purely
            sequence-algebraic; O(L²) to compute, O(L) per MC delta step.

        Parameters
        ----------
        struct:
            Target structure with ``pair_map`` set.
        w_evo, w_ss, w_stack:
            Term weights.
        temperature:
            RNAfold temperature in °C.
        """
        from .energy_terms.ss_rnafold import RNAfoldTerm

        return cls({
            "evo":              (EvoTerm(),                                      w_evo),
            "ss_rnafold":       (RNAfoldTerm.from_structure(struct, temperature), w_ss),
            "stacking_context": (StackingContextTerm(),                          w_stack),
        })

    @classmethod
    def structural(
        cls,
        struct: "Structure",
        tables: "PropensityTables | None" = None,
        w_evo: float = 1.0,
        w_ss: float = 1.0,
        w_stack: float = 1.0,
        w_sa: float = 1.0,
        w_torsion: float = 1.0,
        w_junction: float = 1.0,
        w_hbond: float = 0.0,
        hbond_propensity: "HbondPropensity | str | None" = None,
        w_ensemble: float = 0.0,
        temperature: float = 37.0,
    ) -> "EnergyFunction":
        """Knowledge-based, 3-D-aware energy function (no external folder / NN).

        Combines the evolutionary and stacking terms with the DSSR/FreeSASA-derived
        structural propensity terms:

        evo        PSSM log-odds.
        ss         Base-pair validity at target-paired positions (geometry-level).
        stacking   Stacking-aware base-pair quality.
        sa         log P(nuc | SASA class).
        torsion    log P(nuc | glycosidic-χ bin).
        junction   log P(nuc | DSSR loop/junction element).

        ``hbond`` (weight *w_hbond*) is the optional H-bond satisfaction term:
        it rewards bases that can form the hydrogen bonds the native geometry
        requires at each DSSR contact — including the **non-canonical** (Leontis-
        Westhof) pairs and base multiplets the ``ss``/``stacking`` terms miss.
        O(degree) delta; a no-op when ``struct.contacts`` is empty (no DSSR).

        The three propensity terms read *tables* (defaults to
        :func:`load_default_tables`, i.e. the configured artifact or uniform
        no-op tables when none has been derived yet).  These — like evo/ss/stack —
        have O(1) delta and need no subprocess or torch.

        ``ensemble_defect`` (weight *w_ensemble*) is the optional thermodynamic
        term: the ViennaRNA partition-function ensemble defect vs. the target
        secondary structure (lower = the target dominates the Boltzmann
        ensemble).  It folds the whole sequence, so its delta is O(L^3), not
        O(1); it is included only when ``w_ensemble`` is non-zero and requires
        the ViennaRNA python bindings.
        """
        tables = tables if tables is not None else load_default_tables()
        terms: dict[str, tuple[EnergyTerm, float]] = {
            "evo":              (EvoTerm(),                     w_evo),
            "ss":               (SSTerm(),                      w_ss),
            "stacking_context": (StackingContextTerm(),         w_stack),
            "sa":               (SATerm(tables.sa),             w_sa),
            "torsion":          (TorsionTerm(tables.torsion),   w_torsion),
            "junction":         (JunctionTerm(tables.junction), w_junction),
        }
        if w_hbond:
            prop = hbond_propensity
            if isinstance(prop, str):
                from .energy_terms.dssr_hbond import HbondPropensity
                prop = HbondPropensity.load(prop)
            terms["hbond"] = (DSSRHbondTerm.from_structure(struct, prop), w_hbond)
        if w_ensemble:
            terms["ensemble_defect"] = (
                EnsembleDefectTerm(struct.pair_map, temperature=temperature),
                w_ensemble,
            )
        return cls(terms)

    def set_weight(self, name: str, weight: float) -> None:
        if name in self.terms:
            term, _ = self.terms[name]
            self.terms[name] = (term, weight)

    def weights_spec(self) -> Dict[str, object]:
        """The live weight vector, in the shape ``score()`` keys are scaled by.

        Recorded into ``results.json`` so a results file can never be read
        without knowing which weights produced its energies — uniform and
        calibrated runs are otherwise indistinguishable after the fact, since
        every value ``score()`` emits is already weighted.

        Returns
        -------
        dict
            ``{"blocks": {name: top-level weight},
               "sub_weights": {block: {sub: normalised sub-weight}},
               "effective": {score-key: total scale applied to the raw term}}``

            ``effective`` is the flat map that matters for reading the file:
            for a plain term it is its weight, for a composite leaf
            ``b.<sub>`` it is ``w_b * a_sub`` — exactly the factor ``score()``
            and :class:`EnergyLogger` multiply the raw term score by.
        """
        blocks: Dict[str, float] = {}
        sub_weights: Dict[str, Dict[str, float]] = {}
        effective: Dict[str, float] = {}
        for name, (term, weight) in self.terms.items():
            blocks[name] = float(weight)
            if isinstance(term, CompositeTerm):
                subs = term.weights          # already simplex-normalised
                sub_weights[name] = {k: float(v) for k, v in subs.items()}
                for sub, sub_w in subs.items():
                    effective[f"{name}.{sub}"] = float(weight * sub_w)
            else:
                effective[name] = float(weight)
        return {"blocks": blocks, "sub_weights": sub_weights, "effective": effective}

    def score(self, struct: Structure) -> Dict[str, float]:
        """Evaluate all terms. Returns {name: weighted_score, ..., 'total': sum}.

        Composite blocks are expanded: a block ``b`` contributes both its own
        key (``b``) and one nested key per sub-term (``b.<sub>``), each already
        scaled by the block's top-level weight.
        """
        result: Dict[str, float] = {}
        total = 0.0
        for name, (term, weight) in self.terms.items():
            if isinstance(term, CompositeTerm):
                parts = term.breakdown(struct)  # normalised-weighted, sums to term.score
                for sub, sub_val in parts.items():
                    result[f"{name}.{sub}"] = sub_val * weight
                val = sum(parts.values()) * weight
            else:
                val = term.score(struct) * weight
            result[name] = val
            total += val
        result["total"] = total
        return result

    def delta(self, struct: Structure, pos: int, new_nuc: str) -> float:
        """Total weighted energy change for a single-position mutation."""
        d = 0.0
        for _, (term, weight) in self.terms.items():
            d += term.delta(struct, pos, new_nuc) * weight
        return d
