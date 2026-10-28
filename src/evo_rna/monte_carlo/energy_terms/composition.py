"""GC-composition prior — keeps the optimiser from buying energy with GC.

Why this exists
---------------
G·C pairs contribute roughly three times the stacking free energy of A·U, so
any objective containing a folding-energy term can lower its energy simply by
raising GC content.  A GC-rich sequence has a rigid, unambiguous Boltzmann
ensemble, which drives ``ensemble_defect`` down for *almost any* target — it
folds decisively rather than correctly.

Measured on the exp_1 arms (``scripts/analysis/gc_vs_mfe.py``), holding the
target fixed:

    r(GC, MFE)   = -0.963   (65/75 targets)   GC buys energy, near-determinstically
    r(GC, scMCC) = -0.501   (35/52 targets)   ... and degrades 2-D correctness
    r(GC, TM-3D) = -0.242   (45/77 targets)   ... and 3-D correctness

The harm is monotone across the observed range rather than confined to an
upper tail, so this term pulls composition *toward a reference GC* (the
native's by default) instead of merely capping it.

Shape
-----
    score = max(0, |gc(seq) - target_gc| - tolerance)^2 / tolerance^2

A flat-bottomed quadratic: free movement inside ``±tolerance`` of the
reference, growing cost outside it.  Dividing by ``tolerance^2`` makes the
score ~1 at one tolerance-width beyond the band, so the weight means the same
thing regardless of how wide the band is.  Zero-cost inside the band matters —
designs legitimately differ from the native in composition, and this should
penalise drift, not diversity.

The score is intensive (GC is a fraction, and the term is O(1) in length), so
its weight transfers across targets like the other terms.  Unlike
``ensemble_defect`` its ``delta`` is genuinely O(1): a point mutation shifts GC
by at most 1/L.

Where the reference GC comes from
--------------------------------
**Not from the native sequence.**  In a real design task you have a target
structure and no native, and even in benchmarking the native is held out —
anchoring to its GC would leak the answer into the objective and inflate
``seq_recovery``.  The reference must be derivable from the target structure
or its homologs alone:

``pssm``     expected GC under the PSSM, which is built from *structurally*
             aligned homologs (``evo_rna.profile.pssm.compute_M``).  Target-
             specific and leak-free — it is the same information the ``evo``
             term already consumes.  Falls back to neutral when the PSSM is
             flat (no homologs found), which is the honest answer in that case.
``neutral``  a fixed 0.5.  Use when there is no profile at all.
``native``   oracle only.  Available for ablations that quantify the headroom
             a perfect composition prior would buy; **never** for a design run
             or a reported benchmark, since it leaks the held-out sequence.
"""

from __future__ import annotations

import logging
from typing import Optional

from evo_rna.core.structure import Structure
from evo_rna.monte_carlo.energy_terms.evo import EnergyTerm

log = logging.getLogger(__name__)

_GC = frozenset("GC")

# Minimum number of homologs for a PSSM to be considered reliable. 
MIN_HOMOLOGS = 10

#: PSSM column order (evo_rna.profile.pssm.compute_M): A U G C.
_GC_COLUMNS = (2, 3)

#: Reference GC when nothing target-specific is available. Structured RNAs sit
#: near this, and it is what a flat (homolog-free) PSSM implies anyway.
NEUTRAL_GC = 0.55


def gc_fraction(sequence: str) -> float:
    """Fraction of G + C in *sequence*; 0.0 for an empty sequence."""
    if not sequence:
        return 0.0
    return sum(c in _GC for c in sequence) / len(sequence)


def expected_gc_from_pssm(pssm) -> Optional[float]:
    """Expected GC fraction under *pssm*, or None if it carries no signal.

    ``pssm`` is the ``(L, 4)`` log-odds matrix ``log(w / background)`` in A-U-G-C
    column order.  Per-position probabilities are recovered as a softmax across
    the four columns, which assumes a uniform background — exact for a flat
    PSSM and a close approximation otherwise, and good enough for a scalar
    composition target.

    Returns None when the PSSM is absent, empty, or all-zero (no homologs), so
    the caller can fall back to :data:`NEUTRAL_GC` rather than silently treating
    "no information" as "0.5 is the right answer for this target".
    """
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        return None
    if pssm is None:
        return None
    m = np.asarray(pssm, dtype=float)
    if m.ndim != 2 or m.shape[0] == 0 or m.shape[1] < 4:
        return None
    if not np.any(m):           # all-zero => flat profile, no homologs
        return None
    m = m - m.max(axis=1, keepdims=True)          # stabilise the exponential
    p = np.exp(m)
    denom = p.sum(axis=1, keepdims=True)
    if not np.all(denom > 0):
        return None
    p = p / denom
    return float(p[:, _GC_COLUMNS].sum(axis=1).mean())


class GCContentTerm(EnergyTerm):
    """Penalise GC content drifting away from a reference value.

    Parameters
    ----------
    target_gc:
        Reference GC fraction.  Use the native's (see :meth:`from_structure`) —
        it is the composition the real molecule achieves while satisfying its
        actual functional constraints.
    tolerance:
        Half-width of the free band around *target_gc*.  Inside it the score is
        exactly zero.  Default 0.05 (±5 percentage points); the measured drift
        of the unconstrained arms was +0.076 and +0.102, i.e. 1.5-2 tolerance
        widths out, so this band permits ordinary variation while charging the
        observed shortcut.
    """

    def __init__(
        self,
        target_gc: float,
        tolerance: float = 0.05,
        resolved_from: str = "explicit",
    ) -> None:
        if not 0.0 <= target_gc <= 1.0:
            raise ValueError(f"target_gc must be in [0, 1], got {target_gc}")
        if tolerance <= 0.0:
            raise ValueError(f"tolerance must be > 0, got {tolerance}")
        self._target_gc = float(target_gc)
        self._tolerance = float(tolerance)
        self.resolved_from = resolved_from

    @property
    def name(self) -> str:
        return "gc_content"

    @property
    def target_gc(self) -> float:
        return self._target_gc

    def _penalty(self, gc: float) -> float:
        excess = abs(gc - self._target_gc) - self._tolerance
        if excess <= 0.0:
            return 0.0
        return (excess / self._tolerance) ** 2

    def score(self, struct: Structure) -> float:
        return self._penalty(gc_fraction(struct.sequence))

    def delta(self, struct: Structure, pos: int, new_nuc: str) -> float:
        """O(1): a point mutation moves GC by at most 1/L."""
        seq = struct.sequence
        L = len(seq)
        if L == 0 or pos >= L:
            return 0.0
        old_is_gc = seq[pos] in _GC
        new_is_gc = new_nuc in _GC
        if old_is_gc == new_is_gc:
            return 0.0                      # composition unchanged
        gc = gc_fraction(seq)
        gc_new = gc + (1.0 if new_is_gc else -1.0) / L
        return self._penalty(gc_new) - self._penalty(gc)

    @classmethod
    def from_structure(
        cls,
        struct: Structure,
        tolerance: float = 0.05,
        anchor: str = "pssm",
        native_sequence: Optional[str] = None,
        n_homologs: Optional[float] = None,
        min_homologs: int = MIN_HOMOLOGS,
    ) -> "GCContentTerm":
        """Build a term whose reference GC is derived without the native sequence.

        Parameters
        ----------
        anchor:
            ``"pssm"`` (default) — expected GC under ``struct.pssm``, i.e. from
            structurally aligned homologs.  Leak-free and target-specific;
            falls back to :data:`NEUTRAL_GC` when the profile is flat or too shallow.
            ``"neutral"`` — fixed :data:`NEUTRAL_GC`.
            ``"native"`` — the native's own GC.  **Oracle only**: this leaks the
            held-out sequence, so it is valid for measuring how much a perfect
            composition prior could buy, and invalid for a design run or any
            reported benchmark.  Requires *native_sequence*.
        native_sequence:
            Only read when ``anchor="native"``.  Never defaults to
            ``struct.sequence`` — during MC that is the current candidate, so
            such a fallback would silently anchor to the random start sequence.
        """
        if anchor == "native":
            if not native_sequence:
                raise ValueError(
                    'anchor="native" requires native_sequence. It is an oracle '
                    "setting that leaks the held-out sequence — use "
                    'anchor="pssm" for design runs and reported benchmarks.'
                )
            return cls(target_gc=gc_fraction(native_sequence), tolerance=tolerance,
                       resolved_from="native(oracle)")

        if anchor == "neutral":
            return cls(target_gc=NEUTRAL_GC, tolerance=tolerance,
                       resolved_from="neutral")

        if anchor != "pssm":
            raise ValueError(
                f'anchor must be "pssm", "neutral" or "native"; got {anchor!r}'
            )

        def _neutral(why: str) -> "GCContentTerm":
            log.info("GC anchor: falling back to neutral (%.3f) — %s", NEUTRAL_GC, why)
            return cls(target_gc=NEUTRAL_GC, tolerance=tolerance,
                       resolved_from=f"neutral({why})")

        # Depth gate first: a profile can be numerically non-flat and still rest
        # on too few homologs for its composition to mean anything.
        if n_homologs is not None and n_homologs < min_homologs:
            return _neutral(f"pssm too shallow: n={n_homologs:.0f} < {min_homologs}")

        expected = expected_gc_from_pssm(getattr(struct, "pssm", None))
        if expected is None:
            return _neutral("pssm carries no signal (flat or absent)")

        if n_homologs is None:
            # --pssm supplied a pre-computed profile, so depth is unknown and
            # the gate above could not run. Use it, but say so.
            log.warning(
                "GC anchor: using PSSM expected GC=%.3f, but homolog depth is "
                "unknown (pre-computed profile) — the n>=%d reliability gate "
                "was not applied.", expected, min_homologs,
            )
            return cls(target_gc=expected, tolerance=tolerance,
                       resolved_from="pssm(depth unknown)")

        return cls(target_gc=expected, tolerance=tolerance,
                   resolved_from=f"pssm(n={n_homologs:.0f})")
