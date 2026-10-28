"""Structure-aware MC mutation operator (SAMFEO).

The default :func:`~evo_rna.monte_carlo.mc._pair_aware_move` mutates one
position and, if that position is paired, repairs its partner.  That keeps
pairs valid but never moves a *motif*: it cannot exchange one stack for
another, and it cannot co-adapt a pair with the mismatched bases stacked
against it, because each of those requires several positions to change at
once and every intermediate is uphill.

This module implements the SAMFEO mutation operator, which samples a position
and then mutates the whole local motif that position belongs to.  Six
scenarios, keyed to the diagrams in the paper.  Throughout, ``(i, j)`` is a
target base pair with ``i < j``, and *mismatched* means unpaired **in the
target structure** (``pair_map``), not merely non-complementary in the current
sequence.

Sampled position is **paired**:

======  =========================================  ===========  =========
 diag    geometry                                   mutated      variants
======  =========================================  ===========  =========
 1a      ``(i,j)`` and ``(i+1,j-1)`` both pairs     2 pairs        6·6−1=35
 1b      ``(i,j)`` pair, ``i+1``/``j-1`` unpaired   1 pair + 2   6·4·4−1=95
 1c      1a, and ``i+2``/``j-2`` unpaired           2 pairs + 2  6·6·4·4−1=575
======  =========================================  ===========  =========

Sampled position is **unpaired**:

======  =========================================  ===========  =========
 diag    geometry                                   mutated      variants
======  =========================================  ===========  =========
 1d      ``i+1`` unpaired, ``j-1`` paired           1 pair + 1     4·6−1=23
 1e      ``i+1`` paired, ``j-1`` unpaired           1 pair + 1     4·6−1=23
 1f      ``i+1`` and ``j-1`` both unpaired          1 pair + 2   4·4·6−1=95
======  =========================================  ===========  =========

Diagrams 1b and 1f describe the *same* motif — pair ``(i,j)`` with both
adjacent inner positions mismatched — and mutate the same four positions for
the same 95 variants.  They are listed separately in the paper only because
different sampled positions lead into them, so they share one
:data:`MotifKind` here (``DOUBLE_MISMATCH``); the triggering position is kept
on the motif for logging.

Every scenario excludes the assignment identical to the current sequence (the
``−1`` in each count), so a proposal always changes at least one base.
Positions that happen to keep their current nucleotide within a changed motif
are dropped from the returned move list, since they contribute nothing to the
energy delta.

Position sampling
-----------------
Which position seeds the motif is a separate choice from which motif it
expands to.  The default is uniform.  :class:`DefectWeightedPositions`
implements the paper's eq. 13 instead: a Boltzmann distribution over the
*positional ensemble defect* of the current sequence, so the move budget is
spent where the Boltzmann ensemble actually disagrees with the target rather
than spread evenly over positions that are already right.  Select it with
``SAConfig(position_sampling="defect")`` / ``--position-sampling defect``; it
applies to ``pair_aware`` as well, since the choice of *where* is orthogonal
to the choice of *what*.

Entry point
-----------
:func:`samfeo_move` is signature-compatible with ``mc._pair_aware_move``
(``struct -> ([(pos, nuc), ...], description)``), so wiring it into
:func:`~evo_rna.monte_carlo.mc.run_sa` is a one-line substitution.  ``mc``
applies multi-site moves telescopically, which is what makes these proposals
correct for terms that couple the mutated positions (ss, stacking, hbond).
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from evo_rna.core.structure import NUC_ALPHABET, Structure

log = logging.getLogger(__name__)

# The six base pairs a paired slot may take, in the paper's order.
CANONICAL_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("C", "G"), ("G", "C"), ("A", "U"), ("U", "A"), ("G", "U"), ("U", "G"),
)


class MotifKind:
    """Which diagram a motif corresponds to.  Plain strings, for logging."""

    STACK                 = "stack"                  # 1a
    DOUBLE_MISMATCH       = "double_mismatch"        # 1b / 1f
    STACK_DOUBLE_MISMATCH = "stack_double_mismatch"  # 1c
    SINGLE_MISMATCH_5P    = "single_mismatch_5p"     # 1d
    SINGLE_MISMATCH_3P    = "single_mismatch_3p"     # 1e
    PAIR                  = "pair"                   # paired position, no motif
    SINGLE                = "single"                 # unpaired position, no motif


@dataclass(frozen=True)
class Motif:
    """A set of positions to mutate as a unit.

    ``pairs`` are mutated to one of :data:`CANONICAL_PAIRS` each; ``singles``
    to one of :data:`NUC_ALPHABET` each.  ``trigger`` is the sampled position
    that selected this motif — it is not special during mutation, only
    recorded so a trajectory can say which diagram fired and why.
    """

    kind: str
    pairs: Tuple[Tuple[int, int], ...] = ()
    singles: Tuple[int, ...] = ()
    trigger: int = -1

    @property
    def positions(self) -> Tuple[int, ...]:
        flat = [p for pair in self.pairs for p in pair]
        flat.extend(self.singles)
        return tuple(flat)

    def n_slots(self) -> int:
        """Total assignments over the motif's slots, identity included."""
        return 6 ** len(self.pairs) * 4 ** len(self.singles)

    def n_variants(self) -> int:
        """The paper's nominal count: 35 (1a), 95 (1b/1f), 575 (1c), 23 (1d/1e).

        This subtracts the identity, which assumes the motif currently holds
        an assignment the operator can produce — every pair slot canonical.
        That does not always hold: a non-canonical target pair, or an early MC
        sequence, leaves the identity unreachable and makes all
        :meth:`n_slots` assignments distinct proposals.  Use
        :meth:`n_variants_in` for the true count against a given sequence.
        """
        return self.n_slots() - 1

    def n_variants_in(self, struct: "Structure") -> int:
        """Proposals actually reachable from *struct*'s current sequence.

        Equals :meth:`n_variants` when every pair slot currently holds one of
        :data:`CANONICAL_PAIRS`, and :meth:`n_slots` otherwise.
        """
        seq = struct.sequence
        canonical = all((seq[i], seq[j]) in CANONICAL_PAIRS for i, j in self.pairs)
        return self.n_slots() - 1 if canonical else self.n_slots()


# ---------------------------------------------------------------------------
# Target-structure predicates.  All read `pair_map` — the *target*, never the
# current sequence — and are bounds-safe, so callers may pass i-1 / j+1 freely.
# ---------------------------------------------------------------------------

def _is_pair(struct: Structure, i: int, j: int) -> bool:
    """True if ``(i, j)`` is a base pair in the target structure."""
    return struct.pair_map.get(i) == j


def _is_unpaired(struct: Structure, pos: int) -> bool:
    """True if *pos* is in range and unpaired (mismatched) in the target."""
    return 0 <= pos < struct.length and pos not in struct.pair_map


def _ordered_pair(struct: Structure, pos: int) -> Optional[Tuple[int, int]]:
    """The target pair containing *pos*, as ``(i, j)`` with ``i < j``."""
    partner = struct.paired_with(pos)
    if partner is None:
        return None
    return (pos, partner) if pos < partner else (partner, pos)


def _stacks(struct: Structure, i: int, j: int) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Stacks the pair ``(i, j)`` belongs to, as ``(outer, inner)`` pairs.

    A pair in the middle of a helix belongs to two stacks — one on each side —
    and the paper's rule ("either i or j of the stacking pair positions is
    sampled") admits both.  Both are returned so the caller can choose without
    biasing toward the inward one.
    """
    found = []
    if _is_pair(struct, i + 1, j - 1):
        found.append(((i, j), (i + 1, j - 1)))
    if _is_pair(struct, i - 1, j + 1):
        found.append(((i - 1, j + 1), (i, j)))
    return found


def _mismatched_flank(struct: Structure, i: int, j: int) -> Optional[Tuple[int, int]]:
    """``(i+1, j-1)`` if both are mismatched and distinct, else ``None``.

    The distinctness guard matters for a hairpin closing pair with a
    single-nucleotide loop, where ``i+1 == j-1``: without it the same position
    would occupy two slots and the variant count would be wrong.
    """
    a, b = i + 1, j - 1
    if a < b and _is_unpaired(struct, a) and _is_unpaired(struct, b):
        return (a, b)
    return None


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_paired(struct: Structure, pos: int, rng=random) -> Motif:
    """Diagrams 1a / 1b / 1c — the sampled position is paired in the target.

    Precedence follows the paper: 1c is "the situation with both (1) and (2)",
    so a stack whose flanks are mismatched takes 1c over 1a.  A pair in no
    stack falls to 1b if its inner flanks are mismatched, else to a plain
    two-position pair move.
    """
    pair = _ordered_pair(struct, pos)
    if pair is None:
        raise ValueError(f"position {pos} is not paired in the target structure")
    i, j = pair

    stacks = _stacks(struct, i, j)
    if stacks:
        outer, inner = stacks[0] if len(stacks) == 1 else rng.choice(stacks)
        # 1c: mismatches immediately inside the stack's inner pair.
        flank = _mismatched_flank(struct, *inner)
        if flank is not None:
            return Motif(MotifKind.STACK_DOUBLE_MISMATCH, (outer, inner), flank, pos)
        return Motif(MotifKind.STACK, (outer, inner), (), pos)

    flank = _mismatched_flank(struct, i, j)
    if flank is not None:
        return Motif(MotifKind.DOUBLE_MISMATCH, ((i, j),), flank, pos)   # 1b
    return Motif(MotifKind.PAIR, ((i, j),), (), pos)


def classify_unpaired(struct: Structure, pos: int, rng=random) -> Motif:
    """Diagrams 1d / 1e / 1f — the sampled position is unpaired in the target.

    *pos* qualifies when it sits immediately inside a pair: as ``i+1`` (its 5'
    neighbour ``pos-1`` is the pair's opening base) or as ``j-1`` (its 3'
    neighbour ``pos+1`` is the closing base).  A position in a 1-nt bulge can
    be both, so the eligible pairs are collected and one is chosen at random.
    """
    candidates: List[Tuple[int, int]] = []

    # pos == i+1  =>  the pair opens at pos-1 and closes downstream.
    left = struct.paired_with(pos - 1) if pos - 1 >= 0 else None
    if left is not None and left > pos:
        candidates.append((pos - 1, left))

    # pos == j-1  =>  the pair closes at pos+1 and opens upstream.
    right = struct.paired_with(pos + 1) if pos + 1 < struct.length else None
    if right is not None and right < pos:
        candidates.append((right, pos + 1))

    if not candidates:
        return Motif(MotifKind.SINGLE, (), (pos,), pos)

    i, j = candidates[0] if len(candidates) == 1 else rng.choice(candidates)

    # 1f: the pair's other inner flank is mismatched too — mutate both.
    flank = _mismatched_flank(struct, i, j)
    if flank is not None:
        return Motif(MotifKind.DOUBLE_MISMATCH, ((i, j),), flank, pos)

    # 1d / 1e: only this side is mismatched; the other inner position is paired.
    kind = MotifKind.SINGLE_MISMATCH_5P if pos == i + 1 else MotifKind.SINGLE_MISMATCH_3P
    return Motif(kind, ((i, j),), (pos,), pos)


def classify(struct: Structure, pos: int, rng=random) -> Motif:
    """Dispatch to :func:`classify_paired` / :func:`classify_unpaired`."""
    if struct.paired_with(pos) is None:
        return classify_unpaired(struct, pos, rng)
    return classify_paired(struct, pos, rng)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def _current_assignment(struct: Structure, motif: Motif) -> Tuple:
    seq = struct.sequence
    return (
        tuple((seq[i], seq[j]) for i, j in motif.pairs),
        tuple(seq[p] for p in motif.singles),
    )


def sample_mutation(
    struct: Structure,
    motif: Motif,
    rng=random,
) -> List[Tuple[int, str]]:
    """Draw one proposal for *motif*, uniform over its ``n_variants()``.

    Rejection-samples the identity assignment, which is exactly uniform over
    the non-identity outcomes and terminates quickly: the identity has
    probability ``1/(n_variants+1) <= 1/4``.  When the motif's current bases
    are not a canonical assignment — a non-canonical target pair, or a random
    starting sequence — the identity is unreachable and the first draw is
    always accepted.

    Returns ``[(pos, nucleotide), ...]`` with unchanged positions removed, so
    the list is never empty.
    """
    current = _current_assignment(struct, motif)
    while True:
        pairs = tuple(rng.choice(CANONICAL_PAIRS) for _ in motif.pairs)
        singles = tuple(rng.choice(NUC_ALPHABET) for _ in motif.singles)
        if (pairs, singles) != current:
            break

    seq = struct.sequence
    moves: List[Tuple[int, str]] = []
    for (i, j), (nuc_i, nuc_j) in zip(motif.pairs, pairs):
        if seq[i] != nuc_i:
            moves.append((i, nuc_i))
        if seq[j] != nuc_j:
            moves.append((j, nuc_j))
    for pos, nuc in zip(motif.singles, singles):
        if seq[pos] != nuc:
            moves.append((pos, nuc))
    return moves


# ---------------------------------------------------------------------------
# Position sampling
# ---------------------------------------------------------------------------
#: A position sampler is ``(struct, rng) -> position``.  It chooses *where* to
#: mutate; the motif classification above then decides *what* moves with it.
PositionSampler = Callable[[Structure, "random.Random"], int]


def uniform_position(struct: Structure, rng=random) -> int:
    """Every position equally likely — the operator's original behaviour."""
    return rng.randint(0, struct.length - 1)


class DefectWeightedPositions:
    """Sample positions by positional ensemble defect (SAMFEO eq. 13).

    Uniform position sampling spends the move budget evenly over the sequence,
    including on positions the Boltzmann ensemble already places exactly where
    the target wants them.  SAMFEO instead draws the position from

        p_i = exp(eps_i(x, y) / T0) / sum_j exp(eps_j(x, y) / T0)

    where ``eps_i`` is the positional ensemble defect of the *current* sequence
    ``x`` against the target ``y`` -- the probability, over the ensemble, that
    position *i* is **not** in its target pairing state.  Positions the
    ensemble is getting wrong are proposed more often; positions it already has
    right are not starved (``eps_i = 0`` still carries weight ``exp(0) = 1``),
    so every position keeps positive probability and the chain stays
    irreducible.

    ``T0`` controls flatness exactly as in the paper: ``T0 -> inf`` recovers
    uniform sampling, ``T0 -> 0`` concentrates on the single worst position.
    Since ``eps`` lies in ``[0, 1]``, the most a position can be favoured over
    a perfect one is ``exp(1 / T0)`` -- 148x at the default ``T0 = 0.2``, 7.4x
    at ``T0 = 0.5``, 2.7x at ``T0 = 1.0``.  That bound makes ``T0`` easy to
    reason about; tune it there rather than by rescaling the defect.

    Not a Metropolis-Hastings proposal
    ----------------------------------
    The bias is applied **without** a Hastings correction, so the annealing
    loop no longer samples the Boltzmann distribution of the energy exactly --
    it is a biased optimiser, which is what SAMFEO is too.  Correcting it would
    need ``q(x' -> x)``, i.e. the defect vector of every *proposed* sequence
    (one extra partition-function fold per proposal) plus a sum over the
    positions that map to the same motif.  That is deliberately not done here;
    if a run needs a properly balanced chain, use ``uniform`` sampling.

    Cost
    ----
    One partition-function fold per *distinct* sequence sampled (O(L^3)):
    measured at ~0.5 ms for L = 40, ~2.3 ms for L = 70, ~9 ms for L = 120 and
    ~30 ms for L = 200.  The vector is cached by sequence, so rejected
    proposals are free and only accepted moves pay.  When ``ensemble_defect``
    is already in the energy objective this is nearly free in relative terms --
    that term folds the *mutant* on every proposal, so the sampler adds roughly
    one fold per accepted move on top.  ``refresh_interval > 1`` reuses a
    slightly stale vector for that many proposals, trading exactness for a
    proportional saving on long sequences.

    Falls back to uniform sampling (warning once) if ViennaRNA is missing or a
    fold fails, so a run never dies over the proposal distribution.
    """

    def __init__(
        self,
        pair_map: dict,
        temperature: float = 0.2,
        fold_temperature: float = 37.0,
        refresh_interval: int = 1,
    ) -> None:
        if temperature < 0.0:
            raise ValueError(f"defect temperature T0 must be >= 0, got {temperature}")
        if refresh_interval < 1:
            raise ValueError(
                f"refresh_interval must be >= 1, got {refresh_interval}")
        self.pair_map = dict(pair_map)
        self.temperature = float(temperature)
        self.fold_temperature = float(fold_temperature)
        self.refresh_interval = int(refresh_interval)
        # Bookkeeping, reported by the caller so a run can say how the
        # proposal distribution actually behaved.
        self.n_folds = 0
        self.n_calls = 0
        self.n_fallback = 0
        self._weights: Optional[List[float]] = None
        self._weights_seq: Optional[str] = None
        self._since_refresh = 0
        self._warned = False

    # -- weights ---------------------------------------------------------
    def _boltzmann(self, defect: List[float]) -> List[float]:
        """exp(eps_i / T0), shifted by max(eps) so the exponent cannot overflow."""
        if self.temperature <= 0.0:
            # T0 -> 0: all mass on the worst position(s), ties shared.
            worst = max(defect)
            return [1.0 if d >= worst else 0.0 for d in defect]
        top = max(defect)
        return [math.exp((d - top) / self.temperature) for d in defect]

    def weights(self, struct: Structure) -> Optional[List[float]]:
        """Current per-position weights, refolding only when they are stale."""
        seq = struct.sequence
        cached = (self._weights is not None
                  and len(self._weights) == struct.length)
        if cached and self._weights_seq == seq:
            return self._weights                    # nothing changed, no fold
        if cached and self._since_refresh + 1 < self.refresh_interval:
            # The sequence moved but we are inside the staleness window: reuse
            # the vector rather than pay another fold.
            self._since_refresh += 1
            return self._weights

        from evo_rna.monte_carlo.energy_terms.ensemble import (
            positional_ensemble_defect,
        )
        try:
            defect = positional_ensemble_defect(
                seq, self.pair_map, self.fold_temperature)
        except (ImportError, RuntimeError, ValueError) as exc:
            self.n_fallback += 1
            if not self._warned:
                self._warned = True
                log.warning(
                    "Defect-weighted position sampling unavailable (%s); "
                    "falling back to uniform sampling for this run.", exc)
            return None
        self.n_folds += 1
        self._weights = self._boltzmann(defect)
        self._weights_seq = seq
        self._since_refresh = 0
        return self._weights

    # -- sampling --------------------------------------------------------
    def __call__(self, struct: Structure, rng=random) -> int:
        self.n_calls += 1
        w = self.weights(struct)
        if not w or not any(w):
            return uniform_position(struct, rng)
        return rng.choices(range(struct.length), weights=w, k=1)[0]

    def stats(self) -> dict:
        """Counters for the run record: how often the sampler actually folded."""
        return {
            "position_sampling":   "defect",
            "defect_temperature":  self.temperature,
            "refresh_interval":    self.refresh_interval,
            "calls":               self.n_calls,
            "folds":               self.n_folds,
            "uniform_fallbacks":   self.n_fallback,
        }


def samfeo_move(
    struct: Structure,
    rng=random,
    position_sampler: Optional[PositionSampler] = None,
) -> Tuple[List[Tuple[int, str]], str]:
    """Propose one structure-aware mutation.

    Drop-in replacement for ``mc._pair_aware_move``: samples a position,
    classifies the motif it belongs to, and mutates that motif as a unit.
    Returns ``([(pos, nuc), ...], motif_kind)``.

    *position_sampler* chooses where to mutate; the default is uniform, which
    is the operator's original behaviour.  Pass a
    :class:`DefectWeightedPositions` to sample by positional ensemble defect
    (SAMFEO eq. 13) instead.
    """
    sampler = position_sampler or uniform_position
    pos = sampler(struct, rng)
    motif = classify(struct, pos, rng)
    return sample_mutation(struct, motif, rng), motif.kind
