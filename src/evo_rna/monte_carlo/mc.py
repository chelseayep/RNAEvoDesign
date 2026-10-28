"""Simulated-annealing Monte Carlo for RNA sequence design.

Operates on core.Structure with the energy terms defined in energy.py.
The backbone is fixed; only the sequence mutates.
"""

from __future__ import annotations

import dataclasses
import inspect
import logging
import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from evo_rna.core.structure import NUC_ALPHABET, Structure

from .energy import EnergyFunction

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SAConfig:
    """Simulated-annealing hyperparameters."""

    t_max: float = 5.0          # lowered from 50.0; overridden by adaptive_t_max
    t_min: float = 0.001        # absolute quench floor; ignored when t_min_ratio is set
    t_min_ratio: float | None = 0.005
    cool_factor: float = 0.9
    steps_per_temp: int = 20_000
    num_cycles: int = 3
    seed: int | None = None
    adaptive_t_max: bool = True  # set t_max from initial ΔE distribution
    adaptive_sample_steps: int = 500  # moves to sample for adaptive t_max
    adaptive_target_accept: float = 0.5  # target acceptance rate at t_max
    greedy_init: bool = True        # initialise with a greedy left-to-right sweep
    greedy_quench: bool = True      # polish with zero-T passes after annealing
    greedy_quench_max_passes: int = 5  # upper bound on quench passes
    seed_cycles_from_best: bool = True  # restart each cycle from best sequence found
    # Which proposal distribution to sample moves from:
    #   "pair_aware"  one position (+ its partner, repaired) — the original.
    #   "samfeo"      the whole local motif the position belongs to, so stacks
    #                 and their flanking mismatches co-adapt in one move
    #                 (evo_rna.monte_carlo.samfeo_mutations).
    move_operator: str = "pair_aware"
    # Where in the sequence a proposal is centred, for either operator:
    #   "uniform"  every position equally likely — the original behaviour.
    #   "defect"   Boltzmann over the positional ensemble defect of the current
    #              sequence (SAMFEO eq. 13), so the budget goes to the
    #              positions the ensemble is actually getting wrong.  Needs the
    #              ViennaRNA bindings; falls back to uniform with a warning if
    #              they are missing.  NOTE: applied without a Hastings
    #              correction, so the chain is a biased optimiser rather than
    #              an exact Boltzmann sampler — see DefectWeightedPositions.
    position_sampling: str = "uniform"
    # T0 in eq. 13.  eps is in [0, 1], so a maximally-defective position is at
    # most exp(1/T0) times as likely as a perfect one: 148x at 0.2, 7.4x at
    # 0.5, 2.7x at 1.0.  Larger flattens toward uniform; 0 is argmax.
    defect_temperature: float = 0.2
    # Proposals a defect vector is reused for before refolding.  1 = exact
    # (refold whenever the sequence changes); larger trades a slightly stale
    # proposal distribution for proportionally fewer O(L^3) folds.
    defect_refresh: int = 1


# ---------------------------------------------------------------------------
# Design state
# ---------------------------------------------------------------------------

@dataclass
class DesignResult:
    """Returned by run_sa after the design finishes."""

    best_sequence: str
    best_energy: Dict[str, float]
    trajectory: List[Dict[str, float]] = field(default_factory=list)
    accept_rate: float = 0.0

    # Move-operator bookkeeping: how often each proposal *kind* was drawn and
    # how often it was accepted.  The kinds are the operator's own labels --
    # "single"/"pair" for pair_aware, the MotifKind strings for samfeo (see
    # samfeo_mutations.MotifKind) -- so the two arms are NOT directly
    # comparable key-for-key; compare the distributions, not the labels.
    # Counts cover the annealing loop only, never the adaptive-t_max sampler,
    # whose proposals are calibration and not part of the move budget.
    #   proposed_by_kind[k]  proposals drawn of kind k
    #   accepted_by_kind[k]  of those, accepted by the Metropolis criterion
    #   sites_by_kind[k]     positions actually mutated, summed over proposals
    #                        of kind k (a samfeo motif move is up to 6)
    proposed_by_kind: Dict[str, int] = field(default_factory=dict)
    accepted_by_kind: Dict[str, int] = field(default_factory=dict)
    sites_by_kind: Dict[str, int] = field(default_factory=dict)

    # Position-sampler counters (empty for uniform sampling): which sampler
    # ran, its T0, and how many partition-function folds it actually paid for.
    # Recorded because "defect-weighted" silently degrades to uniform when
    # ViennaRNA is missing, and a run must be able to say which it got.
    position_sampler: Dict[str, float] = field(default_factory=dict)

    # Time-resolved version of the three *_by_kind dicts: one entry per
    # temperature level, in schedule order, each
    #   {"cycle", "level", "temperature", "steps",
    #    "proposed": {kind: n}, "accepted": {kind: n}, "sites": {kind: n}}
    # The run-level dicts cannot show whether the kind mix or a kind's
    # acceptance rate *moves* as the anneal cools -- which is the whole
    # question when comparing proposal distributions -- so the same counters
    # are also kept per level.  Summing any field over levels reproduces the
    # run-level dict exactly.
    by_kind_by_level: List[Dict[str, object]] = field(default_factory=list)

    # Where the move budget landed: proposed_sites_by_position[i] counts every
    # proposal that would have mutated position i (accepted or not), so a
    # defect-weighted arm can be shown to have actually concentrated its
    # budget, and compared against a uniform arm on the same axis.  Counts
    # mutated positions, not the sampled seed position, so it is defined
    # identically for both move operators.
    proposed_sites_by_position: List[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# MC moves
# ---------------------------------------------------------------------------

def _random_position(struct: Structure) -> int:
    return random.randint(0, struct.length - 1)


def _random_nuc_except(current: str) -> str:
    choices = [n for n in NUC_ALPHABET if n != current]
    return random.choice(choices)


def _pair_aware_move(
    struct: Structure,
    position: int | None = None,
) -> tuple[list[tuple[int, str]], str]:
    """Propose a mutation.  If the position is paired, also propose a
    compensatory mutation at the partner so the pair stays valid.

    *position* overrides the uniform draw, so this operator can share the
    defect-weighted position sampler with ``samfeo``.

    Returns ([(pos, new_nuc), ...], description).
    """
    pos = _random_position(struct) if position is None else position
    new_nuc = _random_nuc_except(struct.sequence[pos])
    partner = struct.paired_with(pos)

    if partner is None:
        return [(pos, new_nuc)], "single"

    # For a paired position, pick a valid partner nucleotide.
    _COMPLEMENT = {"A": "U", "U": "A", "G": "C", "C": "G"}
    comp = _COMPLEMENT.get(new_nuc)
    if comp and comp != struct.sequence[partner]:
        return [(pos, new_nuc), (partner, comp)], "pair"
    return [(pos, new_nuc)], "single"


def n_temperature_levels(t_max: float, t_min: float, cool_factor: float) -> int:
    """How many temperature levels the cooling loop will run."""
    n, temp = 0, t_max
    while temp > t_min:
        temp *= cool_factor
        n += 1
    return n


def planned_moves(cfg: "SAConfig") -> int:
    """Total proposals the schedule in *cfg* will make.

    The loop has no convergence test — this is the exact, hard budget.
    Only valid once ``t_max``/``t_min`` are resolved (i.e. after
    :func:`run_sa` has applied ``adaptive_t_max`` and ``t_min_ratio``).
    """
    return (cfg.num_cycles
            * n_temperature_levels(cfg.t_max, cfg.t_min, cfg.cool_factor)
            * cfg.steps_per_temp)


MOVE_OPERATORS = ("pair_aware", "samfeo")
POSITION_SAMPLERS = ("uniform", "defect")


def _resolve_position_sampler(cfg: "SAConfig", struct: Structure):
    """Map ``SAConfig.position_sampling`` to a ``(struct, rng) -> pos`` callable.

    Returns ``None`` for uniform sampling, which lets each operator keep its
    own cheap inline draw.
    """
    if cfg.position_sampling == "uniform":
        return None
    if cfg.position_sampling == "defect":
        from .samfeo_mutations import DefectWeightedPositions
        return DefectWeightedPositions(
            struct.pair_map,
            temperature=cfg.defect_temperature,
            refresh_interval=cfg.defect_refresh,
        )
    raise ValueError(
        f"unknown position_sampling {cfg.position_sampling!r}; "
        f"expected one of {POSITION_SAMPLERS}"
    )


def _resolve_move_operator(cfg: "SAConfig", struct: Structure):
    """Map *cfg* to a ``struct -> (moves, kind)`` callable, plus its sampler.

    Resolved once per run and shared by the adaptive-t_max sampler and the main
    loop: calibrating t_max against a different proposal distribution than the
    one being sampled would set the temperature from the wrong |ΔE| scale.
    The position sampler is returned alongside so the caller can read its
    counters after the run.
    """
    sampler = _resolve_position_sampler(cfg, struct)

    if cfg.move_operator == "pair_aware":
        if sampler is None:
            return _pair_aware_move, None

        def _move(struct: Structure):
            return _pair_aware_move(struct, position=sampler(struct, random))

        return _move, sampler

    if cfg.move_operator == "samfeo":
        from .samfeo_mutations import samfeo_move

        def _move(struct: Structure):
            return samfeo_move(struct, random, position_sampler=sampler)

        return _move, sampler

    raise ValueError(
        f"unknown move_operator {cfg.move_operator!r}; "
        f"expected one of {MOVE_OPERATORS}"
    )


# ---------------------------------------------------------------------------
# Metropolis criterion
# ---------------------------------------------------------------------------

def _callback_takes_move(callback) -> bool:
    """Does *callback* accept the extra ``(kind, positions)`` move arguments?

    Probed once per run rather than caught per call: the 3-argument
    ``(step, energy, sequence)`` signature is the documented one and stays
    supported, so a caller with an older callback keeps working unchanged.
    """
    if callback is None:
        return False
    try:
        sig = inspect.signature(callback)
    except (TypeError, ValueError):        # builtins / C callables
        return False
    params = list(sig.parameters.values())
    if any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in params):
        return True
    positional = [p for p in params
                  if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                                inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    return len(positional) >= 5


def _accept(delta_e: float, temperature: float) -> bool:
    if delta_e <= 0.0:
        return True
    if temperature < 1e-12:
        return False
    return random.random() < math.exp(-delta_e / temperature)


# ---------------------------------------------------------------------------
# Greedy helpers
# ---------------------------------------------------------------------------

def _greedy_sweep(struct: Structure, efn: EnergyFunction) -> float:
    """One left-to-right greedy pass: set each position to its lowest-energy
    nucleotide given the current sequence state.  Returns total delta applied."""
    total_delta = 0.0
    for pos in range(struct.length):
        current_nuc = struct.sequence[pos]
        best_nuc = current_nuc
        best_delta = 0.0
        for nuc in NUC_ALPHABET:
            if nuc == current_nuc:
                continue
            d = efn.delta(struct, pos, nuc)
            if d < best_delta:
                best_delta = d
                best_nuc = nuc
        if best_nuc != current_nuc:
            struct.mutate(pos, best_nuc)
            total_delta += best_delta
    return total_delta


def _greedy_quench(struct: Structure, efn: EnergyFunction, max_passes: int) -> float:
    """Repeat greedy sweeps until no improvement or max_passes reached.
    Returns total energy improvement (negative = better)."""
    total_delta = 0.0
    for _ in range(max_passes):
        delta = _greedy_sweep(struct, efn)
        total_delta += delta
        if delta >= 0.0:
            break
    return total_delta


# ---------------------------------------------------------------------------
# Simulated annealing driver
# ---------------------------------------------------------------------------

def run_sa(
    struct: Structure,
    energy_fn: EnergyFunction | None = None,
    config: SAConfig | None = None,
    callback: Callable[[int, float, str], None] | None = None,
) -> DesignResult:
    """Run simulated annealing on *struct* (mutates struct.sequence in place).

    Parameters
    ----------
    struct : Structure
        Must be fully populated (ss, sa, torsions, pssm, pair_map).
    energy_fn : EnergyFunction, optional
        Defaults to EnergyFunction.default() (unit weights).
    config : SAConfig, optional
        Defaults to SAConfig().
    callback : callable(step, energy, sequence), optional
        Called after every accepted move.

    Returns
    -------
    DesignResult with the best sequence found and its per-term energies.
    """
    cfg = config or SAConfig()
    efn = energy_fn or EnergyFunction.default(struct)
    propose, position_sampler = _resolve_move_operator(cfg, struct)

    if cfg.seed is not None:
        random.seed(cfg.seed)

    # Adaptive t_max: sample moves to estimate typical |ΔE|, then set T so
    # that uphill moves are accepted at the target rate.
    if cfg.adaptive_t_max:
        deltas = []
        for _ in range(cfg.adaptive_sample_steps):
            moves, _ = propose(struct)
            # Telescope exactly as the main loop does, then revert: summing
            # per-position deltas against the pre-move state understates a
            # multi-site move whenever a term couples the mutated positions
            # (ss, stacking, hbond).  Calibrating t_max on that understated
            # scale sets the temperature too low — increasingly so for the
            # samfeo operator, which moves up to six positions at once.
            old_nucs = [(pos, struct.sequence[pos]) for pos, _ in moves]
            d = 0.0
            for pos, nuc in moves:
                d += efn.delta(struct, pos, nuc)
                struct.mutate(pos, nuc)
            for pos, old_nuc in reversed(old_nucs):
                struct.mutate(pos, old_nuc)
            deltas.append(abs(d))
        mean_delta = sum(deltas) / len(deltas) if deltas else 1.0
        if mean_delta > 1e-12:
            # dataclasses.replace, not a field-by-field rebuild: the latter
            # silently reset any field it forgot to copy.
            cfg = dataclasses.replace(
                cfg,
                t_max=-mean_delta / math.log(cfg.adaptive_target_accept),
                adaptive_t_max=False,
            )
        log.info(
            "Adaptive t_max=%.4f  (mean_|ΔE|=%.4f  target_accept=%.2f)",
            cfg.t_max, mean_delta, cfg.adaptive_target_accept,
        )

    # Quench floor proportional to t_max, so the level count — and therefore
    # the move budget — does not move when the energy scale does.
    if cfg.t_min_ratio is not None:
        cfg = dataclasses.replace(cfg, t_min=cfg.t_max * cfg.t_min_ratio)

    log.info(
        "SA schedule: t_max=%.4f  t_min=%.4f  cool=%.2f  levels=%d  "
        "steps/level=%d  cycles=%d  -> %d moves (hard budget, no early stop)",
        cfg.t_max, cfg.t_min, cfg.cool_factor,
        n_temperature_levels(cfg.t_max, cfg.t_min, cfg.cool_factor),
        cfg.steps_per_temp, cfg.num_cycles, planned_moves(cfg),
    )

    # Greedy initialisation: one left-to-right sweep before any MC.
    if cfg.greedy_init:
        _greedy_sweep(struct, efn)
        log.info("Greedy init done  energy=%.4f", efn.score(struct)["total"])

    # Initial energy
    current_scores = efn.score(struct)
    current_e = current_scores["total"]

    best_seq = struct.sequence
    best_e = current_e
    best_scores = dict(current_scores)

    trajectory: list[dict[str, float]] = [dict(current_scores)]
    total_accepted = 0
    total_steps = 0
    proposed_by_kind: dict[str, int] = {}
    accepted_by_kind: dict[str, int] = {}
    sites_by_kind: dict[str, int] = {}
    by_kind_by_level: list[dict[str, object]] = []
    sites_by_position = [0] * struct.length
    callback_takes_move = _callback_takes_move(callback)

    for cycle in range(cfg.num_cycles):
        # Seed each cycle after the first from the best sequence found so far.
        if cycle > 0 and cfg.seed_cycles_from_best:
            struct.sequence = best_seq
            current_scores = efn.score(struct)
            current_e = current_scores["total"]

        temp = cfg.t_max
        level = 0

        while temp > cfg.t_min:
            # Per-level counters, so the kind mix and each kind's acceptance
            # rate can be read as a function of annealing progress.
            lvl_proposed: dict[str, int] = {}
            lvl_accepted: dict[str, int] = {}
            lvl_sites: dict[str, int] = {}

            for _ in range(cfg.steps_per_temp):
                total_steps += 1
                moves, kind = propose(struct)
                proposed_by_kind[kind] = proposed_by_kind.get(kind, 0) + 1
                sites_by_kind[kind] = sites_by_kind.get(kind, 0) + len(moves)
                lvl_proposed[kind] = lvl_proposed.get(kind, 0) + 1
                lvl_sites[kind] = lvl_sites.get(kind, 0) + len(moves)
                for _pos, _ in moves:
                    sites_by_position[_pos] += 1

                # Telescoping delta: apply each mutation *before* measuring the
                # next, so a multi-site (pair) move's delta is exact even for
                # terms that couple the two positions (ss, stacking) or depend on
                # the whole sequence (ensemble).  Summing independent per-position
                # deltas against the pre-move state would otherwise drift.
                old_nucs = [(pos, struct.sequence[pos]) for pos, _ in moves]
                delta = 0.0
                for pos, new_nuc in moves:
                    delta += efn.delta(struct, pos, new_nuc)
                    struct.mutate(pos, new_nuc)

                if _accept(delta, temp):
                    current_e += delta
                    total_accepted += 1
                    accepted_by_kind[kind] = accepted_by_kind.get(kind, 0) + 1
                    lvl_accepted[kind] = lvl_accepted.get(kind, 0) + 1

                    if current_e < best_e:
                        best_e = current_e
                        best_seq = struct.sequence
                        best_scores = efn.score(struct)

                    if callback is not None:
                        if callback_takes_move:
                            callback(total_steps, current_e, struct.sequence,
                                     kind, [pos for pos, _ in moves])
                        else:
                            callback(total_steps, current_e, struct.sequence)
                else:
                    # Reject: restore the original nucleotides (reverse order).
                    for pos, old_nuc in reversed(old_nucs):
                        struct.mutate(pos, old_nuc)

            by_kind_by_level.append({
                "cycle":       cycle,
                "level":       level,
                "temperature": float(temp),
                "steps":       cfg.steps_per_temp,
                "proposed":    dict(lvl_proposed),
                "accepted":    dict(lvl_accepted),
                "sites":       dict(lvl_sites),
            })
            level += 1
            temp *= cfg.cool_factor

        trajectory.append(efn.score(struct))
        log.info(
            "cycle %d/%d  energy=%.4f  best=%.4f",
            cycle + 1, cfg.num_cycles, current_e, best_e,
        )

    accept_rate = total_accepted / total_steps if total_steps > 0 else 0.0

    # Restore best sequence, then polish with zero-temperature greedy passes.
    struct.sequence = best_seq
    if cfg.greedy_quench:
        delta = _greedy_quench(struct, efn, cfg.greedy_quench_max_passes)
        if delta < 0.0:
            best_scores = efn.score(struct)
            best_e = best_scores["total"]
            best_seq = struct.sequence
            log.info("Greedy quench improved energy by %.4f  final=%.4f", delta, best_e)

    if position_sampler is not None:
        log.info("Position sampler: %s", position_sampler.stats())

    return DesignResult(
        best_sequence=best_seq,
        best_energy=best_scores,
        trajectory=trajectory,
        accept_rate=accept_rate,
        proposed_by_kind=proposed_by_kind,
        accepted_by_kind=accepted_by_kind,
        sites_by_kind=sites_by_kind,
        position_sampler=(position_sampler.stats()
                          if position_sampler is not None else {}),
        by_kind_by_level=by_kind_by_level,
        proposed_sites_by_position=sites_by_position,
    )
