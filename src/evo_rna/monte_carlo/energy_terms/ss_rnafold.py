"""RNAfold-based secondary structure energy term for Monte Carlo sequence design.

Uses ViennaRNA's RNAfold (subprocess) to predict the minimum free energy (MFE)
secondary structure for a candidate sequence.  The energy term penalises
deviation of the predicted structure from a supplied target structure, or
(when no target is given) directly uses the negative MFE as the score.

Two modes
---------
Structure-matching  (target_db provided)
    score = 1 − MCC(predicted_structure, target_structure)  ∈ [0, 2]
    Zero means the predicted structure exactly matches the target.  The MCC
    over the base-pair classification is *intensive* (it lives in [-1, 1] for
    any length), so unlike a raw ``bp_distance`` count this term needs no
    explicit size divisor and its weight transfers across targets.  It is also
    the same quantity reported as scMCC by the evaluation pipeline, so the
    search objective and the reported 2-D metric agree.

MFE maximisation  (no target_db)
    score = −mfe / L  (kcal/mol per nucleotide)
    Since MFE is negative for stable structures, this rewards sequences that
    fold stably without a specific target; the /L keeps it intensive too.

Install
-------
RNAfold must be on $PATH::

    conda install -c bioconda viennarna
    # or: https://www.tbi.univie.ac.at/RNA/

Performance note
----------------
Each score/delta evaluation spawns a RNAfold subprocess (O(N³)).  This term
is substantially more expensive than the analytic terms (EvoTerm, SSTerm,
etc.).  Use a reduced weight or a less-frequent evaluation strategy for long
sequences.
"""

from __future__ import annotations

import subprocess
from typing import Dict, Optional

from evo_rna.core.structure import Structure
from evo_rna.monte_carlo.energy_terms.evo import EnergyTerm


# ---------------------------------------------------------------------------
# RNAfold subprocess wrapper
# ---------------------------------------------------------------------------

def fold(sequence: str, temperature: float = 37.0) -> tuple[str, float]:
    """Fold *sequence* with RNAfold and return ``(dot_bracket, mfe_kcal_mol)``.

    Parameters
    ----------
    sequence:
        RNA sequence over {A, U, G, C}.
    temperature:
        Folding temperature in °C (passed to RNAfold via ``-T``).

    Returns
    -------
    dot_bracket:
        MFE secondary structure in dot-bracket notation.
    mfe:
        Minimum free energy in kcal/mol (negative = stable fold).

    Raises
    ------
    RuntimeError
        If RNAfold is not found on ``$PATH`` or exits with a non-zero code.
    """
    cmd = ["RNAfold", "--noPS", f"-T{temperature:.1f}"]
    try:
        result = subprocess.run(
            cmd,
            input=sequence,
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "RNAfold not found. Install ViennaRNA and ensure it is on $PATH."
        ) from None
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"RNAfold failed (exit {exc.returncode}):\n{exc.stderr}"
        ) from exc

    return _parse_rnafold_output(result.stdout, len(sequence))


def _parse_rnafold_output(stdout: str, seq_len: int) -> tuple[str, float]:
    """Parse RNAfold stdout into ``(dot_bracket, mfe)``.

    RNAfold stdout format::

        <sequence>
        <dot_bracket> (<mfe>)
    """
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        # The structure line starts with a dot-bracket token of exactly seq_len chars.
        token = line.split()[0]
        if len(token) == seq_len and all(c in ".()" for c in token):
            # Energy is everything after the dot-bracket token, inside parentheses.
            rest = line[len(token):].strip()
            mfe_str = rest.strip("() ")
            try:
                mfe = float(mfe_str)
            except ValueError:
                mfe = 0.0
            return token, mfe

    raise RuntimeError(f"Could not parse RNAfold output:\n{stdout!r}")


# ---------------------------------------------------------------------------
# Dot-bracket utilities
# ---------------------------------------------------------------------------

def dot_bracket_to_pair_map(db: str) -> Dict[int, int]:
    """Convert a dot-bracket string to a ``{i: j, j: i}`` pair map (0-indexed).

    Supports standard ``()`` notation only (no pseudoknots).
    """
    stack: list[int] = []
    pair_map: Dict[int, int] = {}
    for i, c in enumerate(db):
        if c == "(":
            stack.append(i)
        elif c == ")":
            if stack:
                j = stack.pop()
                pair_map[j] = i
                pair_map[i] = j
    return pair_map


def pair_map_to_dot_bracket(pair_map: Dict[int, int], length: int) -> str:
    """Convert a ``{i: j}`` pair map back to a dot-bracket string.

    Parameters
    ----------
    pair_map:
        ``{i: j, j: i}`` as returned by ``dot_bracket_to_pair_map`` or
        ``evo_rna.core.structure.Structure.pair_map``.
    length:
        Total sequence length (number of nucleotides).
    """
    db = ["."] * length
    for i, j in pair_map.items():
        if i < j:
            db[i] = "("
            db[j] = ")"
    return "".join(db)


def bp_distance(db1: str, db2: str) -> int:
    """Symmetric base-pair distance between two dot-bracket structures.

    Counts the number of base pairs present in one structure but absent in the
    other (i.e. the size of the symmetric difference of the two pair sets).
    This is the standard metric used by RNAdistance.

    Parameters
    ----------
    db1, db2:
        Dot-bracket strings of equal length.

    Raises
    ------
    ValueError
        If the two strings differ in length.
    """
    if len(db1) != len(db2):
        raise ValueError(
            f"Dot-bracket lengths differ: {len(db1)} vs {len(db2)}"
        )
    pairs1 = frozenset(
        (i, j) for i, j in dot_bracket_to_pair_map(db1).items() if i < j
    )
    pairs2 = frozenset(
        (i, j) for i, j in dot_bracket_to_pair_map(db2).items() if i < j
    )
    return len(pairs1.symmetric_difference(pairs2))


def base_pair_mcc(db1: str, db2: str) -> float:
    """Matthews correlation coefficient between two structures' base-pair sets.

    Each of the ``N*(N-1)/2`` candidate position pairs (``N`` = sequence length)
    is a binary instance — paired or unpaired — with *db2* as the ground truth::

        TP = pairs in both                 FP = in db1 but not db2
        FN = in db2 but not db1            TN = candidate pairs in neither

        MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))

    Returns 0.0 when any marginal is zero (the denominator vanishes) — the
    conventional MCC fallback for a degenerate confusion matrix.

    Unlike :func:`bp_distance` this is *intensive*: it lives in [-1, 1]
    regardless of sequence length, which is why :class:`RNAfoldTerm` scores as
    ``1 - MCC`` rather than as a raw pair count.

    .. note::
       The standard base-pair MCC (the
       reported scMCC), duplicated so the MC hot path does not import the
       evaluation package.  Keep the two in sync.
    """
    if len(db1) != len(db2):
        raise ValueError(f"Dot-bracket lengths differ: {len(db1)} vs {len(db2)}")
    pairs1 = frozenset(
        (i, j) for i, j in dot_bracket_to_pair_map(db1).items() if i < j
    )
    pairs2 = frozenset(
        (i, j) for i, j in dot_bracket_to_pair_map(db2).items() if i < j
    )
    total = len(db1) * (len(db1) - 1) / 2.0
    tp = len(pairs1 & pairs2)
    fp = len(pairs1 - pairs2)
    fn = len(pairs2 - pairs1)
    tn = total - tp - fp - fn

    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denom <= 0:
        return 0.0
    return (tp * tn - fp * fn) / (denom ** 0.5)


# ---------------------------------------------------------------------------
# EnergyTerm subclass
# ---------------------------------------------------------------------------

class RNAfoldTerm(EnergyTerm):
    """Energy term driven by RNAfold MFE secondary structure prediction.

    Parameters
    ----------
    target_db:
        Target dot-bracket string.  When provided the term scores as
        ``1 - MCC(predicted, target)`` — zero is a perfect match, 1 is the
        no-better-than-chance baseline.  When ``None`` the term scores as
        ``-mfe / L``, rewarding stable folds.  Both forms are intensive.
    temperature:
        Folding temperature in °C forwarded to RNAfold (default 37.0).

    Examples
    --------
    Plug into an existing :class:`~evo_rna.monte_carlo.energy.EnergyFunction`::

        from evo_rna.monte_carlo.energy import EnergyFunction
        from evo_rna.monte_carlo.ss_rnafold import RNAfoldTerm

        efn = EnergyFunction.default()
        efn.terms["rnafold"] = (RNAfoldTerm(target_db="(((...)))"), 2.0)

    Or build one automatically from a :class:`~evo_rna.core.structure.Structure`::

        term = RNAfoldTerm.from_structure(struct, weight=2.0)
    """

    def __init__(
        self,
        target_db: Optional[str] = None,
        temperature: float = 37.0,
    ) -> None:
        self._target_db = target_db
        self._temperature = temperature
        # Small cache: sequence string → (dot_bracket, mfe).
        # Avoids redundant subprocess calls within a single MC step when both
        # score() and delta() inspect the same sequence.
        self._cache: dict[str, tuple[str, float]] = {}

    # ------------------------------------------------------------------
    # EnergyTerm interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "rnafold"

    def score(self, struct: Structure) -> float:
        """Full-sequence RNAfold energy for the current sequence."""
        return self._evaluate(struct.sequence)

    def delta(self, struct: Structure, pos: int, new_nuc: str) -> float:
        """Energy change if ``struct.sequence[pos]`` were mutated to *new_nuc*.

        Requires a full RNAfold call on the hypothetical mutant sequence.

        .. note::
           For pair-aware moves in :func:`~evo_rna.monte_carlo.mc.run_sa`,
           the MC driver sums ``delta(pos_i)`` + ``delta(pos_j)`` independently.
           The true delta for both mutations simultaneously may differ slightly.
        """
        seq_list = list(struct.sequence)
        seq_list[pos] = new_nuc
        mutant_seq = "".join(seq_list)
        return self._evaluate(mutant_seq) - self._evaluate(struct.sequence)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fold_cached(self, sequence: str) -> tuple[str, float]:
        if sequence not in self._cache:
            if len(self._cache) >= 4:
                oldest = next(iter(self._cache))
                del self._cache[oldest]
            self._cache[sequence] = fold(sequence, self._temperature)
        return self._cache[sequence]

    def _evaluate(self, sequence: str) -> float:
        """Scalar energy for *sequence* — intensive in both modes."""
        db, mfe = self._fold_cached(sequence)
        if self._target_db is not None:
            return 1.0 - base_pair_mcc(db, self._target_db)
        L = len(sequence)
        return -mfe / L if L else 0.0

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_structure(
        cls,
        struct: Structure,
        temperature: float = 37.0,
    ) -> "RNAfoldTerm":
        """Build a structure-matching term using the target encoded in *struct*.

        Converts ``struct.pair_map`` to dot-bracket notation and uses it as
        the reference structure.

        Parameters
        ----------
        struct:
            A fully populated :class:`~evo_rna.core.structure.Structure` whose
            ``pair_map`` encodes the desired secondary structure.
        temperature:
            RNAfold temperature in °C.
        """
        target_db = pair_map_to_dot_bracket(struct.pair_map, struct.length)
        return cls(target_db=target_db, temperature=temperature)


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # A short hairpin with a clear stem-loop.
    TEST_SEQ = "GGGAAACCC"
    # Known target for comparison (3-bp stem, 3-nt loop).
    TARGET_DB = "(((...)))"

    print(f"Sequence : {TEST_SEQ}")
    print(f"Target   : {TARGET_DB}")
    print()

    # --- raw fold ---
    db, mfe = fold(TEST_SEQ)
    print(f"RNAfold MFE structure : {db}")
    print(f"MFE                   : {mfe:.2f} kcal/mol")
    print()

    # --- utility checks ---
    pm = dot_bracket_to_pair_map(db)
    print(f"Pair map (0-indexed)  : {pm}")
    roundtrip = pair_map_to_dot_bracket(pm, len(TEST_SEQ))
    print(f"Pair map → dot-bracket: {roundtrip}")
    print(f"Roundtrip OK          : {roundtrip == db}")
    print()

    # --- base-pair distance vs target ---
    dist = bp_distance(db, TARGET_DB)
    print(f"BP distance to target : {dist}")
    print()

    # --- RNAfoldTerm (structure-matching mode) ---
    term_match = RNAfoldTerm(target_db=TARGET_DB)
    print(f"RNAfoldTerm (match) score vs target : {term_match._evaluate(TEST_SEQ):.1f}  (0 = perfect)")

    # --- RNAfoldTerm (MFE mode, no target) ---
    term_mfe = RNAfoldTerm()
    print(f"RNAfoldTerm (MFE)   score           : {term_mfe._evaluate(TEST_SEQ):.2f}  (= -MFE)")
