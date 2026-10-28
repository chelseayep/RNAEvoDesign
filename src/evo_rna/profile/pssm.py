"""PSSM computation from RNA structural alignments.

RnaAlignmentHit  →  compute_w (frequencies)  →  compute_M (log-odds)

``RnaAlignmentHit`` is the single record type produced by RNAalign.
Use in-memory hits to feed ``compute_M`` directly, or persist with
``write_msa_txt`` / ``parse_msa`` for debugging.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Union

import numpy as np

NUCLEOTIDES = ["a", "u", "g", "c"]
NUC_IDX = {n: i for i, n in enumerate(NUCLEOTIDES)}


@dataclass
class RnaAlignmentHit:
    """One RNAalign hit: gapped alignment rows + metadata."""

    query_id: str
    target_id: str
    tm_score: float
    seq_id: float = 0.0
    rmsd: float = 0.0
    aln: int = 0
    query_row: str = ""
    target_row: str = ""
    query_aligned: str = ""
    template_aligned: str = ""


Alignment = RnaAlignmentHit


def projected_query_sequence(query_row: str, target_row: str) -> str:
    """Project gapped alignment onto query coordinates (template gap columns dropped)."""
    return "".join(qc for tc, qc in zip(target_row, query_row) if tc != "-")


# ------------------------------------------------------------------
# MSA serialisation
# ------------------------------------------------------------------

def format_msa_txt(
    hits: Sequence[RnaAlignmentHit],
    *,
    include_seqid_in_header: bool = True,
) -> str:
    """Serialize hits to block ``msa.txt`` text (parseable by ``parse_msa``)."""
    chunks: List[str] = []
    for h in hits:
        head = (
            f">{h.query_id}|vs|{h.target_id}"
            f"|TM={h.tm_score:.5f}|RMSD={h.rmsd:.3f}|aln={h.aln}"
        )
        if include_seqid_in_header:
            head += f"|seqid={h.seq_id:.5f}"
        chunks.append(f"{head}\nquery   {h.query_row}\ntarget  {h.target_row}")
    return "\n\n".join(chunks) + "\n"


def write_msa_txt(path: Union[str, Path], hits: Sequence[RnaAlignmentHit], **kwargs) -> None:
    """Write ``format_msa_txt`` to *path*."""
    Path(path).write_text(format_msa_txt(hits, **kwargs))


def parse_msa(filepath: str) -> List[RnaAlignmentHit]:
    """Load block-format ``msa.txt`` written by ``format_msa_txt``."""
    with open(filepath) as f:
        content = f.read()

    alignments: List[RnaAlignmentHit] = []
    blocks = [b.strip() for b in content.split("\n\n") if b.strip()]

    for block in blocks:
        lines = block.strip().splitlines()
        header = lines[0].lstrip(">")
        parts: dict = {}
        for item in header.split("|"):
            if "=" in item:
                k, _, v = item.partition("=")
                parts[k.strip()] = v.strip()

        ids = header.split("|vs|")
        query_id = ids[0].strip()
        rest = ids[1] if len(ids) > 1 else ""
        target_id = rest.split("|")[0].strip() if rest else ""

        rows: dict = {}
        for line in lines[1:]:
            if not line.strip():
                continue
            label, seq = line.split(None, 1)
            rows[label] = seq.upper()

        def _float(key: str, default: float = 0.0) -> float:
            try:
                return float(parts.get(key, default))
            except (TypeError, ValueError):
                return default

        def _int(key: str, default: int = 0) -> int:
            try:
                return int(float(parts.get(key, default)))
            except (TypeError, ValueError):
                return default

        alignments.append(
            RnaAlignmentHit(
                query_id=query_id,
                target_id=target_id,
                tm_score=_float("TM"),
                rmsd=_float("RMSD"),
                aln=_int("aln"),
                seq_id=_float("seqid"),
                query_row=rows.get("query", ""),
                target_row=rows.get("target", ""),
            )
        )

    return alignments


def fetch_top_hits(
    alignments: List[RnaAlignmentHit],
    n: int | None = None,
) -> List[RnaAlignmentHit]:
    """Return alignment hits sorted by TM-score, excluding perfect self-matches.

    Parameters
    ----------
    alignments:
        Candidate hits (already filtered by TM-score cutoff).
    n:
        Maximum number of hits to return.  ``None`` (default) returns all
        hits above the cutoff — the recommended setting so that the PSSM
        uses the full homologous set rather than an arbitrary top-N slice.
    """
    top = []
    for a in sorted(alignments, key=lambda x: x.tm_score, reverse=True):
        if a.tm_score == 1.0 or a.seq_id == 1.0:
            continue
        top.append(a)
        if n is not None and len(top) >= n:
            break
    return top


# ------------------------------------------------------------------
# PSSM computation
# ------------------------------------------------------------------

def compute_w(
    alignments: List[RnaAlignmentHit],
    L: int,
    alpha: float = 0.5,
    weight_by_tm: bool = True,
) -> np.ndarray:
    """Position-specific nucleotide frequencies, shape (L, 4)."""
    counts = np.zeros((L, 4))
    weights = np.zeros(L)

    for hit in alignments:
        if L <= 0:
            break
        w = hit.tm_score if weight_by_tm else 1.0
        p = 0
        for q_char, t_char in zip(hit.query_row, hit.target_row):
            if q_char != "-":
                if t_char != "-" and t_char.lower() in NUC_IDX:
                    counts[p, NUC_IDX[t_char.lower()]] += w
                    weights[p] += w
                p += 1
                if p >= L:
                    break

    w_matrix = (counts + alpha) / (weights[:, None] + 4 * alpha)
    return w_matrix


def compute_background(alignments: List[RnaAlignmentHit], beta: float = 0.5) -> np.ndarray:
    """Background nucleotide frequencies across all aligned target positions.

    Laplace-smoothed (pseudocount *beta*) so no background probability is ever
    exactly zero.  Without this, a nucleotide unobserved across the (often few)
    aligned positions gives ``background = 0`` and ``log(w / 0) = +inf`` in
    :func:`compute_M`, which propagates to an ``-inf`` evolutionary energy for
    any sequence using that nucleotide.  Falls back to uniform when there are no
    aligned positions at all.
    """
    counts = np.zeros(4)
    for aln in alignments:
        for q_char, t_char in zip(aln.query_row, aln.target_row):
            if q_char != "-" and t_char != "-" and t_char.lower() in NUC_IDX:
                counts[NUC_IDX[t_char.lower()]] += 1
    total = counts.sum()
    if total == 0:
        return np.full(4, 0.25)
    return (counts + beta) / (total + 4 * beta)


def compute_M(
    alignments: List[RnaAlignmentHit],
    L: int,
    background: np.ndarray | None = None,
    clip: float = 10.0,
) -> np.ndarray:
    """Log-odds PSSM, shape (L, 4).  Column order: A U G C.

    Log-odds are clipped to ``[-clip, clip]`` so a single sparse position can
    never inject ``±inf`` into the evolutionary term (belt-and-suspenders on top
    of the smoothed :func:`compute_background`).
    """
    w = compute_w(alignments, L)
    if background is None:
        background = compute_background(alignments)
    return np.clip(np.log(w / background), -clip, clip)


def score_sequence(seq: str, M: np.ndarray) -> float:
    """Score a candidate RNA sequence against a PSSM."""
    return sum(
        M[p, NUC_IDX[nt]]
        for p, nt in enumerate(seq.upper())
        if nt in NUC_IDX
    )


# ------------------------------------------------------------------
# prf.txt I/O  (the "pos a u g c" text format)
# ------------------------------------------------------------------

def write_prf(M: np.ndarray, path: Union[str, Path]) -> None:
    """Write an (L, 4) PSSM to the ``prf.txt`` text format."""
    with open(path, "w") as f:
        f.write("pos\ta\tu\tg\tc\n")
        for i in range(M.shape[0]):
            f.write(f"{i + 1}\t{M[i, 0]:.6f}\t{M[i, 1]:.6f}\t{M[i, 2]:.6f}\t{M[i, 3]:.6f}\n")


def load_prf(path: Union[str, Path]) -> np.ndarray:
    """Load an (L, 4) PSSM from the ``prf.txt`` text format.

    Expected format: header line, then one row per position with
    ``pos  a  u  g  c`` (whitespace-separated).
    """
    rows: list[list[float]] = []
    with open(path) as f:
        header = f.readline()  # skip header
        for line in f:
            toks = line.split()
            if len(toks) < 5:
                continue
            rows.append([float(toks[1]), float(toks[2]), float(toks[3]), float(toks[4])])
    return np.array(rows, dtype=np.float64)
