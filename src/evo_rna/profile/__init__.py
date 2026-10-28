"""Profile module — structural-alignment PSSM and feature extraction.

Key entry points
----------------
features.build_structure   PDB + prf.txt → fully populated core.Structure
pssm.compute_M             alignment hits → (L,4) log-odds PSSM
pssm.score_sequence        candidate sequence → scalar PSSM score
"""

from .pssm import (
    RnaAlignmentHit,
    Alignment,
    compute_background,
    compute_M,
    compute_w,
    fetch_top_hits,
    format_msa_txt,
    load_prf,
    parse_msa,
    score_sequence,
    write_msa_txt,
    write_prf,
)

from .features import (
    build_structure,
    compute_pssm,
    extract_sequence_and_atoms,
    extract_ss_and_pairs,
    extract_torsions,
    load_pssm_from_prf,
)

__all__ = [
    # PSSM
    "RnaAlignmentHit",
    "Alignment",
    "compute_background",
    "compute_M",
    "compute_w",
    "fetch_top_hits",
    "format_msa_txt",
    "load_prf",
    "parse_msa",
    "score_sequence",
    "write_msa_txt",
    "write_prf",
    # Feature extraction
    "build_structure",
    "compute_pssm",
    "extract_sequence_and_atoms",
    "extract_torsions",
    "load_pssm_from_prf",
    "extract_ss_and_pairs",
]
