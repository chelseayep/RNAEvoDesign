"""RNA structure representation for EvoDesign.

A single flat dataclass that holds everything the Monte Carlo sequence
design loop needs: identity, sequence, per-position structural properties
(extracted once from DSSR / Biopython), and precomputed scoring inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np

# Nucleotide alphabet used throughout — order matches PSSM columns.
NUC_ALPHABET = ("A", "U", "G", "C")
NUC_TO_INDEX: Dict[str, int] = {n: i for i, n in enumerate(NUC_ALPHABET)}


@dataclass
class Atom:
    """Lightweight atom — just what's needed for coordinate-level operations."""

    name: str          # e.g. "C1'", "N1", "P"
    element: str       # e.g. "C", "N", "P", "O"
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    @property
    def xyz(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float64)


@dataclass
class Structure:
    """Flat representation of an RNA structure for sequence design.

    All per-position arrays are indexed 0 … length-1 and are **read-only
    during the MC run** (the backbone never moves; only the sequence
    changes).
    """

    # --- Identity ---
    pdb_id: str = ""
    chain_id: str = ""

    # --- Sequence (mutable during MC) ---
    sequence: str = ""

    @property
    def length(self) -> int:
        return len(self.sequence)

    # --- Per-position structural context (set once at load time) ---

    # Secondary structure label per position: 0 = unpaired, 1 = paired.
    ss: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int8))

    # DSSR secondary-structure element class per position.
    # Codes (see profile.dssr.SS_ELEMENTS):
    #   0 = other/unpaired-dangling, 1 = stem, 2 = hairpin,
    #   3 = internal loop, 4 = bulge, 5 = junction.
    # Empty array when no DSSR annotation is available.
    ss_element: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int8))

    # Base-pair map: position -> paired position (only for paired residues).
    # Includes non-canonical pairs when DSSR-derived.
    pair_map: Dict[int, int] = field(default_factory=dict)

    # Full DSSR base-pair contact list: (i, j, lw) for every annotated pair —
    # canonical, non-canonical and base-multiplet partners — keeping the
    # Leontis-Westhof code.  Unlike pair_map this is NOT collapsed to one partner
    # per position; it feeds the H-bond satisfaction term.  Empty when no DSSR.
    contacts: list = field(default_factory=list)  # list[tuple[int, int, str]]

    # Solvent-accessible surface area per nucleotide (Å²).
    sa: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))

    # Backbone torsion angles per position: α, β, γ, δ, ε, ζ, χ (radians).
    torsions: np.ndarray = field(default_factory=lambda: np.empty((0, 7), dtype=np.float64))

    # --- Atomic coordinates (optional, for fragment / clash terms) ---

    # Per-residue atom lookup: position -> {atom_name: xyz array}.
    atom_coords: Dict[int, Dict[str, np.ndarray]] = field(default_factory=dict)

    # Flat atom list (kept for PDB round-tripping / visualisation).
    atoms: list = field(default_factory=list)  # list[Atom]

    # --- Precomputed scoring inputs ---

    # Position-specific scoring matrix (log-odds), shape (L, 4).
    # Column order follows NUC_ALPHABET: A=0, U=1, G=2, C=3.
    pssm: np.ndarray = field(default_factory=lambda: np.empty((0, 4), dtype=np.float64))

    # Background nucleotide frequencies, shape (4,).
    background: np.ndarray = field(
        default_factory=lambda: np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)
    )

    # --- Homology metadata ---

    # Number of homologous structures used to build the PSSM (after TM-score
    # filtering and self-match removal).  0 when no profile was available.
    n_alignments: int = 0

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def nuc_index(self, pos: int) -> int:
        """Return the alphabet index (0-3) of the nucleotide at *pos*."""
        return NUC_TO_INDEX.get(self.sequence[pos], -1)

    def paired_with(self, pos: int) -> Optional[int]:
        """Return the paired partner of *pos*, or None if unpaired."""
        return self.pair_map.get(pos)

    def is_paired(self, pos: int) -> bool:
        return pos in self.pair_map

    def get_atom_xyz(self, pos: int, atom_name: str) -> Optional[np.ndarray]:
        """Look up coordinates of a specific atom at a given residue position."""
        residue_atoms = self.atom_coords.get(pos)
        if residue_atoms is None:
            return None
        return residue_atoms.get(atom_name)

    def mutate(self, pos: int, new_nuc: str) -> None:
        """Swap the nucleotide at *pos* (the fundamental MC move)."""
        seq = list(self.sequence)
        seq[pos] = new_nuc
        self.sequence = "".join(seq)

    def copy(self) -> "Structure":
        """Deep copy suitable for storing best-seen states during MC."""
        return Structure(
            pdb_id=self.pdb_id,
            chain_id=self.chain_id,
            sequence=self.sequence,
            ss=self.ss.copy(),
            ss_element=self.ss_element.copy(),
            pair_map=dict(self.pair_map),
            contacts=list(self.contacts),
            sa=self.sa.copy(),
            torsions=self.torsions.copy(),
            atom_coords={
                pos: {name: xyz.copy() for name, xyz in atoms.items()}
                for pos, atoms in self.atom_coords.items()
            },
            atoms=[Atom(a.name, a.element, a.x, a.y, a.z) for a in self.atoms],
            pssm=self.pssm.copy(),
            background=self.background.copy(),
            n_alignments=self.n_alignments,
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> list[str]:
        """Return a list of problems (empty = OK)."""
        problems: list[str] = []
        L = self.length
        if L == 0:
            problems.append("sequence is empty")
            return problems
        if self.ss.shape != (L,):
            problems.append(f"ss shape {self.ss.shape} != ({L},)")
        if self.ss_element.size and self.ss_element.shape != (L,):
            problems.append(f"ss_element shape {self.ss_element.shape} != ({L},)")
        if self.sa.shape != (L,):
            problems.append(f"sa shape {self.sa.shape} != ({L},)")
        if self.torsions.shape != (L, 7):
            problems.append(f"torsions shape {self.torsions.shape} != ({L}, 7)")
        if self.pssm.shape != (L, 4):
            problems.append(f"pssm shape {self.pssm.shape} != ({L}, 4)")
        if self.background.shape != (4,):
            problems.append(f"background shape {self.background.shape} != (4,)")
        bad_nucs = [c for c in self.sequence if c not in NUC_TO_INDEX]
        if bad_nucs:
            problems.append(f"non-standard nucleotides: {set(bad_nucs)}")
        for pos, partner in self.pair_map.items():
            if self.pair_map.get(partner) != pos:
                problems.append(f"pair_map asymmetry: {pos}->{partner}")
                break
        return problems
