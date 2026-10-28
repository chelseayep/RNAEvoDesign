"""Knowledge-based propensity tables for the structural energy terms.

A propensity table is ``log P(nucleotide | structural-context class)`` counted
from a structure set (see the propensity-table derivation project) with
pseudocounts.  The SA, torsion, and junction/loop-context terms consult these
tables on every MC move via an O(1) lookup.

The context **discretizations live here** so the derivation script and the
energy terms assign identical classes.  Columns follow ``NUC_ALPHABET``
(A, U, G, C).

When no table artifact is available the tables are uniform (all zeros), so the
terms are well-defined no-ops until a real table is derived.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from evo_rna.core.structure import NUC_ALPHABET
from evo_rna.profile.dssr import SS_ELEMENTS

log = logging.getLogger(__name__)

N_NUC = len(NUC_ALPHABET)

# --- Solvent-accessibility classes (absolute SASA, Å²) ----------------------
# Provisional bounds; the derivation step may refine them (and must use the
# same sasa_to_class so train counts match inference).
SASA_CLASS_BOUNDS = (30.0, 90.0)
SASA_CLASSES = ("buried", "intermediate", "exposed")
N_SA = len(SASA_CLASSES)


def sasa_to_class(sasa: float) -> int:
    """Discretize absolute SASA (Å²) into 0=buried, 1=intermediate, 2=exposed."""
    if sasa is None or (isinstance(sasa, float) and np.isnan(sasa)):
        return 1
    if sasa < SASA_CLASS_BOUNDS[0]:
        return 0
    if sasa < SASA_CLASS_BOUNDS[1]:
        return 1
    return 2


# --- Torsion (glycosidic χ) bins, in DEGREES --------------------------------
# extract_torsions returns degrees; purine anti ~ -160°, pyrimidine anti ~ -120°.
CHI_BOUNDARY_DEG = -140.0
TORSION_BINS = ("chi_low", "chi_high")
N_TOR = len(TORSION_BINS)
CHI_INDEX = 6  # χ column in the (L, 7) torsion array


def chi_to_bin(chi_deg: float) -> int:
    """Bin the glycosidic χ angle (degrees). NaN/undefined → bin 0."""
    if chi_deg is None or np.isnan(chi_deg):
        return 0
    return 0 if chi_deg < CHI_BOUNDARY_DEG else 1


# --- Junction/loop element classes --------------------------------------------
# Reuse the DSSR element vocabulary directly (profile.dssr.SS_ELEMENTS):
#   0 other, 1 stem, 2 hairpin, 3 internal, 4 bulge, 5 junction.
N_ELEM = len(SS_ELEMENTS)


@dataclass
class PropensityTables:
    """log P(nuc | class) tables, columns = A,U,G,C (NUC_ALPHABET order)."""

    sa: np.ndarray        # (N_SA, 4)
    torsion: np.ndarray   # (N_TOR, 4)
    junction: np.ndarray  # (N_ELEM, 4)

    @classmethod
    def uniform(cls) -> "PropensityTables":
        return cls(
            sa=np.zeros((N_SA, N_NUC)),
            torsion=np.zeros((N_TOR, N_NUC)),
            junction=np.zeros((N_ELEM, N_NUC)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "PropensityTables":
        d = np.load(path)
        return cls(sa=d["sa"], torsion=d["torsion"], junction=d["junction"])

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, sa=self.sa, torsion=self.torsion, junction=self.junction)


def load_default_tables(path: str | Path | None = None) -> PropensityTables:
    """Load propensity tables from *path* (or config ``propensity_table``).

    Falls back to uniform (no-op) tables when the artifact is absent — so the
    structural terms are usable before the derivation project has produced one.
    """
    if path is None:
        try:
            from evo_rna.config import PATHS
            path = getattr(PATHS, "propensity_table", None)
        except Exception:
            path = None
    if path and Path(path).exists():
        log.info("Loaded propensity tables from %s", path)
        return PropensityTables.load(path)
    log.info("No propensity table at %s — using uniform (no-op) tables.", path)
    return PropensityTables.uniform()
