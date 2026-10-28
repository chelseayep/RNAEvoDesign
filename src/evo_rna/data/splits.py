"""Load published gRNAde train/val/test splits.

EvoDesign evaluates on the gRNAde splits so results are directly comparable to
gRNAde and related inverse-folding methods.  Propensity tables and any tuning
must be derived from the **train** subset only; the **test** subset is held out.

gRNAde ID format:  {PDB_ID}_{MODEL}_{CHAIN(S)}
    6Q97_1_4        single chain "4"
    4V6T_1_AV       single chain named "AV"
    3B58_1_B-C-A    one design unit spanning chains B, C, A

Chains are the design unit: a multi-chain ID is extracted (and DSSR-annotated)
together, which is how duplex/complex targets keep their inter-chain pairing.

Vendored split files live in ``evo_rna/data/splits/``:
  grnade_das        — gRNAde diversity-augmented split
  grnade_structsim  — gRNAde structure-similarity split
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

SPLITS_DIR = Path(__file__).parent / "splits"

# Filenames inside each gRNAde split directory.
_GRNADE_FILES = {
    "grnade_das": {
        "train": "train_ids_das.txt",
        "val": "val_ids_das.txt",
        "test": "test_ids_das.txt",
    },
    "grnade_structsim": {
        "train": "train_ids_structsim_v2.txt",
        "val": "val_ids_structsim_v2.txt",
        "test": "test_ids_structsim_v2.txt",
    },
}


@dataclass(frozen=True)
class StructureID:
    """A design target: a PDB accession plus the chain(s) forming one unit."""

    pdb_id: str                    # upper-cased PDB accession
    chains: tuple[str, ...]        # one or more chain IDs (original case)
    model: Optional[int] = field(default=None, compare=False, hash=False)

    @classmethod
    def from_grnade(cls, raw: str) -> "StructureID":
        """Parse gRNAde format ``PDB_MODEL_CHAIN(S)`` (chains split on ``-``)."""
        parts = raw.strip().split("_", 2)
        if len(parts) != 3:
            raise ValueError(f"Cannot parse gRNAde ID: {raw!r}")
        pdb_id, model_str, chain_str = parts
        return cls(
            pdb_id=pdb_id.upper(),
            chains=tuple(chain_str.split("-")),
            model=int(model_str),
        )

    def __str__(self) -> str:
        prefix = f"{self.pdb_id}_{self.model}" if self.model is not None else self.pdb_id
        return f"{prefix}_{'-'.join(self.chains)}"


def load_grnade(name: str = "grnade_structsim") -> dict[str, list[StructureID]]:
    """Load a gRNAde split. Returns ``{'train', 'val', 'test'}`` -> [StructureID]."""
    if name not in _GRNADE_FILES:
        raise ValueError(f"Unknown gRNAde split {name!r}. Choose from {list(_GRNADE_FILES)}")
    split_dir = SPLITS_DIR / name
    result: dict[str, list[StructureID]] = {}
    for subset, fname in _GRNADE_FILES[name].items():
        ids = [
            StructureID.from_grnade(line)
            for line in (split_dir / fname).read_text().splitlines()
            if line.strip()
        ]
        result[subset] = ids
    return result


def all_ids(split: dict[str, list[StructureID]]) -> list[StructureID]:
    """Flat list of all IDs across a loaded split dict."""
    return split["train"] + split["val"] + split["test"]


def unique_pdb_ids(split: dict[str, list[StructureID]]) -> set[str]:
    """Unique PDB accessions across all subsets (ignores chain/model)."""
    return {sid.pdb_id for sid in all_ids(split)}
