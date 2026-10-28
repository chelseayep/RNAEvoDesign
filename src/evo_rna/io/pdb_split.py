"""Split a multi-chain structure (PDB or mmCIF) into per-chain files and classify
each chain's type.

- Both ``.pdb`` and ``.cif`` input files are supported.  Per-chain output is
  written as PDB where possible; chains with multi-character IDs or residue
  numbers > 9999 (common in large CIF assemblies) fall back to ``.cif`` output.
- A structure with only one chain is handled without splitting; the single chain
  is written to ``<PDBID>_<chain>.pdb`` for consistency.
- DNA chains are skipped for RNA processing (a warning is logged).
- MIXED_RNA_DNA chains are written but flagged with a warning.
- Protein, ligand, and ion-only chains are written but classified accordingly.
- ``split_pdb`` is kept as a backward-compatible alias for ``split_structure``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from Bio.PDB import MMCIFIO, PDBIO, MMCIFParser, PDBParser, Select
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Residue name sets
# ---------------------------------------------------------------------------

RNA_RESNAMES: frozenset[str] = frozenset({
    "A", "U", "G", "C",                    # standard single-letter PDB names
    "ADE", "URA", "URI", "GUA", "CYT",     # three-letter variants
})

DNA_RESNAMES: frozenset[str] = frozenset({
    "DA", "DT", "DG", "DC", "DU",
})

PROTEIN_RESNAMES: frozenset[str] = frozenset({
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
    "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
    "THR", "TRP", "TYR", "VAL",
    # common non-standard residues
    "MSE", "SEP", "TPO", "PTR", "HSD", "HSE", "HSP",
})


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

class ChainType(Enum):
    RNA = "RNA"
    DNA = "DNA"
    PROTEIN = "protein"
    MIXED_RNA_DNA = "mixed_RNA_DNA"
    OTHER = "other"            # ligands, ions, water-only chains


@dataclass
class ChainInfo:
    """Metadata for one chain extracted from a multi-chain structure."""

    chain_id: str
    chain_type: ChainType
    #: Number of nucleotide / amino-acid residues (HETATM waters/ions excluded).
    n_residues: int
    #: Path to the per-chain file written by :func:`split_structure` (.pdb or .cif).
    pdb_path: Path


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

class _ChainSelect(Select):
    """BioPython PDBIO selector that keeps only one chain."""

    def __init__(self, chain_id: str) -> None:
        self._chain_id = chain_id

    def accept_chain(self, chain) -> bool:
        return chain.id == self._chain_id


def _needs_cif_format(chain) -> bool:
    """Return True if this chain cannot be represented in PDB format.

    PDB format constraints that trigger CIF fallback:
    - Chain ID longer than one character
    - Any residue sequence number outside [-999, 9999] (4-char field)

    BioPython's PDBIO may or may not raise on these violations depending on
    version; checking proactively avoids silently writing malformed files that
    downstream tools (e.g. RNAalign) cannot parse.
    """
    if len(chain.id) > 1:
        return True
    for res in chain.get_residues():
        seq_num = res.get_id()[1]
        if seq_num > 9999 or seq_num < -999:
            return True
    return False


def classify_chain(chain) -> tuple[ChainType, int]:
    """Classify a Bio.PDB ``Chain`` object and count its biological residues.

    Parameters
    ----------
    chain:
        A ``Bio.PDB.Chain.Chain`` instance.

    Returns
    -------
    (ChainType, n_residues)
        *n_residues* counts only nucleotide / amino-acid residues; waters,
        ions, and unrecognised HETATM records are excluded.
    """
    n_rna = n_dna = n_protein = 0

    for res in chain.get_residues():
        resname = res.get_resname().strip()
        if resname in RNA_RESNAMES:
            n_rna += 1
        elif resname in DNA_RESNAMES:
            n_dna += 1
        elif resname in PROTEIN_RESNAMES:
            n_protein += 1

    n_residues = n_rna + n_dna + n_protein

    if n_rna > 0 and n_dna == 0:
        chain_type = ChainType.RNA
    elif n_dna > 0 and n_rna == 0:
        chain_type = ChainType.DNA
    elif n_rna > 0 and n_dna > 0:
        chain_type = ChainType.MIXED_RNA_DNA
    elif n_protein > 0:
        chain_type = ChainType.PROTEIN
    else:
        chain_type = ChainType.OTHER

    return chain_type, n_residues


def _parse_entity_ids_cif(cif_path: Path) -> dict[str, int]:
    """Return a ``chain_id -> entity_id`` mapping from an mmCIF file.

    mmCIF groups chains by ``entity_id``, which plays the same role as
    ``MOL_ID`` in PDB ``COMPND`` records.  Multiple chains with the same
    entity_id represent different copies of the same molecule.

    Uses author chain IDs (``auth_asym_id``) to match BioPython's chain
    identifiers, which are set from the author records by default.
    """
    try:
        d = MMCIF2Dict(str(cif_path))
    except Exception as exc:
        log.warning("parse_mol_ids: could not read mmCIF dict from %s: %s", cif_path.name, exc)
        return {}

    auth_chains = d.get("_atom_site.auth_asym_id", [])
    entity_ids  = d.get("_atom_site.label_entity_id", [])

    if not auth_chains or not entity_ids:
        log.warning(
            "parse_mol_ids: _atom_site.auth_asym_id / label_entity_id not found in %s",
            cif_path.name,
        )
        return {}

    chain_to_entity: dict[str, int] = {}
    for auth_chain, eid in zip(auth_chains, entity_ids):
        if auth_chain not in chain_to_entity:
            try:
                chain_to_entity[auth_chain] = int(eid)
            except (ValueError, TypeError):
                log.warning(
                    "parse_mol_ids: unexpected entity_id %r for chain %s in %s",
                    eid, auth_chain, cif_path.name,
                )

    return chain_to_entity


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_mol_ids(path: str | Path) -> dict[str, int]:
    """Return a ``chain_id -> group_id`` mapping for deduplication.

    For PDB files this parses ``COMPND`` records (``MOL_ID``).
    For mmCIF files this reads ``_atom_site.label_entity_id``, which plays the
    same role — chains sharing a group_id are copies of the same molecule.

    Parameters
    ----------
    path:
        Path to a ``.pdb`` or ``.cif`` structure file.

    Returns
    -------
    dict[str, int]
        Maps each chain identifier (e.g. ``"A"``) to its group ID.
        Returns an empty dict when the relevant records are absent.
    """
    path = Path(path)

    if path.suffix.lower() == ".cif":
        return _parse_entity_ids_cif(path)

    # --- PDB: parse COMPND records ---
    chain_to_mol: dict[str, int] = {}
    compnd_lines: list[str] = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("COMPND"):
                compnd_lines.append(line[10:].strip())

    if not compnd_lines:
        return chain_to_mol

    full = " ".join(compnd_lines)
    current_mol_id: int | None = None
    for token in (t.strip() for t in full.split(";") if t.strip()):
        if ":" not in token:
            continue
        key, _, value = token.partition(":")
        key = key.strip().upper()
        value = value.strip()
        if key == "MOL_ID":
            try:
                current_mol_id = int(value)
            except ValueError:
                log.warning("parse_mol_ids: unexpected MOL_ID value %r in %s", value, path.name)
        elif key == "CHAIN" and current_mol_id is not None:
            for cid in value.split(","):
                cid = cid.strip()
                if cid:
                    chain_to_mol[cid] = current_mol_id

    return chain_to_mol


def split_structure(
    struct_path: str | Path,
    output_dir: str | Path,
) -> list[ChainInfo]:
    """Split *struct_path* (PDB or mmCIF) into one PDB file per chain.

    Output files are named ``<PDBID>_<chain_id>.pdb`` and written in PDB format
    where possible.  When a chain cannot be written as PDB (multi-character chain
    ID or residue number > 9999, common in large CIF assemblies), it is written
    as ``<PDBID>_<chain_id>.cif`` instead.  All downstream code uses
    :func:`~evo_rna.profile.utils._get_parser` so both formats are readable.

    Each chain is written regardless of type so downstream tools can inspect
    the full structure.  Use ``ChainInfo.chain_type`` to filter to RNA chains
    before calling :func:`~evo_rna.profile.features.build_structure`.

    Parameters
    ----------
    struct_path:
        Path to the input structure file (``.pdb`` or ``.cif``).
    output_dir:
        Directory where per-chain PDB files will be written.  Created if it
        does not exist.

    Returns
    -------
    list[ChainInfo]
        One entry per chain, in the order they appear in the structure.

    Raises
    ------
    ValueError
        If the structure contains no chains at all.
    """
    struct_path = Path(struct_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdb_id = struct_path.stem.upper()

    if struct_path.suffix.lower() == ".cif":
        parser = MMCIFParser(QUIET=True)
        log.debug("split_structure: reading %s as mmCIF", struct_path.name)
    else:
        parser = PDBParser(QUIET=True)
        log.debug("split_structure: reading %s as PDB", struct_path.name)

    structure = parser.get_structure(pdb_id, str(struct_path))
    model = structure[0]

    chains = list(model.get_chains())
    if not chains:
        raise ValueError(f"No chains found in {struct_path}")

    log.info(
        "split_structure: %s — %d chain(s): %s",
        struct_path.name,
        len(chains),
        [c.id for c in chains],
    )

    pdb_io = PDBIO()
    pdb_io.set_structure(structure)
    cif_io = MMCIFIO()
    cif_io.set_structure(structure)

    results: list[ChainInfo] = []
    for chain in chains:
        cid = chain.id
        chain_type, n_residues = classify_chain(chain)

        # Use CIF if the chain exceeds PDB format limits; otherwise use PDB.
        # The check is proactive rather than relying on PDBIO raising an
        # exception — older BioPython versions silently write malformed residue
        # numbers that downstream tools (e.g. RNAalign) cannot parse.
        if _needs_cif_format(chain):
            out_path = output_dir / f"{pdb_id}_{cid}.cif"
            cif_io.save(str(out_path), _ChainSelect(cid))
            log.debug(
                "  chain %s: PDB format limit exceeded — written as CIF (%s)",
                cid, out_path.name,
            )
        else:
            out_path = output_dir / f"{pdb_id}_{cid}.pdb"
            try:
                pdb_io.save(str(out_path), _ChainSelect(cid))
            except Exception as exc:
                # Last-resort fallback for any other PDBIO failure.
                out_path = output_dir / f"{pdb_id}_{cid}.cif"
                cif_io.save(str(out_path), _ChainSelect(cid))
                log.debug(
                    "  chain %s: PDBIO raised %s — written as CIF (%s)",
                    cid, exc, out_path.name,
                )

        if chain_type == ChainType.RNA:
            log.info("  chain %s → %s  [RNA, %d nt]", cid, out_path.name, n_residues)
        elif chain_type == ChainType.DNA:
            log.warning(
                "  chain %s → %s  [DNA, %d nt] — will be excluded from RNA processing",
                cid, out_path.name, n_residues,
            )
        elif chain_type == ChainType.MIXED_RNA_DNA:
            log.warning(
                "  chain %s → %s  [MIXED RNA+DNA, %d residues] — treating as RNA; "
                "DNA backbone geometry differs and results may be unreliable",
                cid, out_path.name, n_residues,
            )
        elif chain_type == ChainType.PROTEIN:
            log.info(
                "  chain %s → %s  [protein, %d aa] — skipped for RNA processing",
                cid, out_path.name, n_residues,
            )
        else:
            log.debug("  chain %s → %s  [other — likely ligand/ion/water]", cid, out_path.name)

        results.append(ChainInfo(
            chain_id=cid,
            chain_type=chain_type,
            n_residues=n_residues,
            pdb_path=out_path,
        ))

    rna_count = sum(1 for c in results if c.chain_type in (ChainType.RNA, ChainType.MIXED_RNA_DNA))
    log.info("split_structure: %d RNA chain(s) available for processing", rna_count)

    return results


# Backward-compatible alias — existing code that imports split_pdb continues to work.
split_pdb = split_structure
