from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser

from evo_rna.core.structure import Atom, Structure

log = logging.getLogger(__name__)


def _get_parser(path: str) -> PDBParser | MMCIFParser:
    """Return a BioPython parser appropriate for *path* (.pdb → PDBParser, .cif → MMCIFParser)."""
    if str(path).lower().endswith(".cif"):
        return MMCIFParser(QUIET=True)
    return PDBParser(QUIET=True)


def _resolve_chain(model, chain_id: str | None) -> str:
    """Return *chain_id* if present; otherwise return the first chain in *model*.

    Raises ValueError if the model contains no chains at all.
    """
    available = [c.id for c in model.get_chains()]
    if not available:
        raise ValueError("PDB model contains no chains.")
    if chain_id is not None and chain_id in available:
        return chain_id
    if chain_id is not None:
        import logging
        logging.getLogger(__name__).warning(
            "Chain '%s' not found in PDB (available: %s); using '%s'.",
            chain_id, available, available[0],
        )
    return available[0]


# RNA residue names only.  DNA variants (DA, DT, DG, DC, DU) are intentionally
# excluded — DNA backbone geometry (no 2'-OH) differs from RNA and must not be
# silently treated as RNA.  Use evo_rna.io.pdb_split to classify chains before
# calling extract_sequence_and_atoms.
RESNAME_TO_NUC = {
    "A": "A", "ADE": "A",
    "U": "U", "URA": "U", "URI": "U",
    "G": "G", "GUA": "G",
    "C": "C", "CYT": "C",
}

# DNA residue names — used for detection / warnings only.
DNA_RESNAMES: frozenset[str] = frozenset({"DA", "DT", "DG", "DC", "DU"})


def extract_sequence_and_atoms(
    pdb_path: str,
    chain_id: str | None = None,
) -> tuple[str, list[Atom], Dict[int, Dict[str, np.ndarray]]]:
    """Parse a PDB or mmCIF file and return (sequence, flat atom list, per-residue atom coords).

    Only standard nucleotide residues are kept (water / ions / ligands skipped).
    Residues are re-indexed 0 … L-1 regardless of PDB numbering.

    If *chain_id* is None or not present in the file the first available chain
    is used automatically.
    """
    struct = _get_parser(pdb_path).get_structure(Path(pdb_path).stem, pdb_path)
    model = struct[0]
    chain = model[_resolve_chain(model, chain_id)]

    sequence_chars: list[str] = []
    all_atoms: list[Atom] = []
    atom_coords: Dict[int, Dict[str, np.ndarray]] = {}

    # Warn once if the chain contains DNA residues — they are skipped below
    # because DNA geometry (no 2'-OH) is incompatible with RNA features.
    # Run evo_rna.io.pdb_split.split_pdb first to classify chains properly.
    dna_found = [
        r.get_resname().strip()
        for r in chain.get_residues()
        if r.get_resname().strip() in DNA_RESNAMES
    ]
    if dna_found:
        log.warning(
            "extract_sequence_and_atoms: chain '%s' in %s contains %d DNA residue(s) "
            "(%s) — DNA residues are skipped.  Use split_pdb to classify chains before "
            "processing.",
            chain.id, Path(pdb_path).name, len(dna_found), sorted(set(dna_found)),
        )

    idx = 0
    for residue in chain.get_residues():
        resname = residue.get_resname().strip()
        nuc = RESNAME_TO_NUC.get(resname)
        if nuc is None:
            continue

        sequence_chars.append(nuc)
        residue_coords: Dict[str, np.ndarray] = {}

        for bio_atom in residue.get_atoms():
            name = bio_atom.get_name()
            element = bio_atom.element.strip() if bio_atom.element else name[0]
            x, y, z = bio_atom.get_vector().get_array()
            all_atoms.append(Atom(name=name, element=element, x=float(x), y=float(y), z=float(z)))
            residue_coords[name] = np.array([x, y, z], dtype=np.float64)

        atom_coords[idx] = residue_coords
        idx += 1

    return "".join(sequence_chars), all_atoms, atom_coords


def _normalize_chains(chains: "str | tuple[str, ...] | None") -> "tuple[str, ...] | None":
    """Normalize a chain spec to an ordered tuple of chain IDs.

    A hyphen-separated string (the RNAsolo / gRNAde-DAS multi-chain convention,
    e.g. ``"D-C-A-B"``) is split into ``("D", "C", "A", "B")`` so a multi-chain
    design unit is extracted chain-by-chain in the given order.  A plain string
    (``"A"``) becomes a single-element tuple; an iterable is returned as-is.
    """
    if chains is None:
        return None
    if isinstance(chains, str):
        return tuple(c for c in chains.split("-") if c)
    return tuple(chains)


def strand_breaks_from_keys(keys: "list[tuple[str, int]]") -> list[int]:
    """0-indexed positions where a new chain begins (chain boundaries).

    From the ordered ``[(chain, resseq), ...]`` keys of a design unit, return the
    local indices at which the chain changes (excluding 0).  E.g. an 11+11 duplex
    gives ``[11]``.  Single-chain units return ``[]``.  Used to insert ViennaRNA
    strand separators (``&``) for cofolding multi-strand units.
    """
    breaks: list[int] = []
    for i in range(1, len(keys)):
        if keys[i][0] != keys[i - 1][0]:
            breaks.append(i)
    return breaks


def insert_strand_breaks(sequence: str, breaks: "list[int]", sep: str = "&") -> str:
    """Insert *sep* into *sequence* at each 0-indexed boundary in *breaks*.

    ``insert_strand_breaks("AAAUUU", [3])`` → ``"AAA&UUU"``.  No-op when *breaks*
    is empty.
    """
    if not breaks:
        return sequence
    out: list[str] = []
    prev = 0
    for b in breaks:
        out.append(sequence[prev:b])
        prev = b
    out.append(sequence[prev:])
    return sep.join(out)


def extract_chain_unit(
    struct_path: str,
    chains: "str | tuple[str, ...] | None" = None,
) -> tuple[str, list[Atom], Dict[int, Dict[str, np.ndarray]], list[tuple[str, int]]]:
    """Extract a design unit spanning one or more chains, with residue keys.

    Returns ``(sequence, atoms, atom_coords, keys)`` where ``keys[i] = (chain_id,
    resnum)`` is the PDB chain + residue number for local position *i*.  These
    keys let DSSR annotations (which reference ``chain.resnum``) be aligned onto
    the local 0…L-1 indexing.

    Chains are processed in the given order; for a multi-chain unit (e.g. a
    duplex) the residues are concatenated chain-by-chain.  ``chains=None`` uses
    the first chain in the file.
    """
    chains = _normalize_chains(chains)
    model = _get_parser(struct_path).get_structure(Path(struct_path).stem, struct_path)[0]
    if chains is None:
        chains = (_resolve_chain(model, None),)

    sequence_chars: list[str] = []
    all_atoms: list[Atom] = []
    atom_coords: Dict[int, Dict[str, np.ndarray]] = {}
    keys: list[tuple[str, int]] = []

    idx = 0
    for chain_id in chains:
        if chain_id not in [c.id for c in model.get_chains()]:
            log.warning("extract_chain_unit: chain '%s' not in %s — skipped",
                        chain_id, Path(struct_path).name)
            continue
        for residue in model[chain_id].get_residues():
            if residue.get_id()[0] != " ":   # skip HETATM (ligands, ions, water)
                continue
            nuc = RESNAME_TO_NUC.get(residue.get_resname().strip())
            if nuc is None:
                continue
            sequence_chars.append(nuc)
            keys.append((chain_id, residue.get_id()[1]))
            residue_coords: Dict[str, np.ndarray] = {}
            for bio_atom in residue.get_atoms():
                name = bio_atom.get_name()
                element = bio_atom.element.strip() if bio_atom.element else name[0]
                x, y, z = bio_atom.get_vector().get_array()
                all_atoms.append(Atom(name=name, element=element, x=float(x), y=float(y), z=float(z)))
                residue_coords[name] = np.array([x, y, z], dtype=np.float64)
            atom_coords[idx] = residue_coords
            idx += 1

    return "".join(sequence_chars), all_atoms, atom_coords, keys


if __name__ == "__main__":
    seq, atoms, coords = extract_sequence_and_atoms(
        "data/test/PDB_00009PDG_2_A.pdb", "A"
    )
    print(f"sequence ({len(seq)} nt): {seq}")
    print(f"atoms: {len(atoms)}")
    print(f"residues with coords: {len(coords)}")
