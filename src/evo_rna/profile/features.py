"""Feature extraction to populate a core.Structure from an input PDB.

Each extractor fills one field of Structure.  Extractors that depend on
external tools (DSSR, FreeSASA, …) are placeholder stubs returning dummy
values — swap in real implementations as tools are integrated.

Typical usage
-------------
    struct = build_structure(
        "1ehz.pdb",
        chain_id="A",
        prf_path="data/test_output/prf.txt",
    )
    # struct is now fully populated and ready for MC
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Sequence

import numpy as np

from evo_rna.core.structure import NUC_ALPHABET, NUC_TO_INDEX, Atom, Structure
from evo_rna.io.pdb_split import ChainInfo, ChainType, parse_mol_ids, split_pdb

from .pssm import (
    RnaAlignmentHit,
    compute_background,
    compute_M,
    load_prf,
    parse_msa,
)
from .ss import extract_ss_and_pairs, save_ss
from .sa import extract_sasa, extract_sasa_unit, save_sa
from .torsions import extract_torsions, extract_torsions_unit, save_torsions
from .utils import extract_chain_unit, extract_sequence_and_atoms, _normalize_chains
from .dssr import contacts_aligned, ss_pairs_elements
log = logging.getLogger(__name__)



def load_pssm_from_prf(prf_path: str) -> np.ndarray:
    """Load PSSM directly from ``prf.txt`` produced by GenerateProfile.

    Returns (L, 4) log-odds matrix.
    """
    return load_prf(prf_path)


def compute_pssm(
    alignments: Sequence[RnaAlignmentHit],
    length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build PSSM and background from in-memory alignment hits.

    Returns (pssm, background).
    """
    alns = list(alignments)
    bg = compute_background(alns)
    M = compute_M(alns, length, bg)
    return M, bg



# ======================================================================
# Pipeline: build a fully populated Structure in one call
# ======================================================================


def save_metadata(output_dir: str, n_alignments: int) -> None:
    """Write per-chain metadata to ``metadata.json``.

    Currently records the number of homologous structures used for PSSM
    construction so it can be used as a covariate in downstream analysis.
    """
    import json
    meta = {"n_alignments": n_alignments}
    with open(Path(output_dir) / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


def build_structure(
    pdb_path: str,
    output_dir: str,
    chain_id: str | None = None,
    chains: "str | tuple[str, ...] | None" = None,
    prf_path: str | None = None,
    msa_path: str | None = None,
    alignments: Sequence[RnaAlignmentHit] | None = None,
    n_alignments: int = 0,
) -> Structure:
    """PDB or mmCIF → fully populated Structure ready for MC.

    PSSM source priority: prf_path > alignments > msa_path.

    Parameters
    ----------
    pdb_path : str
        Path to input PDB or mmCIF file.
    chain_id : str
        Chain to extract.
    prf_path : str, optional
        Path to ``prf.txt`` (preferred — avoids recomputation).
    msa_path : str, optional
        Path to ``msa.txt``.  PSSM will be recomputed from alignments.
    alignments : list[RnaAlignmentHit], optional
        Pre-loaded alignment hits.
    n_alignments : int, optional
        Number of homologous structures used to build the PSSM.  Stored as
        metadata on the Structure and written to ``metadata.json``.
    """
    pdb_id = Path(pdb_path).stem

    # Resolve the design unit: explicit `chains` wins, else single `chain_id`.
    chain_tuple = _normalize_chains(chains) or _normalize_chains(chain_id)

    # 1. Sequence + atoms + residue keys (multi-chain aware)
    sequence, atoms, atom_coords, keys = extract_chain_unit(pdb_path, chain_tuple)
    L = len(sequence)

    if L == 0:
        log.error("No residues extracted from %s chains %s", pdb_path, chain_tuple)
        return Structure(pdb_id=pdb_id, chain_id=chain_id or "")

    # 2. Secondary structure + base pairs + element labels (DSSR, geometry fallback)
    ss, pair_map, ss_element, ss_source = ss_pairs_elements(
        pdb_path, chain_tuple, keys, cache_dir=output_dir,
    )
    log.info("SS source: %s  (%d pairs, %d paired positions)", ss_source, len(pair_map) // 2, int(ss.sum()))

    # 2b. Full DSSR contact list (incl. non-canonical / multiplets) for the
    #     H-bond satisfaction term. Reuses the cached DSSR JSON; [] without DSSR.
    contacts = contacts_aligned(pdb_path, chain_tuple, keys, cache_dir=output_dir)

    # 3. Solvent accessibility + 4. backbone torsions, aligned to `keys` by
    #    (chain, resseq) rather than concatenated per chain.  The per-chain
    #    concatenation assumed every extractor selects the same residues, which
    #    they do not: extract_torsions takes residues with a C4', while
    #    extract_chain_unit takes those with a known RNA base, so a DNA or
    #    modified residue made the arrays different lengths and the guard below
    #    "aligned" them by truncation — shifting every torsion onto the wrong
    #    residue from the first extra entry on.  Key-based mapping cannot drift,
    #    and it runs FreeSASA once for the unit instead of once per chain.
    sa = extract_sasa_unit(pdb_path, keys)
    torsions = extract_torsions_unit(pdb_path, keys)

    # Guard against any residue-filtering disagreement between extractors so the
    # per-position arrays always align with the L residues in `keys`.
    if sa.shape[0] != L:
        log.warning("SASA length %d != %d residues — padding/truncating to align.", sa.shape[0], L)
        sa = np.resize(sa, L) if sa.size else np.zeros(L)
    if torsions.shape[0] != L:
        log.warning("torsions length %d != %d residues — padding/truncating.", torsions.shape[0], L)
        fixed = np.full((L, 7), np.nan)
        fixed[: min(L, torsions.shape[0])] = torsions[: min(L, torsions.shape[0])]
        torsions = fixed

    # 5. PSSM  (prf.txt preferred → no recomputation)
    pssm = np.zeros((L, 4), dtype=np.float64)
    background = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)

    if prf_path is not None:
        pssm = load_pssm_from_prf(prf_path)
        log.info("Loaded PSSM from %s  shape=%s", prf_path, pssm.shape)
    elif alignments is not None:
        pssm, background = compute_pssm(alignments, L)

    struct = Structure(
        pdb_id=pdb_id,
        chain_id=chain_id,
        sequence=sequence,
        ss=ss,
        ss_element=ss_element,
        pair_map=pair_map,
        contacts=contacts,
        sa=sa,
        torsions=torsions,
        atom_coords=atom_coords,
        atoms=atoms,
        pssm=pssm,
        background=background,
        n_alignments=n_alignments,
    )

    problems = struct.validate()
    if problems:
        for p in problems:
            log.warning("Structure validation: %s", p)
    ss_labels = [(c, sequence[i], r) for i, (c, r) in enumerate(keys)]
    save_ss(ss, pair_map, ss_labels, output_dir)
    save_sa(sa, output_dir)
    save_torsions(torsions, output_dir)
    save_metadata(output_dir, n_alignments)
    return struct

def build_structures_from_pdb(
    pdb_path: str,
    output_dir: str,
    prf_path: str | None = None,
    msa_path: str | None = None,
    alignments: Sequence[RnaAlignmentHit] | None = None,
    include_mixed: bool = False,
    deduplicate: bool = True,
) -> list[tuple[ChainInfo, "Structure"]]:
    """Split a (possibly multi-chain) PDB and build a Structure for each RNA chain.

    Each chain is written to ``<output_dir>/<PDBID>_<chain>.pdb`` before
    feature extraction.  Non-RNA chains (DNA, protein, ligand) are classified
    and skipped; a warning is logged for DNA chains.

    Parameters
    ----------
    pdb_path : str
        Path to the input PDB file.
    output_dir : str
        Directory for per-chain PDB files and feature outputs.
    prf_path : str, optional
        Path to a ``prf.txt`` PSSM file (applied to all chains; only useful
        when the input is a single-chain structure).
    msa_path : str, optional
        Path to ``msa.txt``; used when *alignments* is None and *prf_path*
        is None.
    alignments : list[RnaAlignmentHit], optional
        Pre-loaded alignment hits.
    include_mixed : bool
        If True, also process MIXED_RNA_DNA chains (default False — they are
        skipped with a warning).
    deduplicate : bool
        If True (default), skip chains that share a ``MOL_ID`` with an
        already-processed chain, using the ``COMPND`` header records.
        Useful for structures like 1ET4 that contain multiple copies of the
        same molecule.  Falls back to processing all chains when no ``COMPND``
        records are present.

    Returns
    -------
    list of (ChainInfo, Structure)
        One entry per processed RNA chain, in PDB chain order.  When
        *deduplicate* is True, only the first representative of each unique
        ``MOL_ID`` is included.
    """
    chains = split_pdb(pdb_path, output_dir)

    processable = {ChainType.RNA}
    if include_mixed:
        processable.add(ChainType.MIXED_RNA_DNA)

    # Build MOL_ID map once from the original (unsplit) PDB.
    mol_id_map: dict[str, int] = {}
    if deduplicate:
        mol_id_map = parse_mol_ids(pdb_path)
        if not mol_id_map:
            log.warning(
                "deduplicate=True but no COMPND records found in %s — "
                "all chains will be processed",
                Path(pdb_path).name,
            )

    results: list[tuple[ChainInfo, Structure]] = []
    seen_mol_ids: dict[int, str] = {}  # mol_id -> canonical chain_id

    for info in chains:
        if info.chain_type not in processable:
            if info.chain_type == ChainType.MIXED_RNA_DNA:
                log.warning(
                    "Skipping mixed RNA/DNA chain %s (pass include_mixed=True to process it).",
                    info.chain_id,
                )
            continue

        if deduplicate and mol_id_map:
            mol_id = mol_id_map.get(info.chain_id)
            if mol_id is not None:
                if mol_id in seen_mol_ids:
                    log.info(
                        "Skipping duplicate chain %s (MOL_ID %d, same as chain %s)",
                        info.chain_id, mol_id, seen_mol_ids[mol_id],
                    )
                    continue
                seen_mol_ids[mol_id] = info.chain_id

        log.info("Building Structure for chain %s from %s", info.chain_id, info.pdb_path.name)
        struct = build_structure(
            pdb_path=str(info.pdb_path),
            output_dir=output_dir,
            chain_id=info.chain_id,
            prf_path=prf_path,
            msa_path=msa_path,
            alignments=alignments,
        )
        results.append((info, struct))

    return results


if __name__ == "__main__":
    import sys
    pdb_path = sys.argv[1] if len(sys.argv) > 1 else "baseline/data/2GCS.pdb"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "data/test_output"
    structure = build_structure(pdb_path, output_dir)
