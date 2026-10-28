"""RNA per-nucleotide solvent-accessible surface area (SASA) from PDB coordinates.

Uses FreeSASA (Lee-Richards algorithm, probe radius 1.4 Å) to compute the
absolute SASA in Å² for each nucleotide residue.

Note
----
FreeSASA uses generic van-der-Waals radii that are not RNA-specific.  The
values are a reasonable baseline for benchmarking; an RNA-tailored predictor
can replace this module later.

Outputs
-------
sasa : (L,) float64 — absolute SASA in Å² per residue
"""

from __future__ import annotations

import copy
import logging
import string
from pathlib import Path

import numpy as np

from evo_rna.profile.utils import RESNAME_TO_NUC, _get_parser, _resolve_chain

log = logging.getLogger(__name__)

# FreeSASA builds its structure through a PDB-format representation, whose chain
# field is a SINGLE character.  A two-character chain id such as "Sm" is not
# merely dropped — it is collapsed onto "S", which in an entry that also has a
# real chain "S" silently merges two chains' areas.  Roughly a fifth of the
# gRNAde split units carry multi-character chain ids ("Sm", "RB", "g2", "1w"),
# and every one of them came back with SASA = 0 for every residue.
#
# So every chain is relabelled to a unique single character before the call and
# mapped back afterwards.  62 symbols are available; entries with more chains
# than that keep the unit plus its nearest neighbours (see _relabelled_model).
_FS_CHAIN_POOL = string.ascii_uppercase + string.ascii_lowercase + string.digits

# FreeSASA raises AssertionError ("Radius is <= 0") on residues its classifier
# has no radius for — UNK in particular — and the assertion kills the whole
# entry, not the offending residue.
#
# `skip-unknown` does NOT prevent this on the structureFromBioPDB path: 3KFU
# still fails with "Radius is <= 0 (-1.0) for the residue: UNK, atom: N" with
# the option set, because the option is honoured when FreeSASA parses a PDB
# file itself, not when atoms are handed to it one at a time from a BioPython
# model.  The offending residues are therefore detached from the copy before
# the call (see _drop_unclassifiable).  The options are kept as a second line
# of defence for any build that does respect them.
_FS_OPTIONS = {
    "hetatm": False,
    "hydrogen": False,
    "join-models": False,
    "skip-unknown": True,
    "halt-at-unknown": False,
}

# Residue names FreeSASA's default classifier cannot assign a radius to.  UNK is
# the common one — ribosome models trace unsequenced polypeptide as poly-UNK on
# ATOM records, so `hetatm: False` does not exclude it.
_FS_UNKNOWN_RESNAMES = frozenset({"UNK", "UNL", "UNX", "DN", "N"})


def _has_unclassifiable(model) -> bool:
    return any(res.get_resname().strip().upper() in _FS_UNKNOWN_RESNAMES
               for res in model.get_residues())


def _drop_unclassifiable(model) -> int:
    """Detach residues FreeSASA has no radius for.  Returns how many went."""
    n = 0
    for chain in list(model.get_chains()):
        for res in list(chain.get_residues()):
            if res.get_resname().strip().upper() in _FS_UNKNOWN_RESNAMES:
                chain.detach_child(res.get_id())
                n += 1
    return n


def _chain_centroid(chain) -> "np.ndarray | None":
    coords = [a.get_coord() for a in chain.get_atoms()]
    return np.mean(coords, axis=0) if coords else None


def _relabelled_model(bio_struct, needed: "tuple[str, ...]"):
    """A copy of the structure FreeSASA can consume, with single-char chain ids.

    Returns ``(structure_copy, fs_id_by_original)``.  Two things are made safe
    here.  Chain ids are relabelled to unique single characters, because
    FreeSASA round-trips through a PDB representation whose chain field is one
    character wide.  And residues its classifier has no radius for are detached,
    because the resulting assertion kills the entry outright.

    When the entry has more chains than the 62 available symbols — ribosome
    entries routinely do — the copy keeps the design unit's own chains plus the
    nearest others by centroid distance, so the burial that actually matters is
    still represented and the result stays interpretable.  Chains dropped that
    way are logged.
    """
    model = bio_struct[0]
    chains = list(model.get_chains())
    ids = [c.id for c in chains]

    labels_safe = (len(ids) <= len(_FS_CHAIN_POOL)
                   and all(len(i) == 1 for i in ids)
                   and len(set(ids)) == len(ids))
    if labels_safe and not _has_unclassifiable(model):
        return bio_struct, {i: i for i in ids}   # already safe — no copy needed

    keep = [c for c in chains if c.id in set(needed)]
    others = [c for c in chains if c.id not in set(needed)]
    room = len(_FS_CHAIN_POOL) - len(keep)
    if room < 0:
        raise ValueError(f"design unit spans {len(keep)} chains, more than FreeSASA can label")
    if len(others) > room:
        anchor = [_chain_centroid(c) for c in keep]
        anchor = [a for a in anchor if a is not None]
        if anchor:
            centre = np.mean(anchor, axis=0)
            def dist(c):
                cc = _chain_centroid(c)
                return float(np.linalg.norm(cc - centre)) if cc is not None else float("inf")
            others = sorted(others, key=dist)
        log.warning(
            "SASA: entry has %d chains, more than the %d FreeSASA labels — keeping the "
            "unit plus its %d nearest chains and dropping %d distant ones.",
            len(chains), len(_FS_CHAIN_POOL), room, len(others) - room,
        )
        others = others[:room]

    selected = keep + others
    struct_copy = copy.deepcopy(bio_struct)
    model_copy = struct_copy[0]
    for c in list(model_copy.get_chains()):
        if c.id not in {s.id for s in selected}:
            model_copy.detach_child(c.id)

    n_dropped = _drop_unclassifiable(model_copy)
    if n_dropped:
        log.warning("SASA: dropped %d residue(s) FreeSASA cannot classify (UNK and "
                    "the like); their neighbours' areas are computed without them.",
                    n_dropped)

    # Detach every chain before renaming: assigning ids in place would collide in
    # the parent's child_dict as soon as a new id matches an existing one.
    detached = []
    for c in list(model_copy.get_chains()):
        model_copy.detach_child(c.id)
        detached.append(c)
    fs_id_by_original: dict[str, str] = {}
    order = {s.id: n for n, s in enumerate(selected)}
    for c in sorted(detached, key=lambda ch: order.get(ch.id, 1 << 30)):
        original = c.id
        label = _FS_CHAIN_POOL[len(fs_id_by_original)]
        fs_id_by_original[original] = label
        c.id = label
        model_copy.add(c)
    return struct_copy, fs_id_by_original


def _freesasa_areas(bio_struct, needed: "tuple[str, ...]", params):
    """residueAreas() keyed by ORIGINAL chain id, or {} if FreeSASA fails."""
    import freesasa

    prepared, fs_id = _relabelled_model(bio_struct, needed)
    try:
        fs_struct = freesasa.structureFromBioPDB(prepared, options=_FS_OPTIONS)
        result = freesasa.calc(fs_struct, params)
    except Exception as exc:  # noqa: BLE001 — one unclassifiable residue must not kill the entry
        log.warning("SASA: FreeSASA failed (%s: %s) — areas unavailable.",
                    type(exc).__name__, exc)
        return {}
    areas = result.residueAreas()
    return {orig: areas.get(label, {}) for orig, label in fs_id.items()}


def extract_sasa(
    pdb_path: str,
    chain_id: str | None = None,
    probe_radius: float = 1.4,
    n_points: int = 100,
) -> np.ndarray:
    """Compute per-nucleotide absolute SASA (Å²) using FreeSASA.

    Parameters
    ----------
    pdb_path : str
        Path to a PDB or mmCIF file (HETATM records are skipped when matching
        residues; FreeSASA processes all heavy atoms via the BioPython model).
    chain_id : str
        Chain identifier to extract.
    probe_radius : float
        Solvent probe radius in Å.  Default 1.4 Å (water molecule).
    n_points : int
        Number of test points per atom used in the Lee-Richards numerical
        integration.  Higher values give more accurate areas at greater cost.

    Returns
    -------
    sasa : (L,) float64
        Per-residue absolute SASA in Å².  Residues not resolved by FreeSASA
        (e.g. residues with no heavy atoms present) are set to 0.0 and a
        warning is emitted.
    """
    try:
        import freesasa
    except ImportError as exc:
        raise ImportError(
            "FreeSASA Python bindings are required.  "
            "Install with:  pip install freesasa"
        ) from exc

    # Silence FreeSASA's per-atom "guessing that atom 'C2' is symbol ' C'"
    # warnings (RNA atom names trip its element guesser) while keeping errors.
    freesasa.setVerbosity(freesasa.nowarnings)

    # --- ordered nucleotide residues via BioPython (ATOM records only) -------
    bio_struct = _get_parser(pdb_path).get_structure(Path(pdb_path).stem, pdb_path)
    chain_id = _resolve_chain(bio_struct[0], chain_id)
    chain = bio_struct[0][chain_id]
    residues = [
        res
        for res in chain.get_residues()
        if res.get_id()[0] == " "                               # ATOM, not HETATM
        and RESNAME_TO_NUC.get(res.get_resname().strip()) is not None
    ]
    L = len(residues)
    if L == 0:
        log.warning("extract_sasa: no RNA residues found in chain %s of %s", chain_id, pdb_path)
        return np.empty(0, dtype=np.float64)

    # --- run FreeSASA --------------------------------------------------------
    # Use structureFromBioPDB so FreeSASA works from the already-parsed model
    # rather than re-reading the file; this supports both PDB and mmCIF input.
    params = freesasa.Parameters({
        "algorithm": freesasa.LeeRichards,
        "probe-radius": probe_radius,
        "n-points": n_points,
    })
    # Chains are relabelled to single characters first; see _relabelled_model.
    chain_areas = _freesasa_areas(bio_struct, (chain_id,), params).get(chain_id, {})

    # --- map FreeSASA per-residue totals onto the ordered residue list -------
    sasa = np.zeros(L, dtype=np.float64)
    n_missing = 0
    for idx, res in enumerate(residues):
        seq_num = res.get_id()[1]
        icode = res.get_id()[2].strip()
        # FreeSASA typically keys on the residue number as a plain string;
        # append the insertion code when present.
        key = f"{seq_num}{icode}" if icode else str(seq_num)
        area = chain_areas.get(key)
        if area is None and icode:
            # Some FreeSASA builds drop the insertion code — try without it.
            area = chain_areas.get(str(seq_num))
        if area is not None:
            sasa[idx] = area.total
        else:
            n_missing += 1
            log.debug(
                "SASA: no FreeSASA entry for chain %s residue %s — set to 0.0",
                chain_id, key,
            )

    if n_missing:
        log.warning(
            "SASA: %d/%d residues had no FreeSASA entry and were set to 0.0.",
            n_missing, L,
        )

    return sasa


def extract_sasa_unit(
    pdb_path: str,
    keys: "list[tuple[str, int]]",
    probe_radius: float = 1.4,
    n_points: int = 100,
) -> np.ndarray:
    """Per-nucleotide SASA for a multi-chain design unit, aligned to *keys*.

    Runs FreeSASA **once** over the whole structure (so inter-chain interface
    burial is reflected) and maps the per-residue areas onto the ordered
    ``keys`` list (``[(chain, resseq), ...]`` from
    :func:`evo_rna.profile.utils.extract_chain_unit`).  This replaces calling
    :func:`extract_sasa` once per chain, which recomputed the full SASA N times.

    Returns an ``(len(keys),)`` float64 array; residues FreeSASA can't resolve
    are 0.0.
    """
    try:
        import freesasa
    except ImportError as exc:
        raise ImportError(
            "FreeSASA Python bindings are required.  Install with:  pip install freesasa"
        ) from exc

    freesasa.setVerbosity(freesasa.nowarnings)

    bio_struct = _get_parser(pdb_path).get_structure(Path(pdb_path).stem, pdb_path)
    model = bio_struct[0]
    params = freesasa.Parameters({
        "algorithm": freesasa.LeeRichards,
        "probe-radius": probe_radius,
        "n-points": n_points,
    })
    unit_chains = tuple(dict.fromkeys(c for c, _ in keys))
    residue_areas = _freesasa_areas(bio_struct, unit_chains, params)  # keyed by ORIGINAL id

    # Build (chain, resseq) → total SASA, mirroring extract_sasa's icode handling.
    area_by_key: dict[tuple[str, int], float] = {}
    for chain in model.get_chains():
        chain_areas = residue_areas.get(chain.id, {})
        for res in chain.get_residues():
            if res.get_id()[0] != " ":
                continue
            if RESNAME_TO_NUC.get(res.get_resname().strip()) is None:
                continue
            seq_num = res.get_id()[1]
            icode = res.get_id()[2].strip()
            key = f"{seq_num}{icode}" if icode else str(seq_num)
            area = chain_areas.get(key) or (chain_areas.get(str(seq_num)) if icode else None)
            area_by_key[(chain.id, seq_num)] = area.total if area is not None else 0.0

    sasa = np.zeros(len(keys), dtype=np.float64)
    n_missing = 0
    for i, k in enumerate(keys):
        if k in area_by_key:
            sasa[i] = area_by_key[k]
        else:
            n_missing += 1
    if n_missing:
        log.warning("SASA: %d/%d unit residues had no FreeSASA entry (set to 0.0).",
                    n_missing, len(keys))
    return sasa


def save_sa(sasa: np.ndarray, output_dir: str) -> None:
    with open(Path(output_dir) / "sa.txt", "w") as f:
        f.write("pos\tsa\n")
        for i in range(sasa.shape[0]):
            f.write(f"{i + 1}\t{sasa[i]:.6f}\n")





if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    pdb = sys.argv[1] if len(sys.argv) > 1 else "data/test/PDB_00009PDG_2_A.pdb"
    chain = sys.argv[2] if len(sys.argv) > 2 else "A"

    sasa = extract_sasa(pdb, chain)
    L = len(sasa)
    print(f"chain: {chain}  length: {L}")
    print(f"total SASA : {sasa.sum():.1f} Å²")
    print(f"mean  SASA : {sasa.mean():.1f} Å²  (per nucleotide)")
    print(f"min   SASA : {sasa.min():.1f} Å²")
    print(f"max   SASA : {sasa.max():.1f} Å²")
    print()
    print(f"{'idx':>4}  {'SASA (Å²)':>10}")
    print("-" * 18)
    for i, val in enumerate(sasa):
        print(f"{i:>4}  {val:>10.2f}")
