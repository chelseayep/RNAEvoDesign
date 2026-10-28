"""RNA secondary structure extraction from PDB coordinates.

Detects base pairs by:
  1. Glycosidic N-N distance filter (coarse)
  2. Base-plane coplanarity check
  3. Hydrogen-bond donor/acceptor distance validation

Canonical pairs:  A-U, G-C  (Watson-Crick)
Wobble pairs:     G-U

Outputs
-------
ss        : (L,) int8 — 0 = unpaired, 1 = paired
pair_map  : {i: j, j: i, ...}  (0-indexed)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import numpy as np

from evo_rna.profile.utils import RESNAME_TO_NUC, _get_parser, _resolve_chain

_GLYCOSIDIC_N = {"A": "N9", "G": "N9", "U": "N1", "C": "N1"}

_CANONICAL_PAIRS = {
    frozenset(("A", "U")),
    frozenset(("G", "C")),
    frozenset(("G", "U")),
}

_NN_DIST_MAX = 13.0  # Å
_NN_DIST_MIN = 7.0   # Å

_BASE_ATOMS = {
    "A": ["N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4"],
    "G": ["N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4"],
    "U": ["N1", "C2", "O2", "N3", "C4", "O4", "C5", "C6"],
    "C": ["N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"],
}

HBOND_ATOMS = {
    frozenset(("A", "U")): [
        ("A", "N6", "U", "O4"),
        ("A", "N1", "U", "N3"),
    ],
    frozenset(("G", "C")): [
        ("G", "O6", "C", "N4"),
        ("G", "N1", "C", "N3"),
        ("G", "N2", "C", "O2"),
    ],
    frozenset(("G", "U")): [
        ("G", "O6", "U", "N3"),
        ("G", "N1", "U", "O2"),
    ],
}

HBOND_DIST_MAX = 3.5  # Å
HBOND_MIN_SATISFIED = 2

# ------------------------------------------------------------------
# Geometry helpers
# ------------------------------------------------------------------

def _get_glycosidic_n_coord(residue, nuc: str) -> np.ndarray | None:
    atom_name = _GLYCOSIDIC_N.get(nuc)
    if atom_name is None:
        return None
    try:
        return residue[atom_name].get_vector().get_array()
    except KeyError:
        return None


def base_plane_normal(residue, nuc: str) -> np.ndarray | None:
    """Fit a plane to the base heavy atoms and return the unit normal."""
    atom_names = _BASE_ATOMS.get(nuc, [])
    coords = []
    for name in atom_names:
        try:
            coords.append(residue[name].get_vector().get_array())
        except KeyError:
            continue
    if len(coords) < 3:
        return None
    coords = np.array(coords)
    centroid = coords.mean(axis=0)
    _, _, vh = np.linalg.svd(coords - centroid)
    return vh[-1]


def coplanar(n1: np.ndarray, n2: np.ndarray, max_angle_deg: float = 35.0) -> bool:
    """True if two plane normals are roughly parallel or antiparallel."""
    cos = abs(np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2)))
    angle = np.degrees(np.arccos(np.clip(cos, 0.0, 1.0)))
    return angle < max_angle_deg


def count_hbonds(res_i, nuc_i: str, res_j, nuc_j: str) -> int:
    """Count satisfied H-bonds between two nucleotide residues."""
    key = frozenset((nuc_i, nuc_j))
    bonds = HBOND_ATOMS.get(key, [])
    count = 0
    for na, atom_a, nb, atom_b in bonds:
        if nuc_i == na:
            ra, rb = res_i, res_j
        else:
            ra, rb = res_j, res_i
        try:
            ca = ra[atom_a].get_vector().get_array()
            cb = rb[atom_b].get_vector().get_array()
        except KeyError:
            continue
        if np.linalg.norm(ca - cb) <= HBOND_DIST_MAX:
            count += 1
    return count


# ------------------------------------------------------------------
# Main extraction
# ------------------------------------------------------------------

@dataclass(order=True)
class _Candidate:
    score: float
    i: int = field(compare=False)
    j: int = field(compare=False)
    hbonds: int = field(compare=False)


def extract_ss_and_pairs(
    pdb_path: str,
    chain_id: str | None = None,
) -> tuple[np.ndarray, Dict[int, int]]:
    """Extract secondary structure labels and base-pair map from PDB geometry.

    Returns (ss, pair_map) where:
      ss       : (L,) int8 — 0 = unpaired, 1 = paired
      pair_map : {i: j, j: i, ...}   (0-indexed)

    If *chain_id* is None or not present in the PDB the first available chain
    is used automatically.
    """
    struct = _get_parser(pdb_path).get_structure(Path(pdb_path).stem, pdb_path)
    model = struct[0]
    chain = model[_resolve_chain(model, chain_id)]

    # Collect nucleotide residues: (nuc letter, glycosidic-N coord, Bio.PDB residue, PDB resseq).
    residues: list[tuple[str, np.ndarray, object, int]] = []
    for res in chain.get_residues():
        nuc = RESNAME_TO_NUC.get(res.get_resname().strip())
        if nuc is None:
            continue
        coord = _get_glycosidic_n_coord(res, nuc)
        if coord is None:
            continue
        resseq = res.get_id()[1]
        residues.append((nuc, coord, res, resseq))

    L = len(residues)

    # 1. Collect all valid candidates.
    candidates: list[_Candidate] = []
    for i in range(L):
        nuc_i, coord_i, res_i, _ = residues[i]
        normal_i = base_plane_normal(res_i, nuc_i)
        if normal_i is None:
            continue

        for j in range(i + 4, L):
            nuc_j, coord_j, res_j, _ = residues[j]
            if frozenset((nuc_i, nuc_j)) not in _CANONICAL_PAIRS:
                continue

            d = float(np.linalg.norm(coord_i - coord_j))
            if not (_NN_DIST_MIN <= d <= _NN_DIST_MAX):
                continue

            normal_j = base_plane_normal(res_j, nuc_j)
            if normal_j is None:
                continue
            if not coplanar(normal_i, normal_j):
                continue

            hb = count_hbonds(res_i, nuc_i, res_j, nuc_j)
            if hb < HBOND_MIN_SATISFIED:
                continue

            candidates.append(_Candidate(score=d, i=i, j=j, hbonds=hb))

    # 2. Greedy assignment: best (shortest distance) first, one pair per position.
    candidates.sort()
    pair_map: Dict[int, int] = {}
    paired: set[int] = set()
    for c in candidates:
        if c.i in paired or c.j in paired:
            continue
        pair_map[c.i] = c.j
        pair_map[c.j] = c.i
        paired.add(c.i)
        paired.add(c.j)

    ss = np.zeros(L, dtype=np.int8)
    for pos in paired:
        ss[pos] = 1

    # Build index → (chain_id, nuc, resseq) lookup for display / comparison.
    residue_labels = [
        (chain_id, nuc, resseq) for nuc, _, _, resseq in residues
    ]

    return ss, pair_map, residue_labels


def extract_ss_and_pairs_unit(
    pdb_path: str,
    keys: "list[tuple[str, int]]",
) -> tuple[np.ndarray, Dict[int, int]]:
    """Geometry SS / base pairs for a multi-chain design unit, aligned to *keys*.

    *keys* is the ordered ``[(chain_id, resseq), ...]`` list returned by
    :func:`evo_rna.profile.utils.extract_chain_unit`; the local index is the
    list position, so the returned ``ss`` (length ``len(keys)``) and ``pair_map``
    line up with the concatenated multi-chain sequence.

    Unlike :func:`extract_ss_and_pairs` (single chain only), this detects
    **inter-chain** pairs — essential for duplexes and multi-strand units (e.g.
    the RNAsolo / gRNAde-DAS ``D-C-A-B`` targets) — by applying the minimum
    sequence-separation filter only *within* a chain.
    """
    model = _get_parser(pdb_path).get_structure(Path(pdb_path).stem, pdb_path)[0]

    # (chain, resseq) → Bio.PDB residue, so we can walk *keys* in their exact
    # order and keep local indices aligned with the extracted sequence.
    res_by_key: dict[tuple[str, int], object] = {}
    for chain in model.get_chains():
        for res in chain.get_residues():
            if res.get_id()[0] != " ":
                continue
            res_by_key[(chain.id, res.get_id()[1])] = res

    L = len(keys)
    nucs: list[str | None] = [None] * L
    coords: list[np.ndarray | None] = [None] * L
    res_objs: list[object | None] = [None] * L
    chain_of: list[str] = [c for c, _ in keys]
    for i, key in enumerate(keys):
        res = res_by_key.get(key)
        if res is None:
            continue
        nuc = RESNAME_TO_NUC.get(res.get_resname().strip())
        if nuc is None:
            continue
        nucs[i] = nuc
        coords[i] = _get_glycosidic_n_coord(res, nuc)
        res_objs[i] = res

    candidates: list[_Candidate] = []
    for i in range(L):
        nuc_i, coord_i, res_i = nucs[i], coords[i], res_objs[i]
        if nuc_i is None or coord_i is None:
            continue
        normal_i = base_plane_normal(res_i, nuc_i)
        if normal_i is None:
            continue
        for j in range(i + 1, L):
            # Min sequence separation applies only within the same chain;
            # inter-chain partners (duplex strands) may be adjacent in index.
            if chain_of[i] == chain_of[j] and j - i < 4:
                continue
            nuc_j, coord_j, res_j = nucs[j], coords[j], res_objs[j]
            if nuc_j is None or coord_j is None:
                continue
            if frozenset((nuc_i, nuc_j)) not in _CANONICAL_PAIRS:
                continue
            d = float(np.linalg.norm(coord_i - coord_j))
            if not (_NN_DIST_MIN <= d <= _NN_DIST_MAX):
                continue
            normal_j = base_plane_normal(res_j, nuc_j)
            if normal_j is None or not coplanar(normal_i, normal_j):
                continue
            hb = count_hbonds(res_i, nuc_i, res_j, nuc_j)
            if hb < HBOND_MIN_SATISFIED:
                continue
            candidates.append(_Candidate(score=d, i=i, j=j, hbonds=hb))

    candidates.sort()
    pair_map: Dict[int, int] = {}
    paired: set[int] = set()
    for c in candidates:
        if c.i in paired or c.j in paired:
            continue
        pair_map[c.i] = c.j
        pair_map[c.j] = c.i
        paired.update((c.i, c.j))

    ss = np.zeros(L, dtype=np.int8)
    for pos in paired:
        ss[pos] = 1
    return ss, pair_map


def save_ss(
    ss: np.ndarray,
    pair_map: Dict[int, int],
    labels: list[tuple[str, str, int]],
    output_dir: str,
) -> None:
    """Write secondary structure to a human-readable text file.

    Parameters
    ----------
    ss : (L,) int8
        0 = unpaired, 1 = paired.
    pair_map : {i: j, j: i, ...}
        0-indexed base-pair map.
    labels : list of (chain_id, nuc, resseq)
        Residue labels in sequence order, as returned by extract_ss_and_pairs.
    output_dir : str
        Destination file path (e.g. "output/ss.txt").
    """
    L = len(ss)
    paired_count = int(ss.sum())
    n_pairs = len(pair_map) // 2

    with open(Path(output_dir) / "ss.txt", "w") as f:
        f.write("# Secondary structure\n")
        f.write(f"# length:   {L}\n")
        f.write(f"# paired:   {paired_count}\n")
        f.write(f"# unpaired: {L - paired_count}\n")
        f.write(f"# pairs:    {n_pairs}\n")
        f.write("#\n")

        # --- per-residue table ---
        f.write(f"{'pos':>4}\t{'ss':>2}\t{'label':<10}\t{'partner':>7}\n")
        for i in range(L):
            label = "paired" if ss[i] == 1 else "unpaired"
            partner = pair_map.get(i)
            partner_str = str(partner + 1) if partner is not None else "—"
            f.write(f"{i + 1:>4}\t{int(ss[i]):>2}\t{label:<10}\t{partner_str:>7}\n")

        # --- base-pair list ---
        f.write(f"\n# Base pairs ({n_pairs} total)\n")
        f.write(f"{'i':>4}  {'j':>4}  {'i-label':>10}  {'j-label':<10}  {'type'}\n")
        f.write("-" * 46 + "\n")
        seen: set[tuple[int, int]] = set()
        for i, j in sorted(pair_map.items()):
            if (i, j) not in seen:
                seen.add((i, j))
                seen.add((j, i))
                ci, ni, ri = labels[i]
                cj, nj, rj = labels[j]
                label_i = f"{ci}.{ni}{ri}"
                label_j = f"{cj}.{nj}{rj}"
                f.write(f"{i + 1:>4}  {j + 1:>4}  {label_i:>10}  {label_j:<10}  {ni}-{nj}\n")


if __name__ == "__main__":
    import sys

    pdb = sys.argv[1] if len(sys.argv) > 1 else "data/test/PDB_00009PDG_2_A.pdb"
    chain = sys.argv[2] if len(sys.argv) > 2 else "A"
    out = sys.argv[3] if len(sys.argv) > 3 else "data/test/ss.txt"

    ss, pair_map, labels = extract_ss_and_pairs(pdb, chain)
    save_ss(ss, pair_map, labels, out)
    print(f"Wrote {out}")
