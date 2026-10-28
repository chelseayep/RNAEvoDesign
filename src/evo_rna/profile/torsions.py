import logging

import numpy as np
from pathlib import Path

from evo_rna.profile.utils import _get_parser, _resolve_chain

log = logging.getLogger(__name__)

# Column order for the (L, 7) output array.
TORSION_COLS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "chi"]
_COL_IDX = {name: i for i, name in enumerate(TORSION_COLS)}


def dihedral(p1, p2, p3, p4) -> float:
    """Dihedral angle in degrees defined by four coordinate arrays."""
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3

    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    n1 /= np.linalg.norm(n1)
    n2 /= np.linalg.norm(n2)
    b2 /= np.linalg.norm(b2)

    m1 = np.cross(n1, b2)
    return float(np.degrees(np.arctan2(np.dot(m1, n2), np.dot(n1, n2))))


def get_coord(residue, atom_name) -> np.ndarray | None:
    try:
        return residue[atom_name].get_vector().get_array()
    except KeyError:
        return None


def infer_glycosidic_atoms(residue) -> tuple[str, str] | None:
    """Return (N_atom, C_atom) for the chi angle; infers purine vs pyrimidine."""
    if "N9" in residue:
        return ("N9", "C4")   # purine
    if "N1" in residue:
        return ("N1", "C2")   # pyrimidine
    return None


def pseudorotation_P(residue) -> float | None:
    """Altona-Sundaralingam pseudorotation phase angle P in degrees [0, 360)."""
    pucker_atoms = [
        ("C4'", "O4'", "C1'", "C2'"),  # ν0
        ("O4'", "C1'", "C2'", "C3'"),  # ν1
        ("C1'", "C2'", "C3'", "C4'"),  # ν2
        ("C2'", "C3'", "C4'", "O4'"),  # ν3
        ("C3'", "C4'", "O4'", "C1'"),  # ν4
    ]
    nu = []
    for atoms in pucker_atoms:
        coords = [get_coord(residue, a) for a in atoms]
        if all(x is not None for x in coords):
            nu.append(dihedral(*coords))
        else:
            return None
    num = (nu[4] + nu[1]) - (nu[3] + nu[0])
    den = 2 * nu[2] * (np.sin(np.radians(36)) + np.sin(np.radians(72)))
    return float(np.degrees(np.arctan2(num, den)) % 360)


def extract_torsions(pdb_path: str, chain_id: str | None = None) -> np.ndarray:
    """Extract backbone torsion angles from PDB coordinates.

    Parameters
    ----------
    pdb_path : str
        Path to a PDB or mmCIF file.
    chain_id : str
        Chain to extract. If ``None``, the first available chain is used.

    Returns
    -------
    torsions : (L, 7) float64
        Torsion angles in degrees for each nucleotide.  Columns follow
        ``TORSION_COLS`` order: alpha, beta, gamma, delta, epsilon, zeta, chi.
        Angles that cannot be computed (missing atoms or terminal residues)
        are set to NaN.
    """
    struct = _get_parser(pdb_path).get_structure(Path(pdb_path).stem, pdb_path)
    chain_id = _resolve_chain(struct[0], chain_id)
    chain = struct[0][chain_id]

    residues = [
        res for res in chain.get_residues()
        if res.get_id()[0] == " " and "C4'" in res   # ATOM records only
    ]
    L = len(residues)
    if L == 0:
        return np.empty((0, 7), dtype=np.float64)

    out = np.full((L, 7), np.nan, dtype=np.float64)

    for idx, res in enumerate(residues):
        prev = residues[idx - 1] if idx > 0 else None
        nxt  = residues[idx + 1] if idx < L - 1 else None

        def c(r, name):
            return get_coord(r, name) if r is not None else None

        def store(col: str, atoms: list):
            if all(x is not None for x in atoms):
                out[idx, _COL_IDX[col]] = dihedral(*atoms)

        # α: O3'(i-1) – P – O5' – C5'
        store("alpha",   [c(prev,"O3'"), c(res,"P"),   c(res,"O5'"), c(res,"C5'")])
        # β: P – O5' – C5' – C4'
        store("beta",    [c(res,"P"),    c(res,"O5'"), c(res,"C5'"), c(res,"C4'")])
        # γ: O5' – C5' – C4' – C3'
        store("gamma",   [c(res,"O5'"), c(res,"C5'"),  c(res,"C4'"), c(res,"C3'")])
        # δ: C5' – C4' – C3' – O3'
        store("delta",   [c(res,"C5'"), c(res,"C4'"),  c(res,"C3'"), c(res,"O3'")])
        # ε: C4' – C3' – O3' – P(i+1)
        store("epsilon", [c(res,"C4'"), c(res,"C3'"),  c(res,"O3'"), c(nxt,"P")])
        # ζ: C3' – O3' – P(i+1) – O5'(i+1)
        store("zeta",    [c(res,"C3'"), c(res,"O3'"),  c(nxt,"P"),   c(nxt,"O5'")])
        # χ: O4' – C1' – N(glyco) – C(glyco)
        glyco = infer_glycosidic_atoms(res)
        if glyco is not None:
            n_atom, c_atom = glyco
            store("chi", [c(res,"O4'"), c(res,"C1'"), c(res,n_atom), c(res,c_atom)])

    return out


def _residue_keys(chain) -> "list[tuple[str, int]]":
    """(chain, resseq) for the residues extract_torsions computes rows for."""
    return [
        (chain.id, res.get_id()[1])
        for res in chain.get_residues()
        if res.get_id()[0] == " " and "C4'" in res
    ]


def extract_torsions_unit(pdb_path: str, keys: "list[tuple[str, int]]") -> np.ndarray:
    """Per-residue torsions for a multi-chain design unit, aligned to *keys*.

    ``extract_torsions`` selects residues with ``"C4'" in res`` while
    ``extract_chain_unit`` selects those with a known RNA base, so the two
    disagree whenever a chain carries a DNA or modified residue with a ribose:
    the arrays come out different lengths and ``build_structure`` "fixes" the
    mismatch by truncating from the end, which silently shifts every torsion
    onto the wrong residue from the first extra entry onward.

    Mapping by ``(chain, resseq)`` instead — the same alignment
    :func:`evo_rna.profile.sa.extract_sasa_unit` uses — makes the correspondence
    explicit and cannot drift.  Neighbouring residues are still taken from the
    full chain, so α/ε/ζ keep the sequential context they need.

    Returns ``(len(keys), 7)``; residues without a computed row are NaN.
    """
    model = _get_parser(pdb_path).get_structure(Path(pdb_path).stem, pdb_path)[0]
    row_by_key: dict[tuple[str, int], np.ndarray] = {}
    for chain_id in dict.fromkeys(c for c, _ in keys):
        if chain_id not in [c.id for c in model.get_chains()]:
            log.warning("extract_torsions_unit: chain '%s' not in %s", chain_id, Path(pdb_path).name)
            continue
        chain = model[chain_id]
        rows = extract_torsions(pdb_path, chain_id)
        for k, row in zip(_residue_keys(chain), rows):
            row_by_key[k] = row

    out = np.full((len(keys), 7), np.nan, dtype=np.float64)
    n_missing = 0
    for i, k in enumerate(keys):
        row = row_by_key.get(k)
        if row is None:
            n_missing += 1
        else:
            out[i] = row
    if n_missing:
        log.warning("torsions: %d/%d unit residues had no torsion row (left NaN).",
                    n_missing, len(keys))
    return out


def save_torsions(torsions: np.ndarray, output_dir: str) -> None:
    """Write a (L, 7) torsions array to a human-readable tab-separated file."""
    L = torsions.shape[0]
    with open(Path(output_dir) / "torsions.txt", "w") as f:
        f.write("# Backbone torsion angles (degrees); NaN = undefined\n")
        f.write(f"# length: {L}\n")
        f.write("#\n")
        header = "\t".join(["pos"] + TORSION_COLS)
        f.write(f"{header}\n")
        for i in range(L):
            vals = "\t".join(
                f"{v:>8.2f}" if not np.isnan(v) else f"{'—':>8}"
                for v in torsions[i]
            )
            f.write(f"{i + 1:>4}\t{vals}\n")


if __name__ == "__main__":
    torsions = extract_torsions("data/test/PDB_00008ZNQ_1_A.pdb", "A")
    print(f"shape: {torsions.shape}")
    print(f"defined: {(~np.isnan(torsions)).sum()} / {torsions.size} values")
    save_torsions(torsions, "data/test/PDB_00008ZNQ_1_A_torsions.txt")
