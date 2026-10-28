#!/usr/bin/env python3
"""Data-quality QC on the feature extraction, before committing 8,294 structures to it.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))


from evo_rna.config import PATHS  # noqa: E402
from evo_rna.data.splits import load_grnade  # noqa: E402
from evo_rna.profile.features import build_structure  # noqa: E402
from evo_rna.profile.sa import extract_sasa_unit  # noqa: E402
from evo_rna.profile.utils import extract_chain_unit  # noqa: E402

from evo_rna.learned.features import resolve_path  # noqa: E402

log = logging.getLogger("check_extraction")


def odd_chain(chains: tuple[str, ...]) -> bool:
    """True if any chain id cannot survive a PDB-format round-trip."""
    return any(len(c) != 1 or not c.isalpha() for c in chains)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="grnade_structsim")
    ap.add_argument("--subset", default="train", choices=("train", "val", "test"))
    ap.add_argument("--pdb-dir", type=Path, default=Path(PATHS.pdb_dir))
    ap.add_argument("--work-dir", type=Path,
                    default=Path(__file__).resolve().parent / "cache" / "qc_work")
    ap.add_argument("--n", type=int, default=40, help="random sample size")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent / "results" / "extraction_qc.csv")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s %(levelname)-8s %(message)s")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    ids = load_grnade(args.split)[args.subset]
    rng = random.Random(args.seed)
    sample = rng.sample(ids, min(args.n, len(ids)))

    rows = []
    for n, sid in enumerate(sample, 1):
        path = resolve_path(sid, args.pdb_dir)
        if path is None:
            print(f"  [{n:3d}/{len(sample)}] {sid}: no structure")
            continue
        try:
            struct = build_structure(str(path), output_dir=str(args.work_dir),
                                     chains=sid.chains)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{n:3d}/{len(sample)}] {sid}: {type(exc).__name__}: {exc}")
            continue

        L = len(struct.sequence)
        if L == 0:
            print(f"  [{n:3d}/{len(sample)}] {sid}: no residues")
            continue

        sa = struct.sa
        chi = struct.torsions[:, 6] if struct.torsions.size else np.array([])
        frac_sa_zero = float(np.mean(sa == 0.0)) if sa.size else 1.0
        frac_chi_nan = float(np.mean(~np.isfinite(chi))) if chi.size else 1.0

        # Decisive test of the chain-id hypothesis: same unit, but SASA computed
        # once over the structure and mapped by (chain, resnum) keys rather than
        # looked up by chain id.
        try:
            _seq, _atoms, _coords, keys = extract_chain_unit(str(path), sid.chains)
            sa_unit = extract_sasa_unit(str(path), keys)
            frac_sa_zero_unit = float(np.mean(sa_unit == 0.0)) if sa_unit.size else 1.0
        except Exception as exc:  # noqa: BLE001
            frac_sa_zero_unit = float("nan")
            log.warning("extract_sasa_unit failed for %s: %s", sid, exc)

        rows.append({
            "target": str(sid),
            "chains": "-".join(sid.chains),
            "odd_chain_id": odd_chain(sid.chains),
            "n_residues": L,
            "n_pairs": len(struct.pair_map) // 2,
            "n_contacts": len(struct.contacts),
            "frac_sasa_zero_perchain": frac_sa_zero,
            "frac_sasa_zero_unit": frac_sa_zero_unit,
            "frac_chi_nan": frac_chi_nan,
        })
        print(f"  [{n:3d}/{len(sample)}] {str(sid):<16} L={L:<5d} pairs={rows[-1]['n_pairs']:<4d} "
              f"contacts={rows[-1]['n_contacts']:<4d} "
              f"sasa0(per-chain)={frac_sa_zero:.2f} sasa0(unit)={frac_sa_zero_unit:.2f}")

    if not rows:
        print("nothing extracted — check --pdb-dir")
        return 1

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    print("\n" + "=" * 72)
    print(f"{len(df)} units sampled from {args.split}:{args.subset}")
    print(f"  residues        median {df.n_residues.median():.0f}   total {df.n_residues.sum():,}")
    print(f"  pairs/unit      median {df.n_pairs.median():.0f}   "
          f"units with 0 pairs: {(df.n_pairs == 0).mean():.0%}")
    print(f"  contacts/unit   median {df.n_contacts.median():.0f}   "
          f"units with 0 contacts: {(df.n_contacts == 0).mean():.0%}")
    print(f"  chi undefined   mean {df.frac_chi_nan.mean():.1%} of residues")

    print("\nSASA — the hypothesis under test:")
    print(f"  fully-zero SASA, per-chain lookup (current path): "
          f"{(df.frac_sasa_zero_perchain > 0.99).mean():.0%} of units")
    print(f"  fully-zero SASA, unit-keyed lookup (extract_sasa_unit): "
          f"{(df.frac_sasa_zero_unit > 0.99).mean():.0%} of units")
    odd = df[df.odd_chain_id]
    if len(odd):
        print(f"  units with a non-single-letter chain id: {len(odd)}/{len(df)}"
              f" — fully-zero SASA in {(odd.frac_sasa_zero_perchain > 0.99).mean():.0%}"
              f" of them vs {(df[~df.odd_chain_id].frac_sasa_zero_perchain > 0.99).mean():.0%}"
              f" of the rest")
    else:
        print("  no odd chain ids in this sample — rerun with a larger --n to test that arm")

    print("\nRead:")
    print("  * If 0-pair units are common but contacts track residue count, the sparse")
    print("    first-5 was sampling, not breakage — inter-unit pairs are correctly excluded.")
    print("  * If per-chain SASA is fully zero far more often than unit-keyed SASA, the")
    print("    chain-id lookup is the bug and build_structure should use extract_sasa_unit.")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
