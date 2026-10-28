#!/usr/bin/env python3
"""
Parse a locally downloaded RNAsolo mirror into a minimal, flat-output manifest.

This script walks a directory like:
  data/rnasolo/local/<archive_dir>/<PDB_XXXXXXXX_model_chains[__range].pdb>

And writes a CSV/TSV manifest that can be used on a cluster (or by collaborators)
to fetch the same set of structures into a *flat* directory. Output filenames are
normalized to:
  <PDB_ID>_<model>_<chains_raw>[__<rfam_range>].pdb

Examples:
  python scripts/parse_rnasolo_local_manifest.py \
    --root data/rnasolo/local \
    --out data/rnasolo/rnasolo_manifest.csv

  python scripts/parse_rnasolo_local_manifest.py \
    --root data/rnasolo/local \
    --out rnasolo_manifest.tsv \
    --format tsv
"""

import argparse
import csv
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple


_PDB_RE = re.compile(
    r"""
    ^
    PDB_
    (?P<id_blob>[0-9A-Za-z]{8})       # e.g. 00001A3M, 00008RUI
    _
    (?P<model>\d+)
    _
    (?P<chains>[^._]+?)               # e.g. A, A-B, 1, N, etc.
    (?:__ (?P<rfam_range>\d+-\d+))?   # optional range suffix (RFAM sets)
    \.pdb
    $
    """,
    re.VERBOSE,
)


def _chains_to_list(chains_raw: str) -> Sequence[str]:
    # Chains are often "A" or "A-B" (meaning multiple chains).
    # Keep conservative splitting: treat '-' and ',' as separators.
    # If a chain identifier itself contains '-', this will over-split, but that
    # seems unlikely for PDB chain IDs in these RNAsolo exports.
    parts = re.split(r"[-,]+", chains_raw.strip())
    return [p for p in parts if p]


@dataclass(frozen=True)
class ManifestRow:
    pdb_id: str
    model: int
    chains_raw: str
    chains: str
    rfam_range: str
    out_name: str
    local_path: str


def _iter_pdb_files(root: Path) -> Iterable[Path]:
    # Avoid Path.rglob("*.pdb") overhead on some filesystems by manual walk.
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(".pdb"):
                yield Path(dirpath) / fn


def _parse_pdb_filename(filename: str) -> Optional[Tuple[str, int, str, str]]:
    """
    Returns (pdb_id, model, chains_raw, rfam_range) or None if not matched.
    """
    m = _PDB_RE.match(filename)
    if not m:
        return None
    id_blob = m.group("id_blob")
    pdb_id = id_blob[-4:].upper()
    model = int(m.group("model"))
    chains_raw = m.group("chains")
    rfam_range = m.group("rfam_range") or ""
    return pdb_id, model, chains_raw, rfam_range


def build_manifest(root: Path) -> Tuple[Sequence[ManifestRow], Sequence[str]]:
    rows: list[ManifestRow] = []
    skipped: list[str] = []

    for path in _iter_pdb_files(root):
        parsed = _parse_pdb_filename(path.name)
        if parsed is None:
            skipped.append(str(path.relative_to(root)))
            continue

        pdb_id, model, chains_raw, rfam_range = parsed
        chains_list = _chains_to_list(chains_raw)
        local_path = str(path.relative_to(root))
        out_name = "{}_{}_{}".format(pdb_id, model, chains_raw)
        if rfam_range:
            out_name += "__{}".format(rfam_range)
        out_name += ".pdb"

        rows.append(
            ManifestRow(
                pdb_id=pdb_id,
                model=model,
                chains_raw=chains_raw,
                chains=",".join(chains_list),
                rfam_range=rfam_range,
                out_name=out_name,
                local_path=local_path,
            )
        )

    # Stable ordering: pdb_id, model, chains_raw, rfam_range
    rows.sort(key=lambda r: (r.pdb_id, r.model, r.chains_raw, r.rfam_range))
    return rows, skipped


def write_manifest(rows: Sequence[ManifestRow], out_path: Path, fmt: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt not in {"csv", "tsv"}:
        raise ValueError("Unknown format: {} (expected csv or tsv)".format(fmt))

    delimiter = "," if fmt == "csv" else "\t"
    fieldnames = [
        "pdb_id",
        "model",
        "chains_raw",
        "chains",
        "rfam_range",
        "out_name",
        "local_path",
    ]

    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Parse locally downloaded RNAsolo PDB archives into a CSV/TSV manifest."
    )
    ap.add_argument(
        "--root",
        default="data/rnasolo/local",
        help="Root directory containing RNAsolo archive folders (default: %(default)s)",
    )
    ap.add_argument(
        "--out",
        default="data/rnasolo/rnasolo_manifest.csv",
        help="Output manifest path (default: %(default)s)",
    )
    ap.add_argument(
        "--format",
        default=None,
        choices=["csv", "tsv"],
        help="Output format. If omitted, inferred from --out extension.",
    )
    ap.add_argument(
        "--write-skipped",
        default=None,
        metavar="PATH",
        help="Optional path to write skipped (unparsed) file paths, one per line.",
    )
    args = ap.parse_args()

    root = Path(os.path.expanduser(args.root)).resolve()
    if not root.exists():
        raise SystemExit("Root not found: {}".format(root))

    out_path = Path(os.path.expanduser(args.out)).resolve()
    fmt = args.format
    if fmt is None:
        fmt = "tsv" if out_path.suffix.lower() == ".tsv" else "csv"

    rows, skipped = build_manifest(root)
    write_manifest(rows, out_path, fmt=fmt)

    if args.write_skipped:
        skipped_path = Path(os.path.expanduser(args.write_skipped)).resolve()
        skipped_path.parent.mkdir(parents=True, exist_ok=True)
        with skipped_path.open("w") as f:
            for p in skipped:
                f.write(p + "\n")

    print("Root: {}".format(root))
    print("Wrote: {} ({} rows)".format(out_path, len(rows)))
    if skipped:
        print(
            "Skipped: {} file(s) (pass --write-skipped PATH to save list)".format(
                len(skipped)
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

