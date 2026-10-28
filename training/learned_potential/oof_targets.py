#!/usr/bin/env python3
"""Which targets fell OUTSIDE the learned potential's own training fold?

The weight fit runs on the 04 pools, which are gRNAde structsim **train** --
and that is exactly the split ``train.py`` fitted the heads on.  So a fit over
all 80 asks ``learned_tertiary`` to rank natives it was trained to reconstruct,
and rewards it for memorising them.  Measured on the current manifest: 52 of the
80 were in the head's own train fold, 19 in its val fold, 9 in its test fold.

That is the same shape as the ``Z = 1/sigma`` pathology -- an artefact of how the
number was produced rather than a property of the term -- and it lands on the
weight the paper leads with, so it has to be measured rather than argued about.
Fit on the 28 out-of-fold targets as well and compare: if ``learned_tertiary``
holds its weight, the leak does not bite; if it drops, the honest weight is the
out-of-fold one.

The subset is thin -- 28 targets, and only **18 unique native sequences** -- so
it can detect a large shift and cannot set a precise weight.  Use it as a check
ON the full fit, never as a replacement for it.

How the fold is recovered
-------------------------
``train.py::assign_folds`` is a pure function of the PDB ENTRY: ``crc32(entry) /
2^32`` against ``test_frac`` then ``test_frac + val_frac``, at ``--split-seed 0``.
Nothing about the trained model is needed to reproduce it, so this script
recomputes the assignment rather than reading it out of a checkpoint that never
stored it.

The defaults here MUST match the training run (report.json ``args``):
``val_frac 0.15``, ``test_frac 0.15``, ``split_seed 0``.  They are flags so a
re-trained model with different fractions can be handled, not so they can be
tuned -- changing them silently changes which targets count as clean.

Usage
-----
    # from a manifest (target_id, ...) or a plain list of ids
    python training/learned_potential/oof_targets.py \\
        --manifest my_targets.tsv \\
        --out oof_targets.txt
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import zlib
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]

FOLD_NAMES = {0: "train", 1: "val", 2: "test"}


def fold_of(entry: str, val_frac: float, test_frac: float, seed: int) -> int:
    """Reproduce ``train.py::assign_folds`` for one PDB entry.

    Kept byte-identical to the training code, including the ``+ seed`` (which
    barely moves the ratio -- see the note in that function); this has to match
    what the model was actually trained with, not what would be better.
    """
    u = ((zlib.crc32(str(entry).encode()) + seed) & 0xFFFFFFFF) / 0xFFFFFFFF
    if u < test_frac:
        return 2
    if u < test_frac + val_frac:
        return 1
    return 0


def entry_of(target_id: str) -> str:
    """``7URI_1_y`` -> ``7URI``. The grouping key extract_features.py stored."""
    return target_id.split("_")[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest", type=Path,
                     help="pools manifest.tsv (reads the target_id column)")
    src.add_argument("--targets", type=Path, help="file of target ids, one per line")
    ap.add_argument("--out", type=Path, default=_HERE / "oof_targets.txt")
    ap.add_argument("--models", type=Path, default=_HERE / "models",
                    help="model dir whose report.json records the training args; "
                         "the fractions below are checked against it.")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--include-val", action="store_true",
                    help="Count the head's VAL fold as out-of-fold too (the "
                         "default). Early stopping selected on val, so val "
                         "targets are mildly optimistic but were never fitted; "
                         "--strict drops them instead.")
    ap.add_argument("--strict", action="store_true",
                    help="Only the head's TEST fold counts as clean. Safest and "
                         "much thinner -- 9 of 80 on the current manifest.")
    args = ap.parse_args()

    # The fractions must match the run that produced the tables, or this reports
    # a fold assignment the model never used.
    report = args.models / "report.json"
    if report.exists():
        targs = json.loads(report.read_text()).get("args", {})
        for flag, key in (("val_frac", "val_frac"), ("test_frac", "test_frac"),
                          ("split_seed", "split_seed")):
            want, got = getattr(args, flag), targs.get(key)
            if got is not None and float(want) != float(got):
                sys.exit(f"--{flag.replace('_', '-')} {want} does not match the "
                         f"trained model's {key}={got} ({report}). The fold "
                         f"assignment would not be the one the head was trained "
                         f"with.")
    else:
        print(f"[warn] no {report}: cannot verify the split fractions match the "
              f"trained model.", file=sys.stderr)

    if args.manifest:
        with args.manifest.open(newline="") as fh:
            ids = [r["target_id"] for r in csv.DictReader(fh, delimiter="\t")]
    else:
        ids = [l.strip() for l in args.targets.read_text().splitlines() if l.strip()]

    counts = {0: 0, 1: 0, 2: 0}
    keep: list[str] = []
    for t in ids:
        f = fold_of(entry_of(t), args.val_frac, args.test_frac, args.split_seed)
        counts[f] += 1
        if f == 2 or (f == 1 and not args.strict):
            keep.append(t)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(keep) + "\n")

    print(f"{len(ids)} targets, {len({entry_of(t) for t in ids})} unique PDB entries")
    for f in (0, 1, 2):
        print(f"  02 {FOLD_NAMES[f]:>5} fold: {counts[f]:>3}"
              + ("   <- the head was FITTED on these" if f == 0 else ""))
    print(f"\nkept {len(keep)} out-of-fold target(s) -> {args.out}"
          + ("  [--strict: test fold only]" if args.strict else
             "  [val + test folds; --strict for test only]"))
    if len(keep) < 20:
        print("  [warn] that is very few targets. Treat the resulting weights as "
              "a\n  direction-of-travel check on the full fit, not as weights to "
              "run arms with.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
