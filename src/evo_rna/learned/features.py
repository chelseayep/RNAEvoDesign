#!/usr/bin/env python3
"""Extract per-residue and per-contact geometry features for the learned potential.

Step 1 of 02_learned_potential.  Turns the TRAIN split into a flat feature table
whose labels are the native nucleotides — the same data the 236-parameter binned
tables were fit from, but without the binning.

Why the features look like this
-------------------------------
The binned tables discretise continuous geometry into 2-6 buckets:
``sasa_to_class`` gives 3 SASA bins, ``chi_to_bin`` gives 2 chi bins, and the
wiring check showed the second chi bin spans 0.086 log-odds — inert.  Everything
here is kept continuous, and angles are carried as (sin, cos) pairs so that the
-180/+180 wrap is not a discontinuity in feature space.

LEAKAGE RULE (this is the important part)
-----------------------------------------
``hbond`` reaches AUC 1.000 separating native from decoys using annotations
derived from the native structure.  Much of that is the Leontis-Westhof class:
a tHS geometry is often achievable only by particular bases, so the label nearly
names the answer.  A head trained on LW class would inherit the leak and produce
an excellent recovery number that does not refold — precisely the failure the
diagnostic already found in ``hbond``.

So the LW code is extracted and stored, but in its own column, **excluded from
the default feature set**.  Training reads it only under an explicit flag, so the
with/without-LW ablation is a one-line change and the default is the clean one.
Geometry that is legitimately sequence-independent — C1'-C1' distance, glycosidic
angles, relative orientation, sequence separation — carries the signal instead.

Sharding
--------
Feature extraction runs DSSR, FreeSASA and the torsion parser per structure, so
it is the expensive step.  Each shard writes one ``.npz`` per structure into a
cache directory and is safely resumable: a structure whose cache file exists is
skipped.  ``--collect`` then merges the cache into two tables.

Usage
-----
    # one shard of an array job
    python training/learned_potential/extract_features.py \\
        --split grnade_structsim --subset train \\
        --shard $SLURM_ARRAY_TASK_ID --n-shards 64 \\
        --cache training/learned_potential/cache

    # after all shards finish
    python training/learned_potential/extract_features.py --collect \\
        --cache training/learned_potential/cache \\
        --out training/learned_potential/data
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import zlib
from pathlib import Path

import numpy as np


from evo_rna.config import PATHS  # noqa: E402
from evo_rna.core.structure import NUC_TO_INDEX, Structure  # noqa: E402
from evo_rna.monte_carlo.energy_terms.propensities import CHI_INDEX  # noqa: E402
from evo_rna.data.splits import StructureID, load_grnade  # noqa: E402
from evo_rna.profile.dssr import SS_ELEMENTS  # noqa: E402
from evo_rna.profile.features import build_structure  # noqa: E402

log = logging.getLogger("extract_features")

# evo_rna.profile.torsions.extract_torsions returns DEGREES (the Structure
# field's docstring says radians — it is wrong; see README).  Kept explicit
# here because a silent unit change would corrupt every angle feature.
TORSION_COLS = ("alpha", "beta", "gamma", "delta", "epsilon", "zeta", "chi")
TORSION_UNITS = "degrees"

# Bump whenever the extractors or the feature layout change meaning.  A cache
# entry stamped with an older version is re-extracted rather than trusted: the
# features are only correct as of c8688f6 (SASA chain-label collapse + torsion
# misalignment) and acf3e05 (residues FreeSASA cannot classify), and a file
# written before those loads perfectly while carrying zeroed SASA and torsions
# shifted onto the wrong residues.  Nothing downstream could detect that.
#   1  first extraction (pre-c8688f6) -- DO NOT TRUST
#   2  post-c8688f6 + acf3e05, key-based sa/torsion alignment
FEATURE_VERSION = 2

RESIDUE_FEATURES = (
    "sasa",              # Å², raw — no binning
    "sasa_log1p",        # compresses the long tail
    *[f"{a}_{fn}" for a in TORSION_COLS for fn in ("sin", "cos")],
    *[f"{a}_defined" for a in TORSION_COLS],   # NaN indicator per angle
    *[f"elem_{e}" for e in SS_ELEMENTS],       # one-hot DSSR element
    "is_paired",
    "contact_degree",    # number of annotated contacts at this position
    "rel_position",      # i / (L-1) — 5'/3' end effects
    "log_length",        # log10 L, so a per-residue model can see target size
    "dist_prev_c1",      # C1'(i-1) - C1'(i), Å; NaN-filled at the 5' end
    "dist_next_c1",
)

PAIR_FEATURES = (
    "c1_distance",       # C1'(i) - C1'(j), Å
    "log_seq_sep",       # log10(|i-j|)
    "chi_i_sin", "chi_i_cos", "chi_j_sin", "chi_j_cos",
    "chi_defined_i", "chi_defined_j",
    "sasa_i", "sasa_j",
    "is_wc_position",    # both in pair_map with each other (geometry, not identity)
    *[f"elem_i_{e}" for e in SS_ELEMENTS],
    *[f"elem_j_{e}" for e in SS_ELEMENTS],
)


def _sin_cos(deg: float) -> tuple[float, float, float]:
    """(sin, cos, defined) for an angle in degrees; NaN -> (0, 0, 0)."""
    if deg is None or not np.isfinite(deg):
        return 0.0, 0.0, 0.0
    r = np.radians(deg)
    return float(np.sin(r)), float(np.cos(r)), 1.0


def _c1(struct: Structure, pos: int) -> np.ndarray | None:
    return struct.get_atom_xyz(pos, "C1'")


def _dist(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return float("nan")
    return float(np.linalg.norm(a - b))


def residue_rows(struct: Structure) -> tuple[np.ndarray, np.ndarray]:
    """(n_residues, n_features) matrix and (n_residues,) native-nucleotide labels."""
    L = len(struct.sequence)
    degree = np.zeros(L, dtype=np.float64)
    for i, j, _lw in struct.contacts:
        if 0 <= i < L:
            degree[i] += 1
        if 0 <= j < L:
            degree[j] += 1

    X = np.zeros((L, len(RESIDUE_FEATURES)), dtype=np.float32)
    y = np.full(L, -1, dtype=np.int8)
    log_len = float(np.log10(max(L, 1)))

    for i in range(L):
        f: list[float] = []
        sasa = float(struct.sa[i]) if i < struct.sa.shape[0] else float("nan")
        f.append(0.0 if not np.isfinite(sasa) else sasa)
        f.append(0.0 if not np.isfinite(sasa) else float(np.log1p(max(sasa, 0.0))))

        sins_coss: list[float] = []
        defined: list[float] = []
        for c in range(len(TORSION_COLS)):
            ang = float(struct.torsions[i, c]) if i < struct.torsions.shape[0] else float("nan")
            s, co, d = _sin_cos(ang)
            sins_coss += [s, co]
            defined.append(d)
        f += sins_coss + defined

        elem = int(struct.ss_element[i]) if i < struct.ss_element.shape[0] else 0
        f += [1.0 if e == elem else 0.0 for e in range(len(SS_ELEMENTS))]

        f.append(1.0 if i in struct.pair_map else 0.0)
        f.append(float(degree[i]))
        f.append(float(i) / (L - 1) if L > 1 else 0.0)
        f.append(log_len)
        f.append(_dist(_c1(struct, i - 1), _c1(struct, i)) if i > 0 else float("nan"))
        f.append(_dist(_c1(struct, i), _c1(struct, i + 1)) if i < L - 1 else float("nan"))

        X[i] = np.nan_to_num(np.asarray(f, dtype=np.float32), nan=0.0)
        y[i] = NUC_TO_INDEX.get(struct.sequence[i], -1)

    return X, y


def kept_contacts(struct: Structure) -> list[tuple[int, int, str]]:
    """The contacts :func:`contact_rows` emits, in emission order.

    Exported as a function because export_tables.py has to reproduce the (i, j)
    indices that line up with the emitted rows.  A predicate copied into two
    files drifts; when it does, the per-pair log-odds silently attach to the
    wrong residues and every lookup in the MC loop is wrong-but-plausible.
    """
    L = len(struct.sequence)
    return [(i, j, lw) for i, j, lw in struct.contacts
            if 0 <= i < L and 0 <= j < L and i != j]


def contact_rows(struct: Structure) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-contact features, (i, j) native label pair, and the held-out LW codes.

    The LW code is returned separately and deliberately NOT placed in the feature
    matrix — see the leakage rule in the module docstring.
    """
    L = len(struct.sequence)
    rows: list[list[float]] = []
    labels: list[tuple[int, int]] = []
    lws: list[str] = []

    for i, j, lw in kept_contacts(struct):
        ci, cj = _c1(struct, i), _c1(struct, j)
        si, coi, di = _sin_cos(float(struct.torsions[i, CHI_INDEX]) if i < struct.torsions.shape[0] else float("nan"))
        sj, coj, dj = _sin_cos(float(struct.torsions[j, CHI_INDEX]) if j < struct.torsions.shape[0] else float("nan"))
        ei = int(struct.ss_element[i]) if i < struct.ss_element.shape[0] else 0
        ej = int(struct.ss_element[j]) if j < struct.ss_element.shape[0] else 0

        f = [
            _dist(ci, cj),
            float(np.log10(max(abs(i - j), 1))),
            si, coi, sj, coj, di, dj,
            float(struct.sa[i]) if i < struct.sa.shape[0] else 0.0,
            float(struct.sa[j]) if j < struct.sa.shape[0] else 0.0,
            1.0 if struct.pair_map.get(i) == j else 0.0,
        ]
        f += [1.0 if e == ei else 0.0 for e in range(len(SS_ELEMENTS))]
        f += [1.0 if e == ej else 0.0 for e in range(len(SS_ELEMENTS))]

        rows.append(list(np.nan_to_num(np.asarray(f, dtype=np.float64), nan=0.0)))
        labels.append((NUC_TO_INDEX.get(struct.sequence[i], -1),
                       NUC_TO_INDEX.get(struct.sequence[j], -1)))
        lws.append(lw)

    if not rows:
        return (np.zeros((0, len(PAIR_FEATURES)), dtype=np.float32),
                np.zeros((0, 2), dtype=np.int8),
                np.array([], dtype=object))
    return (np.asarray(rows, dtype=np.float32),
            np.asarray(labels, dtype=np.int8),
            np.asarray(lws, dtype=object))


def candidate_names(sid: StructureID) -> list[str]:
    """Filenames to try for *sid*, most specific first.

    Two layouts coexist in this repo and both are supported:

    * **per-design-unit** files (``2OIU_1_P.pdb``) — what the eval subset and the
      wiring check use;
    * **whole-entry** mmCIF (``2OIU.cif``) — what ``data/pdb`` holds for the split
      structures, and what ``derive_hbond_propensity.py`` resolves against.

    The whole-entry path is the one that matters here: the learned potential is
    trained on the same TRAIN structures the binned tables were derived from, and
    those were read as whole entries with the chains selected afterwards.
    ``build_structure(..., chains=sid.chains)`` does that selection, so a
    whole-entry file gives the identical design unit.
    """
    joined = "-".join(sid.chains)
    return [
        f"{sid}.pdb", f"{sid}.cif",
        f"{sid.pdb_id}_{joined}.pdb", f"{sid.pdb_id}_{joined}.cif",
        f"{sid.pdb_id}.cif", f"{sid.pdb_id}.pdb",
        f"{sid.pdb_id.lower()}.cif", f"{sid.pdb_id.lower()}.pdb",
    ]


def resolve_path(sid: StructureID, pdb_dir: Path) -> Path | None:
    for name in candidate_names(sid):
        p = pdb_dir / name
        if p.exists():
            return p
    return None


def process_one(sid: StructureID, pdb_dir: Path, work_dir: Path, cache: Path) -> str:
    """Extract and cache one structure. Returns a short status string."""
    out = cache / f"{sid}.npz"
    if out.exists():
        # A shard killed mid-write (walltime, preemption) leaves a truncated
        # .npz that "exists" and would be skipped forever, then break --collect
        # after the whole array has already been paid for.  Touch one member to
        # prove it is readable; a bad file is deleted and re-extracted.
        try:
            with np.load(out, allow_pickle=True) as d:
                _ = d["residue_X"].shape
                version = int(d["feature_version"]) if "feature_version" in d else 1
            if version == FEATURE_VERSION:
                return "cached"
            log.warning("cache %s is feature_version %d, want %d - re-extracting",
                        out.name, version, FEATURE_VERSION)
            out.unlink(missing_ok=True)
        except Exception as exc:                      # noqa: BLE001
            log.warning("corrupt cache %s (%s) - re-extracting", out.name, exc)
            out.unlink(missing_ok=True)

    path = resolve_path(sid, pdb_dir)
    if path is None:
        return "no-structure"

    struct = build_structure(str(path), output_dir=str(work_dir), chains=sid.chains)
    if not struct.sequence:
        return "no-residues"

    Xr, yr = residue_rows(struct)
    Xc, yc, lw = contact_rows(struct)
    # Write-then-rename: os.replace is atomic within a filesystem, so a kill
    # during the write leaves the partial file under .tmp and never a
    # half-written cache entry that a later shard would trust.
    tmp = cache / f".{sid}.npz.tmp"
    with open(tmp, "wb") as fh:
        np.savez_compressed(
            fh,
            residue_X=Xr, residue_y=yr,
            contact_X=Xc, contact_y=yc, contact_lw=lw,
            target=str(sid), pdb_id=sid.pdb_id, length=len(struct.sequence),
            residue_features=np.array(RESIDUE_FEATURES, dtype=object),
            pair_features=np.array(PAIR_FEATURES, dtype=object),
            torsion_units=TORSION_UNITS,
            feature_version=FEATURE_VERSION,
        )
    os.replace(tmp, out)
    return f"ok ({Xr.shape[0]} residues, {Xc.shape[0]} contacts)"


def collect(cache: Path, out_dir: Path) -> None:
    """Merge per-structure caches into two tables, keeping the grouping key.

    ``pdb_id`` is carried through so held-out splits can be grouped by structure
    rather than by residue — residues from one chain are not independent, and a
    random residue split would leak badly.
    """
    files = sorted(cache.glob("*.npz"))
    if not files:
        raise SystemExit(f"no cached shards under {cache}")

    rX, ry, rgroup, rentry = [], [], [], []
    cX, cy, clw, cgroup, centry = [], [], [], [], []
    bad: list[str] = []
    stale: list[str] = []
    for f in files:
        try:
            d = np.load(f, allow_pickle=True)
            target = str(d["target"])
            _ = d["residue_X"].shape
            version = int(d["feature_version"]) if "feature_version" in d else 1
        except Exception as exc:                      # noqa: BLE001
            bad.append(f"{f.name}: {exc}")
            continue
        if version != FEATURE_VERSION:
            stale.append(f"{f.name}: feature_version {version}")
            continue
        n_r = d["residue_X"].shape[0]
        n_c = d["contact_X"].shape[0]
        entry = str(d["pdb_id"])
        rX.append(d["residue_X"]); ry.append(d["residue_y"])
        rgroup.append(np.full(n_r, target, dtype=object))
        rentry.append(np.full(n_r, entry, dtype=object))
        if n_c:
            cX.append(d["contact_X"]); cy.append(d["contact_y"]); clw.append(d["contact_lw"])
            cgroup.append(np.full(n_c, target, dtype=object))
            centry.append(np.full(n_c, entry, dtype=object))

    if stale:
        raise SystemExit(
            f"{len(stale)} cache files predate FEATURE_VERSION {FEATURE_VERSION} "
            f"(e.g. {stale[0]}). They were written by extractors with the SASA "
            f"and torsion bugs, and merging them would silently poison the "
            f"training set. Extraction re-does a stale entry automatically, so "
            f"just resubmit the array (it is resumable and will skip the good "
            f"files), then --collect again.")
    if not rX:
        raise SystemExit(
            f"no usable cache entries under {cache} "
            f"({len(bad)} unreadable, {len(stale)} stale of {len(files)} files)")

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "residues.npz",
        X=np.concatenate(rX), y=np.concatenate(ry),
        group=np.concatenate(rgroup),
        entry=np.concatenate(rentry),
        features=np.array(RESIDUE_FEATURES, dtype=object),
        torsion_units=TORSION_UNITS,
    )
    np.savez_compressed(
        out_dir / "contacts.npz",
        X=np.concatenate(cX) if cX else np.zeros((0, len(PAIR_FEATURES)), dtype=np.float32),
        y=np.concatenate(cy) if cy else np.zeros((0, 2), dtype=np.int8),
        lw=np.concatenate(clw) if clw else np.array([], dtype=object),
        group=np.concatenate(cgroup) if cgroup else np.array([], dtype=object),
        entry=np.concatenate(centry) if centry else np.array([], dtype=object),
        features=np.array(PAIR_FEATURES, dtype=object),
        torsion_units=TORSION_UNITS,
    )
    if bad:
        print(f"WARNING: skipped {len(bad)} unreadable cache files:")
        for b in bad[:10]:
            print(f"  {b}")
        print("  re-run the array to regenerate them (they are deleted on retry), "
              "then --collect again.")
    n_res = sum(x.shape[0] for x in rX)
    n_con = sum(x.shape[0] for x in cX)
    print(f"collected {len(files)} structures -> {n_res:,} residues, {n_con:,} contacts")
    print(f"  {out_dir/'residues.npz'}  ({len(RESIDUE_FEATURES)} features)")
    print(f"  {out_dir/'contacts.npz'}  ({len(PAIR_FEATURES)} features, LW held out separately)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="grnade_structsim")
    ap.add_argument("--subset", default="train", choices=("train", "val", "test"))
    ap.add_argument("--pdb-dir", type=Path, default=Path(PATHS.pdb_dir))
    ap.add_argument("--cache", type=Path,
                    default=Path(__file__).resolve().parent / "cache")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent / "data")
    ap.add_argument("--work-dir", type=Path, default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--collect", action="store_true",
                    help="merge cached shards into data/ instead of extracting")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-8s %(message)s")

    if args.collect:
        collect(args.cache, args.out)
        return 0

    args.cache.mkdir(parents=True, exist_ok=True)
    work_dir = args.work_dir or (args.cache / "work")
    work_dir.mkdir(parents=True, exist_ok=True)

    sids = load_grnade(args.split)[args.subset]
    if args.limit:
        sids = sids[: args.limit]
    # Shard by PDB ENTRY, not by list position.  run_dssr caches its JSON at
    # <work_dir>/<file stem>.dssr.json, and design units of one entry all resolve
    # to the same whole-entry file -- so with k % n_shards, units of the same
    # entry land in different concurrent shards and race on that one path.
    # write_text is not atomic, so a reader gets truncated JSON, run_dssr raises
    # DSSRUnavailable, and contacts_aligned CATCHES it and returns [] (dssr.py:334).
    # The structure is then cached with zero contacts and no error anywhere.
    # 5,537 of the 8,294 train units live in a multi-unit entry (4V99 alone has
    # 120), so this is the common case, not an edge case.
    # Hashing the entry keeps its units together: no race, and DSSR runs once
    # per entry instead of once per unit.
    mine = [s for s in sids
            if (zlib.crc32(s.pdb_id.encode()) & 0xFFFFFFFF) % args.n_shards == args.shard]
    log.info("shard %d/%d: %d of %d structures (%d entries)", args.shard,
             args.n_shards, len(mine), len(sids),
             len({s.pdb_id for s in mine}))

    tally: dict[str, int] = {}
    for n, sid in enumerate(mine, 1):
        try:
            status = process_one(sid, args.pdb_dir, work_dir, args.cache)
        except Exception as exc:  # noqa: BLE001 — one bad structure must not abort a shard
            status = f"error: {type(exc).__name__}"
            log.warning("skip %s: %s: %s", sid, type(exc).__name__, exc)
        key = status.split(" ")[0]
        tally[key] = tally.get(key, 0) + 1
        if n % 25 == 0 or n == len(mine):
            log.info("%d/%d  %s", n, len(mine), tally)

    log.info("shard %d done: %s", args.shard, tally)

    if tally.get("no-structure", 0) == len(mine) and mine:
        # Every target missed: almost always the wrong --pdb-dir, so say exactly
        # what was looked for and what is actually there instead of failing mute.
        log.error("no structure resolved for ANY of the %d targets in this shard.", len(mine))
        log.error("--pdb-dir was: %s (exists=%s)", args.pdb_dir, args.pdb_dir.is_dir())
        log.error("tried, for %s: %s", mine[0], ", ".join(candidate_names(mine[0])))
        if args.pdb_dir.is_dir():
            listing = sorted(p.name for p in args.pdb_dir.iterdir() if p.is_file())
            log.error("directory holds %d files; first 10: %s",
                      len(listing), listing[:10] or "(empty)")
        log.error("data/pdb holds whole-entry <PDB_ID>.cif for the split structures; "
                  "point --pdb-dir at that, or at the directory used by "
                  "scripts/propensity/derive_hbond_propensity.py --pdb-dir.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
