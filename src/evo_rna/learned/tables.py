#!/usr/bin/env python3
"""Emit per-target log-odds arrays from the trained heads.

Step 3 of 02_learned_potential, and the step that keeps the MC hot loop
unchanged.  The network runs ONCE per target, at initialisation, and writes:

    residue_logodds   (L, 4)      log P(nuc | geometry_i) - log P(nuc)
    contact_logodds   (C, 4, 4)   log P(nuc_i, nuc_j | geometry) - log P(nuc_i, nuc_j)
    contacts          (C, 2)      the (i, j) positions those pairs refer to

Both are exactly the shapes the existing terms already index.  ``residue_logodds``
is indexed ``[i, nuc]`` -- the same access pattern ``EvoTerm`` uses on the PSSM --
and ``contact_logodds`` is indexed ``[c, nuc_i, nuc_j]``, which is what
``DSSRHbondTerm`` does today through ``propensity.score(lw, base_i, base_j)``.
So the runtime term stays an O(1) table lookup per mutated position plus O(degree)
over its contacts; no torch, no network, nothing per-move that was not there
before.  The ablation is then a one-flag switch between these arrays and the
binned tables, which is what makes the paired arm affordable.

Columns follow NUC_ALPHABET (A, U, G, C) -- the same order the labels were built
with in extract_features.py, and the order NUC_TO_INDEX gives the runtime.  The
checkpoint carries the alphabet and this script asserts on it rather than
trusting the coincidence.

Log-odds, not log-probabilities: the background is the TRAIN-fold marginal stored
in the checkpoint at fit time.  Recomputing a background here would be a silent
train/serve skew, which is the same class of bug as refitting a scaler at
inference.

Usage
-----
    python training/learned_potential/export_tables.py \\
        --targets my_targets.txt \\
        --models models/learned_tertiary \\
        --out    data/learned_tables

    # the whole test split
    python training/learned_potential/export_tables.py \\
        --split grnade_structsim --subset test
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

from evo_rna.config import PATHS
from evo_rna.core.structure import NUC_ALPHABET, NUC_TO_INDEX
from evo_rna.data.splits import StructureID, load_grnade
from evo_rna.profile.features import build_structure

from evo_rna.learned.features import (
    PAIR_FEATURES,
    RESIDUE_FEATURES,
    contact_rows,
    kept_contacts,
    residue_rows,
    resolve_path,
)
from evo_rna.learned.model import Head

log = logging.getLogger("export_tables")
N_NUC = len(NUC_ALPHABET)


def load_head(path: Path) -> dict:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if list(ck["nuc_alphabet"]) != list(NUC_ALPHABET):
        raise SystemExit(
            f"{path}: checkpoint alphabet {ck['nuc_alphabet']} != runtime "
            f"{list(NUC_ALPHABET)} -- the emitted columns would be permuted "
            f"and every lookup silently wrong.")
    model = Head(len(ck["features"]), ck["n_out"], tuple(ck["hidden"]), ck["dropout"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    ck["model"] = model
    return ck


def log_odds(ck: dict, X: np.ndarray) -> np.ndarray:
    """Select the fit-time columns, standardise with the fit-time scaler, run.

    ``feature_index`` replays whatever --drop-features excluded during training.
    Selecting by stored index rather than by recomputing the drop rule here means
    a checkpoint and this script cannot disagree about which columns were used.
    """
    idx = ck.get("feature_index")
    if idx is not None and len(idx) != X.shape[1]:
        X = X[:, list(idx)]
    if X.shape[1] != len(ck["features"]):
        raise SystemExit(
            f"feature width {X.shape[1]} != checkpoint's {len(ck['features'])}. "
            f"extract_features.py and the checkpoint disagree; re-extract or "
            f"retrain rather than padding.")
    z = torch.tensor((X - ck["mean"]) / ck["std"], dtype=torch.float32)
    with torch.no_grad():
        logits = ck["model"](z).numpy()
    logp = logits - logits.max(axis=1, keepdims=True)
    logp = logp - np.log(np.exp(logp).sum(axis=1, keepdims=True))
    return (logp - ck["background"]).astype(np.float32)


def self_check(seq: str, res_lo: np.ndarray, con: np.ndarray,
               con_lo: np.ndarray, rng: np.random.Generator) -> dict:
    """Score the native against random sequences on these very arrays.

    Cheap, but it is the same question 01_wiring_check asked of the binned terms,
    and it catches a transposed axis or a permuted alphabet immediately -- a
    potential that cannot beat random on the native's own backbone is miswired,
    whatever its held-out NLL said.
    """
    idx = np.array([NUC_TO_INDEX.get(c, -1) for c in seq])
    ok = idx >= 0

    def energy(v: np.ndarray) -> float:
        e = -res_lo[np.arange(len(v))[ok], v[ok]].sum()
        if len(con):
            i, j = con[:, 0], con[:, 1]
            good = (v[i] >= 0) & (v[j] >= 0)
            e -= con_lo[np.arange(len(con))[good], v[i][good], v[j][good]].sum()
        return float(e / max(len(v), 1))

    native = energy(idx)
    decoys = [energy(rng.integers(0, N_NUC, size=len(idx))) for _ in range(24)]
    return {"native": native, "decoy_mean": float(np.mean(decoys)),
            "beats_random": bool(native < np.mean(decoys))}


def export_one(sid: StructureID, path: Path, work_dir: Path, out_dir: Path,
               res_ck: dict, pair_ck: dict | None,
               rng: np.random.Generator) -> dict | None:
    return export_path(path, str(sid), work_dir, out_dir, res_ck, pair_ck, rng,
                       chains=sid.chains)


def export_path(path: Path, target: str, work_dir: Path, out_dir: Path,
                res_ck: dict, pair_ck: dict | None,
                rng: np.random.Generator,
                chains: "str | None" = None) -> dict | None:
    """Write ``<out_dir>/<target>.npz`` for one structure file.

    The ``sid``-free form: the design pipeline knows a path and a stem, not a
    split entry, and must be able to build a table for a target that is in no
    split at all.
    """
    struct = build_structure(str(path), output_dir=str(work_dir), chains=chains)
    if not struct.sequence:
        return None

    Xr, _ = residue_rows(struct)
    res_lo = log_odds(res_ck, Xr)

    contacts = np.zeros((0, 2), dtype=np.int32)
    con_lo = np.zeros((0, N_NUC, N_NUC), dtype=np.float32)
    if pair_ck is not None:
        Xc, _, lw = contact_rows(struct)
        if len(Xc):
            if pair_ck.get("with_lw"):
                # Ablation checkpoint: rebuild the same one-hot, in the same
                # vocabulary order, appended last.  An unseen class is all-zero,
                # which is what the model saw for rare codes anyway.
                vocab = list(pair_ck["lw_vocab"])
                oh = np.zeros((len(lw), len(vocab)), dtype=np.float32)
                for k, code in enumerate(map(str, lw)):
                    if code in vocab:
                        oh[k, vocab.index(code)] = 1.0
                Xc = np.hstack([Xc, oh])
            flat = log_odds(pair_ck, Xc)
            con_lo = flat.reshape(-1, N_NUC, N_NUC).astype(np.float32)
            # Same helper contact_rows uses, so row c of contact_logodds is
            # guaranteed to be the pair contacts[c] -- not merely likely to be.
            contacts = np.array([[i, j] for i, j, _ in kept_contacts(struct)],
                                dtype=np.int32)
            if len(contacts) != len(con_lo):
                raise SystemExit(
                    f"{sid}: {len(contacts)} contacts but {len(con_lo)} rows -- "
                    f"contact_rows and this filter disagree; they must apply the "
                    f"identical predicate or the (i, j) indices are misaligned.")

    check = self_check(struct.sequence, res_lo, contacts, con_lo, rng)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / f"{target}.npz",
        residue_logodds=res_lo,
        contact_logodds=con_lo,
        contacts=contacts,
        sequence=str(struct.sequence),
        target=str(target),
        nuc_alphabet=np.array(list(NUC_ALPHABET), dtype=object),
        residue_features=np.array(RESIDUE_FEATURES, dtype=object),
        pair_features=np.array(PAIR_FEATURES, dtype=object),
        with_lw=bool(pair_ck.get("with_lw")) if pair_ck else False,
        dropped=np.array(list(res_ck.get("dropped", [])), dtype=object),
        model_commit=str(res_ck.get("git_commit", "unknown")),
    )
    return {"target": str(target), "L": len(struct.sequence),
            "contacts": int(len(contacts)), **check}


def default_models_dir() -> Path:
    """Trained residue/pair heads. ``paths.learned_models`` in the config."""
    return Path(PATHS.learned_models)


def default_tables_dir() -> Path:
    """Where per-target tables are cached. ``paths.learned_tables``."""
    return Path(PATHS.learned_tables)


def ensure_table(pdb_path: "str | Path", target: str,
                 tables_dir: "str | Path | None" = None,
                 models_dir: "str | Path | None" = None,
                 chains: "str | None" = None,
                 work_dir: "str | Path | None" = None,
                 seed: int = 7) -> Path:
    """Return ``<tables_dir>/<target>.npz``, building it from the heads if absent.

    The A2 arm needs one table per design target, and a user designing on a
    structure that is in no split has no batch export to run first -- so the
    pipeline builds it on demand.  An existing table is reused untouched: the
    network is deterministic, and re-deriving it per run would add a torch load
    to every job for no change in output.
    """
    tables_dir = Path(tables_dir) if tables_dir is not None else default_tables_dir()
    out = tables_dir / f"{target}.npz"
    if out.exists():
        log.info("learned tertiary table: reusing %s", out)
        return out

    models_dir = Path(models_dir) if models_dir is not None else default_models_dir()
    res_path = models_dir / "residue_head.pt"
    if not res_path.exists():
        raise SystemExit(
            f"no learned-potential checkpoint at {res_path}. The A2 arm needs "
            f"the trained heads: point --learned-models at them, set "
            f"paths.learned_models in the config, or train them with "
            f"training/learned_potential/train.py.")
    res_ck = load_head(res_path)
    pair_path = models_dir / "pair_head.pt"
    pair_ck = load_head(pair_path) if pair_path.exists() else None
    if pair_ck is None:
        log.warning("no pair_head.pt in %s -- the table carries residue log-odds "
                    "only, so the contact part of the leaf is empty.", models_dir)

    work = Path(work_dir) if work_dir is not None else tables_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    log.info("learned tertiary table: building %s from %s", out, models_dir)
    row = export_path(Path(pdb_path), target, work, tables_dir, res_ck, pair_ck,
                      np.random.default_rng(seed), chains=chains)
    if row is None:
        raise SystemExit(f"{target}: no residues extracted from {pdb_path}")
    if not row["beats_random"]:
        # Same self-check the batch export prints, promoted to a warning here:
        # a table the native loses on is miswired, and the MC would optimise it.
        log.warning("%s: native (%.4f) does NOT beat random decoys (%.4f) on this "
                    "table -- check the structure and the checkpoint before "
                    "trusting the run.", target, row["native"], row["decoy_mean"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--targets", type=Path, help="file of structure ids, one per line")
    src.add_argument("--split", default=None, help="e.g. grnade_structsim")
    ap.add_argument("--subset", default="test", choices=("train", "val", "test"))
    ap.add_argument("--pdb-dir", type=Path, default=Path(PATHS.pdb_dir))
    ap.add_argument("--models", type=Path, default=default_models_dir())
    ap.add_argument("--out", type=Path, default=default_tables_dir())
    ap.add_argument("--work-dir", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-8s %(message)s")

    res_ck = load_head(args.models / "residue_head.pt")
    pair_path = args.models / "pair_head.pt"
    pair_ck = load_head(pair_path) if pair_path.exists() else None
    if pair_ck is None:
        log.warning("no pair_head.pt in %s -- emitting residue log-odds only, "
                    "so the hbond leaf is NOT replaced by this export.", args.models)
    if res_ck.get("dropped"):
        log.info("residue head was fit without: %s", ",".join(res_ck["dropped"]))
    if pair_ck is not None and pair_ck.get("dropped"):
        log.info("pair head was fit without: %s", ",".join(pair_ck["dropped"]))
    if pair_ck is not None and pair_ck.get("with_lw"):
        log.warning("pair head was trained WITH Leontis-Westhof features: this is "
                    "the ablation arm, not the default potential.")

    if args.targets:
        sids = [StructureID.from_grnade(ln.split()[0])
                for ln in args.targets.read_text().splitlines() if ln.strip()]
    else:
        sids = load_grnade(args.split)[args.subset]
    if args.limit:
        sids = sids[: args.limit]

    work_dir = args.work_dir or (args.out / "work")
    work_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    rows, missing, failed = [], [], []
    for n, sid in enumerate(sids, 1):
        path = resolve_path(sid, args.pdb_dir)
        if path is None:
            missing.append(str(sid))
            continue
        try:
            r = export_one(sid, path, work_dir, args.out, res_ck, pair_ck, rng)
        except SystemExit:
            raise
        except Exception as exc:                      # noqa: BLE001
            failed.append(f"{sid}: {type(exc).__name__}: {exc}")
            continue
        if r is None:
            failed.append(f"{sid}: no residues extracted")
            continue
        rows.append(r)
        log.info("[%d/%d] %s  L=%d contacts=%d  native=%.4f decoys=%.4f %s",
                 n, len(sids), sid, r["L"], r["contacts"], r["native"],
                 r["decoy_mean"], "ok" if r["beats_random"] else "*** WORSE ***")

    print(f"\nexported {len(rows)} targets -> {args.out}")
    if missing:
        print(f"  {len(missing)} with no structure file: {', '.join(missing[:5])}"
              f"{' ...' if len(missing) > 5 else ''}")
    for f in failed:
        print(f"  FAILED {f}")
    if rows:
        beat = sum(r["beats_random"] for r in rows)
        print(f"  self-check: native beats random decoys on {beat}/{len(rows)} targets")
        if beat < len(rows):
            print("  *** a target where the native loses to random is miswired -- "
                  "check the alphabet order and the contact index alignment before "
                  "wiring this into the MC loop.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
