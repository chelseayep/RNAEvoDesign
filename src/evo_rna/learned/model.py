#!/usr/bin/env python3
"""Train the learned tertiary potential on the extracted geometry features.

Step 2 of 02_learned_potential.  Fits two heads against native nucleotides:

* **residue head** — 35 continuous geometry features -> P(nuc | geometry), the
  replacement for the ``sa`` (3 bins), ``torsion`` (2 bins) and ``junction``
  (6 bins) tables.  Those three leaves collapse into one head, which is the
  point: the diagnostic showed their weighted sum lands wrong-signed because
  ``sa`` is right-signed while ``hbond`` and ``torsion`` are not, so a model
  that learns the combination internally fixes the cancellation by construction.
* **pair head** — 23 contact-geometry features -> P(nuc_i, nuc_j | geometry),
  the replacement for the LW-keyed ``hbond`` table.

Both generalise per-RESIDUE rather than per-target, which is what makes this
trainable at all: the design evaluation has n=18 clean targets, but the train
split has thousands of structures and hundreds of thousands of residues.

The comparison that decides the contribution
--------------------------------------------
Beating a random baseline is not the bar — the binned tables already do that.
The bar is beating *them*, so this script refits the binned tables on the same
training fold and scores both on the same held-out groups:

    marginal   P(nuc) from train counts.  Floor.
    binned     log P(nuc|sasa_class) + log P(nuc|chi_bin) + log P(nuc|element),
               summed and renormalised — exactly what the three current leaves
               compute, refit here so the comparison is not confounded by the
               older derivation set.
    learned    this model.

If the MLP cannot beat ~236 parameters of binned counts, the functional form was
not the problem and the paper's primary claim does not survive.  Report that
outcome rather than tuning until it goes away.

Leaky features and --drop-features
----------------------------------
Two feature groups encode the answer they are meant to predict, and both are
droppable by name so the ablation is one flag:

* **chi** — ``evo_rna/profile/torsions.py:_chi_atoms`` selects (N9, C4) for a
  purine and (N1, C2) for a pyrimidine, so the glycosidic angle **cannot be
  computed without already knowing the base type**.  It is not backbone geometry
  that correlates with identity; its definition is chosen by the answer, and at
  design time that answer does not exist.
* **sasa** — computed over all of a residue's atoms, so a purine presents more
  area than a pyrimidine.  A milder version of the same problem.

    --drop-features chi          residue: chi_{sin,cos,defined}
                                 pair:    chi_{i,j}_{sin,cos}, chi_defined_{i,j}
    --drop-features chi,sasa     also sasa, sasa_log1p, sasa_{i,j}
    --drop-features ss           is_paired, is_wc_position, elem_* -- a different
                                 question: not "does this leak the answer" but
                                 "is this just restating the `ss` term we already
                                 have for free"

Perfect purine/pyrimidine discrimination on a near-uniform composition scores
NLL = log 2 = 0.693, so a head at ~0.88 is consistent with chi carrying most of
the signal.  If dropping it costs little, the potential rests on legitimate
geometry; if NLL degrades toward the binned tables, the headline was the leak.

The leakage rule
----------------
The Leontis-Westhof class is NOT in the default pair features.  ``hbond`` scores
AUC 1.000 at native-vs-decoy through LW annotations derived from the native
structure — a tHS geometry is often reachable only by particular bases, so the
label half-names the answer, which is why that term recovers natives beautifully
and is wrong-signed against refold TM.  ``--with-lw`` adds it for the ablation.
Expect the LW run to look better on held-out accuracy and to be the WORSE
potential; that gap is the leak, and it is a result worth reporting.

Splitting
---------
Grouped by PDB ENTRY, never by residue and never by design unit: residues in one
chain share a backbone, a fold and an evolutionary history, so a random residue
split leaks massively and would report an accuracy the MC loop can never realise.
The entry, not the unit, is the grouping key because one entry contributes up to
120 units to the train split (4V99 does) and its units share chains -- splitting
between them puts near-identical residues on both sides of the held-out line.  Groups are assigned
by ``crc32(target)``, not the builtin ``hash()`` — str hashing is salted by
PYTHONHASHSEED, so ``hash()`` would silently reshuffle the split on every run and
no two runs would be comparable.  (This repo has already been bitten by exactly
that in the decoy pools; see 01_wiring_check.)

Usage
-----
    python training/learned_potential/train.py \\
        --data training/learned_potential/data \\
        --out  training/learned_potential/models

    # the ablation the leakage rule requires
    python training/learned_potential/train.py --with-lw \\
        --out training/learned_potential/models_lw
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


from evo_rna.core.structure import NUC_ALPHABET  # noqa: E402
from evo_rna.monte_carlo.energy_terms.propensities import (  # noqa: E402
    N_ELEM,
    N_SA,
    N_TOR,
    chi_to_bin,
    sasa_to_class,
)
from evo_rna.profile.dssr import SS_ELEMENTS  # noqa: E402

log = logging.getLogger("train")

N_NUC = len(NUC_ALPHABET)
N_PAIR_CLASSES = N_NUC * N_NUC

# Feature groups that can be dropped by name.  Keyed on the column names
# extract_features.py emits, matched exactly, so a rename fails loudly here
# rather than silently dropping nothing.
DROPPABLE = {
    "chi": ("chi_sin", "chi_cos", "chi_defined",
            "chi_i_sin", "chi_i_cos", "chi_j_sin", "chi_j_cos",
            "chi_defined_i", "chi_defined_j"),
    "sasa": ("sasa", "sasa_log1p", "sasa_i", "sasa_j"),
    "torsions": tuple(f"{a}_{fn}" for a in
                      ("alpha", "beta", "gamma", "delta", "epsilon", "zeta", "chi")
                      for fn in ("sin", "cos", "defined")),
    # Everything derived from the TARGET SECONDARY STRUCTURE, which the `ss`
    # term in the secondary block already scores -- for free, O(1), leak-free.
    # `hbond` failed exactly here: 0.88 within-target Spearman with `ss`, i.e.
    # most of its apparent power was `ss` counted twice at a larger weight.  If
    # the learned head loses little without these, its gain is 3D geometry; if it
    # collapses, it is re-deriving a term the objective already has.
    #
    # contact_degree is deliberately NOT in this group: it counts DSSR tertiary
    # contacts, which no existing leaf scores.
    "ss": ("is_paired", "is_wc_position",
           *[f"elem_{e}" for e in SS_ELEMENTS],
           *[f"elem_i_{e}" for e in SS_ELEMENTS],
           *[f"elem_j_{e}" for e in SS_ELEMENTS]),
}


def select_columns(names: list[str], drop: tuple[str, ...]) -> list[int]:
    """Indices of the columns to KEEP after dropping the named groups.

    Returns indices rather than a boolean mask because the checkpoint stores
    them and export_tables.py replays the identical selection: it rebuilds the
    full feature matrix from extract_features.py and then takes these columns,
    so train and serve cannot disagree about what was dropped.
    """
    unknown = set(drop) - set(DROPPABLE)
    if unknown:
        raise SystemExit(f"unknown --drop-features group(s): {sorted(unknown)}. "
                         f"Known: {sorted(DROPPABLE)}")
    banned = {c for g in drop for c in DROPPABLE[g]}
    return [k for k, n in enumerate(names) if n not in banned]


# ---------------------------------------------------------------------------
# Grouped split
# ---------------------------------------------------------------------------

def assign_folds(groups: np.ndarray, val_frac: float, test_frac: float,
                 seed: int) -> np.ndarray:
    """Map each row to 0=train, 1=val, 2=test by hashing its GROUP, not its row.

    crc32 is used rather than hash() so the split is identical across processes
    and machines; ``seed`` shifts it for a genuinely independent re-split.
    """
    uniq = np.unique(groups)
    bucket = {
        g: ((zlib.crc32(str(g).encode()) + seed) & 0xFFFFFFFF) / 0xFFFFFFFF
        for g in uniq
    }
    fold_of = {}
    for g, u in bucket.items():
        if u < test_frac:
            fold_of[g] = 2
        elif u < test_frac + val_frac:
            fold_of[g] = 1
        else:
            fold_of[g] = 0
    return np.array([fold_of[g] for g in groups], dtype=np.int8)


# ---------------------------------------------------------------------------
# Baselines — the things the learned head has to beat
# ---------------------------------------------------------------------------

def log_marginal(y: np.ndarray, n_classes: int, alpha: float = 0.5) -> np.ndarray:
    """log P(class) with a Dirichlet(alpha) prior.

    Never log(0): a nucleotide absent from the train fold would otherwise give
    -inf, which propagates into the stored background and makes every exported
    log-odds for that base +inf.
    """
    counts = np.bincount(y, minlength=n_classes).astype(np.float64) + alpha
    return np.log(counts / counts.sum())


def _log_counts(idx: np.ndarray, y: np.ndarray, n_classes: int,
                alpha: float = 0.5) -> np.ndarray:
    """log P(nuc | class) with a symmetric Dirichlet(alpha), rows = classes."""
    counts = np.full((n_classes, N_NUC), alpha, dtype=np.float64)
    for c, nuc in zip(idx, y):
        if 0 <= c < n_classes and nuc >= 0:
            counts[c, nuc] += 1.0
    return np.log(counts / counts.sum(axis=1, keepdims=True))


def residue_bin_indices(X: np.ndarray, feature_names: list[str]) -> dict[str, np.ndarray]:
    """Recover the three binned contexts from the continuous feature matrix.

    The binning functions are imported from the runtime module rather than
    reimplemented, so the baseline is the *actual* current discretisation and a
    future change to a bin boundary cannot silently desynchronise the comparison.
    """
    col = {n: k for k, n in enumerate(feature_names)}
    sasa = X[:, col["sasa"]]
    # chi is stored as (sin, cos); invert to degrees for chi_to_bin, which is
    # defined on degrees (extract_torsions returns degrees -- see README).
    chi_deg = np.degrees(np.arctan2(X[:, col["chi_sin"]], X[:, col["chi_cos"]]))
    chi_undef = X[:, col["chi_defined"]] < 0.5
    elem_cols = [col[f"elem_{e}"] for e in SS_ELEMENTS]
    elem = np.argmax(X[:, elem_cols], axis=1)

    sa_idx = np.array([sasa_to_class(float(s)) for s in sasa], dtype=np.int64)
    chi_idx = np.array(
        [0 if u else chi_to_bin(float(d)) for d, u in zip(chi_deg, chi_undef)],
        dtype=np.int64,
    )
    return {"sa": sa_idx, "torsion": chi_idx, "junction": elem.astype(np.int64)}


def binned_residue_logits(Xtr, ytr, Xev, names) -> np.ndarray:
    """Refit the three binned tables on train, score eval rows as their sum.

    Summing three independent log P(nuc|class) tables is precisely what the
    current energy does (each leaf contributes its own log-odds, simplex-weighted
    within the tertiary block); renormalising makes it a distribution so the NLL
    is comparable with the learned head's.
    """
    tr = residue_bin_indices(Xtr, names)
    ev = residue_bin_indices(Xev, names)
    tables = {
        "sa": _log_counts(tr["sa"], ytr, N_SA),
        "torsion": _log_counts(tr["torsion"], ytr, N_TOR),
        "junction": _log_counts(tr["junction"], ytr, N_ELEM),
    }
    logits = (tables["sa"][ev["sa"]]
              + tables["torsion"][ev["torsion"]]
              + tables["junction"][ev["junction"]])
    return logits - logits.max(axis=1, keepdims=True)


def binned_pair_logits(lw_tr, ytr, lw_ev) -> np.ndarray:
    """Refit the LW-keyed pair table on train and score eval contacts.

    This baseline DOES use LW -- it is the current ``hbond`` term. That is the
    honest comparator, and it is also why the default learned head starting from
    geometry alone is the harder, leak-free bar.
    """
    vocab = {c: k for k, c in enumerate(sorted(set(map(str, lw_tr))))}
    idx_tr = np.array([vocab.get(str(c), -1) for c in lw_tr], dtype=np.int64)
    counts = np.full((len(vocab) + 1, N_PAIR_CLASSES), 0.5, dtype=np.float64)
    for c, lab in zip(idx_tr, ytr):
        if c >= 0 and lab >= 0:
            counts[c, lab] += 1.0
    table = np.log(counts / counts.sum(axis=1, keepdims=True))
    idx_ev = np.array([vocab.get(str(c), len(vocab)) for c in lw_ev], dtype=np.int64)
    return table[idx_ev]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Head(nn.Module):
    """Small MLP.  Deliberately small: 42k residues of signal, not 42M."""

    def __init__(self, n_in: int, n_out: int, hidden: tuple[int, ...],
                 dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def standardise(Xtr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Column mean/std from the TRAIN fold only; zero-variance columns -> std 1.

    Stored in the checkpoint because export_tables.py must apply the identical
    transform -- a scaler refit at inference is a silent train/serve skew.
    """
    mean = Xtr.mean(axis=0)
    std = Xtr.std(axis=0)
    std[std < 1e-8] = 1.0
    return mean, std


def metrics(logits: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict:
    """NLL in nats, top-1 accuracy, and accuracy macro-averaged over targets.

    The macro number matters because target sizes span 23-300+ nt: a micro
    average is dominated by a handful of large ribosomal fragments.
    """
    if len(y) == 0:
        return {"nll": float("nan"), "acc": float("nan"),
                "acc_macro": float("nan"), "n": 0}
    z = logits - logits.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    nll = float(-logp[np.arange(len(y)), y].mean())
    pred = logits.argmax(axis=1)
    acc = float((pred == y).mean())
    per_target = [
        float((pred[groups == g] == y[groups == g]).mean())
        for g in np.unique(groups)
    ]
    return {"nll": nll, "acc": acc, "acc_macro": float(np.mean(per_target)),
            "n": int(len(y))}


def train_head(Xtr, ytr, Xva, yva, gva, n_out, args, tag: str):
    dev = torch.device(args.device)
    mean, std = standardise(Xtr)
    xtr = torch.tensor((Xtr - mean) / std, dtype=torch.float32, device=dev)
    xva = torch.tensor((Xva - mean) / std, dtype=torch.float32, device=dev)
    ttr = torch.tensor(ytr, dtype=torch.long, device=dev)
    tva = torch.tensor(yva, dtype=torch.long, device=dev)

    model = Head(Xtr.shape[1], n_out, tuple(args.hidden), args.dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    lossf = nn.CrossEntropyLoss()

    best = (float("inf"), None, -1)
    n = len(xtr)
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n)
        for k in range(0, n, args.batch_size):
            sel = perm[k:k + args.batch_size]
            opt.zero_grad()
            loss = lossf(model(xtr[sel]), ttr[sel])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            vloss = float(lossf(model(xva), tva))
        if vloss < best[0] - 1e-5:
            best = (vloss, {k: v.clone() for k, v in model.state_dict().items()}, epoch)
        if epoch - best[2] >= args.patience:
            log.info("[%s] early stop at epoch %d (best %d, val nll %.4f)",
                     tag, epoch, best[2], best[0])
            break
        if epoch % 10 == 0:
            log.info("[%s] epoch %3d  val nll %.4f", tag, epoch, vloss)

    if best[1] is None:
        raise SystemExit(
            f"[{tag}] validation loss was never finite ({best[0]}), so no epoch "
            f"was ever the best and there is nothing to save. Almost always the "
            f"fold is empty or degenerate: {len(xtr)} train rows, {len(xva)} val "
            f"rows. Check that data/ holds the whole split -- the train subset is "
            f"8,294 units, and a few hundred rows means the extraction array has "
            f"not landed yet.")
    model.load_state_dict(best[1])
    model.eval()
    with torch.no_grad():
        va_logits = model(xva).cpu().numpy()
    # Checkpoints are written and reloaded on CPU: export_tables.py runs at MC
    # init on whatever node the design job landed on, which need not have a GPU.
    return model.cpu(), mean, std, metrics(va_logits, yva, gva)


def evaluate(model, mean, std, X, y, groups) -> dict:
    model.eval()
    dev = next(model.parameters()).device
    with torch.no_grad():
        x = torch.tensor((X - mean) / std, dtype=torch.float32, device=dev)
        logits = model(x).cpu().numpy()
    return metrics(logits, y, groups)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent, text=True).strip()
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path,
                    default=Path(__file__).resolve().parent / "data")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent / "models")
    ap.add_argument("--with-lw", action="store_true",
                    help="add the Leontis-Westhof class to the PAIR features "
                         "(ablation only -- see the leakage rule)")
    ap.add_argument("--hidden", type=int, nargs="+", default=[128, 128])
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--torch-seed", type=int, default=7)
    ap.add_argument("--drop-features", default="",
                    help="comma-separated feature groups to exclude: "
                         "chi (the glycosidic angle, whose definition requires "
                         "knowing purine vs pyrimidine), sasa (scales with base "
                         "size), torsions (all backbone angles), ss (everything "
                         "from the target secondary structure, which the free "
                         "O(1) `ss` term already scores). Applied to both heads "
                         "and recorded in the checkpoint so export_tables.py "
                         "masks identically.")
    ap.add_argument("--skip-pairs", action="store_true")
    ap.add_argument("--device", default="cpu",
                    help="cpu (default) or cuda. Two 128-unit MLPs over ~1e6 rows "
                         "train in minutes on CPU -- do NOT hold a GPU for this; "
                         "the expensive step is the extraction array, which is "
                         "pure CPU (DSSR/FreeSASA).")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-8s %(message)s")
    torch.manual_seed(args.torch_seed)
    drop = tuple(g.strip() for g in args.drop_features.split(",") if g.strip())
    args.out.mkdir(parents=True, exist_ok=True)
    report: dict = {"git_commit": git_commit(), "dropped": list(drop),
                    "args": vars(args) | {
        "data": str(args.data), "out": str(args.out), "hidden": list(args.hidden)}}

    # ---------------- residue head ----------------
    d = np.load(args.data / "residues.npz", allow_pickle=True)
    X, y, groups = d["X"], d["y"].astype(np.int64), d["group"]
    entries = d["entry"] if "entry" in d else np.array(
        [str(g).split("_")[0] for g in groups], dtype=object)
    names = [str(s) for s in d["features"]]
    # The binned baseline IS the current energy, which reads sasa/chi/element
    # whatever the learned head is allowed to see.  Keep the full matrix for it,
    # so --drop-features changes only the model under test and the comparator
    # stays fixed across arms.
    full_names = list(names)
    cols = select_columns(names, drop)
    X_full = X
    if len(cols) != len(names):
        log.warning("dropping %d of %d residue features (%s): %s",
                    len(names) - len(cols), len(names), ",".join(drop),
                    [n for k, n in enumerate(names) if k not in set(cols)])
        X, names = X[:, cols], [names[k] for k in cols]
    keep = y >= 0                       # drop non-AUGC (modified/unknown) residues
    X, X_full = X[keep], X_full[keep]
    y, groups, entries = y[keep], groups[keep], entries[keep]
    log.info("residues: %d rows, %d units, %d entries, %d features "
             "(dropped %d non-AUGC)", len(y), len(np.unique(groups)),
             len(np.unique(entries)), X.shape[1], int((~keep).sum()))

    fold = assign_folds(entries, args.val_frac, args.test_frac, args.split_seed)
    tr, va, te = fold == 0, fold == 1, fold == 2
    for nm, m in (("train", tr), ("val", va), ("test", te)):
        log.info("  %-5s %7d rows  %4d units  %4d entries", nm, int(m.sum()),
                 len(np.unique(groups[m])), len(np.unique(entries[m])))
    for nm, m in (("train", tr), ("val", va), ("test", te)):
        if not m.any():
            raise SystemExit(
                f"the {nm} fold is empty: {len(np.unique(entries))} entries in "
                f"data/ cannot be split {1 - args.val_frac - args.test_frac:.0%}/"
                f"{args.val_frac:.0%}/{args.test_frac:.0%} by entry. The train "
                f"subset has 4,574 entries; if data/ has a handful, --collect ran "
                f"against an incomplete cache -- check the extraction array "
                f"finished and re-run --collect.")
    shared = (set(np.unique(entries[tr])) & set(np.unique(entries[te])))
    if shared:
        raise SystemExit(f"{len(shared)} entries on both sides of the split "
                         f"(e.g. {sorted(shared)[:3]}) -- folds are not grouped")

    model, mean, std, va_m = train_head(X[tr], y[tr], X[va], y[va], groups[va],
                                        N_NUC, args, "residue")
    res = {
        "learned": evaluate(model, mean, std, X[te], y[te], groups[te]),
        "binned": metrics(
            binned_residue_logits(X_full[tr], y[tr], X_full[te], full_names),
            y[te], groups[te]),
        "marginal": metrics(np.tile(log_marginal(y[tr], N_NUC),
                                    (int(te.sum()), 1)), y[te], groups[te]),
        "val": va_m,
    }
    report["residue"] = res
    background = log_marginal(y[tr], N_NUC)
    torch.save({"state_dict": model.state_dict(), "mean": mean, "std": std,
                "background": background, "dropped": list(drop),
                "feature_index": cols,
                "features": names, "hidden": list(args.hidden),
                "dropout": args.dropout, "n_out": N_NUC,
                "nuc_alphabet": list(NUC_ALPHABET),
                "torsion_units": str(d["torsion_units"]),
                "git_commit": report["git_commit"]},
               args.out / "residue_head.pt")

    # ---------------- pair head ----------------
    if not args.skip_pairs:
        c = np.load(args.data / "contacts.npz", allow_pickle=True)
        cX, cy2, cg, clw = c["X"], c["y"].astype(np.int64), c["group"], c["lw"]
        centry = c["entry"] if "entry" in c else np.array(
            [str(g).split("_")[0] for g in cg], dtype=object)
        cnames = [str(s) for s in c["features"]]
        ok = (cy2[:, 0] >= 0) & (cy2[:, 1] >= 0)
        cX, cy2, cg, clw, centry = cX[ok], cy2[ok], cg[ok], clw[ok], centry[ok]
        cy = cy2[:, 0] * N_NUC + cy2[:, 1]      # joint 16-way label

        lw_vocab: list[str] = []
        if args.with_lw:
            # One-hot appended LAST so the leak-free columns keep their indices
            # and a leak-free checkpoint stays a prefix of this one.
            lw_vocab = sorted(set(map(str, clw)))
            onehot = np.zeros((len(clw), len(lw_vocab)), dtype=np.float32)
            for k, code in enumerate(map(str, clw)):
                onehot[k, lw_vocab.index(code)] = 1.0
            cX = np.hstack([cX, onehot])
            cnames = cnames + [f"lw_{v}" for v in lw_vocab]
            log.warning("--with-lw: %d LW classes added. This is the ABLATION arm; "
                        "expect better accuracy and a worse potential.", len(lw_vocab))

        ccols = select_columns(cnames, drop)
        if len(ccols) != len(cnames):
            log.warning("dropping %d of %d pair features (%s): %s",
                        len(cnames) - len(ccols), len(cnames), ",".join(drop),
                        [n for k, n in enumerate(cnames) if k not in set(ccols)])
            cX, cnames = cX[:, ccols], [cnames[k] for k in ccols]

        cfold = assign_folds(centry, args.val_frac, args.test_frac, args.split_seed)
        ctr, cva, cte = cfold == 0, cfold == 1, cfold == 2
        log.info("contacts: %d rows, %d targets, %d features",
                 len(cy), len(np.unique(cg)), cX.shape[1])

        pmodel, pmean, pstd, pva = train_head(cX[ctr], cy[ctr], cX[cva], cy[cva],
                                              cg[cva], N_PAIR_CLASSES, args, "pair")
        report["pair"] = {
            "learned": evaluate(pmodel, pmean, pstd, cX[cte], cy[cte], cg[cte]),
            "binned_lw": metrics(binned_pair_logits(clw[ctr], cy[ctr], clw[cte]),
                                 cy[cte], cg[cte]),
            "marginal": metrics(np.tile(log_marginal(cy[ctr], N_PAIR_CLASSES),
                                        (int(cte.sum()), 1)),
                                cy[cte], cg[cte]),
            "val": pva,
            "with_lw": bool(args.with_lw),
        }
        pbackground = log_marginal(cy[ctr], N_PAIR_CLASSES)
        torch.save({"state_dict": pmodel.state_dict(), "mean": pmean, "std": pstd,
                    "background": pbackground, "dropped": list(drop),
                    "feature_index": ccols,
                    "features": cnames, "hidden": list(args.hidden),
                    "dropout": args.dropout, "n_out": N_PAIR_CLASSES,
                    "nuc_alphabet": list(NUC_ALPHABET), "with_lw": bool(args.with_lw),
                    "lw_vocab": lw_vocab,
                    "torsion_units": str(c["torsion_units"]),
                    "git_commit": report["git_commit"]},
                   args.out / "pair_head.pt")

    (args.out / "report.json").write_text(json.dumps(report, indent=2, default=str))

    print("\n=== held-out (grouped by structure) ===")
    print(f"{'head':8s} {'model':11s} {'NLL':>7s} {'acc':>7s} {'acc_macro':>10s}")
    for head in ("residue", "pair"):
        if head not in report:
            continue
        for k in [x for x in report[head] if isinstance(report[head][x], dict)
                  and x != "val"]:
            m = report[head][k]
            print(f"{head:8s} {k:11s} {m['nll']:7.4f} {m['acc']:7.4f} "
                  f"{m['acc_macro']:10.4f}")
    r = report["residue"]
    verdict = ("BEATS the binned tables" if r["learned"]["nll"] < r["binned"]["nll"]
               else "DOES NOT beat the binned tables -- the functional form was "
                    "not the problem; report this")
    print(f"\nresidue head {verdict} "
          f"(NLL {r['learned']['nll']:.4f} vs {r['binned']['nll']:.4f})")
    print(f"report -> {args.out / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
