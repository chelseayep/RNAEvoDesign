# RNAEvoDesign

Monte Carlo RNA inverse folding: given a target backbone, design a sequence
that folds to it. The objective combines an evolutionary profile built from
structural homologues with secondary-structure terms and a **learned tertiary
potential** over continuous geometry.

The defaults are the arm the paper reports (**A2**): TM-refit block weights
over the learned tertiary leaf, SAMFEO moves, uniform position sampling.
Running `rna-evo-design <target.pdb>` reproduces that configuration.

## Install

```bash
git clone https://github.com/chelseayep/RNAEvoDesign.git
cd RNAEvoDesign
pip install -e .
```

Python 3.10+. ViennaRNA, NumPy, Biopython, FreeSASA and PyTorch come with the
install. Two external binaries are fetched separately:

```bash
bash scripts/setup/setup_RNAfold.sh      # ViennaRNA CLI
```

- **RNAalign** (`bin/RNAalign/RNAalign`) aligns the target against the template
  library to build the PSSM. Without it, pass `--pssm` or run with a flat
  profile (the Evo term goes inert).
- **X3DNA-DSSR** (`tools/x3dna-dssr`, licensed separately) annotates base pairs
  and contacts. Without it the pipeline falls back to geometric pair detection
  and the contact part of the tertiary block is empty.

Point the config at whatever paths you have — see `configs/default.yaml`, or
override any single one with `EVO_RNA_<KEY>` (e.g. `EVO_RNA_DSSR_BIN=...`).

## Checkpoints

The A2 arm needs two things that are not in the repo:

| what | where it goes | notes |
|---|---|---|
| trained heads | `models/learned_tertiary/{residue_head,pair_head}.pt` | see `models/README.md` |
| fitted weights | `weights/a2_learned.json` | see `weights/README.md` |

Both can be retrained from scratch with `training/` (below). Without the heads
the A2 run stops immediately with a message saying so, rather than quietly
designing against a different objective.

## Design

```bash
# A2 (default): learned tertiary leaf, PSSM from a template library
rna-evo-design data/rna_pdb/8G4W_1_R.pdb --pdb-lib data/rna_pdb_lib

# pre-computed PSSM instead of an RNAalign sweep
rna-evo-design target.pdb --pssm profile/prf.txt

# no profile at all — flat PSSM, Evo term inert
rna-evo-design target.pdb

# quick check that everything is wired (seconds, not hours)
rna-evo-design target.pdb --steps-per-temp 100
```

Output lands in `runs/<target>/`:

- **`design.json`** — the designed sequence, plus the full record of the run:
  seed, accept rate, per-move-kind counters, the resolved block weights, the
  PSSM depth behind the Evo term, and the final per-term energies.
- `energy_log.json` — per-term energies at every accepted move, with
  `--log-energy`.
- `profile/` — the PSSM and structural features, when generated here.

The native sequence is read from the target only to be recorded in
`design.json` for comparison. It never enters the objective and is never the
starting point: the MC starts from a random sequence.

**No evaluation is included.** The pipeline ends at the designed sequence;
fold and score it with whatever you prefer.

### The three arms

A1 is the paired control for A2 — same fit procedure, same pools, same dropped
leaves, differing only in the tertiary leaf. A0 shows what the refit alone
bought.

```bash
# A2  learned     (the method, default)
rna-evo-design target.pdb

# A1  reweighted  (the control)
rna-evo-design target.pdb --no-learned-tertiary --weights weights/a1_reweighted.json

# A0  baseline
rna-evo-design target.pdb --no-learned-tertiary --weights weights/a0_baseline.json
```

### The objective

Three gauge-fixed blocks, `w_evo·E_evo + w_secondary·E_secondary +
w_tertiary·E_tertiary`. Each block is a simplex-normalised mix of its leaves,
so the `--w-*` flags carry block magnitude and the `--*-weights` flags carry
the within-block mix; the two levels are separately identifiable.

| block | leaves |
|---|---|
| evo | PSSM log-odds from structural homologues |
| secondary | `ss`, `stacking_context`, `ensemble_defect`, `junction` |
| tertiary | the learned potential (A2) — or binned `hbond`, `sa`, `torsion` (A1/A0) |

The learned leaf replaces all three binned ones with a single head over
continuous geometry. The binned leaves are kept rather than deleted: they are
the control arm, and the question "did the learned potential help, or would
reweighting alone have done it?" is unanswerable once they are gone.
`evo_rna/monte_carlo/ENERGY_TERMS.md` documents each leaf.

A GC-composition prior (`--w-composition`) and a depth-adaptive Evo weight
(`--w-evo-shallow`) are available but **off** by default: neither was in the
weight fit, so enabling one perturbs the fitted balance.

## Retraining the potential

`training/learned_potential/` retrains the tertiary potential from structures:
extract the geometry features, fit the residue and pair heads, export tables.
It needs nothing beyond what a design run already needs (DSSR, FreeSASA), but
the extraction sweeps the training split and is cluster-scale. Its README has
the exact sequence. None of it is needed to *run* a design.

`oof_targets.py` there reports which fold each target fell in, so you can tell
whether the shipped checkpoint was trained on a structure you intend to design
against.

### The block weights

The weight files in `weights/` are released artifacts, not reproducible from
this repository. They were produced in two stages: a contrastive fit over
native-vs-decoy rankings within each target, then refinement of that warm start
against mean refold TM-score. Both stages need machinery outside the scope of a
design-only release — decoy pools from several external inverse-folding models,
and a structure-prediction and alignment harness to score against. The fitted
result ships instead.

If you want to weight a *different* leaf set, the loader
(`evo_rna/monte_carlo/weights.py`) documents both accepted file shapes; any
fit of your own that emits them will load.

## Layout

```
src/evo_rna/
  main/design.py      the design pipeline (CLI entry point)
  monte_carlo/        energy terms, simulated annealing, weight loading
  learned/            the learned tertiary potential: features, head, tables
  profile/            PSSM generation, DSSR, SASA, torsions
  core/  io/  data/   structures, PDB handling, gRNAde split id-lists
configs/default.yaml  paths + the A2 run defaults
scripts/setup/        external tool installers
scripts/data/         structure download and design-unit preparation
training/             retrain the learned tertiary potential
weights/              fitted block weights, one file per arm
```

## Citation

Paper in preparation.

## License

MIT — see `LICENSE`.
