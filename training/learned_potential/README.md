# Learned tertiary potential

Replaces the three binned tertiary leaves (`hbond`, `sa`, `torsion`) with one
learned potential over continuous geometry. **Primary contribution of the paper.**


## Running

Extraction runs DSSR, FreeSASA and the torsion parser per structure, so it is the
expensive step. Shards are resumable — a structure whose cache file exists is skipped.

```bash
# shard the train split however your cluster prefers; 64 shards is a reasonable
# split of the work, and shards are independent
python training/learned_potential/extract_features.py --shard 0 --n-shards 64
# ... shards 1..63 ...

# merge once every shard has finished
python training/learned_potential/extract_features.py --collect
```

**Check that DSSR resolves before launching a large extraction.** This is not
paranoia: `contacts_aligned` catches `DSSRUnavailable` and returns `[]`,
so a missing binary yields structures with zero contacts and no error anywhere —
visible only much later as an implausibly small contact total.

Cache entries carry a `FEATURE_VERSION`; anything written before the SASA and
torsion fixes is re-extracted automatically, and `--collect` refuses to merge a
stale entry rather than silently poisoning the training set.

Sanity-check on a handful of structures first:

```bash
python training/learned_potential/extract_features.py --limit 5 --n-shards 1
python training/learned_potential/extract_features.py --collect
```

## Training and export

```bash
# 2. fit both heads (grouped held-out split, LW excluded by default).
#    CPU, minutes -- do not request a GPU.
python training/learned_potential/train.py \
    --data training/learned_potential/data \
    --out  models/learned_tertiary

# the ablation arms: --with-lw, --drop-features chi,
# --drop-features chi,sasa, --drop-features ss

# 3. (optional) pre-emit per-target log-odds arrays. The design pipeline builds
#    a missing table on demand, so this is only worth doing for a batch of
#    targets you already know.
python training/learned_potential/export_tables.py \
    --split grnade_structsim --subset test \
    --models models/learned_tertiary \
    --out    data/learned_tables
```

`train.py` refits the binned tables on the same training fold and scores both on
the same held-out groups, so the printed comparison is against the 236 parameters
this is meant to replace, not against chance. Splits are grouped by structure and
assigned by `crc32(target)` — not the builtin `hash()`, which is salted by
PYTHONHASHSEED and would reshuffle the split on every run.

`export_tables.py` runs each head ONCE per target and writes `residue_logodds`
(L, 4) and `contact_logodds` (C, 4, 4) plus the (i, j) `contacts` index. Those are
the shapes the existing terms already index — `EvoTerm` accesses the PSSM as
`[i, nuc]`, `DSSRHbondTerm` as a per-contact base-pair lookup — so the MC hot loop
stays an O(1) table read per move and needs no torch. Each export self-checks the
native against random decoys on its own arrays; a target where the native loses is
miswired regardless of held-out NLL.

Still to write: the runtime `LearnedTertiaryTerm` that reads these arrays behind
the ablation flag.

