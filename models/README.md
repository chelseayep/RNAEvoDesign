# Trained checkpoints

The learned tertiary potential's heads live here:

```
models/learned_tertiary/residue_head.pt   # required by the A2 arm
models/learned_tertiary/pair_head.pt      # contact part of the leaf
```

They are not tracked in git. Either train them yourself —
`training/learned_potential/README.md` has the extract → train → export
sequence — or drop in the released checkpoints.

At design time the heads run **once**, to build a per-target lookup table; the
Monte Carlo loop itself never touches torch. Tables are cached under
`paths.learned_tables` (`data/learned_tables/` by default) and reused across
runs, so the cost is paid once per target rather than once per run.
