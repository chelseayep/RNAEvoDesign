# Fitted block weights

`a2_learned.json` is the default the pipeline loads (`paths.default_weights`):
the TM-refit weights over the learned tertiary leaf — the A2 arm.

The other two arms, for comparison:

| file | arm | tertiary block |
|---|---|---|
| `a2_learned.json`    | A2 learned (default) | learned potential |
| `a1_reweighted.json` | A1 reweighted        | binned hbond/sa/torsion, same fit procedure |
| `a0_baseline.json`   | A0 baseline          | binned, standardised (1/σ) weights |

A1 is the paired control for A2: same fit, same pools, same dropped leaves —
only the tertiary leaf differs. Run a control arm with
`--no-learned-tertiary --weights weights/a1_reweighted.json`.

These are released artifacts. The fitting chain that produced them —
contrastive native-vs-decoy fitting, then refinement against refold TM-score —
is not part of this repository: it needs decoy pools from several external
inverse-folding models and a structure-prediction harness to score against.
`README.md` summarises the procedure.
