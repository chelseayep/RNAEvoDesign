# Data directory

Everything here is generated or downloaded; nothing is tracked.

| path | what | produced by |
|---|---|---|
| `rna_pdb/` | target structures | `scripts/data/fetch_rnapdb_from_manifest.py` |
| `rna_pdb_lib/` | PDB-format template library for the PSSM sweep | `scripts/data/make_unit_files.py` |
| `pssm_cache/` | `<target>.prf`, keyed by target and shared across runs | the pipeline |
| `learned_tables/` | per-target tables for the learned tertiary leaf | built on demand at design time |
| `propensities.npz`, `hbond_propensity.npz` | binned tertiary tables | only needed for the A0/A1 control arms |

The template library must be **PDB format**: RNAalign cannot read mmCIF, and a
library of mmCIF files silently yields an empty homolog pool rather than an
error.
