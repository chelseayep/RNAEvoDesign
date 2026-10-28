"""Central path / run configuration for RNAEvoDesign.

Avoids hard-coding filesystem paths in scripts.  Paths come from
``configs/default.yaml`` (overridable with the ``EVO_RNA_CONFIG`` env var or a
``load_config(path)`` argument), are resolved relative to the repo root, and any
individual entry can be overridden by an ``EVO_RNA_<KEY>`` environment variable.

Typical use
-----------
    from evo_rna.config import PATHS, CONFIG

    rnaalign = PATHS.rnaalign_bin        # absolute Path, repo-root resolved
    steps    = CONFIG.mc["steps_per_temp"]

Or load an alternate config explicitly::

    from evo_rna.config import load_config
    cfg = load_config("configs/cluster.yaml")
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# repo root = two levels above this package (src/evo_rna/config.py -> repo/)
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def _resolve(value: Any, root: Path = REPO_ROOT) -> Optional[Path]:
    """Expand ~ and make *value* absolute relative to *root* (None passes through)."""
    if value is None or value == "":
        return None
    p = Path(os.path.expanduser(str(value)))
    return p if p.is_absolute() else (root / p)


@dataclass
class Paths:
    """Resolved (absolute) filesystem locations used across the pipeline."""

    bin_dir: Path
    rnaalign_bin: Path   # RNAalign: template alignment for PSSM/profile generation
    dssr_bin: Path
    pdb_lib: Path
    pdb_dir: Path
    data_dir: Path
    rna_fm_model: Path
    propensity_table: Path
    hbond_propensity: Path
    learned_models: Path   # trained residue_head.pt / pair_head.pt (A2 arm)
    learned_tables: Path   # per-target exported tables, built on demand
    default_weights: Optional[Path]  # fitted block weights the default arm uses
    pssm_cache: Path       # shared <target>.prf cache — target-keyed, not experiment-keyed
    output_root: Path


# Default relative paths, used when a key is missing from the YAML.
_PATH_DEFAULTS: Dict[str, Optional[str]] = {
    "bin_dir": "bin",
    "rnaalign_bin": "bin/RNAalign/RNAalign",
    "dssr_bin": "tools/x3dna-dssr",
    "pdb_lib": "data/rna_pdb_lib",
    "pdb_dir": "data/rna_pdb",
    "data_dir": "data",
    "rna_fm_model": "rnafm_weights",
    "propensity_table": "data/propensities.npz",
    "hbond_propensity": "data/hbond_propensity.npz",
    "learned_models": "models/learned_tertiary",
    "learned_tables": "data/learned_tables",
    "default_weights": "weights/a2_learned.json",
    "pssm_cache": "data/pssm_cache",
    "output_root": "runs",
}


@dataclass
class Config:
    paths: Paths
    mc: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)


def load_config(path: "str | Path | None" = None) -> Config:
    """Load and resolve the EvoDesign configuration.

    Resolution order for *path*:
    explicit argument > ``$EVO_RNA_CONFIG`` > ``configs/default.yaml``.
    A missing file is tolerated (built-in defaults are used).
    """
    cfg_path = Path(path or os.environ.get("EVO_RNA_CONFIG") or DEFAULT_CONFIG)
    data: Dict[str, Any] = {}
    if cfg_path.exists():
        data = yaml.safe_load(cfg_path.read_text()) or {}

    paths_raw: Dict[str, Any] = dict(data.get("paths", {}))

    def _value(key: str) -> Any:
        # env override (EVO_RNA_PDB_LIB, ...) wins over the YAML
        env = os.environ.get(f"EVO_RNA_{key.upper()}")
        if env:
            return env
        return paths_raw.get(key, _PATH_DEFAULTS[key])

    paths = Paths(
        bin_dir=_resolve(_value("bin_dir")),
        rnaalign_bin=_resolve(_value("rnaalign_bin")),
        dssr_bin=_resolve(_value("dssr_bin")),
        pssm_cache=_resolve(_value("pssm_cache")),
        pdb_lib=_resolve(_value("pdb_lib")),
        pdb_dir=_resolve(_value("pdb_dir")),
        data_dir=_resolve(_value("data_dir")),
        rna_fm_model=_resolve(_value("rna_fm_model")),
        propensity_table=_resolve(_value("propensity_table")),
        hbond_propensity=_resolve(_value("hbond_propensity")),
        learned_models=_resolve(_value("learned_models")),
        learned_tables=_resolve(_value("learned_tables")),
        default_weights=_resolve(_value("default_weights")),
        output_root=_resolve(_value("output_root")),
    )
    return Config(paths=paths, mc=dict(data.get("mc", {})), raw=data)


# Module-level singletons for the common case.
CONFIG = load_config()
PATHS = CONFIG.paths
