"""Load an energy-weight file into ``EnergyFunction.default`` keyword params.

Two producers write weight files, in two different shapes:

``flat``
    The contrastive and TM fits emit the constructor params directly:
    ``w_evo`` / ``w_secondary`` / ``w_tertiary`` plus ``secondary_weights`` /
    ``tertiary_weights`` list.  This is the shape of the shipped arm weights.

``standardised`` (``schema: evo_rna.standardised_weights/1``)
    The older scale-standardisation output (Z = 1/sigma), which nests the same
    numbers one level down::

        blocks.<name>.weight_raw        -> w_<name>
        blocks.<name>.sub_weights_raw   -> <name>_weights
        blocks.<name>.leaves            -> the order those sub-weights are in

Reading a ``standardised`` file with a ``flat`` reader finds none of the keys
it wants.  Because the loader that shipped first used ``spec.get(key)`` with a
uniform default, that produced a run that logged "loaded weights" and then
silently sampled with uniform weights — the failure this module exists to make
impossible.  :func:`load_weight_spec` raises on a shape it does not recognise
rather than returning a partial dict.

Sub-weight order is taken from the file's own ``leaves`` list and reordered to
:data:`BLOCK_LEAVES`, never assumed positionally: the ``*_weights`` tuples are
positional in ``EnergyFunction.default``, so a mismatched order would silently
mis-assign every sub-weight.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

# Positional order of the *_weights tuples in EnergyFunction.default.
# Must match the CompositeTerm construction order in EnergyFunction._secondary
# / ._tertiary.
BLOCK_LEAVES: Dict[str, tuple[str, ...]] = {
    "secondary": ("ss", "stacking_context", "ensemble_defect", "junction"),
    "tertiary":  ("hbond", "sa", "torsion"),
}

# The keys EnergyFunction.default takes, and that the design pipeline resolves.
PARAM_KEYS = ("w_evo", "w_secondary", "w_tertiary",
              "secondary_weights", "tertiary_weights", "w_ss_head")


class WeightSpecError(ValueError):
    """A weight file could not be interpreted as either known schema."""


def _leaf(name: str, block: str) -> str:
    """``'secondary.ss'`` / ``'ss'`` -> ``'ss'`` (tolerate both spellings)."""
    return name.split(".", 1)[1] if name.startswith(f"{block}.") else name


def _from_standardised(spec: Mapping[str, Any]) -> Dict[str, Any]:
    """Parse ``evo_rna.standardised_weights/1`` (apply_scales.py)."""
    out: Dict[str, Any] = {}
    blocks = spec["blocks"]
    for block, entry in blocks.items():
        if "weight_raw" not in entry:
            raise WeightSpecError(
                f"block {block!r} has no 'weight_raw'; not a standardised-weights file"
            )
        out[f"w_{block}"] = float(entry["weight_raw"])
        if block not in BLOCK_LEAVES:
            continue  # single-leaf block (evo) — magnitude only, no mix tuple
        leaves = [_leaf(n, block) for n in entry["leaves"]]
        sub = entry["sub_weights_raw"]
        if len(leaves) != len(sub):
            raise WeightSpecError(
                f"block {block!r}: {len(leaves)} leaves but {len(sub)} sub-weights"
            )
        by_name = dict(zip(leaves, (float(x) for x in sub)))
        expected = BLOCK_LEAVES[block]
        missing = [l for l in expected if l not in by_name]
        extra   = [l for l in by_name if l not in expected]
        if missing or extra:
            raise WeightSpecError(
                f"block {block!r} leaves do not match EnergyFunction.default: "
                f"missing={missing} unexpected={extra}"
            )
        # Reorder to the constructor's positional order.
        out[f"{block}_weights"] = tuple(by_name[l] for l in expected)
    return out


def _from_flat(spec: Mapping[str, Any]) -> Dict[str, Any]:
    """Parse the flat fitted-weights shape."""
    out: Dict[str, Any] = {}
    for key in PARAM_KEYS:
        val = spec.get(key)
        if val is None:
            continue
        out[key] = tuple(float(x) for x in val) if key.endswith("_weights") else float(val)
    return out


def parse_weight_spec(spec: Mapping[str, Any], *, source: str = "<dict>") -> Dict[str, Any]:
    """Normalise a loaded weight document to ``EnergyFunction.default`` params.

    Returns only the keys the file actually specifies, so a caller can layer
    it under explicit CLI flags.  Raises :class:`WeightSpecError` if the
    document matches neither schema — never returns an empty dict, because a
    silent no-op here is indistinguishable from a uniform-weight run.
    """
    schema = str(spec.get("schema", ""))
    if schema.startswith("evo_rna.standardised_weights") or "blocks" in spec:
        params = _from_standardised(spec)
    else:
        params = _from_flat(spec)
    if not params:
        raise WeightSpecError(
            f"{source}: no recognised weight keys. Expected either the flat shape "
            f"({', '.join(PARAM_KEYS)}), "
            f"or 'blocks.<name>.weight_raw' (schema evo_rna.standardised_weights/1). "
            f"Found keys: {sorted(spec)}"
        )
    return params


def load_weight_spec(path: "str | Path") -> Dict[str, Any]:
    """Read *path* and normalise it via :func:`parse_weight_spec`."""
    path = Path(path)
    return parse_weight_spec(json.loads(path.read_text()), source=str(path))
