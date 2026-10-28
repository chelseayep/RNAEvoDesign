"""DSSR structural annotation: run x3dna-dssr and parse base pairs + SS elements.

DSSR reads a 3-D structure (mmCIF/PDB with base atoms) and reports observed base
pairs (with Leontis-Westhof class), secondary-structure elements (stems,
hairpins, internal loops, bulges, junctions), and per-nucleotide backbone
geometry.

This module produces an alignment-ready :class:`DSSRAnnotation` keyed by
``(chain, resnum)`` so callers can map onto their own residue ordering.

The binary is resolved from (in order):
    EVO_RNA_DSSR_BIN env var > config PATHS.dssr_bin (default tools/x3dna-dssr).

If the binary is missing or fails, :class:`DSSRUnavailable` is raised so callers
can fall back to geometry-based extraction.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


class DSSRUnavailable(RuntimeError):
    """Raised when DSSR cannot be run (binary missing or execution failed)."""


# Standard Watson-Crick pairs in DSSR's LW vocabulary; everything else is
# treated as non-canonical (wobble, Hoogsteen, sugar-edge, ...).
_CANONICAL_LW = {"cWW", "cWw", "cww"}

# Per-position secondary-structure element classes (integer codes).
# 0 = other/unpaired-dangling, 1 = stem, then loop subtypes.
SS_ELEMENTS = ("other", "stem", "hairpin", "internal", "bulge", "junction")
SS_ELEMENT_CODE = {name: i for i, name in enumerate(SS_ELEMENTS)}


def resolve_dssr_bin() -> Optional[str]:
    """Return the DSSR binary path, or None if not found/executable."""
    candidates = [
        os.environ.get("EVO_RNA_DSSR_BIN"),
    ]
    try:
        from evo_rna.config import PATHS
        candidates.append(str(PATHS.dssr_bin) if PATHS.dssr_bin else None)
    except Exception:  # pragma: no cover - config optional
        pass
    for c in candidates:
        if c and Path(c).exists() and os.access(c, os.X_OK):
            return c
    return None


@dataclass
class DSSRPair:
    chain1: str
    resnum1: int
    chain2: str
    resnum2: int
    lw: str
    is_canonical: bool


@dataclass
class DSSRAnnotation:
    """Parsed DSSR output, keyed by ``(chain, resnum)`` for alignment.

    pairs:        list of :class:`DSSRPair`
    element_of:   {(chain, resnum): element_class_name}
    torsions:     {(chain, resnum): {alpha..chi: deg|None}}
    """

    pairs: list[DSSRPair] = field(default_factory=list)
    element_of: dict[tuple[str, int], str] = field(default_factory=dict)
    torsions: dict[tuple[str, int], dict[str, Optional[float]]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# nt_id parsing
# ---------------------------------------------------------------------------

_NT_ID_RE = re.compile(r"^(?P<chain>[^.]+)\.[A-Za-z0-9]*?(?P<resnum>-?\d+)(?:\^.)?$")


def _split_nt_id(nt_id: str) -> Optional[tuple[str, int]]:
    """Map a DSSR nt_id (e.g. ``A.A43``, ``B.U-7``, ``A.PSU101``) to (chain, resnum).

    DSSR nt_ids are ``<chain>.<name><resnum>[^insertion]``: chain before the
    first dot, trailing signed integer as the residue number.
    """
    m = _NT_ID_RE.match(nt_id)
    if not m:
        return None
    return m.group("chain"), int(m.group("resnum"))


def _in_chains(nt_id: str, chains: Optional[tuple[str, ...]]) -> bool:
    if chains is None:
        return True
    return nt_id.split(".", 1)[0] in chains


# ---------------------------------------------------------------------------
# DSSR runner
# ---------------------------------------------------------------------------

def run_dssr(struct_path: str | Path, cache_dir: str | Path | None = None) -> dict:
    """Run DSSR on *struct_path* and return the parsed JSON dict.

    If *cache_dir* is given, the JSON is cached at ``<cache_dir>/dssr.json`` and
    reused on subsequent calls.  Raises :class:`DSSRUnavailable` on any failure.
    """
    struct_path = Path(struct_path)
    cached = Path(cache_dir) / f"{struct_path.stem}.dssr.json" if cache_dir is not None else None
    if cached is not None and cached.exists():
        return json.loads(cached.read_text())

    binary = resolve_dssr_bin()
    if binary is None:
        raise DSSRUnavailable(
            "DSSR binary not found. Set EVO_RNA_DSSR_BIN or configs/default.yaml "
            "paths.dssr_bin (default tools/x3dna-dssr)."
        )

    cmd = [binary, f"--input={struct_path.resolve()}", "--json", "--more", "--non-pair"]
    try:
        # Run in a temp dir so DSSR's auxiliary files don't pollute the cwd.
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=tmp)
    except (FileNotFoundError, OSError) as exc:
        raise DSSRUnavailable(f"Could not execute DSSR ({binary!r}): {exc}") from exc

    if result.returncode != 0:
        raise DSSRUnavailable(
            f"DSSR failed on {struct_path.name} (exit {result.returncode}): {result.stderr[:300]}"
        )
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise DSSRUnavailable(f"DSSR produced invalid JSON for {struct_path.name}: {exc}") from exc

    if cached is not None:
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps(data))
    return data


# ---------------------------------------------------------------------------
# Parsing into an alignment-ready annotation
# ---------------------------------------------------------------------------

def _nt_ids_from_motif(item: dict, chains: Optional[tuple[str, ...]]) -> list[str]:
    """Extract nt_ids from a DSSR motif object (nts_long preferred)."""
    raw = item.get("nts_long") or item.get("nts", "")
    sep = "," if "," in raw else " "
    ids = [s.strip() for s in raw.split(sep) if s.strip()]
    if chains is not None:
        ids = [n for n in ids if _in_chains(n, chains)]
    return ids


def parse_annotation(dssr: dict, chains: Optional[tuple[str, ...]] = None) -> DSSRAnnotation:
    """Parse a DSSR JSON dict into a :class:`DSSRAnnotation`.

    *chains* restricts pairs/elements to the design unit.  A pair is kept only
    when **both** partners lie in *chains* (intra-unit pairing); for a multi-chain
    design unit this naturally includes inter-chain pairs within the unit.
    """
    ann = DSSRAnnotation()

    # --- pairs (both partners in the design unit) ---------------------------
    for p in dssr.get("pairs", []):
        nt1, nt2 = p.get("nt1", ""), p.get("nt2", "")
        if chains is not None and not (_in_chains(nt1, chains) and _in_chains(nt2, chains)):
            continue
        k1, k2 = _split_nt_id(nt1), _split_nt_id(nt2)
        if k1 is None or k2 is None:
            continue
        lw = p.get("LW", "")
        ann.pairs.append(
            DSSRPair(k1[0], k1[1], k2[0], k2[1], lw, lw in _CANONICAL_LW)
        )

    # --- per-position SS element class --------------------------------------
    # Priority: a position labelled by a more specific loop wins over "stem".
    def _assign(nt_ids: list[str], element: str) -> None:
        for nt_id in nt_ids:
            key = _split_nt_id(nt_id)
            if key is None:
                continue
            cur = ann.element_of.get(key)
            if cur is None or SS_ELEMENT_CODE[element] > SS_ELEMENT_CODE[cur]:
                ann.element_of[key] = element

    # stems first (lowest priority among assigned), then loops override.
    for stem in dssr.get("stems", []):
        nt_ids = [
            nt for pr in stem.get("pairs", []) for key in ("nt1", "nt2")
            for nt in (pr.get(key, ""),) if nt and _in_chains(nt, chains)
        ]
        _assign(nt_ids, "stem")
    for hp in dssr.get("hairpins", []):
        _assign(_nt_ids_from_motif(hp, chains), "hairpin")
    for il in dssr.get("iloops", []):
        _assign(_nt_ids_from_motif(il, chains), "internal")
    for bg in dssr.get("bulges", []):
        _assign(_nt_ids_from_motif(bg, chains), "bulge")
    for jn in dssr.get("junctions", []):
        _assign(_nt_ids_from_motif(jn, chains), "junction")

    # --- per-nt torsions ----------------------------------------------------
    for nt in dssr.get("nts", []):
        nt_id = nt.get("nt_id", "")
        if not _in_chains(nt_id, chains):
            continue
        key = _split_nt_id(nt_id)
        if key is None:
            continue

        def _angle(name: str):
            v = nt.get(name)
            if v is None or v == "---":
                return None
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        ann.torsions[key] = {a: _angle(a) for a in
                             ("alpha", "beta", "gamma", "delta", "epsilon", "zeta", "chi")}
    return ann


def annotate(struct_path: str | Path, chains: Optional[tuple[str, ...]] = None,
             cache_dir: str | Path | None = None) -> DSSRAnnotation:
    """Run DSSR and return a parsed :class:`DSSRAnnotation` (convenience wrapper)."""
    return parse_annotation(run_dssr(struct_path, cache_dir=cache_dir), chains=chains)


def ss_pairs_elements(
    struct_path: str | Path,
    chains: Optional[tuple[str, ...]],
    keys: list[tuple[str, int]],
    cache_dir: str | Path | None = None,
):
    """DSSR-derived (ss, pair_map, ss_element) aligned to *keys*, with fallback.

    *keys* is the ordered ``[(chain, resnum), ...]`` list from
    :func:`evo_rna.profile.utils.extract_chain_unit` (local index = list index).

    Returns ``(ss, pair_map, ss_element, source)`` where:
      ss          (L,) int8 — 0 unpaired, 1 paired
      pair_map    {i: j, j: i} including non-canonical pairs
      ss_element  (L,) int8 — element class codes (SS_ELEMENT_CODE)
      source      "dssr" or "geometry" (fallback)

    On :class:`DSSRUnavailable` it falls back to the geometry-based
    ``extract_ss_and_pairs`` (single-chain only; multi-chain pairs are lost) and
    returns an all-zero ``ss_element``.
    """
    import numpy as np

    L = len(keys)
    key_to_idx = {k: i for i, k in enumerate(keys)}

    try:
        ann = annotate(struct_path, chains=chains, cache_dir=cache_dir)
    except DSSRUnavailable as exc:
        log.warning("DSSR unavailable (%s) — falling back to geometry-based SS.", exc)
        from evo_rna.profile.ss import extract_ss_and_pairs_unit
        # Multi-chain aware and aligned to *keys*, so inter-chain (duplex) pairs
        # are kept and ss has length L for multi-strand DAS units.
        ss, pair_map = extract_ss_and_pairs_unit(str(struct_path), keys)
        return ss, pair_map, np.zeros(L, dtype=np.int8), "geometry"

    # Build a clean involution (one partner per position).  DSSR reports
    # multiplets/base-triples where a base pairs with several partners; keeping
    # them all would break pair_map symmetry.  Add canonical pairs first, then
    # non-canonical, skipping any endpoint that is already paired.  Extra
    # tertiary partners remain available via the full DSSRAnnotation.
    pair_map: dict[int, int] = {}
    for p in sorted(ann.pairs, key=lambda q: not q.is_canonical):
        i = key_to_idx.get((p.chain1, p.resnum1))
        j = key_to_idx.get((p.chain2, p.resnum2))
        if i is None or j is None or i == j:
            continue
        if i in pair_map or j in pair_map:
            continue
        pair_map[i] = j
        pair_map[j] = i

    ss_element = np.zeros(L, dtype=np.int8)
    for key, element in ann.element_of.items():
        idx = key_to_idx.get(key)
        if idx is not None:
            ss_element[idx] = SS_ELEMENT_CODE[element]

    ss = np.zeros(L, dtype=np.int8)
    for i in pair_map:
        ss[i] = 1

    return ss, pair_map, ss_element, "dssr"


def contacts_aligned(
    struct_path: str | Path,
    chains: Optional[tuple[str, ...]],
    keys: list[tuple[str, int]],
    cache_dir: str | Path | None = None,
) -> list[tuple[int, int, str]]:
    """All DSSR base-pair contacts aligned to *keys*, as ``(i, j, lw)``.

    Unlike :func:`ss_pairs_elements` (which collapses pairing to a one-partner-
    per-position involution for ``pair_map``), this keeps **every** annotated
    pair — canonical, non-canonical and base-multiplet partners — with its
    Leontis-Westhof code, for the H-bond satisfaction term.  ``i``/``j`` are
    local indices (``i`` = nt1, ``j`` = nt2); the unordered pair is deduplicated.

    On :class:`DSSRUnavailable` an empty list is returned (the term no-ops).
    The DSSR JSON is cached, so calling this alongside :func:`ss_pairs_elements`
    does not re-run the binary.
    """
    try:
        ann = annotate(struct_path, chains=chains, cache_dir=cache_dir)
    except DSSRUnavailable as exc:
        log.warning("DSSR unavailable (%s) — no tertiary H-bond contacts.", exc)
        return []
    return contacts_from_annotation(ann, keys)


def contacts_from_annotation(
    ann: DSSRAnnotation,
    keys: list[tuple[str, int]],
) -> list[tuple[int, int, str]]:
    """Align a parsed :class:`DSSRAnnotation`'s pairs to *keys* as ``(i, j, lw)``.

    The index-alignment + unordered dedup half of :func:`contacts_aligned`,
    split out so the propensity-derivation script can reuse it on cached DSSR
    JSON without re-running the binary.  ``i`` = nt1 (LW edge1), ``j`` = nt2.
    """
    key_to_idx = {k: i for i, k in enumerate(keys)}
    contacts: list[tuple[int, int, str]] = []
    seen: set[tuple[int, int]] = set()
    for p in ann.pairs:
        i = key_to_idx.get((p.chain1, p.resnum1))
        j = key_to_idx.get((p.chain2, p.resnum2))
        if i is None or j is None or i == j:
            continue
        undirected = (i, j) if i < j else (j, i)
        if undirected in seen:
            continue
        seen.add(undirected)
        contacts.append((i, j, p.lw))
    return contacts


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "baseline/_raw_pdbs/1Y26.pdb"
    chs = tuple(sys.argv[2].split("-")) if len(sys.argv) > 2 else None
    a = annotate(path, chains=chs)
    print(f"pairs: {len(a.pairs)}  (non-canonical: {sum(not p.is_canonical for p in a.pairs)})")
    from collections import Counter
    print("element classes:", dict(Counter(a.element_of.values())))
