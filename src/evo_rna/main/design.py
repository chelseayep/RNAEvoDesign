"""Monte Carlo RNA sequence design from a target backbone.

Reads a target structure, derives its secondary structure, solvent
accessibility, torsions and DSSR contacts, builds a PSSM from structural
homologues, and anneals a random sequence against the composite energy
(evo + secondary + tertiary).  The product is ``design.json``: the designed
sequence and the record of how it was produced.  Nothing is folded or scored
here — bring your own evaluation.

Arms
----
The defaults are the **A2 arm**: TM-refit block weights over the *learned*
tertiary potential, SAMFEO moves, uniform position sampling.  This is the
method the paper reports.  The other two arms are one flag each:

    A2  learned     (default)                       the method
    A1  reweighted  --no-learned-tertiary           same weights, binned leaves
    A0  baseline    --no-learned-tertiary --weights <standardised.json>

Usage
-----
    # A2 on a target, with a template library for the PSSM
    rna-evo-design target.pdb --pdb-lib data/rna_pdb_lib

    # equivalently, from a source checkout
    python src/evo_rna/main/design.py target.pdb --pdb-lib data/rna_pdb_lib

    # pre-computed PSSM instead of an RNAalign sweep
    rna-evo-design target.pdb --pssm profile/prf.txt

    # no PSSM source at all: flat profile, Evo term inert
    rna-evo-design target.pdb

    # quick smoke run (seconds rather than hours)
    rna-evo-design target.pdb --steps-per-temp 100

The learned tertiary leaf needs a per-target table.  It is built on demand from
the trained heads (``paths.learned_models``) and cached under
``paths.learned_tables``, so no batch export step is required.

Run  rna-evo-design --help  for all options.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import os
from pathlib import Path

# ── path setup ───────────────────────────────────────────────────────────────
# Running this file directly out of a source checkout (src/evo_rna/main/design.py
# → 3 levels up = repo root, src/ is the import root).  A pip-installed copy
# already has the package on the path and this is a no-op.
_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
# ─────────────────────────────────────────────────────────────────────────────

import random

import numpy as np

from evo_rna.config import CONFIG, PATHS
from evo_rna.core.structure import NUC_ALPHABET, Structure
from evo_rna.monte_carlo.energy import EnergyFunction
from evo_rna.monte_carlo.mc import (MOVE_OPERATORS, POSITION_SAMPLERS,
                                    SAConfig, run_sa)
from evo_rna.profile.features import build_structure
from evo_rna.profile.dssr import ss_pairs_elements, contacts_aligned
from evo_rna.profile.sa import extract_sasa_unit
from evo_rna.profile.torsions import extract_torsions
from evo_rna.profile.utils import extract_chain_unit
from evo_rna.monte_carlo.energy_terms.composition import  gc_fraction

def random_sequence(length: int, seed: int | None = None) -> str:
    """Uniformly random RNA sequence of given length."""
    rng = random.Random(seed)
    return "".join(rng.choice(NUC_ALPHABET) for _ in range(length))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------
_parser = argparse.ArgumentParser(
    description="End-to-end RNA design: Monte Carlo sequence design from a PDB structure.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
_parser.add_argument(
    "pdb",
    help="Path to input PDB file (the target structure).",
)
_parser.add_argument(
    "--chain", "-c",
    default=None,
    metavar="CHAIN",
    help="Chain ID to extract from the PDB. Auto-detects the first chain if omitted.",
)
_parser.add_argument(
    "--pdb-lib", "-l",
    default=str(PATHS.pdb_lib),
    metavar="DIR",
    help=(
        "Directory of template PDB files used to generate a PSSM via RNAalign. "
        "When provided, the PSSM is computed automatically and the Evo term is active. "
        "Skipped if --pssm is also given."
    ),
)
_parser.add_argument(
    "--bin-path",
    default=str(PATHS.bin_dir),
    metavar="DIR",
    help="Directory containing the RNAalign binary (bin/RNAalign/RNAalign). "
         "Used only when --pdb-lib is supplied.",
)
_parser.add_argument(
    "--cutoff",
    type=float,
    default=CONFIG.mc.get("cutoff", 0.5),
    metavar="F",
    help="TM-score cutoff for RNAalign template filtering (used with --pdb-lib).",
)
_parser.add_argument(
    "--pssm", "-p",
    default=None,
    metavar="PRF_PATH",
    help="Path to a pre-computed prf.txt PSSM. Takes priority over --pdb-lib.",
)
_parser.add_argument(
    "--steps-per-temp", "-s",
    type=int,
    default=CONFIG.mc.get("steps_per_temp", 200),
    metavar="N",
    help="MC steps per temperature level (increase for longer runs).",
)
_parser.add_argument(
    "--seed",
    type=int,
    default=CONFIG.mc.get("seed", 42),
    metavar="SEED",
    help="Random seed for reproducibility.",
)
_parser.add_argument(
    "--t-min-ratio",
    type=float,
    default=CONFIG.mc.get("t_min_ratio", 0.005),
    metavar="R",
    help="Quench floor as a fraction of the (adaptive) t_max. Keeps the "
         "temperature-level count — and so the total move budget — constant "
         "no matter how large the energy weights are, so arms that differ "
         "only in weights get equal sampling. Pass 0 for an absolute t_min "
         "(the old scale-coupled behaviour). [default: %(default)s]",
)
_parser.add_argument(
    "--move-operator",
    choices=MOVE_OPERATORS,
    default=CONFIG.mc.get("move_operator", "pair_aware"),
    help="MC proposal distribution. 'pair_aware' mutates one position and "
         "repairs its partner. 'samfeo' mutates the whole local motif "
         "(stack / mismatch flanks) as a unit, so a stack and the bases "
         "stacked against it can co-adapt in a single move. [default: %(default)s]",
)
_parser.add_argument(
    "--w-evo-shallow",
    type=float,
    default=CONFIG.mc.get("w_evo_shallow"),
    metavar="FACTOR",
    help="Depth-adaptive Evo weight: multiply the resolved w_evo by FACTOR on "
         "targets whose PSSM is built from fewer than --shallow-depth-threshold "
         "homologues. A shallow profile gives noisy PSSM log-odds, so Eevo may "
         "be adding noise rather than signal there; this down-weights the block "
         "only where that applies and leaves deep targets at their standardised "
         "weight. FACTOR is a share of the resolved weight, matching exp_6's W "
         "(0.5 halves it). The rule needs the in-loop PSSM depth, so it cannot "
         "fire for --pssm runs or when every template misses the TM cutoff; "
         "those keep the full weight and say so in design.json. Off by default.",
)
_parser.add_argument(
    "--shallow-depth-threshold",
    type=int,
    default=CONFIG.mc.get("shallow_depth_threshold", 50),
    metavar="N",
    help="Homologue count below which --w-evo-shallow applies. Compared "
         "against pssm_depth.n_homologs, the cutoff-filtered hit set behind "
         "this run's PSSM. [default: %(default)s]",
)
_parser.add_argument(
    "--position-sampling",
    choices=POSITION_SAMPLERS,
    default=CONFIG.mc.get("position_sampling", "uniform"),
    help="Where a proposal is centred, for either move operator. 'uniform' "
         "draws every position equally. 'defect' draws from a Boltzmann "
         "distribution over the positional ensemble defect of the current "
         "sequence (SAMFEO eq. 13), so the move budget goes to the positions "
         "the Boltzmann ensemble is actually getting wrong instead of being "
         "spread evenly over positions that are already right. Costs one "
         "ViennaRNA partition-function fold per accepted move (~2 ms at 70 nt, "
         "~30 ms at 200 nt) and falls back to uniform with a warning if the "
         "bindings are missing. Applied without a Hastings correction — the "
         "chain becomes a biased optimiser, not an exact Boltzmann sampler. "
         "[default: %(default)s]",
)
_parser.add_argument(
    "--defect-temperature",
    type=float,
    default=CONFIG.mc.get("defect_temperature", 0.2),
    metavar="T0",
    help="T0 in SAMFEO eq. 13, the flatness of the position distribution. "
         "The defect is in [0, 1], so a maximally-defective position is at "
         "most exp(1/T0) times as likely as a perfect one: 148x at 0.2, 7.4x "
         "at 0.5, 2.7x at 1.0; large T0 recovers uniform sampling and 0 is "
         "argmax. Ignored unless --position-sampling defect. "
         "[default: %(default)s]",
)
_parser.add_argument(
    "--defect-refresh",
    type=int,
    default=CONFIG.mc.get("defect_refresh", 1),
    metavar="N",
    help="Proposals a defect vector is reused for before refolding. 1 refolds "
         "whenever the sequence changes (exact); larger trades a slightly "
         "stale proposal distribution for proportionally fewer O(L^3) folds — "
         "worth it on long targets. [default: %(default)s]",
)

# ── Energy-term weights for the composite EnergyFunction.default ─────────────
# Three gauge-fixed blocks: evo + secondary + tertiary.  The three --w-* set the
# block magnitudes; the --*-weights set the within-block mix (simplex-normalised
# inside each block, so the two levels are identifiable).  See
# evo_rna/monte_carlo/energy.py::EnergyFunction.default.
_parser.add_argument(
    "--weights", default=(str(PATHS.default_weights)
                          if PATHS.default_weights
                          and Path(PATHS.default_weights).exists() else None),
    metavar="JSON",
    help=(
        "Load block weights from a fitted-weights JSON "
        "(reads w_evo/w_secondary/w_tertiary + secondary_weights/tertiary_weights). "
        "Precedence: an explicit --w-*/--*-weights flag overrides the file, which "
        "overrides the built-in uniform default.  For an ablation, load the refined "
        "weights and zero one block, e.g. --weights refined.json --w-secondary 0."
    ),
)
# Weight flags default to None so we can tell 'user set it' from 'use --weights /
# the uniform fallback'.  Resolved just after parse_args (see _resolve_weights).
_parser.add_argument(
    "--w-evo", type=float, default=None, metavar="W",
    help="Magnitude of the evolutionary block Eevo (PSSM log-odds). [default: 1.0]",
)
_parser.add_argument(
    "--w-secondary", type=float, default=None, metavar="W",
    help="Magnitude of the secondary block (ss, stacking, ensemble_defect, junction). [default: 1.0]",
)
_parser.add_argument(
    "--w-tertiary", type=float, default=None, metavar="W",
    help="Magnitude of the tertiary block (hbond, sa, torsion). [default: 1.0]",
)
_parser.add_argument(
    "--learned-tertiary", default=None, metavar="DIR",
    help=(
        "Directory of exported per-target tables for the learned tertiary "
        "potential, read as <DIR>/<pdb_stem>.npz.  A missing table is built "
        "from --learned-models and cached here. "
        "[default: paths.learned_tables]"
    ),
)
_parser.add_argument(
    "--learned-models", default=None, metavar="DIR",
    help=(
        "Trained residue_head.pt / pair_head.pt for the learned potential, "
        "used when a target's table has to be built.  "
        "[default: paths.learned_models]"
    ),
)
_parser.add_argument(
    "--no-learned-tertiary", dest="learned_on", action="store_false",
    default=CONFIG.mc.get("learned_tertiary", True),
    help=(
        "Fall back to the three binned tertiary leaves (hbond, sa, torsion) "
        "instead of the single learned leaf.  This is the A1/A0 control arm: "
        "the leaves are kept, not deleted, because they are the answer to "
        "'did the learned potential help, or would reweighting alone have "
        "done it?'.  --tertiary-weights applies only in this mode (a one-leaf "
        "simplex is 1.0); --w-tertiary carries the block magnitude either way, "
        "so a saved --weights file stays valid for both."
    ),
)
_parser.add_argument(
    "--w-composition", type=float, default=0.0, metavar="W",
    help=(
        "Magnitude of the GC-composition prior, anchored to the native's GC. "
        "OFF by default (0.0) so existing runs reproduce exactly. Turn it on "
        "when the energy is being bought with GC rather than structure — on the "
        "exp_1 arms, within-target r(GC,MFE)=-0.963 while r(GC,scMCC)=-0.501 "
        "and r(GC,TM)=-0.242. Diagnose with scripts/analysis/gc_vs_mfe.py. "
        "Try 0.5-2.0."
    ),
)
_parser.add_argument(
    "--gc-anchor", choices=("pssm", "neutral", "native"), default="pssm",
    help=(
        "Where the reference GC comes from. 'pssm' (default): expected GC "
        "under the homolog-derived PSSM — needs no native sequence and leaks "
        "nothing, falling back to 0.5 when the profile is flat. 'neutral': "
        "fixed 0.5. 'native': the held-out native's GC — an ORACLE that leaks "
        "the answer; use only to measure the headroom a perfect composition "
        "prior would buy, never for a reported design result."
    ),
)
_parser.add_argument(
    "--gc-tolerance", type=float, default=0.05, metavar="F",
    help="Half-width of the free band around the reference GC; the composition "
         "term is exactly zero inside it. Only used with --w-composition.",
)
_parser.add_argument(
    "--gc-min-homologs", type=int, default=10, metavar="N",
    help="Minimum homologs before --gc-anchor pssm is trusted. Below this the "
         "profile's expected GC reflects which few structures cleared the TM "
         "cutoff rather than any family-level constraint, so the anchor falls "
         "back to neutral. Only used with --w-composition --gc-anchor pssm.",
)
_parser.add_argument(
    "--secondary-weights", type=float, nargs=4, default=None,
    metavar=("SS", "STACK", "ENSEMBLE", "JUNCTION"),
    help=(
        "Within-block mix for the secondary block, in order: ss stacking "
        "ensemble_defect junction.  ensemble_defect folds the full sequence "
        "(O(L^3), ViennaRNA) — set its weight to 0 to drop that cost. "
        "[default: 1 1 1 1, or from --weights]"
    ),
)
_parser.add_argument(
    "--tertiary-weights", type=float, nargs=3, default=None,
    metavar=("HBOND", "SA", "TORSION"),
    help="Within-block mix for the tertiary block, in order: hbond sa torsion. "
         "[default: 1 1 1, or from --weights]",
)
_parser.add_argument(
    "--hbond-propensity", default=str(PATHS.hbond_propensity), metavar="NPZ",
    help=(
        "Path to a derived H-bond propensity table (.npz) for the tertiary "
        "hbond term (default: configured paths.hbond_propensity).  Pass '' to "
        "force the hand-built donor/acceptor chemistry capacity instead.  If the "
        "file is absent the pipeline falls back to chemistry with a warning.  "
        "The hbond term is a no-op when DSSR yields no contacts for the target."
    ),
)
_parser.add_argument(
    "--temperature", type=float, default=37.0, metavar="C",
    help="Folding temperature (°C) for the ensemble_defect term.",
)
_parser.add_argument(
    "--log-energy",
    action="store_true",
    help=(
        "Record per-term weighted energies at each accepted MC move and save "
        "to <output-dir>/<pdb>/energy_log.json.  Adds one extra term.score() call "
        "per accepted move."
    ),
)
_parser.add_argument(
    "--output-dir", "-O",
    default=None,
    metavar="DIR",
    help=(
        "Root output directory.  Results are written to <DIR>/<pdb_stem>/. "
        "Defaults to runs/<pdb_stem>/ relative to the repo root."
    ),
)

_args = _parser.parse_args()


def _resolve_weights(args) -> None:
    """Fill the block-weight args in place: CLI flag > --weights JSON > uniform.

    The JSON may be either weight-file shape — the flat fitted-weights output,
    or the older nested standardised-weights schema.
    :func:`~evo_rna.monte_carlo.weights.load_weight_spec` normalises both and
    raises on anything else.

    Logging happens *after* parsing, and reports the resolved values rather
    than just the path: an earlier version logged "Loaded block weights from
    <path>" before reading any key, so a schema mismatch produced a run that
    claimed to be weighted and silently sampled with uniform weights.
    """
    from evo_rna.monte_carlo.weights import load_weight_spec

    _UNIFORM = {
        "w_evo": 1.0, "w_secondary": 1.0, "w_tertiary": 1.0,
        "secondary_weights": (1.0, 1.0, 1.0, 1.0),
        "tertiary_weights": (1.0, 1.0, 1.0),
    }
    _OPT_IN: set[str] = set()
    spec = {}
    if args.weights:
        spec = load_weight_spec(args.weights)      # raises on an unknown shape
        log.info("Loaded block weights from %s: %s", args.weights,
                 ", ".join(f"{k}={spec[k]}" for k in sorted(spec)))
        unset = [k for k in _UNIFORM if k not in spec and k not in _OPT_IN]
        if unset:
            log.warning("Weight file %s does not set %s — those stay uniform.",
                        args.weights, ", ".join(sorted(unset)))
    for key, default in _UNIFORM.items():
        if getattr(args, key) is not None:        # explicit CLI flag wins
            continue
        val = spec.get(key)                        # then the --weights file
        setattr(args, key, val if val is not None else default)
    log.info(
        "Block weights in effect: w_evo=%s w_secondary=%s w_tertiary=%s "
        "secondary_weights=%s tertiary_weights=%s",
        args.w_evo, args.w_secondary, args.w_tertiary,
        tuple(args.secondary_weights), tuple(args.tertiary_weights),
    )


_resolve_weights(_args)


def _chains_from_stem(stem: str) -> str | None:
    """Parse the chain spec from an RNAsolo / gRNAde-DAS filename stem.

    These structures are named ``{PDB}_{MODEL}_{CHAINS}`` where ``CHAINS`` is a
    hyphen-separated, ordered list of chain IDs (e.g. ``2BCZ_1_D-C-A-B`` → the
    design unit spans chains D, C, A, B in that order).  Returns the raw chain
    spec string (``"D-C-A-B"``) for ``--chain`` / :func:`extract_chain_unit`, or
    ``None`` if the stem doesn't match the convention (caller falls back to the
    first chain).
    """
    # Require the full {PDB}_{MODEL}_{CHAINS} shape: a numeric model field guards
    # against mis-parsing arbitrary names like "my_data" as a chain spec.
    parts = stem.rsplit("_", 2)
    if len(parts) != 3 or not parts[1].isdigit():
        return None
    spec = parts[2]
    tokens = spec.split("-")
    # Each token must be a plausible chain id (1-4 alphanumerics).
    if tokens and all(1 <= len(t) <= 4 and t.isalnum() for t in tokens):
        return spec
    return None


PDB_PATH  = _args.pdb
# Chain spec priority: explicit --chain > parsed from the DAS-format filename
# (e.g. 2BCZ_1_D-C-A-B → D-C-A-B) > None (first chain). _normalize_chains splits
# the hyphenated spec into an ordered multi-chain design unit.
CHAIN_ID  = _args.chain or _chains_from_stem(Path(PDB_PATH).stem)
PRF_PATH  = _args.pssm
PDB_LIB   = _args.pdb_lib
BIN_PATH  = _args.bin_path
_out_root = Path(_args.output_dir) if _args.output_dir else PATHS.output_root
OUT_DIR  = str(_out_root / Path(PDB_PATH).stem)
# Created up front, not by whichever stage happens to write first: the DSSR
# cache, the energy log and design.json all land here, and a run that anneals
# for hours must not fail at the final write because the directory is missing.
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Load structure from PDB
# ---------------------------------------------------------------------------
log.info("Loading PDB: %s  chain %s", PDB_PATH, CHAIN_ID or "(auto)")

native_sequence, atoms, atom_coords, keys = extract_chain_unit(PDB_PATH, CHAIN_ID)
L = len(native_sequence)
log.info("Native sequence  (%d nt): %s  [held out — not used as MC start]", L, native_sequence)

# Chains actually present in the design unit (preserves order, de-dups).
_unit_chains = tuple(dict.fromkeys(c for c, _ in keys)) or None

# Secondary structure + base pairs + DSSR element labels (geometry fallback).
ss, pair_map, ss_element, ss_source = ss_pairs_elements(
    PDB_PATH, _unit_chains, keys, cache_dir=OUT_DIR,
)
n_pairs = len(pair_map) // 2
log.info("Secondary structure [%s]: %d paired positions, %d base pairs",
         ss_source, int(ss.sum()), n_pairs)

# DSSR base-pair contacts (canonical + non-canonical + multiplets, with LW codes)
# for the tertiary H-bond term.  Empty (term no-ops) when DSSR is unavailable.
contacts = contacts_aligned(PDB_PATH, _unit_chains, keys, cache_dir=OUT_DIR)
log.info("DSSR H-bond contacts: %d (hbond term %s)",
         len(contacts), "active" if contacts else "no-op — no DSSR contacts")

# Real per-position structural features, aligned to keys.
#   SASA: one FreeSASA pass over the whole structure (keeps inter-chain interface
#   burial), mapped onto the unit — not recomputed per chain.
#   Torsions: per-chain (backbone torsions are intra-strand; per-chain extraction
#   keeps chain breaks NaN instead of inventing a torsion across the junction).
sa = extract_sasa_unit(PDB_PATH, keys) if _unit_chains else np.zeros(L)
torsions = (np.concatenate([extract_torsions(PDB_PATH, c) for c in _unit_chains], axis=0)
            if _unit_chains else np.full((L, 7), np.nan))
if sa.shape[0] != L:
    sa = np.resize(sa, L)
if torsions.shape[0] != L:
    _t = np.full((L, 7), np.nan); _t[:min(L, torsions.shape[0])] = torsions[:min(L, torsions.shape[0])]; torsions = _t

# ---------------------------------------------------------------------------
# 1b. PSSM — three sources in priority order:
#       1. --pssm <prf.txt>    (pre-computed, highest priority)
#       2. --pdb-lib <dir>     (auto-generate via RNAalign)
#       3. flat zeros          (Evo term disabled)
# ---------------------------------------------------------------------------
from evo_rna.profile.pssm import load_prf  # noqa: E402

_profile_out = Path(OUT_DIR) / "profile"
_profile_files: list[str] = []  # paths written during profile generation
# Homolog depth behind the PSSM (the confidence the evo term deserves). Stays
# None for pre-computed --pssm and flat-PSSM runs (no in-loop alignment set).
pssm_depth: dict[str, float] | None = None

if PRF_PATH:
    pssm = load_prf(PRF_PATH)
    log.info("Loaded PSSM from %s  shape=%s", PRF_PATH, pssm.shape)

elif PDB_LIB:
    from evo_rna.profile.generate_profile import GenerateProfileConfig, RNAligningPipeline  # noqa: E402

    log.info(
        "Generating PSSM: RNAalign against %s  (cutoff=%.2f) …",
        PDB_LIB, _args.cutoff,
    )
    _profile_out.mkdir(parents=True, exist_ok=True)
    _gp_cfg = GenerateProfileConfig(
        input_pdb=PDB_PATH,
        output_dir=str(_profile_out),
        bin_path=BIN_PATH,
        pdb_lib=PDB_LIB,
        cutoff=_args.cutoff,
        write_msa_file=True,
    )
    _gp = RNAligningPipeline(_gp_cfg)
    _gp.run_rnaalign()
    _profile_files += [
        str(_profile_out / f)
        for f in ("tmscore.txt", "tmscore_sorted.txt", "seqid.txt", "msa.txt")
    ]

    if not _gp.alignments:
        log.warning(
            "No templates passed TM-score cutoff %.2f — falling back to flat PSSM.",
            _args.cutoff,
        )
        pssm = np.zeros((L, 4), dtype=np.float64)
    else:
        pssm = _gp.get_pssm(target_length=L)
        _profile_files.append(str(_profile_out / "prf.txt"))
        # Record the homolog depth behind this PSSM. _gp.alignments is already
        # the cutoff-filtered, self-match-removed hit set (fetch_top_hits), so
        # these match pssm_depth.py's columns: n_homologs / Σtm / max tm.
        _tms = [h.tm_score for h in _gp.alignments]
        pssm_depth = {
            "n_homologs":        float(len(_tms)),
            "tm_weighted_depth": round(float(sum(_tms)), 4),
            "max_tm":            round(float(max(_tms)), 4) if _tms else 0.0,
        }
        log.info(
            "PSSM generated from %d homologues  shape=%s  saved → %s",
            len(_gp.alignments), pssm.shape,
            _profile_out / "prf.txt",
        )
        build_structure(
            PDB_PATH, str(_profile_out), chain_id=CHAIN_ID,
            alignments=_gp.alignments,
            n_alignments=len(_gp.alignments),
        )
        _profile_files += [
            str(_profile_out / f) for f in ("ss.txt", "sa.txt", "torsions.txt")
        ]
        log.info("Structural features written → %s", _profile_out)

else:
    pssm = np.zeros((L, 4), dtype=np.float64)
    log.info(
        "No PSSM source supplied — using flat PSSM (Evo term = 0). "
        "Pass --pssm <prf.txt> to load one, or --pdb-lib <dir> to auto-generate."
    )

start_sequence = random_sequence(L, seed=_args.seed)
log.info("Random start seq (%d nt): %s", L, start_sequence)

struct = Structure(
    pdb_id=Path(PDB_PATH).stem,
    chain_id=CHAIN_ID,
    sequence=start_sequence,        # random — NOT the native sequence
    ss=ss,
    ss_element=ss_element,
    pair_map=pair_map,
    sa=sa,
    torsions=torsions,
    atom_coords=atom_coords,
    atoms=atoms,
    pssm=pssm,
    contacts=contacts,
)

problems = struct.validate()
if problems:
    for p in problems:
        log.warning("Validation: %s", p)
else:
    log.info("Structure validation: OK")


# ---------------------------------------------------------------------------
# 2. Energy function  (composite: evo + secondary + tertiary)
# ---------------------------------------------------------------------------
log.info("Building composite energy function (evo + secondary + tertiary)…")

# Resolve the hbond propensity table: use it if present, else fall back to the
# hand-built chemistry capacity (None) so a missing artifact never aborts a run.
_hbond_prop = _args.hbond_propensity or None
if _hbond_prop and not Path(_hbond_prop).exists():
    log.warning("H-bond propensity table not found (%s) — using chemistry capacity.", _hbond_prop)
    _hbond_prop = None
elif _hbond_prop:
    log.info("H-bond propensity table: %s", _hbond_prop)
else:
    log.info("H-bond propensity: hand-built chemistry capacity (no table).")

# Depth-adaptive Evo weight.  Applied HERE rather than in _resolve_weights
# because it needs pssm_depth, which only exists once the profile has been
# built -- _resolve_weights runs at parse time, long before that.
w_evo_rule: dict[str, object] | None = None
if _args.w_evo_shallow is not None:
    _n_homologs = (pssm_depth or {}).get("n_homologs")
    _w_evo_before = _args.w_evo
    if _n_homologs is None:
        # No in-loop alignment set (--pssm, or every template missed the TM
        # cutoff and the run fell back to a flat PSSM).  Treating unknown depth
        # as "deep" is a decision, so it is logged and recorded, never silent.
        _applied, _reason = False, "depth unknown (precomputed or flat PSSM)"
        log.warning(
            "--w-evo-shallow %.3f requested but this run has no PSSM depth "
            "(%s) — keeping the full w_evo=%.4f.",
            _args.w_evo_shallow, _reason, _args.w_evo)
    elif _n_homologs < _args.shallow_depth_threshold:
        _applied, _reason = True, "shallow"
        _args.w_evo = _args.w_evo * _args.w_evo_shallow
        log.info(
            "Shallow profile (n_homologs=%.0f < %d): w_evo %.4f -> %.4f "
            "(x%.3f)", _n_homologs, _args.shallow_depth_threshold,
            _w_evo_before, _args.w_evo, _args.w_evo_shallow)
    else:
        _applied, _reason = False, "deep"
        log.info(
            "Deep profile (n_homologs=%.0f >= %d): w_evo stays %.4f.",
            _n_homologs, _args.shallow_depth_threshold, _args.w_evo)
    w_evo_rule = {
        "factor":       float(_args.w_evo_shallow),
        "threshold":    int(_args.shallow_depth_threshold),
        "n_homologs":   (float(_n_homologs) if _n_homologs is not None else None),
        "applied":      _applied,
        "reason":       _reason,
        "w_evo_before": float(_w_evo_before),
        "w_evo_after":  float(_args.w_evo),
    }

# ── Learned tertiary leaf (the A2 arm, on by default) ────────────────────────
# Resolved here rather than at parse time because building a missing table
# needs the target's structure file and its chain spec, which only exist once
# the PDB has been read.
_learned_dir = None
if _args.learned_on:
    from evo_rna.learned.tables import ensure_table  # noqa: E402
    _tables_dir = Path(_args.learned_tertiary or PATHS.learned_tables)
    ensure_table(
        PDB_PATH,
        Path(PDB_PATH).stem,
        tables_dir=_tables_dir,
        models_dir=_args.learned_models or PATHS.learned_models,
        chains=CHAIN_ID,
        work_dir=Path(OUT_DIR) / "learned_work",
    )
    _learned_dir = str(_tables_dir)
    log.info("Tertiary block: LEARNED potential (A2 arm), tables in %s", _tables_dir)
else:
    log.info("Tertiary block: binned leaves hbond/sa/torsion (A1/A0 control arm).")

efn = EnergyFunction.default(
    struct,
    w_evo=_args.w_evo,
    w_secondary=_args.w_secondary,
    w_tertiary=_args.w_tertiary,
    secondary_weights=tuple(_args.secondary_weights),
    tertiary_weights=tuple(_args.tertiary_weights),
    hbond_propensity=_hbond_prop,
    w_composition=_args.w_composition,
    gc_anchor=_args.gc_anchor,
    gc_tolerance=_args.gc_tolerance,
    gc_min_homologs=_args.gc_min_homologs,
    # None for --pssm / flat-PSSM runs; the term then reports depth as unknown
    # rather than pretending the reliability gate ran.
    pssm_n_homologs=(pssm_depth or {}).get("n_homologs"),
    # Only read when --gc-anchor native (an oracle ablation); the default
    # "pssm" anchor never touches the held-out sequence.
    native_sequence=native_sequence,
    temperature=_args.temperature,
    # Per-target tables keyed by the same stem the output tree uses, so an
    # arm is one flag and nothing else in the run changes.
    learned_tertiary=_learned_dir,
    target=Path(PDB_PATH).stem,
)
if _args.w_composition:
    _gc_term = efn.terms["composition"][0]
    log.info("GC-composition prior ON (w=%.3f, anchor=%s): reference GC=%.3f "
             "[resolved: %s], free band ±%.3f  [native GC=%.3f, held out]",
             _args.w_composition, _args.gc_anchor, _gc_term.target_gc,
             _gc_term.resolved_from, _args.gc_tolerance,
             gc_fraction(native_sequence))
    if _args.gc_anchor == "native":
        log.warning("--gc-anchor native LEAKS the held-out sequence into the "
                    "objective. Valid only as an oracle ablation — do not "
                    "report seq_recovery from this run as a design result.")

# ── Optional energy logger ────────────────────────────────────────────────────
_energy_logger = None
if _args.log_energy:
    from evo_rna.monte_carlo.energy_logger import EnergyLogger  # noqa: E402
    _energy_logger = EnergyLogger(struct, efn)
    log.info(
        "Energy logger enabled  → %s/energy_log.json",
        OUT_DIR,
    )
# ─────────────────────────────────────────────────────────────────────────────

# Sanity-check score for the random start sequence.
start_scores = efn.score(struct)
log.info("Start sequence scores (random baseline):")
for k, v in start_scores.items():
    log.info("  %-16s %.4f", k, v)


# ---------------------------------------------------------------------------
# 3. Monte Carlo  (~1 min budget)
# ---------------------------------------------------------------------------
# Cooling schedule.  t_max is auto-adapted to the energy scale at runtime
# (from mean|ΔE|), so the literal t_max here is just an upper bound.  The
# quench floor is set as a *ratio* of t_max rather than an absolute: with an
# absolute floor the number of temperature levels — and hence the total move
# budget — grows with the energy scale, so a run with larger weights would
# quietly get more sampling than an otherwise identical one.  The default
# ratio reproduces the 24 levels (4800 moves at 200/level) that the previous
# t_max~2 / t_min=0.01 pairing produced.
cfg = SAConfig(
    t_max=5.0,
    t_min=0.01,                                    # used only if ratio is off
    t_min_ratio=(_args.t_min_ratio or None),       # 0 -> absolute t_min
    cool_factor=0.8,
    steps_per_temp=_args.steps_per_temp,
    num_cycles=1,
    seed=_args.seed,
    move_operator=_args.move_operator,
    position_sampling=_args.position_sampling,
    defect_temperature=_args.defect_temperature,
    defect_refresh=_args.defect_refresh,
)

log.info(
    "Starting SA  t_max=%.1f  t_min=%.3f  cool=%.2f  steps/T=%d  cycles=%d  "
    "moves=%s  positions=%s%s",
    cfg.t_max, cfg.t_min, cfg.cool_factor, cfg.steps_per_temp, cfg.num_cycles,
    cfg.move_operator, cfg.position_sampling,
    (f" (T0={cfg.defect_temperature:g}, refresh={cfg.defect_refresh})"
     if cfg.position_sampling == "defect" else ""),
)

t0 = time.time()
result = run_sa(struct, energy_fn=efn, config=cfg, callback=_energy_logger)
elapsed = time.time() - t0

if _energy_logger is not None:
    _energy_logger.set_metadata(
        pdb_id          = Path(PDB_PATH).stem,
        best_sequence   = result.best_sequence,
        start_sequence  = start_sequence,
        native_sequence = native_sequence,
        pair_map        = pair_map,
        accept_rate     = result.accept_rate,
        move_operator   = cfg.move_operator,
        position_sampling = cfg.position_sampling,
        position_sampler = result.position_sampler,
        proposed_by_kind = result.proposed_by_kind,
        accepted_by_kind = result.accepted_by_kind,
        sites_by_kind    = result.sites_by_kind,
        by_kind_by_level = result.by_kind_by_level,
        proposed_sites_by_position = result.proposed_sites_by_position,
    )
    _log_path = Path(OUT_DIR) / "energy_log.json"
    _energy_logger.save(_log_path)
    log.info("Energy log saved → %s  (%d accepted moves)", _log_path, len(_energy_logger._accepted_idx))


# Persist what the MC designed.  This is the pipeline's product: design.json
# is what a downstream evaluation of your own reads.
import json as _json  # noqa: E402

_design_path = Path(OUT_DIR) / "design.json"
_design_doc = {
    "pdb_id":          Path(PDB_PATH).stem,
    "chain_id":        CHAIN_ID or "",
    "sequence":        result.best_sequence,
    "native_sequence": native_sequence,
    "start_sequence":  start_sequence,
    "nt_length":       L,
    "seed":            _args.seed,
    "accept_rate":     float(result.accept_rate),
    "move_operator":   cfg.move_operator,
    # What the position sampler actually did — "defect" degrades to uniform
    # when ViennaRNA is missing, so the run must record which it got.
    "position_sampling": cfg.position_sampling,
    "position_sampler":  dict(result.position_sampler),
    "proposed_by_kind": dict(result.proposed_by_kind),
    "accepted_by_kind": dict(result.accepted_by_kind),
    "sites_by_kind":    dict(result.sites_by_kind),
    # The same counters per temperature level, plus the per-position proposal
    # budget: enough to say how the move mix and its acceptance moved as the
    # anneal cooled, and where in the sequence the budget landed.
    "by_kind_by_level": list(result.by_kind_by_level),
    "proposed_sites_by_position": list(result.proposed_sites_by_position),
    "mc_elapsed_s":    float(elapsed),
    "best_energy":     {k: float(v) for k, v in result.best_energy.items()},
    # The Evo weight the MC actually ran with, plus the depth rule that set
    # it: without both, a run under --w-evo-shallow is indistinguishable from
    # one that simply had a different weight on the command line.
    "w_evo_effective": float(_args.w_evo),
    "w_evo_rule":      w_evo_rule,
    "w_composition":   float(_args.w_composition),
    "learned_tertiary": _learned_dir,
    "gc_anchor":       _args.gc_anchor if _args.w_composition else None,
    "gc_reference":    (efn.terms["composition"][0].target_gc
                        if _args.w_composition else None),
    "gc_resolved_from": (efn.terms["composition"][0].resolved_from
                         if _args.w_composition else None),
    "pssm_depth":      pssm_depth,
}
_design_path.write_text(_json.dumps(_design_doc, indent=2))
log.info("Design saved → %s", _design_path)
