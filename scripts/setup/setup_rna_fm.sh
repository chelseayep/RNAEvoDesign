#!/usr/bin/env bash
# =============================================================================
# setup_rna_fm.sh
#
# Install RNA-FM (via multimolecule) into the shared 'rna_design' conda env
# so it can be used as an MC energy term alongside RNAfold.
#
# Prerequisites
# -------------
# Run setup_RNAfold.sh first to create the 'rna_design' conda env, then
# activate it before calling this script:
#
#   bash scripts/setup/setup_RNAfold.sh          # creates rna_design env + RNAfold
#   conda activate rna_design
#   bash scripts/setup/setup_rna_fm.sh           # adds torch + multimolecule (RNA-FM)
#
# HPC (no root, SLURM)
# --------------------
# If the HPC has unreliable internet, download the weights locally first
# and SCP them over, then point the script at the local copy:
#
#   # On your local machine:
#   python - <<'EOF'
#   from multimolecule import RnaFmForMaskedLM, RnaTokenizer
#   RnaTokenizer.from_pretrained("multimolecule/rnafm").save_pretrained("./rnafm_weights")
#   RnaFmForMaskedLM.from_pretrained("multimolecule/rnafm").save_pretrained("./rnafm_weights")
#   EOF
#   scp -r ./rnafm_weights chelsea@rna01:/mnt/rna01/chelsea/models/rnafm_weights
#
#   # On HPC — pass the local path via RNAFM_WEIGHTS_PATH:
#   RNAFM_WEIGHTS_PATH=/mnt/rna01/chelsea/models/rnafm_weights \
#     bash scripts/setup/setup_rna_fm.sh
#
# Otherwise, request a node with internet and let the script download:
#
#   srun --pty --partition=gpu --gres=gpu:1 --time=01:00:00 bash
#   conda activate rna_design
#   bash scripts/setup/setup_rna_fm.sh
#
# The HuggingFace weights (~100 MB) are cached in $HF_HOME or
# ~/.cache/huggingface/ and are shared across jobs automatically.
# Set HF_HOME to a shared project directory if you want all users to
# benefit from a single download:
#
#   export HF_HOME=/path/to/shared/hf_cache
#   bash scripts/setup/setup_rna_fm.sh
#
# Works on
# --------
#   - macOS (CPU / MPS on Apple Silicon)
#   - Linux HPC (CPU / CUDA GPU, no root required)
#
# Override torch variant via env var if auto-detection is wrong:
#   TORCH_INSTALL=cpu   bash scripts/setup/setup_rna_fm.sh   # force CPU-only torch
#   TORCH_INSTALL=cu118 bash scripts/setup/setup_rna_fm.sh   # CUDA 11.8
#   TORCH_INSTALL=cu121 bash scripts/setup/setup_rna_fm.sh   # CUDA 12.1
#   TORCH_INSTALL=cu124 bash scripts/setup/setup_rna_fm.sh   # CUDA 12.4
#
# After running, verify with:
#   python -c "from evo_rna.monte_carlo import RNAFMTerm; RNAFMTerm.from_pretrained(); print('OK')"
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Configurable
# ---------------------------------------------------------------------------
ENV_NAME="${ENV_NAME:-rna_design}"
TORCH_INSTALL="${TORCH_INSTALL:-auto}"   # auto | cpu | cu118 | cu121 | cu124

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
echo_step() { echo ""; echo "=== $* ==="; }
echo_ok()   { echo "  [ok] $*"; }
echo_info() { echo "  [..] $*"; }

# ---------------------------------------------------------------------------
# Detect Python
# ---------------------------------------------------------------------------
echo_step "Detecting Python"

# Prefer the active venv / conda env, then fall back to 'python3'
if [ -n "${VIRTUAL_ENV:-}" ]; then
    PYTHON="$VIRTUAL_ENV/bin/python"
    echo_ok "Active virtualenv: $VIRTUAL_ENV"
elif [ -n "${CONDA_PREFIX:-}" ]; then
    PYTHON="$CONDA_PREFIX/bin/python"
    echo_ok "Active conda env: $CONDA_PREFIX"
else
    # Try to activate the project venv if present
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
    if [ -f "$PROJECT_ROOT/venv/bin/activate" ]; then
        # shellcheck source=/dev/null
        source "$PROJECT_ROOT/venv/bin/activate"
        PYTHON="$PROJECT_ROOT/venv/bin/python"
        echo_ok "Activated project venv: $PROJECT_ROOT/venv"
    elif [ -f "$PROJECT_ROOT/venv/Scripts/activate" ]; then
        # shellcheck source=/dev/null
        source "$PROJECT_ROOT/venv/Scripts/activate"
        PYTHON="$PROJECT_ROOT/venv/Scripts/python"
        echo_ok "Activated project venv (Windows): $PROJECT_ROOT/venv"
    else
        PYTHON="${PYTHON:-python3}"
        echo_info "No venv found — using: $PYTHON"
        echo_info "Create one first with:  python3 -m venv venv && source venv/bin/activate"
    fi
fi

PY_VERSION="$("$PYTHON" --version 2>&1)"
echo_ok "Python: $PY_VERSION  ($PYTHON)"

# Require Python >= 3.10
PY_MINOR="$("$PYTHON" -c 'import sys; print(sys.version_info.minor)')"
PY_MAJOR="$("$PYTHON" -c 'import sys; print(sys.version_info.major)')"
if [ "$PY_MAJOR" -lt 3 ] || ([ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]); then
    echo "ERROR: Python >= 3.10 is required (got $PY_VERSION)"
    exit 1
fi

# ---------------------------------------------------------------------------
# Install / upgrade pip
# ---------------------------------------------------------------------------
echo_step "Upgrading pip"
"$PYTHON" -m pip install --quiet --upgrade pip
echo_ok "pip ready"

# ---------------------------------------------------------------------------
# Detect hardware and choose torch variant
# ---------------------------------------------------------------------------
echo_step "Selecting PyTorch variant"

if [ "$TORCH_INSTALL" = "auto" ]; then
    # Apple Silicon → MPS backend
    if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
        TORCH_INSTALL="cpu"   # standard torch wheel includes MPS on macOS
        echo_info "Apple Silicon detected — using standard torch (MPS included)"
    # Linux with nvidia-smi → try CUDA
    elif command -v nvidia-smi &>/dev/null; then
        # Read CUDA version from nvidia-smi
        CUDA_VER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)"
        if nvidia-smi 2>/dev/null | grep -q "CUDA Version: 12"; then
            TORCH_INSTALL="cu121"
            echo_info "CUDA 12.x detected — using torch+cu121"
        elif nvidia-smi 2>/dev/null | grep -q "CUDA Version: 11"; then
            TORCH_INSTALL="cu118"
            echo_info "CUDA 11.x detected — using torch+cu118"
        else
            TORCH_INSTALL="cpu"
            echo_info "nvidia-smi found but CUDA version unclear — defaulting to CPU torch"
        fi
    else
        TORCH_INSTALL="cpu"
        echo_info "No GPU detected — using CPU-only torch"
    fi
fi

echo_ok "torch variant: $TORCH_INSTALL"

# ---------------------------------------------------------------------------
# Install PyTorch
# ---------------------------------------------------------------------------
echo_step "Installing PyTorch"

TORCH_ALREADY="$("$PYTHON" -c 'import torch; print(torch.__version__)' 2>/dev/null || true)"
if [ -n "$TORCH_ALREADY" ]; then
    echo_ok "torch already installed: $TORCH_ALREADY — skipping"
else
    case "$TORCH_INSTALL" in
        cpu)
            "$PYTHON" -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
            ;;
        cu118)
            "$PYTHON" -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cu118
            ;;
        cu121)
            "$PYTHON" -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cu121
            ;;
        cu124)
            "$PYTHON" -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cu124
            ;;
        *)
            echo "ERROR: Unknown TORCH_INSTALL value: '$TORCH_INSTALL'"
            echo "  Valid options: auto cpu cu118 cu121 cu124"
            exit 1
            ;;
    esac
    TORCH_VER="$("$PYTHON" -c 'import torch; print(torch.__version__)')"
    echo_ok "torch installed: $TORCH_VER"
fi

# ---------------------------------------------------------------------------
# Install multimolecule + transformers (pinned, Python 3.10 compatible)
#
# multimolecule 0.0.6+ requires TransformersKwargs (transformers >=4.46)
# but transformers >=4.46 breaks Python 3.10+ dataclass inheritance rules.
# Pin multimolecule to 0.0.5 (last release before TransformersKwargs was
# introduced) and transformers to <4.46.
#
# Key-name compatibility note
# ---------------------------
# multimolecule >=0.0.6 renamed the backbone prefix in state dicts:
#   old (0.0.4/0.0.5):  rnafm.embeddings.*  /  rnafm.encoder.*
#   new (0.0.6+):       model.embeddings.*  /  model.encoder.*
# It also renamed  rnafm.encoder.emb_layer_norm_after  ->  model.encoder.layer_norm
# and added a secondary-structure head (ss_head.*) not present in older versions.
#
# If weights were saved locally with a newer multimolecule (e.g. installed via
# pip without a pin), this script will auto-remap the keys before loading.
# ---------------------------------------------------------------------------
MM_PIN="multimolecule==0.0.5"
TF_PIN="transformers>=4.40,<4.46"
RNAFM_WEIGHTS_PATH="${RNAFM_WEIGHTS_PATH:-}"   # set to a local dir to skip Hub download

echo_step "Installing transformers (pinned for Python 3.10 + multimolecule)"

"$PYTHON" -m pip install --quiet "$TF_PIN"
TF_VER="$("$PYTHON" -c 'from importlib.metadata import version; print(version("transformers"))')"
echo_ok "transformers: $TF_VER"

# ---------------------------------------------------------------------------
# Install multimolecule (provides RNA-FM via HuggingFace)
# Note: multimolecule does not publish a [torch] extra — install plain.
# ---------------------------------------------------------------------------
echo_step "Installing multimolecule (RNA-FM)"

MM_ALREADY="$("$PYTHON" -c 'from importlib.metadata import version; print(version("multimolecule"))' 2>/dev/null || true)"
if [ "$MM_ALREADY" = "0.0.5" ]; then
    echo_ok "multimolecule 0.0.5 already installed — skipping"
else
    if [ -n "$MM_ALREADY" ]; then
        echo_info "multimolecule $MM_ALREADY installed but need 0.0.5 — reinstalling"
    fi
    "$PYTHON" -m pip install --quiet "$MM_PIN"
    MM_VER="$("$PYTHON" -c 'from importlib.metadata import version; print(version("multimolecule"))')"
    echo_ok "multimolecule installed: $MM_VER"
fi

# ---------------------------------------------------------------------------
# Load / download RNA-FM weights
# ---------------------------------------------------------------------------
echo_step "Loading RNA-FM weights"

if [ -n "$RNAFM_WEIGHTS_PATH" ]; then
    # Use pre-copied local weights — no internet required
    if [ ! -d "$RNAFM_WEIGHTS_PATH" ]; then
        echo "ERROR: RNAFM_WEIGHTS_PATH='$RNAFM_WEIGHTS_PATH' does not exist."
        echo "  Copy weights from your local machine first:"
        echo "    scp -r ./rnafm_weights <user>@<hpc>:$RNAFM_WEIGHTS_PATH"
        exit 1
    fi
    echo_info "Using local weights: $RNAFM_WEIGHTS_PATH"

    # Auto-remap state dict keys if weights were saved with multimolecule >=0.0.6.
    # That version renamed the backbone prefix model.* -> rnafm.* and
    # model.encoder.layer_norm -> rnafm.encoder.emb_layer_norm_after.
    # Also adds safetensors format metadata required by transformers <4.46.
    "$PYTHON" - <<PYEOF
import json, sys
from pathlib import Path
from safetensors.torch import load_file, save_file

weights_dir = Path("$RNAFM_WEIGHTS_PATH")
sf = weights_dir / "model.safetensors"
if not sf.exists():
    sys.exit(0)  # nothing to remap

keys = list(load_file(sf, device="cpu").keys())
needs_remap = any(k.startswith("model.") for k in keys)

# Also check/fix safetensors format metadata
from safetensors import safe_open
with safe_open(str(sf), framework="pt") as f:
    has_format_meta = f.metadata() is not None and "format" in (f.metadata() or {})

if not needs_remap and has_format_meta:
    print("  [ok] weights already in correct format")
    sys.exit(0)

state_dict = load_file(str(sf), device="cpu")

if needs_remap:
    print("  [..] remapping state dict keys from multimolecule >=0.0.6 format ...")
    remapped = {}
    for k, v in state_dict.items():
        if k.startswith("ss_head."):
            continue  # secondary structure head not present in RnaFmForMaskedLM
        elif k.startswith("model.encoder.layer_norm."):
            new_k = k.replace("model.encoder.layer_norm.", "rnafm.encoder.emb_layer_norm_after.")
        elif k.startswith("model."):
            new_k = "rnafm." + k[len("model."):]
        else:
            new_k = k
        remapped[new_k] = v
    state_dict = remapped
    print(f"  [ok] remapped to {len(state_dict)} keys")

save_file(state_dict, str(sf), metadata={"format": "pt"})

# Update transformers_version in config to match installed version
cfg_path = weights_dir / "config.json"
if cfg_path.exists():
    from importlib.metadata import version
    cfg = json.loads(cfg_path.read_text())
    cfg["transformers_version"] = version("transformers")
    cfg_path.write_text(json.dumps(cfg, indent=2))

print("  [ok] weights normalised and saved")
PYEOF

    RNAFM_MODEL_ID="$RNAFM_WEIGHTS_PATH"
else
    # Download from HuggingFace Hub.
    # Wipe any existing cache first — partial downloads leave a corrupted
    # state dict that causes "state dictionary is corrupted" on load.
    HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}/hub/models--multimolecule--rnafm"
    if [ -d "$HF_CACHE" ]; then
        echo_info "Removing existing (possibly partial) weight cache: $HF_CACHE"
        rm -rf "$HF_CACHE"
    fi
    echo_info "Downloading multimolecule/rnafm weights from Hub (this may take a minute) ..."
    RNAFM_MODEL_ID="multimolecule/rnafm"
fi

"$PYTHON" - <<PYEOF
from multimolecule import RnaFmForMaskedLM, RnaTokenizer
model_id = "$RNAFM_MODEL_ID"
from_hub = model_id == "multimolecule/rnafm"
kwargs = {"force_download": True} if from_hub else {"local_files_only": True}
tok = RnaTokenizer.from_pretrained(model_id, **kwargs)
mdl = RnaFmForMaskedLM.from_pretrained(model_id, **kwargs)
print(f"  model params: {sum(p.numel() for p in mdl.parameters()):,}")
print(f"  vocab size  : {tok.vocab_size}")
PYEOF

echo_ok "weights ready"

# ---------------------------------------------------------------------------
# Smoke test: run a forward pass on a short sequence
# ---------------------------------------------------------------------------
echo_step "Smoke test — forward pass on AUGCAUGC"

"$PYTHON" - <<PYEOF
import torch
from multimolecule import RnaFmForMaskedLM, RnaTokenizer

tok = RnaTokenizer.from_pretrained("$RNAFM_MODEL_ID")
mdl = RnaFmForMaskedLM.from_pretrained("$RNAFM_MODEL_ID")
mdl.eval()

seq = "AUGCAUGC"
masked = list(seq)
masked[3] = tok.mask_token   # mask position 3 (C)
enc = tok("".join(masked), return_tensors="pt")

with torch.no_grad():
    out = mdl(**enc)

# Find mask position in token ids
mask_pos = (enc["input_ids"][0] == tok.mask_token_id).nonzero(as_tuple=False)[0].item()
logits = out.logits[0, mask_pos, :]
log_probs = torch.log_softmax(logits, dim=-1)

vocab = tok.get_vocab()
for nuc in ["A", "U", "G", "C"]:
    print(f"  P({nuc} | context) = {log_probs[vocab[nuc]].exp().item():.4f}")

print()
print("  Smoke test passed.")
PYEOF

# ---------------------------------------------------------------------------
echo ""
echo "=== Setup complete ==="
echo ""
echo "RNA-FM is ready.  Use it in your design:"
echo ""
echo "  from evo_rna.monte_carlo import RNAFMTerm, EnergyFunction"
echo ""
echo "  # Load the model once (weights cached in ~/.cache/huggingface/)"
echo "  rna_fm_term = RNAFMTerm.from_pretrained()"
echo ""
echo "  # Build an energy function with RNA-FM + evolutionary PSSM"
echo "  efn = EnergyFunction.with_rna_fm(struct, rna_fm_term)"
echo ""
echo "  # Or adjust weights:"
echo "  efn = EnergyFunction.with_rna_fm(struct, rna_fm_term, w_fm=1.5, w_evo=1.0)"
echo ""
