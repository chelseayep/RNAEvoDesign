#!/usr/bin/env bash
# =============================================================================
# setup_RNAfold.sh
#
# Install ViennaRNA (RNAfold) into a conda environment.
#
# Usage:
#   bash scripts/setup/setup_RNAfold.sh
#
# Designed to work on:
#   - macOS (local dev, with miniconda/miniforge)
#   - Linux HPC clusters (miniforge in $HOME, no root access)
#
# Override defaults via environment variables before running:
#   ENV_NAME=my_env  bash scripts/setup/setup_RNAfold.sh
#
# If conda is not in your PATH after this script, run once:
#   conda init bash    (Linux cluster)
#   conda init zsh     (macOS)
# then restart your shell.
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Configurable — override with env vars if needed
# ---------------------------------------------------------------------------
ENV_NAME="${ENV_NAME:-rna_design}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

# ---------------------------------------------------------------------------
# Locate and initialise conda
# ---------------------------------------------------------------------------
_find_conda_sh() {
    for candidate in \
        "$HOME/miniforge3/etc/profile.d/conda.sh" \
        "$HOME/miniconda3/etc/profile.d/conda.sh" \
        "$HOME/mambaforge/etc/profile.d/conda.sh" \
        "$HOME/anaconda3/etc/profile.d/conda.sh" \
        "$HOME/opt/miniconda3/etc/profile.d/conda.sh" \
        "$HOME/opt/anaconda3/etc/profile.d/conda.sh" \
        "/opt/miniconda3/etc/profile.d/conda.sh" \
        "/opt/anaconda3/etc/profile.d/conda.sh" \
        "/opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh"; do
        [ -f "$candidate" ] && echo "$candidate" && return
    done
}

CONDA_SH="$(_find_conda_sh)"
if [ -z "$CONDA_SH" ]; then
    echo "ERROR: conda/miniforge not found."
    echo "  Install Miniforge: https://github.com/conda-forge/miniforge/releases/latest"
    exit 1
fi

# shellcheck source=/dev/null
source "$CONDA_SH"
echo "conda:  $(conda --version)  ($CONDA_SH)"

# Use mamba if available (ships with miniforge, much faster than conda)
if command -v mamba &>/dev/null; then
    PKG_MANAGER="mamba"
else
    PKG_MANAGER="conda"
fi
echo "pkg manager: $PKG_MANAGER"
echo ""

# ---------------------------------------------------------------------------
# Create env (idempotent)
# ---------------------------------------------------------------------------
echo "=== ViennaRNA / RNAfold ==="

if conda env list | grep -q "^${ENV_NAME} "; then
    echo "  conda env '${ENV_NAME}' already exists — skipping create."
else
    echo "  Creating conda env '${ENV_NAME}' (python=${PYTHON_VERSION}) ..."
    $PKG_MANAGER create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}"
fi

# ---------------------------------------------------------------------------
# Install ViennaRNA
#
# conda-forge carries a recent build for both linux-64 and osx-arm64/osx-64.
# bioconda is added as a fallback channel in case conda-forge lags.
# ---------------------------------------------------------------------------
echo "  Installing ViennaRNA ..."
$PKG_MANAGER install -y -n "${ENV_NAME}" \
    -c conda-forge \
    -c bioconda \
    viennarna

# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
echo ""
echo "  Running smoke test ..."
RNAFOLD_VERSION="$(conda run -n "${ENV_NAME}" RNAfold --version 2>&1 | head -1)"
echo "  ${RNAFOLD_VERSION}"

echo ""
echo "  Folding test sequence: GCGCUUCGGCGC"
RESULT="$(echo 'GCGCUUCGGCGC' | conda run -n "${ENV_NAME}" RNAfold --noPS 2>/dev/null)"
echo "  ${RESULT}"

# ---------------------------------------------------------------------------
echo ""
echo "=== Setup complete ==="
echo "  Conda env  : ${ENV_NAME}"
echo "  ViennaRNA  : ${RNAFOLD_VERSION}"
echo ""
echo "Activate and use:"
echo "  conda activate ${ENV_NAME}"
echo "  echo 'GCGCUUCGGCGC' | RNAfold"
echo ""
echo "Or run without activating:"
echo "  conda run -n ${ENV_NAME} RNAfold < my_sequences.fasta"
