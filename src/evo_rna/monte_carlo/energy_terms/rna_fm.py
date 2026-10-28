"""RNA Foundation Model (RNA-FM) energy term for Monte Carlo sequence design.

Scores sequences via pseudo-log-likelihood (PLL):

    PLL(S) = Σ_i  log P_FM(s_i | S_{-i})
    E_FM(S) = -PLL(S)

Because each MC step mutates a single position k, the delta can be computed
with exactly one forward pass (mask position k, compare log P(old) vs log P(new)):

    ΔE_FM = log P_FM(s_k^old | S_{-k}) - log P_FM(s_k^new | S_{-k})

score() runs L batched forward passes (one mask per position); call it only at
logging checkpoints, not inside the hot MC loop.

Requirements
------------
    pip install multimolecule torch
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from evo_rna.core.structure import NUC_TO_INDEX, Structure

from .evo import EnergyTerm

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

# NUC_TO_INDEX = {A:0, U:1, G:2, C:3}  (matches evo_rna convention)
_NUC_ORDER = ["A", "U", "G", "C"]


class RNAFMTerm(EnergyTerm):
    """RNA-FM masked-LM pseudo-log-likelihood energy term.

    Parameters
    ----------
    model :
        ``RnaFmForMaskedLM`` instance (from multimolecule).
    tokenizer :
        ``RnaTokenizer`` instance (from multimolecule).
    device : str, optional
        PyTorch device string (``"cpu"``, ``"cuda"``, ``"mps"``).
        Defaults to CUDA if available, else CPU.
    score_batch_size : int
        Number of masked sequences to process in one forward pass inside
        ``score()``.  Reduce if you hit GPU OOM on long sequences.
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: str | None = None,
        score_batch_size: int = 32,
    ) -> None:
        import torch

        self._tok = tokenizer
        self._score_batch_size = score_batch_size

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self._device = device

        self._model = model.to(device)
        self._model.eval()

        # Pre-compute token IDs for A, U, G, C in vocab order.
        vocab = tokenizer.get_vocab()
        self._nuc_token_ids: list[int] = []
        for nuc in _NUC_ORDER:
            if nuc not in vocab:
                raise ValueError(
                    f"Nucleotide '{nuc}' not found in RNA-FM tokenizer vocabulary. "
                    f"Available tokens: {sorted(vocab.keys())}"
                )
            self._nuc_token_ids.append(vocab[nuc])

        log.info(
            "RNAFMTerm ready  device=%s  nuc_token_ids=%s",
            self._device,
            dict(zip(_NUC_ORDER, self._nuc_token_ids)),
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name: str | None = None,
        device: str | None = None,
        score_batch_size: int = 32,
    ) -> "RNAFMTerm":
        """Load RNA-FM weights, preferring a local path over the HuggingFace Hub.

        Resolution order:
        1. ``model_name`` argument (if provided)
        2. ``RNAFM_WEIGHTS_PATH`` environment variable (set by ``setup_rna_fm.sh``)
        3. HuggingFace Hub ``"multimolecule/rnafm"`` (requires internet)

        Parameters
        ----------
        model_name :
            Local directory or HuggingFace model identifier.  If ``None``,
            ``RNAFM_WEIGHTS_PATH`` is checked before falling back to the Hub.
        device :
            PyTorch device.  Auto-detected if ``None``.
        score_batch_size :
            Batch size used in ``score()``.

        Raises
        ------
        ImportError
            If ``multimolecule`` is not installed.  Run ``scripts/setup_rna_fm.sh``.
        """
        import os

        try:
            from multimolecule import RnaFmForMaskedLM, RnaTokenizer
        except ImportError as exc:
            raise ImportError(
                "multimolecule is required for RNAFMTerm.  "
                "Run:  bash scripts/setup_rna_fm.sh"
            ) from exc

        if model_name is None:
            model_name = os.environ.get("RNAFM_WEIGHTS_PATH", "multimolecule/rnafm")

        from_hub = model_name == "multimolecule/rnafm"
        kwargs = {"force_download": False} if from_hub else {"local_files_only": True}

        log.info("Loading RNA-FM tokenizer from '%s' ...", model_name)
        tokenizer = RnaTokenizer.from_pretrained(model_name, **kwargs)

        log.info("Loading RNA-FM model weights from '%s' ...", model_name)
        model = RnaFmForMaskedLM.from_pretrained(model_name, **kwargs)

        return cls(model, tokenizer, device=device, score_batch_size=score_batch_size)

    # ------------------------------------------------------------------
    # EnergyTerm interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "rna_fm"

    def delta(self, struct: Structure, pos: int, new_nuc: str) -> float:
        """Energy change for mutating position *pos* to *new_nuc*.

        Cost: exactly one RNA-FM forward pass.

        ΔE = (log P_FM(old_nuc | S_{-pos}) − log P_FM(new_nuc | S_{-pos})) / L

        Dividing by sequence length L keeps the per-step delta on the same
        scale as score() and prevents rna_fm from dominating longer sequences.
        """
        old_nuc = struct.sequence[pos]
        old_j = NUC_TO_INDEX.get(old_nuc, -1)
        new_j = NUC_TO_INDEX.get(new_nuc, -1)
        if old_j < 0 or new_j < 0:
            return 0.0

        log_probs = self._log_probs_at_pos(struct.sequence, pos)
        # E = -PLL/L  →  ΔE = (log P(old) − log P(new)) / L
        return float(log_probs[old_j] - log_probs[new_j]) / len(struct.sequence)

    def score(self, struct: Structure) -> float:
        """Full-sequence PLL energy normalised by sequence length.

        E_FM = -( Σ_i  log P_FM(s_i | S_{-i}) ) / L

        Dividing by L keeps the score comparable across sequences of different
        lengths and consistent with the per-step delta() normalisation.
        Use for logging / cycle checkpoints, not inside the per-step hot path.
        """
        import torch

        seq = struct.sequence
        L = len(seq)
        total = 0.0

        for batch_start in range(0, L, self._score_batch_size):
            positions = range(batch_start, min(batch_start + self._score_batch_size, L))

            masked_seqs = []
            for i in positions:
                chars = list(seq)
                chars[i] = self._tok.mask_token
                masked_seqs.append("".join(chars))

            enc = self._tok(masked_seqs, return_tensors="pt", padding=True)
            enc = {k: v.to(self._device) for k, v in enc.items()}

            with torch.no_grad():
                out = self._model(**enc)
            # out.logits: (batch, seq_len_padded, vocab_size)

            for batch_idx, i in enumerate(positions):
                nuc = seq[i]
                j = NUC_TO_INDEX.get(nuc, -1)
                if j < 0:
                    continue
                tok_pos = self._find_mask_pos(enc["input_ids"][batch_idx])
                if tok_pos is None:
                    log.warning("Mask token not found at batch_idx=%d pos=%d", batch_idx, i)
                    continue
                logits = out.logits[batch_idx, tok_pos, :]
                log_probs = torch.log_softmax(logits, dim=-1)
                total -= log_probs[self._nuc_token_ids[j]].item()

        return total / len(seq)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log_probs_at_pos(self, sequence: str, mask_pos: int) -> np.ndarray:
        """One forward pass with position *mask_pos* masked.

        Returns a (4,) array of log-probabilities for [A, U, G, C].
        """
        import torch

        chars = list(sequence)
        chars[mask_pos] = self._tok.mask_token
        masked_seq = "".join(chars)

        enc = self._tok(masked_seq, return_tensors="pt")
        enc = {k: v.to(self._device) for k, v in enc.items()}

        with torch.no_grad():
            out = self._model(**enc)
        # out.logits: (1, seq_len, vocab_size)

        tok_pos = self._find_mask_pos(enc["input_ids"][0])
        if tok_pos is None:
            log.warning("Mask token not found in encoded sequence at pos=%d", mask_pos)
            return np.zeros(4, dtype=np.float64)

        logits = out.logits[0, tok_pos, :]
        log_probs = torch.log_softmax(logits, dim=-1)
        return np.array(
            [log_probs[tid].item() for tid in self._nuc_token_ids],
            dtype=np.float64,
        )

    def _find_mask_pos(self, input_ids) -> int | None:
        """Return the token index of the [MASK] token, or None if not found."""
        import torch

        mask_id = self._tok.mask_token_id
        positions = (input_ids == mask_id).nonzero(as_tuple=False)
        if positions.numel() == 0:
            return None
        return int(positions[0].item())
