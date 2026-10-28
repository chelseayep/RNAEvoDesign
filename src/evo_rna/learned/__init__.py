"""The learned tertiary potential: feature extraction, the heads, and export.

- :mod:`evo_rna.learned.features` — continuous geometry per residue and per
  contact, the matrices the heads are fit on.
- :mod:`evo_rna.learned.model` — the ``Head`` network and its training loop.
- :mod:`evo_rna.learned.tables` — turn a trained head into the per-target
  log-odds arrays the MC loop reads (``ensure_table`` builds one on demand).

Only :mod:`~evo_rna.learned.tables` is touched at design time, and only once,
at initialisation; the MC loop itself imports no torch.
"""
