"""Monte Carlo sequence design for RNA.

Key entry points
----------------
mc.run_sa                         Run simulated annealing on a core.Structure
energy.EnergyFunction             Weighted composite of energy terms (evo + rnafold + rna_fm)
energy_terms.evo.EvoTerm          PSSM log-odds term
energy_terms.stacking             StackingContextTerm (stacking-aware BP quality)
energy_terms._structural          SSTerm, SATerm, TorsionTerm (require 3-D structural context)
energy_terms.rna_fm               RNAFMTerm (requires multimolecule)
"""

from .energy_terms.evo import EnergyTerm, EvoTerm
from .energy_terms.stacking import StackingContextTerm
from .energy import EnergyFunction
from .mc import (
    DesignResult,
    SAConfig,
    run_sa,
)
from .energy_terms.ss_rnafold import (
    RNAfoldTerm,
    base_pair_mcc,
    bp_distance,
    dot_bracket_to_pair_map,
    fold,
    pair_map_to_dot_bracket,
)

# RNAFMTerm is imported lazily to avoid requiring torch/multimolecule at import
# time.  Import directly when needed:
#   from evo_rna.monte_carlo.energy_terms.rna_fm import RNAFMTerm
# or use EnergyFunction.default() which handles the import internally.
try:
    from .energy_terms.rna_fm import RNAFMTerm
except ImportError:
    pass

__all__ = [
    # Energy function and base class
    "EnergyFunction",
    "EnergyTerm",
    "EvoTerm",
    "StackingContextTerm",
    # RNAfold term
    "RNAfoldTerm",
    "fold",
    "bp_distance",
    "base_pair_mcc",
    "dot_bracket_to_pair_map",
    "pair_map_to_dot_bracket",
    # RNA-FM term (optional dep)
    "RNAFMTerm",
    # MC driver
    "DesignResult",
    "SAConfig",
    "run_sa",
]
