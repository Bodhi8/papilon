"""Papilon: marketing mix modeling, causal data simulation and complex-systems tools."""

from .causal import infer_causal_structure
from .datasets import make_synthetic_mmm
from .energy import energy_score
from .entropy import joint_entropy, shannon_entropy
from .mmm import MMM
from .optimizer import grid_search_optimize
from .relationships import analyze_relationships
from .simulation import simulate_kde_scenarios
from .states import count_microstates, macrostate_entropy

__version__ = "0.3.0"

__all__ = [
    "MMM",
    "make_synthetic_mmm",
    "shannon_entropy",
    "joint_entropy",
    "simulate_kde_scenarios",
    "energy_score",
    "count_microstates",
    "macrostate_entropy",
    "analyze_relationships",
    "infer_causal_structure",
    "grid_search_optimize",
]
