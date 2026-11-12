#   Copyright 2024 - 2026 Brian Curry / Vector1 Research
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""Papilon: marketing mix modeling, causal data simulation and complex-systems tools."""

from .causal import infer_causal_structure
from .datasets import make_synthetic_geo_mmm, make_synthetic_mmm
from .discovery import discover_structure
from .energy import energy_score
from .entropy import joint_entropy, shannon_entropy
from .graph import MarketingGraph
from .loop import Loop
from .mmm import MMM
from .mmm.geo import GeoMMM
from .optimizer import grid_search_optimize
from .relationships import analyze_relationships
from .simulation import simulate_kde_scenarios
from .states import count_microstates, macrostate_entropy

__version__ = "1.0.0"

__all__ = [
    "MMM",
    "GeoMMM",
    "Loop",
    "MarketingGraph",
    "make_synthetic_mmm",
    "make_synthetic_geo_mmm",
    "shannon_entropy",
    "joint_entropy",
    "simulate_kde_scenarios",
    "energy_score",
    "count_microstates",
    "macrostate_entropy",
    "analyze_relationships",
    "infer_causal_structure",
    "discover_structure",
    "grid_search_optimize",
]
