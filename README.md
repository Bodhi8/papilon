# Papilon 🦋

**Papilon** is a Python library for exploring the statistical mechanics of complex systems. Created and developed by Brian Curry and inspired by the butterfly effect and chaos theory, Papilon enables researchers and data scientists to simulate, analyze, and optimize dynamic systems through entropy modeling, causal inference, and scenario generation.

<p align="center">
  <img src="lorenz_attractor.png" alt="Lorenz Attractor - Butterfly Effect" width="500"/>
</p>

<p align="center"><em>“Does the flap of a butterfly’s wings in Brazil set off a tornado in Texas?” — Edward Lorenz</em></p>

## ✨ Features

- 📊 **Entropy & Energy Modeling** — quantify system uncertainty
- 🔄 **Causal Inference** — reveal hidden directional relationships
- 🔬 **Scenario Simulation** — generate alternative state evolutions
- 🧠 **Optimization Engine** — search for efficient configurations
- 🧩 **Relationship Discovery** — mutual information & correlation
- 📈 **Optional MMM & Regression** — model outcome influence

## 📦 Installation

```bash
git clone https://github.com/Bodhi8/papilon.git
cd papilon
pip install -e .
```

> Requires: Python 3.8+, numpy, pandas, matplotlib, seaborn, scikit-learn, networkx, statsmodels

## 🚀 Example Usage

```python
from papilon import (
    shannon_entropy,
    simulate_kde_scenarios,
    analyze_relationships,
    infer_causal_structure,
    grid_search_optimize
)
```

See [`examples/`](examples/) for a full end-to-end complex systems notebook.

## 📘 Use Cases

- Studying dynamic system behavior (ecology, traffic, climate)
- Testing interventions under causal feedback
- Finding stable/efficient parameter regimes
- Teaching entropy, causality, and complexity science

## 🧪 Example Output

<p align="center">
  <img src="sample_causal_graph.png" alt="Sample Causal Graph" width="500"/>
</p>

<p align="center"><em>Example: A simple directed acyclic graph (DAG) used in causal inference.</em></p>

## 📝 License

Apache License 2.0 © 2025 Vector1 Research


---

_Developed with curiosity and chaos in mind._
