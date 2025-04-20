# Papilon 🦋

**Papilon** is a Python library for exploring the statistical mechanics of complex systems. Inspired by the butterfly effect and chaos theory, Papilon enables researchers and data scientists to simulate, analyze, and optimize dynamic systems through entropy modeling, causal inference, and scenario generation.

![Butterfly Effect](https://upload.wikimedia.org/wikipedia/commons/thumb/f/f3/Lorenz_system_r28_s10_b2-6666.png/600px-Lorenz_system_r28_s10_b2-6666.png)

## ✨ Features

- 📊 **Entropy & Energy Modeling** — quantify system uncertainty
- 🔄 **Causal Inference** — reveal hidden directional relationships
- 🔬 **Scenario Simulation** — generate alternative state evolutions
- 🧠 **Optimization Engine** — search for efficient configurations
- 🧩 **Relationship Discovery** — mutual information & correlation
- 📈 **Optional MMM & Regression** — model outcome influence

## 📦 Installation

```bash
git clone https://github.com/YOUR_USERNAME/papilon.git
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

![Sample causal graph](https://upload.wikimedia.org/wikipedia/commons/thumb/3/3b/Simple_dag.svg/500px-Simple_dag.svg.png)

## 📝 License

MIT License

---

_Developed with curiosity and chaos in mind._
