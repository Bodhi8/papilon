# Papilon 2.0

<p align="center">
  <img src="https://img.shields.io/badge/version-2.0.0-blue.svg" alt="Version">
  <img src="https://img.shields.io/badge/python-3.10+-green.svg" alt="Python">
  <img src="https://img.shields.io/badge/license-Proprietary-red.svg" alt="License">
</p>

<p align="center">
  <strong>The Unified Python Framework for Causal Media Modeling, Economic Systems & Complex Adaptive Systems</strong>
</p>

<p align="center">
  <em>"Does the flap of a butterfly's wings in Brazil set off a tornado in Texas?" — Edward Lorenz</em>
</p>

---

##  What is Papilon?

**Papilon** (French for "butterfly") is the most comprehensive Python framework for Marketing Mix Modeling, Causal Inference, and Complex Systems Analysis. Named after the butterfly effect, it captures how small changes in marketing spend can cascade into significant business outcomes.

### Why Papilon?

| Feature | Meridian | PyMC-Marketing | Robyn | **Papilon** |
|---------|----------|----------------|-------|-------------|
| Bayesian MMM | ✅ | ✅ | ❌ | ✅ |
| Causal Discovery | ❌ | ❌ | ❌ | ✅ |
| Causal Inference | ❌ | ❌ | ❌ | ✅ |
| Agent-Based Modeling | ❌ | ❌ | ❌ | ✅ |
| Economic I-O Models | ❌ | ❌ | ❌ | ✅ |
| Complex Systems | ❌ | ❌ | ❌ | ✅ |
| **Autonomous Agents** | ❌ | ❌ | ❌ | ✅ |
| **Workflow Automation** | ❌ | ❌ | ❌ | ✅ |
| Python Native | ✅ | ✅ | ❌ (R) | ✅ |

---

##  Quick Start

### Installation

```bash
pip install papilon
```

### Marketing Mix Modeling

```python
import papilon as pp

# Quick MMM Analysis
model = pp.MMM(
    data=df,
    target="revenue",
    media=["tv_spend", "digital_spend", "radio_spend"],
    date_col="date"
)
model.fit()

# Get ROI estimates
rois = model.get_roi()
print(rois)

# Optimize budget
optimal = model.optimize_budget(total=1_000_000)
print(optimal)
```

### Autonomous Agent (NEW in 2.0!)

```python
from papilon.orchestrator import MMMAgent

# Create an AI-powered agent
agent = MMMAgent(llm="claude")

# Natural language interface
results = agent.run("""
    Load my marketing data,
    fit an MMM model with TV, Digital, and Radio channels,
    optimize budget for $2M quarterly spend,
    and generate a report with visualizations.
""", data="marketing_data.csv")

# Or use quick analysis
results = agent.quick_analysis(
    data="data.csv",
    target_col="revenue",
    media_cols=["tv", "digital", "radio"],
    total_budget=2_000_000
)
```

### Full Automation

```python
from papilon.orchestrator import AutoMMM

# Create automated MMM system
auto = AutoMMM(
    data_source="marketing_data.csv",
    target="revenue",
    media_channels=["tv", "digital", "radio"]
)

# Configure automation
auto.set_retraining_schedule("0 9 * * 1")  # Weekly Monday 9am
auto.set_performance_threshold("mape", max_value=0.15)
auto.set_data_trigger(min_new_rows=7)

# Add notifications
from papilon.orchestrator import SlackNotifier
auto.add_notifier(SlackNotifier(webhook_url="..."))

# Start continuous automation
auto.start()
```

### Causal Discovery & Inference

```python
import papilon as pp

# Discover causal structure
dag = pp.discover_causal_structure(df, method="pc")

# Estimate causal effects
effect = pp.estimate_effect(
    df,
    treatment="marketing_spend",
    outcome="sales",
    method="backdoor"
)

# Synthetic control for geo experiments
sc = pp.SyntheticControl(
    treated_unit="California",
    control_units=["Texas", "Florida", "New York"],
    treatment_time="2024-01-01"
)
sc.fit(panel_data)
causal_effect = sc.estimate_effect()
```

---

##  Architecture

```
papilon/
├── core/                    # Foundation modules
│   ├── entropy.py          # Information theory
│   ├── energy.py           # Statistical mechanics
│   ├── states.py           # State space analysis
│   ├── relationships.py    # Correlation analysis
│   └── topology.py         # Network topology
│
├── mmm/                    # Marketing Mix Modeling
│   ├── transforms/         # Adstock & saturation
│   ├── models/             # Bayesian & frequentist
│   ├── decomposition.py    # Channel attribution
│   ├── response_curves.py  # Marginal response
│   ├── calibration.py      # Lift test integration
│   └── budget_optimizer.py # SLSQP optimization
│
├── causal/                 # Causal Inference
│   ├── discovery/          # PC, FCI, GES, NOTEARS
│   ├── effects/            # ATE, CATE, ITT, LATE
│   ├── methods/            # DiD, synthetic control, RDD, IV
│   ├── interventions.py    # do-calculus
│   └── refutation.py       # Robustness checks
│
├── orchestrator/           #  Autonomous Agents
│   ├── base.py             # Agent framework
│   ├── agents.py           # MMMAgent, CausalAgent
│   ├── tools.py            # Tool library
│   ├── workflows.py        # DAG pipelines
│   └── automation.py       # AutoMMM, triggers, monitors
│
├── economics/              # Economic Modeling
│   ├── io_model.py         # Leontief I-O
│   ├── multipliers.py      # Economic multipliers
│   └── shock_analysis.py   # Shock propagation
│
├── dynamics/               # Complex Systems
│   ├── attractors.py       # Fixed points, chaos
│   ├── bifurcation.py      # Parameter sensitivity
│   └── stability.py        # Lyapunov exponents
│
├── agents/                 # Agent-Based Modeling
│   ├── base.py             # Agent class
│   ├── model.py            # ABM model
│   └── space.py            # Grid, network, continuous
│
├── sensitivity/            # Sensitivity Analysis
│   ├── sobol.py            # Sobol indices
│   └── morris.py           # Morris screening
│
└── viz/                    # Visualization
    ├── dag.py              # Causal graphs
    ├── response.py         # Response curves
    └── decomposition.py    # Waterfall charts
```

---

##  Autonomous Agent Framework

Papilon 2.0 introduces a revolutionary **autonomous agent framework** that can automate your entire MMM workflow:

### Features

- **Natural Language Interface**: Describe tasks in plain English
- **LLM Integration**: Works with Claude, GPT-4, or local models
- **Tool-Based Architecture**: Extensible with custom tools
- **Workflow Orchestration**: DAG-based pipeline execution
- **Event-Driven Automation**: Triggers, monitors, and alerts
- **Human-in-the-Loop**: Approval steps when needed

### Agent Types

| Agent | Purpose |
|-------|---------|
| `MMMAgent` | Marketing Mix Modeling automation |
| `CausalAgent` | Causal discovery and inference |
| `AgentOrchestrator` | Multi-agent coordination |

### Automation Components

| Component | Purpose |
|-----------|---------|
| `Workflow` | DAG-based pipeline definition |
| `AutoMMM` | Fully automated MMM system |
| `DataTrigger` | Retrain on new data |
| `PerformanceMonitor` | Track model metrics |
| `SlackNotifier` | Send alerts to Slack |

---

##  Use Cases

### 1. Media Planning & Budget Optimization
```python
# Optimize $10M annual budget across channels
optimal = model.optimize_budget(
    total=10_000_000,
    constraints={
        "tv": (1_000_000, 5_000_000),
        "digital": (2_000_000, 6_000_000),
    }
)
```

### 2. Incrementality Testing
```python
# Analyze geo-lift experiment
effect = pp.difference_in_differences(
    data=geo_data,
    treatment_group="test_markets",
    control_group="holdout_markets",
    treatment_time="2024-06-01"
)
```

### 3. Scenario Planning
```python
# What-if analysis
scenarios = [
    {"name": "Cut TV 20%", "changes": {"tv": -0.2}},
    {"name": "Double Digital", "changes": {"digital": 1.0}},
]

for scenario in scenarios:
    impact = agent.scenario(scenario["name"], scenario["changes"])
    print(f"{scenario['name']}: {impact['total_revenue_change_pct']:.1f}%")
```

### 4. Economic Impact Analysis
```python
# Input-Output modeling
io = pp.InputOutputModel(transaction_matrix)
multipliers = io.calculate_multipliers()
shock_impact = io.shock_analysis(sector="advertising", shock=-0.1)
```

---

##  Configuration

### Environment Variables

```bash
# LLM API Keys (for agent features)
export ANTHROPIC_API_KEY="sk-..."
export OPENAI_API_KEY="sk-..."

# Optional: GPU acceleration
export JAX_PLATFORM_NAME="gpu"
```

### Model Configuration

```python
model = pp.BayesianMMM(
    # Adstock configuration
    adstock_type="weibull",  # geometric, weibull, delayed
    adstock_max_lag=8,
    
    # Saturation configuration
    saturation_type="hill",  # hill, logistic, tanh
    
    # Priors
    roi_prior="informative",  # informative, weakly_informative, flat
    
    # Inference
    n_samples=2000,
    n_chains=4,
    target_accept=0.9,
)
```

---

##  Performance

- **GPU Acceleration**: JAX/NumPyro backend for 10-100x speedup
- **Parallel Workflows**: Execute independent steps concurrently
- **Incremental Updates**: Update models with new data efficiently
- **Memory Efficient**: Streaming for large datasets

---

##  Enterprise Features

- **Audit Logging**: Track all model changes and decisions
- **Role-Based Access**: Control who can modify models
- **Deployment Ready**: Docker, Kubernetes support
- **API Server**: REST API for integration

---

##  Documentation

- [Full Documentation](https://papilon.ai/docs)
- [API Reference](https://papilon.ai/api)
- [Tutorials](https://papilon.ai/tutorials)
- [Examples](./examples/)

---

##  Support

- **Enterprise Support**: support@vector1.ai
- **Issues**: GitHub Issues (private repo)
- **Consulting**: Available for custom implementations

---

##  License

**Proprietary License** - All rights reserved.

Copyright © 2024-2025 Brian Curry / Vector1 Research

This software is proprietary and confidential. Unauthorized copying, distribution, or use is strictly prohibited. See [LICENSE](./LICENSE) for full terms.

---

##  Acknowledgments

Built with insights from:
- Google Meridian
- PyMC-Marketing
- DoWhy / EconML / CausalML
- Mesa ABM Framework
- SALib

---

<p align="center">
  <strong>Papilon</strong> — Understanding the butterfly effects in your marketing
</p>
