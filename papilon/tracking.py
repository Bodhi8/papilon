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
"""Experiment tracking: log a fitted model to MLflow.

``log_to_mlflow`` records the settings, fit diagnostics and returns of a fitted model
as one MLflow run, and stores the saved model file and the returns table as
artifacts, so model refreshes can be compared and audited. MLflow is optional and
imported only when this is called.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

__all__ = ["log_to_mlflow"]


def log_to_mlflow(model, run_name: str | None = None, experiment: str | None = None,
                  tags: dict | None = None, log_model_file: bool = True) -> str:
    """Log ``model`` as an MLflow run and return the run id. Needs ``mlflow``."""
    try:
        import mlflow
    except ImportError as e:
        raise ImportError('pip install mlflow to log models to MLflow') from e
    model._require_fit()
    if experiment:
        mlflow.set_experiment(experiment)
    best = max(model.family_probabilities, key=model.family_probabilities.get)
    gate = model.gate_results[best]
    with mlflow.start_run(run_name=run_name) as run:
        from . import __version__
        mlflow.set_tags({"papilon_version": __version__, "model_kind": type(model).__name__, **(tags or {})})
        mlflow.log_params({"family": best, "target": model.target, "media": ",".join(model.media),
                           "periods": len(model.data), "l_max": model.l_max, "seed": model.seed})
        metrics = {"log_evidence": gate.log_evidence, "ess_fraction": gate.ess, "gate_passed": float(gate.passed)}
        if gate.method == "nuts":
            metrics.update(rhat=gate.rhat, ess_bulk=gate.ess_bulk, divergences=float(gate.divergences))
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v is not None and v == v})
        roi = model.get_roi()
        for c, r in roi.iterrows():
            mlflow.log_metrics({f"roas_{c}": r.roas_mean, f"roas_p05_{c}": r.roas_p05, f"roas_p95_{c}": r.roas_p95})
        with tempfile.TemporaryDirectory() as tmp:
            roi_path = Path(tmp) / "roi.csv"
            roi.to_csv(roi_path)
            mlflow.log_artifact(str(roi_path))
            if log_model_file:
                path = Path(tmp) / "model.papilon"
                model.save(path)
                mlflow.log_artifact(str(path))
        return run.info.run_id
