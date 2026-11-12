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
"""Saving and loading fitted models.

A saved model is one zip file with three parts: ``model.json`` (settings, model
family, fit diagnostics, format version), ``data.csv`` (the data it was fitted to)
and ``draws.npz`` (the posterior draws). Nothing is pickled, so loading a file
never runs code from it, and the parts can be read without Papilon.
"""
from __future__ import annotations

import io
import json
import zipfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["save_model", "load_model", "FORMAT_VERSION"]

FORMAT_VERSION = 1


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(x))
    return x


def save_model(model, path) -> None:
    """Write a fitted :class:`papilon.MMM` or :class:`papilon.GeoMMM` to ``path``."""
    from . import __version__
    model._require_fit()
    kind = type(model).__name__
    meta = {"format_version": FORMAT_VERSION, "papilon_version": __version__, "kind": kind,
            "config": _jsonable(model._config), "family": model._fam.to_dict(),
            "gate_results": {k: _jsonable(asdict(v)) for k, v in model.gate_results.items()},
            "family_probabilities": _jsonable(model.family_probabilities), "seed": model.seed,
            "date_col": model.date_col}
    if kind == "GeoMMM":
        meta["geo_col"] = model.geo_col
    buf = io.BytesIO()
    np.savez_compressed(buf, **{k: v for k, v in model._draws.items() if isinstance(v, np.ndarray)})
    with zipfile.ZipFile(Path(path), "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("model.json", json.dumps(meta, indent=2))
        z.writestr("data.csv", model.data.to_csv(index=False))
        z.writestr("draws.npz", buf.getvalue())


def load_model(path):
    """Read a model written by :func:`save_model`. No refit is needed or done."""
    from .mmm.model import Family, GateResult, MMM
    with zipfile.ZipFile(Path(path)) as z:
        meta = json.loads(z.read("model.json"))
        if meta.get("format_version", 0) > FORMAT_VERSION:
            raise ValueError(f"file format {meta['format_version']} is newer than this Papilon understands; upgrade")
        data = pd.read_csv(io.BytesIO(z.read("data.csv")))
        npz = np.load(io.BytesIO(z.read("draws.npz")))
        draws = {k: npz[k] for k in npz.files}
    date_col = meta["date_col"]
    data[date_col] = pd.to_datetime(data[date_col])
    cfg = dict(meta["config"])
    for k in ("alpha_prior",):
        if isinstance(cfg.get(k), list):
            cfg[k] = tuple(cfg[k])
    if isinstance(cfg.get("alpha_prior"), dict):
        cfg["alpha_prior"] = {c: tuple(v) for c, v in cfg["alpha_prior"].items()}
    if cfg.get("roi_priors"):
        cfg["roi_priors"] = {c: tuple(v) for c, v in cfg["roi_priors"].items()}
    if cfg.get("reach_frequency"):
        cfg["reach_frequency"] = {c: tuple(v) for c, v in cfg["reach_frequency"].items()}
    fam = meta["family"]
    if meta["kind"] == "GeoMMM":
        from .mmm.geo import GeoMMM
        model = GeoMMM(data, carryover=fam["carryover"], saturation=fam["saturation"], seed=meta["seed"], **cfg)
    else:
        model = MMM(data, carryover=fam["carryover"], saturation=fam["saturation"],
                    time_varying=fam["time_varying"], seed=meta["seed"], **cfg)
    model._draws = draws
    model._fam = Family(model.channels, fam["carryover"], fam["saturation"], fam.get("time_varying"))
    model.gate_results = {k: GateResult(**{**v, "history": [tuple(h) for h in v["history"]],
                                           "pareto_k": float("nan") if v.get("pareto_k") is None else v["pareto_k"]})
                          for k, v in meta["gate_results"].items()}
    model.family_probabilities = meta["family_probabilities"]
    model._roi_cache = model._roas_cache = None
    return model
