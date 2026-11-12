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
"""Convergence diagnostics for Markov chains, in NumPy.

``rhat`` is the split R-hat on rank-normalized draws (the larger of the bulk and the
folded version, so it reacts to chains that disagree in location or in spread).
``ess_bulk`` is the effective sample size of the rank-normalized split chains, with the
autocorrelation sum truncated by Geyer's initial monotone sequence. Both take draws of
shape (chains, draws, parameters) and return one value per parameter.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

__all__ = ["rhat", "ess_bulk", "summarize", "RHAT_MAX", "ESS_PER_CHAIN"]

RHAT_MAX = 1.01        # largest R-hat a fit may have and still pass
ESS_PER_CHAIN = 100.0  # smallest bulk effective sample size per chain a fit may have and still pass


def _split(x: np.ndarray) -> np.ndarray:
    n = x.shape[1] // 2
    return np.concatenate([x[:, :n], x[:, x.shape[1] - n:]], axis=0)


def _rank_normalize(x: np.ndarray) -> np.ndarray:
    m, n, d = x.shape
    r = stats.rankdata(x.reshape(m * n, d), axis=0)
    return stats.norm.ppf((r - 0.375) / (m * n + 0.25)).reshape(m, n, d)


def _plain_rhat(x: np.ndarray) -> np.ndarray:
    n = x.shape[1]
    w = x.var(axis=1, ddof=1).mean(axis=0)
    b = n * x.mean(axis=1).var(axis=0, ddof=1)
    with np.errstate(all="ignore"):
        return np.sqrt(((n - 1) / n * w + b / n) / w)


def _as3d(draws) -> np.ndarray:
    x = np.asarray(draws, dtype=float)
    if x.ndim == 2:
        x = x[:, :, None]
    if x.ndim != 3:
        raise ValueError("draws must have shape (chains, draws, parameters)")
    return x


def rhat(draws) -> np.ndarray:
    """Split R-hat per parameter, rank-normalized; NaN with fewer than four draws per chain."""
    x = _as3d(draws)
    if x.shape[1] < 4:
        return np.full(x.shape[2], np.nan)
    s = _split(x)
    folded = np.abs(s - np.median(s, axis=(0, 1), keepdims=True))
    return np.fmax(_plain_rhat(_rank_normalize(s)), _plain_rhat(_rank_normalize(folded)))


def ess_bulk(draws) -> np.ndarray:
    """Bulk effective sample size per parameter; NaN with fewer than four draws per chain."""
    x = _as3d(draws)
    if x.shape[1] < 4:
        return np.full(x.shape[2], np.nan)
    z = _rank_normalize(_split(x))
    m, n, d = z.shape
    zc = z - z.mean(axis=1, keepdims=True)
    size = int(2 ** np.ceil(np.log2(2 * n)))
    f = np.fft.rfft(zc, n=size, axis=1)
    acov = np.fft.irfft(f * np.conj(f), n=size, axis=1)[:, :n] / n            # (m, n, d)
    w = (acov[:, 0] * n / (n - 1)).mean(axis=0)
    var_plus = w * (n - 1) / n + (z.mean(axis=1).var(axis=0, ddof=1) if m > 1 else 0.0)
    with np.errstate(all="ignore"):
        rho = 1.0 - (w[None, :] - acov.mean(axis=0)) / var_plus[None, :]      # (n, d)
    rho[0] = 1.0
    k = n // 2
    pairs = rho[0:2 * k:2] + rho[1:2 * k:2]                                   # (k, d)
    keep = np.cumprod(pairs > 0, axis=0).astype(bool)
    pairs = np.where(keep, np.minimum.accumulate(np.where(keep, pairs, np.inf), axis=0), 0.0)
    tau = -1.0 + 2.0 * pairs.sum(axis=0)
    total = m * n
    with np.errstate(all="ignore"):
        out = np.minimum(total / tau, total * np.log10(total))
    return np.where(np.isfinite(out) & (tau > 0), out, np.nan)


def summarize(draws, divergences: int = 0) -> dict:
    """Worst R-hat and smallest bulk effective sample size over parameters, and whether the fit passes.

    A fit passes with every R-hat below 1.01, at least 100 effective draws per chain for
    every parameter (400 at the default four chains) and no divergent transitions. One
    chain cannot show that chains agree, so it does not pass.
    """
    x = _as3d(draws)
    chains = x.shape[0]
    r, e = rhat(x), ess_bulk(x)
    worst = float(np.nanmax(r)) if np.any(np.isfinite(r)) else float("nan")
    least = float(np.nanmin(e)) if np.any(np.isfinite(e)) else float("nan")
    ok = bool(chains > 1 and np.all(np.isfinite(r)) and worst < RHAT_MAX and np.all(np.isfinite(e))
              and least >= ESS_PER_CHAIN * chains and int(divergences) == 0)
    return {"rhat": worst, "ess_bulk": least, "divergences": int(divergences), "passed": ok}
